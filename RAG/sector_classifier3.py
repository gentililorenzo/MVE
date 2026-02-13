"""
Enhanced SectorClassifier that maps company descriptions (and optional NACE codes / metadata)
to VSME Basic/Comprehensive Module applicability and recommended datapoints.

Design goals:
- Use any `embedding_model` exposing `.encode(list_of_texts)` -> np.ndarray
- Provide deterministic helpers for VSME calculations (GHG intensity, fuel->MWh, kWh->tCO2e)
- Offer structured recommendations (which B# disclosures apply, priority modules, key metrics)
- Provide configurable thresholds and clear docstrings.

References:
- VSME Standard (EFRAG) Basic Module requirements B1-B11 and guidance (B3-B7, Appendix B). See PDF.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import json

# Constants used based on VSME guidance
TJ_TO_MWH = 277.77777777777777  # 1 TJ = 277.777... MWh

# Default densities / net calorific values (approximate typical values used as examples)
# Values are included as defaults but user should pass supplier-specific values when available.
DEFAULT_FUEL_PROPERTIES = {
    # volume unit -> density (kg/l) and NCV (TJ/m3) approximations
    # net calorific values expressed per m3 for liquids/gases where convenient
    "diesel": {"density_kg_per_l": 0.84, "ncv_TJ_per_m3": 0.043},  # example from VSME guidance
    "fuel_oil": {"density_kg_per_l": 0.94, "ncv_TJ_per_m3": 0.03921},
    "natural_gas": {"density_kg_per_m3": 0.8, "ncv_TJ_per_m3": 0.038},  # illustrative
    # add more as needed
}

# High climate impact NACE sections (VSME: A to H and L). See VSME commentary on "high climate impact".
HIGH_CLIMATE_NACE_SECTIONS = set(list("ABCDEFGHL"))  # letters A-H plus L (per VSME). Note: 'I' omitted per VSME.


class SectorClassifier:
    """
    Maps company free-text descriptions (and optional NACE codes) to VSME-relevant sector profiles.

    Key methods:
    - classify(description, nace_codes=None) -> best sector match + profile + score
    - recommend_reporting(sector, nace_codes=None, sites_geo=None, employee_count=None)
        -> structured recommended VSME datapoints (which B# apply and suggested metrics)
    - helper conversions: fuel_volume_to_mwh, compute_ghg_intensity, electricity_kwh_to_tco2e

    Notes:
    - The mappings and heuristics are derived from the VSME Standard Basic Module guidance
      (B3-B11) and Appendix B (list of possible sustainability issues). See VSME Standard.
      (refer to provided PDF for full normative text).
    """

    def __init__(self, embedding_model: Any, confidence_threshold: float = 0.25):
        """
        embedding_model: any object with method encode(list[str]) -> np.ndarray (n x d)
        confidence_threshold: min cosine similarity to accept a sector match (fallback -> 'General')
        """
        self.model = embedding_model
        self.confidence_threshold = float(confidence_threshold)

        # Knowledge base expanded with more sectors and VSME profiling
        # Each sector entry contains:
        #  - definition: short textual description used for semantic matching
        #  - profile: VSME-specific mapping (VSME_Sector_Type, Priority_Modules, Key_Metrics, Hint, probable_NACE_sections)
        self.knowledge_base: Dict[str, Dict[str, Any]] = {
            "High Impact Manufacturing (NACE C)": {
                "definition": "industrial manufacturing, chemical production, metal processing, heavy industry, factory production, packaging",
                "profile": {
                    "VSME_Sector_Type": "High Climate Impact (NACE C)",
                    "probable_nace_sections": ["C"],
                    "Priority_Modules": ["Basic (B1-B11)", "Pollution (B4)", "Resource Use (B7)", "Scope 3 (C2)"],
                    "Key_Metrics": ["B3 (Energy & GHG)", "B4 (Pollution to air/water/soil)", "B7 (Mass-flow of materials)", "B9 (Health & Safety)"],
                    "Hint": "High likelihood to report B3, B4, B7 and consider C3 transition plan if available."
                }
            },
            "Construction & Real Estate (NACE F, L)": {
                "definition": "building construction, renovation, demolition, real estate activities, infrastructure, site management",
                "profile": {
                    "VSME_Sector_Type": "High Climate Impact (NACE F/L)",
                    "probable_nace_sections": ["F", "L"],
                    "Priority_Modules": ["Basic (B1-B11)", "Resource Use (B7)"],
                    "Key_Metrics": ["B1 (Gen. Info)", "B7 (Construction waste & materials)", "B9 (Accident rate)", "B5 (Land use/Sealing)"],
                    "Hint": "Report mass flows for construction materials and land-use (B5)."
                }
            },
            "Agriculture & Food (NACE A)": {
                "definition": "farming, livestock, crops, food processing, fisheries, forestry, beverage production",
                "profile": {
                    "VSME_Sector_Type": "High Climate Impact (NACE A)",
                    "probable_nace_sections": ["A"],
                    "Priority_Modules": ["Basic", "Pollution (B4)", "Biodiversity (B5)", "Water (B6)"],
                    "Key_Metrics": ["B4 (Pesticides/Nutrients)", "B5 (Biodiversity sensitive areas)", "B6 (Water withdrawal)"],
                    "Hint": "High relevance for water, biodiversity and pollution metrics."
                }
            },
            "Transport & Logistics (NACE H)": {
                "definition": "road, rail, sea, air freight, warehousing, delivery fleets, logistics services",
                "profile": {
                    "VSME_Sector_Type": "High Climate Impact (NACE H)",
                    "probable_nace_sections": ["H"],
                    "Priority_Modules": ["Basic (B3 focus)", "GHG Scope 1"],
                    "Key_Metrics": ["B3 (Fuel consumption & Scope 1)", "B4 (Air pollutants: NOx, SOx)", "B9 (Workforce Safety)"],
                    "Hint": "Fleet emissions and pollutant emissions often material (report Scope 1 and specific pollutants)."
                }
            },
            "Services & Offices (Generic)": {
                "definition": "consulting, it services, software, legal, accounting, education, marketing, retail shops, office-based services",
                "profile": {
                    "VSME_Sector_Type": "Low Environmental Impact / Service",
                    "probable_nace_sections": ["J", "K", "M", "N"],  # business services / professional activities heuristics
                    "Priority_Modules": ["Basic (Simplified)", "Social (B8-B10)"],
                    "Key_Metrics": ["B1 (General)", "B3 (Scope 2 electricity)", "B8 (Workforce)", "B10 (Remuneration)"],
                    "Hint": "Pollution and mass-flow metrics typically not applicable; focus on social and governance."
                }
            },
            "Retail & Wholesale (NACE G)": {
                "definition": "retail stores, wholesale trade, e-commerce, distribution of goods to consumers or businesses",
                "profile": {
                    "VSME_Sector_Type": "Retail/Wholesale (NACE G)",
                    "probable_nace_sections": ["G"],
                    "Priority_Modules": ["Basic (B1-B11)", "Resource Use (B7) if packaging intensive"],
                    "Key_Metrics": ["B3 (Energy & GHG)", "B7 (Packaging mass flows)", "B8 (Workforce)"],
                    "Hint": "Packaging and product-related impacts may trigger B7."
                }
            },
            "Energy Utilities & Extraction (NACE B, D, E)": {
                "definition": "mining, quarrying, energy production, electricity utilities and extraction activities",
                "profile": {
                    "VSME_Sector_Type": "High Climate Impact (NACE B/D/E)",
                    "probable_nace_sections": ["B", "D", "E"],
                    "Priority_Modules": ["Basic (B3-B4)", "Comprehensive (C3 for transition plans)"],
                    "Key_Metrics": ["B3 (Scope 1 & 2)", "B4 (Pollutants)", "B7 (Material flows for extraction)"],
                    "Hint": "Likely to have high Scope 1 emissions and pollution reporting obligations."
                }
            },
            # Add other sectors as necessary...
        }

        # Precompute embeddings for sector definitions to speed up classification
        self.sector_names = list(self.knowledge_base.keys())
        self.definitions = [self.knowledge_base[s]["definition"] for s in self.sector_names]

        # If encoding fails, raise informative error
        try:
            self.doc_embeddings = self.model.encode(self.definitions)
            # ensure shape consistency
            if isinstance(self.doc_embeddings, list):
                self.doc_embeddings = np.array(self.doc_embeddings)
        except Exception as e:
            raise RuntimeError("Embedding model.encode failed during initialization: " + str(e))

    def classify(self, company_description: str, nace_codes: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any], float]:
        """
        Returns (best_sector_label, profile_dict, score).
        If score < confidence_threshold, returns 'Unclassified/General' fallback profile.

        nace_codes: optional list of reported NACE codes (e.g. ['C10', 'C11']) — used to boost matches
        """
        if not company_description or not company_description.strip():
            raise ValueError("company_description must be a non-empty string.")

        try:
            query_vec = self.model.encode([company_description])
            if isinstance(query_vec, list):
                query_vec = np.array(query_vec)
        except Exception as e:
            raise RuntimeError("Embedding model.encode failed for the query: " + str(e))

        # cosine similarity between query and each sector definition
        similarities = cosine_similarity(query_vec, self.doc_embeddings)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_sector = self.sector_names[best_idx]
        profile = self.knowledge_base[best_sector]["profile"].copy()

        # If NACE codes provided, attempt to bump sector if NACE section matches
        if nace_codes:
            normalized = [c.strip().upper() for c in nace_codes if c]
            # extract leading letter from codes like 'C10' -> 'C'
            sections = {c[0] for c in normalized if len(c) > 0 and c[0].isalpha()}
            probable_sections = set(profile.get("probable_nace_sections", []))
            if sections & probable_sections:
                # boost score slightly to reflect authoritative NACE input
                best_score = max(best_score, 0.9)

        # Fallback when confidence low
        if best_score < self.confidence_threshold:
            fallback = {
                "VSME_Sector_Type": "General",
                "Priority_Modules": ["Basic"],
                "Key_Metrics": ["B1", "B2", "B8"],
                "Hint": "Apply Basic Module; evaluate applicability of other disclosures per operations."
            }
            return "Unclassified/General", fallback, best_score

        return best_sector, profile, best_score

    def recommend_reporting(self, sector_label: str, nace_codes: Optional[List[str]] = None,
                             sites_geo: Optional[List[Tuple[float, float]]] = None,
                             employee_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Based on a sector_label (as returned by classify) and optional metadata, recommend which
        VSME Basic disclosures are likely applicable, and list the exact datapoints to gather.

        Returns example structure:
        {
            'sector_label': 'High Impact Manufacturing (NACE C)',
            'apply_B3': True,
            'apply_B4': True,
            'apply_B5': False,
            'recommended_datapoints': {
                'B3': ['total_energy_MWh', 'electricity_renewable_MWh', 'scope1_tCO2e', 'scope2_location_tCO2e'],
                ...
            }
        }
        """
        profile = self.knowledge_base.get(sector_label, {}).get("profile", None)
        rec = {
            "sector_label": sector_label,
            "recommended_modules": [],
            "apply_B3": False,
            "apply_B4": False,
            "apply_B5": False,
            "apply_B6": False,
            "apply_B7": False,
            "apply_B8_to_B10": True,  # workforce metrics often relevant
            "apply_B11": True,  # governance rarely irrelevant
            "recommended_datapoints": {}
        }

        # Determine likely applicability using sector heuristics and optional NACE codes
        def maybe_apply_pollution(sector_profile):
            # sectors with manufacturing, energy, agriculture, transport -> pollution
            label = sector_profile.get("VSME_Sector_Type", "").lower()
            if "manufactur" in label or "energy" in label or "agric" in label or "transport" in label:
                return True
            if nace_codes:
                # if NACE section in high-pollution groups
                sections = {c[0].upper() for c in nace_codes if c}
                if sections & {"A", "B", "C", "D", "E", "F", "H"}:
                    return True
            return False

        if profile:
            # B3 (Energy & GHG) applies widely but is mandatory mainly for energy-consuming activities
            rec["apply_B3"] = True  # VSME: B3 is a core Basic Module metric. See B3 in VSME.
            rec["recommended_modules"].append("B3")

            rec["apply_B4"] = maybe_apply_pollution(profile)
            rec["apply_B7"] = any("manufact" in x.lower() or "construction" in x.lower() or
                                   "packaging" in x.lower() for x in [profile.get("VSME_Sector_Type", "")])
            # B5 biodiversity if sites_geo present or if agriculture / forestry / near sensitive areas
            rec["apply_B5"] = False
            if sites_geo:
                # if sites provided, request user to check proximity to biodiversity sensitive areas
                rec["apply_B5"] = True

            # B6 if agriculture, food processing, energy, or if site in water-stress area
            rec["apply_B6"] = any(s in ["A", "C", "E", "F"] for s in (n[0].upper() for n in (nace_codes or []))) if nace_codes else False
            # always recommend collecting workforce and governance datapoints (B8-B11)
            rec["apply_B8_to_B10"] = True
            rec["apply_B11"] = True

            # Recommended datapoints mapping
            rec["recommended_datapoints"] = {
                "B1": ["legal_form", "nace_codes", "balance_sheet_eur", "turnover_eur", "employees_headcount_or_FTE", "primary_country", "geolocations"],
                "B2": ["policies_public_url", "practices_list", "targets_list"],
                "B3": ["energy_total_MWh", "electricity_renewable_MWh", "fuel_MWh", "scope1_tCO2e", "scope2_location_tCO2e", "ghg_intensity_tCO2e_per_eur"],
                "B4": ["pollutant_list_with_amounts_and_medium"],  # e.g., {'NOx': {'air': kg, 'water': None}}
                "B5": ["sites_in_biodiversity_sensitive_areas_count", "area_hectares"],
                "B6": ["water_withdrawal_m3", "water_withdrawal_in_high_stress_m3", "water_consumption_m3"],
                "B7": ["waste_non_hazardous_t", "waste_hazardous_t", "waste_diverted_recycled_t", "material_mass_flows_t"],
                "B8": ["contract_type_breakdown", "gender_breakdown", "country_of_contract"],
                "B9": ["recordable_accidents_number", "accident_rate", "work_related_fatalities"],
                "B10": ["minimum_wage_compliance_bool", "gender_pay_gap_pct", "collective_bargaining_coverage_pct", "avg_training_hours_per_employee_by_gender"],
                "B11": ["convictions_number", "total_fines_eur"]
            }

        else:
            # Unknown sector -> suggest basic datapoints only
            rec["apply_B3"] = True
            rec["recommended_modules"].append("Basic (B1-B11 minimal)")
            rec["recommended_datapoints"] = {
                "B1": ["legal_form", "nace_codes", "turnover_eur", "employees_headcount"]
            }

        return rec

    # -------------------------------
    # Helper functions for computations recommended by VSME guidance (B3 etc.)
    # -------------------------------

    @staticmethod
    def compute_ghg_intensity(total_ghg_tco2eq: float, turnover_eur: float) -> Optional[float]:
        """
        Compute GHG intensity = total GHG (tCO2e) / turnover (EUR) as required by B3 paragraph 31.
        Returns None if turnover is zero or invalid.
        """
        try:
            total = float(total_ghg_tco2eq)
            turnover = float(turnover_eur)
            if turnover <= 0:
                return None
            return total / turnover
        except Exception:
            return None

    @staticmethod
    def fuel_volume_to_mwh(volume: float, volume_unit: str = "m3", fuel: str = "fuel_oil",
                          fuel_props: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """
        Convert a fuel volume (e.g. in m3 or liters) to MWh using Net Calorific Value (NCV).
        - volume: numeric volume
        - volume_unit: 'm3' or 'l' (liters)
        - fuel: key for DEFAULT_FUEL_PROPERTIES (e.g. 'diesel', 'fuel_oil')
        - fuel_props: optional override dict similar structure to DEFAULT_FUEL_PROPERTIES

        Formula (as in VSME guidance):
        energy_TJ = volume_in_m3 * ncv_TJ_per_m3
        energy_MWh = energy_TJ * TJ_TO_MWH

        Note: If user passes volume in liters, it is converted to m3 (1 m3 = 1000 liters).
        """
        props = fuel_props.get(fuel) if fuel_props else DEFAULT_FUEL_PROPERTIES.get(fuel)
        if not props:
            raise ValueError(f"Unknown fuel '{fuel}' and no fuel_props provided.")
        ncv_TJ_per_m3 = props.get("ncv_TJ_per_m3")
        if ncv_TJ_per_m3 is None:
            raise ValueError("Missing ncv_TJ_per_m3 for fuel properties.")

        # normalize volume to m3
        if volume_unit == "l" or volume_unit == "litre" or volume_unit == "liter":
            volume_m3 = float(volume) / 1000.0
        elif volume_unit == "m3":
            volume_m3 = float(volume)
        else:
            raise ValueError("Unsupported volume_unit. Use 'm3' or 'l'.")

        energy_TJ = volume_m3 * float(ncv_TJ_per_m3)
        energy_MWh = energy_TJ * TJ_TO_MWH
        return float(energy_MWh)

    @staticmethod
    def electricity_kwh_to_tco2e(electricity_kwh: float, emission_factor_g_per_kwh: float) -> float:
        """
        Convert electricity consumption in kWh to tCO2e using a location-based emission factor
        (g CO2e / kWh). Example: 73 g/kWh -> 0.073 kg/kWh -> 0.000073 t/kWh.
        """
        kg = (float(electricity_kwh) * float(emission_factor_g_per_kwh)) / 1000.0  # kg CO2e
        tco2e = kg / 1000.0
        return float(tco2e)

    @staticmethod
    def sum_scope_emissions(scope1: float, scope2: float, scope3: Optional[float] = None) -> float:
        """
        Return sum of disclosed scopes (scope3 may be None).
        """
        total = float(scope1) + float(scope2)
        if scope3 is not None:
            total += float(scope3)
        return float(total)

    # Export helper for structured reporting (JSON)
    @staticmethod
    def export_recommendation_json(recommendation: Dict[str, Any]) -> str:
        """
        Return prettified JSON string of the recommendation structure for saving or API use.
        """
        return json.dumps(recommendation, indent=2, ensure_ascii=False)


# -----------------------------
# Example usage (to be removed when integrating into production)
# -----------------------------
if __name__ == "__main__":
    # Example: embedding_model stub (user should inject real model, e.g. sentence-transformers SentenceTransformer)
    class DummyEmbedder:
        def encode(self, texts):
            # naive: hash-based deterministic vector (only for example)
            out = []
            for t in texts:
                h = abs(hash(t)) % 1000
                vec = np.array([math.sin(h), math.cos(h), (h % 17) / 17.0])
                out.append(vec)
            return np.vstack(out)

    embedder = DummyEmbedder()
    clf = SectorClassifier(embedder)

    desc = "We operate a medium-sized bakery and produce packaged confectionery, use diesel vans for deliveries."
    sector, profile, score = clf.classify(desc, nace_codes=["C10", "G47"])
    print("Sector:", sector)
    print("Profile:", profile)
    print("Score:", score)

    rec = clf.recommend_reporting(sector_label=sector, nace_codes=["C10"], sites_geo=[(45.4642, 9.1900)], employee_count=60)
    print(clf.export_recommendation_json(rec))

    # Example: convert diesel volume to MWh (if we have 10000 liters of diesel)
    mwh = clf.fuel_volume_to_mwh(10000, volume_unit="l", fuel="diesel")
    print(f"10000 L diesel ≈ {mwh:.2f} MWh (example conversion)")

    # Example: compute GHG intensity (tCO2e per EUR)
    intensity = clf.compute_ghg_intensity(total_ghg_tco2eq=301.5, turnover_eur=1_000_000)
    print("GHG intensity (tCO2e / EUR):", intensity)
