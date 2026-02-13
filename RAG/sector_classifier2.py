import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SectorClassifier:
    """
    Classifies the company sector using Semantic Similarity and maps it to 
    VSME Standard specific requirements (High Impact NACE, Pollution applicability, etc.).
    """
    
    def __init__(self, embedding_model):
       
        self.model = embedding_model

        # VSME-based
        self.knowledge_base = {
            'High Impact Manufacturing (NACE C)': {
                'definition': 'Industrial manufacturing, chemical production, metal processing, heavy industry, factory production, packaging.',
                'profile': {
                    'VSME_Sector_Type': 'High Climate Impact (NACE C)',
                    'Priority_Modules': ['Basic (B1-B11)', 'Pollution (B4)', 'Resource Use (B7)', 'Scope 3 (C2)'],
                    'Key_Metrics': ['B3 (Energy & GHG)', 'B4 (Pollution to Air/Water/Soil)', 'B7 (Mass-flow of materials)', 'B9 (Health & Safety)'],
                    'Hint': 'Critical focus on Pollution (B4) and Material Mass-Flow (B7). Transition plan (C3) is strongly recommended for high impact sectors.'
                }
            },
            'Construction & Real Estate (NACE F, L)': {
                'definition': 'Building construction, renovation, demolition, real estate activities, infrastructure, site management.',
                'profile': {
                    'VSME_Sector_Type': 'High Climate Impact (NACE F/L)',
                    'Priority_Modules': ['Basic (B1-B11)', 'Resource Use (B7)'],
                    'Key_Metrics': ['B1 (Gen. Info)', 'B7 (Construction waste & materials)', 'B9 (Accidents rate)', 'B5 (Land use/Sealing)'],
                    'Hint': 'Focus on B7 (Waste diverted from disposal) and B5 (Soil sealing/Land use). High accident risk sector (B9).'
                }
            },
            'Agriculture & Food (NACE A)': {
                'definition': 'Farming, livestock, crops, food processing, fisheries, forestry, beverage production.',
                'profile': {
                    'VSME_Sector_Type': 'High Climate Impact (NACE A)',
                    'Priority_Modules': ['Basic', 'Pollution (B4)', 'Biodiversity (B5)'],
                    'Key_Metrics': ['B4 (Pesticides/Nutrients)', 'B5 (Biodiversity sensitive areas)', 'B6 (Water withdrawal in stress areas)'],
                    'Hint': 'Crucial reporting on Water (B6) and Pollution (B4 - Nitrogen/Phosphorus). Check if operations are near biodiversity sensitive areas (B5).'
                }
            },
            'Transport & Storage (NACE H)': {
                'definition': 'Logistics, trucking, shipping, warehousing, delivery fleets, freight transport.',
                'profile': {
                    'VSME_Sector_Type': 'High Climate Impact (NACE H)',
                    'Priority_Modules': ['Basic', 'GHG Scope 1'],
                    'Key_Metrics': ['B3 (Fuel consumption & Scope 1)', 'B9 (Workforce safety)', 'B4 (Air pollutants: NOx, SOx)'],
                    'Hint': 'Main impact is Energy/GHG (B3). Focus on fleet electrification and air pollutants (B4) from combustion.'
                }
            },
            'Services & Office-based (Generic)': {
                'definition': 'Consulting, IT, software, legal, accounting, education, marketing, administrative, retail shops.',
                'profile': {
                    'VSME_Sector_Type': 'Low Environmental Impact',
                    'Priority_Modules': ['Basic (Simplified)', 'Social Focus'],
                    'Key_Metrics': ['B1 (General)', 'B3 (Scope 2 - Electricity)', 'B8 (Workforce characteristics)', 'B10 (Pay Gap)'],
                    'Hint': 'Pollution (B4) and Mass-flow (B7) are likely NOT applicable. Focus on Social metrics (B8-B10) and Business Conduct (B11).'
                }
            }
        }
        
        # Pre-compute embeddings
        self.sector_names = list(self.knowledge_base.keys())
        self.definitions = [data['definition'] for data in self.knowledge_base.values()]
        self.doc_embeddings = self.model.encode(self.definitions)

    def classify(self, company_description):
        """
        Returns the best matching sector and its VSME profile.
        """
        # TODO rischioso????
        query_vec = self.model.encode([company_description])
        similarities = cosine_similarity(query_vec, self.doc_embeddings)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_sector = self.sector_names[best_idx]
        
        # Fallback se la confidenza è bassa TODO rischioso????
        if best_score < 0.25:
            return "Unclassified/General", {
                'VSME_Sector_Type': 'General',
                'Priority_Modules': ['Basic'],
                'Key_Metrics': ['B1', 'B2', 'B8'],
                'Hint': 'Apply the General VSME Basic Module principles.'
            }
            
        return best_sector, self.knowledge_base[best_sector]['profile']