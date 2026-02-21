# TODO qui ci si puo' lavorare molto --> applicare re-ranking al posto di contare le keywords. 
# Poi questo considera solo il VSME quindi solo la richiesta dell'utente per stilare un report

class SectorClassifier:
    """
    Classify the sector of the company by matching top-k keywords
    """
    
    def __init__(self):
        self.keywords = {
            'manufacturing': {
                'keywords': ['production', 'factory', 'craftsmanship', 'machinery', 
                           'wood', 'furniture', 'metal', 'plastic', 'processing'],
                'VSME_metrics': ['B3', 'B7', 'B9'],
                'hint': 'Focus on the energy efficiency of machinery and the management of production waste'
            },
            'food': {
                'keywords': ['food', 'restaurant', 'food', 'cooking', 'bakery', 'bar',
                           'pastry shop', 'groceries', 'catering'],
                'VSME_metrics': ['B6', 'B7', 'B3', 'B9'],
                'hint': 'Monitor water consumption, organic waste management, and food safety'
            },
            'services': {
                'keywords': ['software', 'consulting', 'office', 'IT', 'computing',
                           'accounting', 'services', 'digital'],
                'VSME_metrics': ['B3', 'B10', 'C9'],
                'hint': 'Focus on office energy efficiency and employee well-being'
            },
            'construction': {
                'keywords': ['building', 'construction site', 'construction', 'renovation',
                                    'real estate', 'masonry'],
                'VSME_metrics': ['B7', 'B3', 'B9', 'B5'],
                            'hint': 'Management of construction materials, construction waste, and construction site safety'
            },
            'commerce': {
                'keywords': ['shop', 'sales', 'retail', 'commerce', 'distribution'],
                'VSME_metrics': ['B3', 'B7'],
                'hint': 'Point of sale energy efficiency and packaging management'
            }
        }
    
    def classify(self, company_description):
        """
        Detect the sector of the company from the description by keywords counting.
        Check if sectoral-keywords appear in the company's description
        """
        description_lower = company_description.lower()
        
        # Match count by sector
        scores = {}
        for sector, data in self.keywords.items():
            score = sum(1 for kw in data['keywords'] if kw in description_lower)
            scores[sector] = score
        
        # Sector with the highest score
        best_sector = max(scores, key=scores.get)
        
        # generic if no match detected
        if scores[best_sector] == 0:
            return 'generic', None
        
        return best_sector, self.keywords[best_sector]

# Test
if __name__ == "__main__":
    classifier = SectorClassifier()
    
    tests = [
        "We are a carpentry workshop that produces custom-made furniture",
        "We run an Italian restaurant with 20 seats",
        "Software house that develops mobile apps"
    ]
    
    for test in tests:
        sector, hints = classifier.classify(test)
        print(f"\nDescription: {test}")
        print(f"Sector: {sector}")
        if hints:
            print(f"Hint: {hints['hint']}")