# TODO qui ci si puo' lavorare molto

class SectorClassifier:
    """Classifica automaticamente il settore aziendale"""
    
    def __init__(self):
        self.keywords = {
            'manifatturiero': {
                'keywords': ['produzione', 'fabbrica', 'artigianato', 'macchinari', 
                           'legno', 'mobili', 'metallo', 'plastica', 'lavorazione'],
                'metriche_focus': ['B3', 'B7', 'B9'],
                'hint': 'Concentrati su efficienza energetica dei macchinari e gestione scarti produzione'
            },
            'alimentare': {
                'keywords': ['cibo', 'ristorante', 'food', 'cucina', 'panificio', 
                           'pasticceria', 'alimentari', 'ristorazione'],
                'metriche_focus': ['B6', 'B7', 'B3', 'B9'],
                'hint': 'Monitora consumo idrico, gestione rifiuti organici e sicurezza alimentare'
            },
            'servizi': {
                'keywords': ['software', 'consulenza', 'ufficio', 'IT', 'informatica',
                           'contabilità', 'servizi', 'digitale'],
                'metriche_focus': ['B3', 'B10', 'C9'],
                'hint': 'Focus su efficienza energetica uffici e benessere lavoratori'
            },
            'costruzioni': {
                'keywords': ['edilizia', 'cantiere', 'costruzioni', 'ristrutturazione',
                           'immobiliare', 'muratura'],
                'metriche_focus': ['B7', 'B3', 'B9', 'B5'],
                'hint': 'Gestione materiali da costruzione, rifiuti edili e sicurezza cantiere'
            },
            'commercio': {
                'keywords': ['negozio', 'vendita', 'retail', 'commercio', 'distribuzione'],
                'metriche_focus': ['B3', 'B7'],
                'hint': 'Efficienza energetica punto vendita e gestione imballaggi'
            }
        }
    
    def classify(self, company_description):
        """Classifica il settore dalla descrizione"""
        description_lower = company_description.lower()
        
        # Conta match per settore
        scores = {}
        for sector, data in self.keywords.items():
            score = sum(1 for kw in data['keywords'] if kw in description_lower)
            scores[sector] = score
        
        # Settore con score più alto
        best_sector = max(scores, key=scores.get)
        
        # Se nessun match, generico
        if scores[best_sector] == 0:
            return 'generico', None
        
        return best_sector, self.keywords[best_sector]

# Test
if __name__ == "__main__":
    classifier = SectorClassifier()
    
    tests = [
        "Siamo una falegnameria che produce mobili su misura",
        "Gestiamo un ristorante italiano con 20 coperti",
        "Software house che sviluppa app mobile"
    ]
    
    for test in tests:
        sector, hints = classifier.classify(test)
        print(f"\nDescrizione: {test}")
        print(f"Settore: {sector}")
        if hints:
            print(f"Hint: {hints['hint']}")