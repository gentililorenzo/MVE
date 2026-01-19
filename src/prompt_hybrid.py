from .rag_system import VSMERagSystem
from .sector_classifier import SectorClassifier

class HybridPromptSystem:
    """Sistema RAG con prompt ibrido settoriale"""
    
    def __init__(self):
        self.rag = VSMERagSystem()
        self.classifier = SectorClassifier()
    
    def generate_prompt(self, company_profile, question, context_chunks):
        """Genera prompt ibrido con hint settoriale"""
        
        # Classifica settore
        sector, hints = self.classifier.classify(
            company_profile['attivita']
        )
        
        # Costruisci contesto VSME
        context = "\n\n".join([
            f"[Riferimento VSME #{i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Prompt base con hint settoriale
        prompt = f"""Sei un consulente di sostenibilità esperto dello standard VSME EFRAG per PMI.

PROFILO AZIENDA:
- Settore: {sector}
- Dimensione: {company_profile['dimensione']} dipendenti
- Attività: {company_profile['attivita']}

{'SUGGERIMENTO SETTORIALE:' if hints else ''}
{hints['hint'] if hints else ''}

{'METRICHE PRIORITARIE PER QUESTO SETTORE:' if hints else ''}
{', '.join(hints['metriche_focus']) if hints else ''}

STANDARD VSME RILEVANTI:
{context}

DOMANDA:
{question}

Fornisci una risposta PRATICA e CONCRETA basata sullo standard VSME.

STRUTTURA:
1. 🎯 PRIORITÀ IMMEDIATE (Quick wins)
2. 📊 METRICHE VSME DA MONITORARE (con codici es. B3, B7)
3. 📝 AZIONI CONCRETE (passo per passo)
4. 📄 DOCUMENTI NECESSARI

Sii specifico per il settore {sector}. Non essere generico.

RISPOSTA:"""
        
        return prompt
    
    def consult(self, company_profile, question):
        """Consulenza completa"""
        print(f"\n{'='*60}")
        print(f"🏢 Azienda: {company_profile['attivita']}")
        print(f"👥 Dipendenti: {company_profile['dimensione']}")
        print(f"❓ Domanda: {question}")
        print(f"{'='*60}")
        
        # 1. Retrieval
        print("\n📚 Ricerca informazioni VSME rilevanti...")
        query_enriched = f"{company_profile['attivita']} {question}"
        chunks, _ = self.rag.retrieve(query_enriched, n_results=5)
        
        # 2. Genera prompt ibrido
        print("🎨 Generazione prompt settoriale...")
        prompt = self.generate_prompt(company_profile, question, chunks)
        
        # 3. LLM response
        print("🤖 Elaborazione risposta...")
        import ollama
        response = ollama.chat(
            model='gemma3:4b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']

# Test completo
if __name__ == "__main__":
    system = HybridPromptSystem()
    
    # Test casi diversi
    test_cases = [
        {
            'company': {
                'dimensione': 15,
                'attivita': 'Falegnameria artigianale che produce mobili su misura'
            },
            'question': 'Come posso rendere più sostenibile la mia produzione?'
        },
        {
            'company': {
                'dimensione': 20,
                'attivita': 'Panificio con vendita diretta'
            },
            'question': 'Quali metriche ambientali devo monitorare?'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'#'*60}")
        print(f"TEST CASO {i}")
        print(f"{'#'*60}")
        
        response = system.consult(test['company'], test['question'])
        
        print("\n📋 RISPOSTA:")
        print(response)