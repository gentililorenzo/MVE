from src.prompt_hybrid import HybridPromptSystem

def main():
    print("""
    ╔═══════════════════════════════════════════╗
    ║  CONSULENTE SOSTENIBILITÀ VSME - v1.0     ║
    ║  Sistema RAG per PMI (Offline)            ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # Inizializza sistema
    print("⚙️  Caricamento sistema...")
    system = HybridPromptSystem()
    print("✅ Sistema pronto!\n")
    
    # Input profilo azienda
    print("📝 PROFILO AZIENDA")
    print("-" * 40)
    dimensione = input("Numero dipendenti: ")
    attivita = input("Descrizione attività: ")
    
    company_profile = {
        'dimensione': int(dimensione),
        'attivita': attivita
    }
    
    print("\n✅ Profilo registrato\n")
    
    # Loop domande
    while True:
        print("\n" + "="*60)
        question = input("\n💬 La tua domanda (o 'esci' per uscire): ")
        
        if question.lower() in ['esci', 'exit', 'quit']:
            print("\n👋 Arrivederci!")
            break
        
        # Genera consulenza asdasdasdas
        response = system.consult(company_profile, question)
        
        print("\n" + "="*60)
        print("📋 RISPOSTA:")
        print("="*60)
        print(response)
        print("="*60)

if __name__ == "__main__":
    main()