from .rag_system import VSMERagSystem
from .sector_classifier import SectorClassifier

class HybridPromptSystem:
    """ RAG with sectoral hints """
    
    def __init__(self):
        self.rag = VSMERagSystem()
        self.classifier = SectorClassifier()
    
    def generate_prompt(self, company_profile, question, context_chunks):
        
        # Classify the sector based on the keywords encountered
        sector, hints = self.classifier.classify(
            company_profile['activity']
        )
        
        # Construct VSME context
        context = "\n\n".join([
            f"[VSME reference #{i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Prompt with secotral hint
        prompt = f"""You are a sustainability consultant with expertise in the EFRAG VSME standard for SMEs.

COMPANY PROFILE:
- Sector: {sector}
- Size: {company_profile['num_employees']} employees
- Activity: {company_profile['activity']}

{'SECTOR TIP:' if hints else ''}
{hints['hint'] if hints else ''}

{'PRIORITY METRICS FOR THIS SECTOR:' if hints else ''}
{', '.join(hints['VSME_metrics']) if hints else ''}

RELEVANT VSME STANDARDS:
{context}

QUESTION:
{question}

Provide a PRACTICAL and CONCRETE answer based on the VSME standard.

STRUCTURE:
1. 🎯 IMMEDIATE PRIORITIES (Quick wins)
2. 📊 VSME METRICS TO MONITOR (with codes, e.g., B3, B7)
3. 📝 CONCRETE ACTIONS (step by step)
4. 📄 NECESSARY DOCUMENTS

Be specific for the {sector} sector. Do not be generic.

ANSWER:"""
        
        return prompt
    
    def consult(self, company_profile, question):
        """ Chat with prompted Ollama to obtain a complete consultation """
        
        # 1. Retrieval
        query_enriched = f"{company_profile['activity']} {question}"
        chunks, _ = self.rag.retrieve(query_enriched, n_results=5)
        
        # 2. Augmentation (prompting) + Retrieval
        prompt = self.generate_prompt(company_profile, question, chunks)
        
        # 3. LLM response
        import ollama
        response = ollama.chat(
            model='gemma3:4b', # TODO fare config.js? necessario?
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']

# Test
if __name__ == "__main__":
    system = HybridPromptSystem()
    
    # different use cases
    test_cases = [
        {
            'company': {
                'num_employees': 15,
                'activity': 'Artisanal carpentry workshop producing custom-made furniture'
            },
            'question': 'How can I make my production more sustainable?'
        },
        {
            'company': {
                'num_employees': 20,
                'activity': 'Bakery with direct sales'
            },
            'question': 'What environmental KPIs should I monitor??'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'#'*60}")
        print(f"TEST {i}")
        print(f"{'#'*60}")
        
        response = system.consult(test['company'], test['question'])
        
        print("\n📋 Answer:")
        print(response)