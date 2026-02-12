import json
import logging
from pathlib import Path
import sys
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
 
# root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import mve_config
from RAG.sector_classifier import SectorClassifier

EMBEDDING_MODEL = mve_config.embedding_model
DB_PATH = mve_config.db_path()
COLLECTION = mve_config.collection
DEVICE = mve_config.device
LLM_MODEL = mve_config.llm_model
CHUNKS_IN_PROMPT = mve_config.chunks_in_prompt
LOG_PATH = mve_config.log_path()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class rag:    
    def __init__(self):
                
        # Load embedding model
        logger.info(f"Loading: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )
        
        # Load vectorial database
        # Nota: CONFIG['DB_PATH'] è ora un oggetto Path corretto grazie a load_config()
        logger.info(f"Connecting at DB in: {DB_PATH}")
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        
        try:
            self.collection = self.client.get_collection(COLLECTION)
        except chromadb.errors.NotFoundError as e:
            logging.error(f"{COLLECTION} was not found! {e}")
            raise e
        
        self.llm_model = LLM_MODEL
        self.classifier = SectorClassifier()
    
    def retrieve(self, query, n_results=5):
        """
        Retrieve relevant chunks
        """
        # Query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Database search - get the n_results nearest neighbor embeddings for provided query_embeddings
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results['documents'][0], results['metadatas'][0]
    
    def consult(self, company_profile, question, scope, interview_history=None):
        """ 
        Chat with prompted LLM. 
        If interview_history is present, it enriches the context.
        """
        
        # 1. Retrieval enrichment
        # Se c'è una storia di intervista, usiamola per cercare chunk più specifici
        interview_text = ""
        if interview_history:
            interview_text = " ".join([f"Q: {q} A: {a}" for q, a in interview_history])
        
        # Query arricchita: Attività + Domanda utente + Dettagli emersi nell'intervista
        query_enriched = f"{company_profile['activity']} {question} {interview_text}"
        
        # Retrieval
        chunks, _ = self.retrieve(query_enriched, n_results=CHUNKS_IN_PROMPT)
        
        # 2. Augmentation (prompting)
        prompt = self.generate_prompt(company_profile, question, chunks, scope, interview_history)
        
        # 3. Generation
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    def generate_prompt(self, company_profile, question, context_chunks, scope, interview_history=None):
        
        sector, hints = self.classifier.classify(company_profile['activity'])
        
        context = "\n\n".join([f"[reference #{i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Formattazione dell'intervista per il prompt
        interview_section = ""
        if interview_history:
            interview_str = "\n".join([f"- Q: {item[0]}\n  A: {item[1]}" for item in interview_history])
            interview_section = f"""
COMPANY DEEP-DIVE (INTERVIEW DATA):
The user has provided specific details about their operations:
{interview_str}
"""

        prompt = f"""You are a sustainability consultant for micro and SMEs.

COMPANY PROFILE:
- Sector: {sector}
- Size: {company_profile['num_employees']} employees
- Activity: {company_profile['activity']}

{interview_section}

{'You are specialized in:' if scope else ''}
{'- the EFRAG VSME standard' if 'VSME oriented' in scope else ''}
{'- the domain of sustainability practices and ESG domain' if 'ESG oriented' in scope else ''}
{'- the financial-related ESG domain' if 'SFDR oriented' in scope else ''}

{'SECTOR TIP:' if hints else ''}
{hints['hint'] if hints else ''}

RELEVANT VSME STANDARDS REFERENCES:
{context}

USER REQUEST:
{question}

Based on the COMPANY PROFILE and the INTERVIEW DATA provided above, provide a tailored answer.

STRUCTURE:
1. 🎯 DIAGNOSIS (Based on interview answers)
2. 📊 METRICS TO MONITOR
3. 📝 ACTION PLAN
4. 📄 DOCUMENTATION NEEDED

Be specific for the {sector} sector.

ANSWER:"""
        log_prompt(prompt)
        return prompt
    
def log_prompt(prompt):
    try:
        LOG_PATH.mkdir(exist_ok=True, parents=True)
        with open(LOG_PATH / "prompt_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n\nPrompt with {CHUNKS_IN_PROMPT} chunks\n")
            f.write("="*60 + "\n")
            f.write(prompt)
                
    except IOError as e:
        logger.error(f"❌ Error saving the .txt log file: {e}")

# Test
if __name__ == "__main__":
    system = rag()
    
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