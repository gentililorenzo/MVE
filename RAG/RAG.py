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
from sector_classifier import SectorClassifier #RAG.sector_classifier

EMBEDDING_MODEL = mve_config.embedding_model
DB_PATH = mve_config.db_path()
COLLECTION = mve_config.collection
DEVICE = mve_config.device
LLM_MODEL = mve_config.llm_model

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
    
    def consult(self, company_profile, question):
        """ Chat with prompted Ollama to obtain a complete consultation """
        
        # 1. Retrieval
        query_enriched = f"{company_profile['activity']} {question}"
        chunks, _ = self.retrieve(query_enriched, n_results=2)  # 5 chunks as default
        
        # 2. Augmentation (prompting)
        prompt = self.generate_prompt(company_profile, question, chunks)
        
        # 3. Generation - LLM response
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    def generate_prompt(self, company_profile, question, context_chunks):
        
        # Classify the sector based on the keywords encountered
        sector, hints = self.classifier.classify(
            company_profile['activity']
        )
        
        # Construct the context TODO qui aggiornare pesantemente
        context = "\n\n".join([
            f"[reference #{i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Prompt with secotral hint # TODO attenzione a VSME_metrics in sector_classifier.py
        prompt = f"""You are a sustainability consultant with expertise in the EFRAG VSME standard for micro and SMEs.

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
        
        return prompt # TODO salvare il prompt in ../prompt_logs/ per vedere cosa viene fuori
    
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