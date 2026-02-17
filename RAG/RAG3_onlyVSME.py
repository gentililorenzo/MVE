import json
import logging
import os
from pathlib import Path
import sys
os.environ["ANONYMIZED_TELEMETRY"] = "False" # offline usage
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# root setup (invariato)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import mve_config
from RAG.sector_classifier2 import SectorClassifier

# Embedding model offline usage
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

EMBEDDING_MODEL = mve_config.embedding_model_path()
DB_PATH = mve_config.db_path()
COLLECTION = mve_config.collection
DEVICE = mve_config.device
LLM_MODEL = mve_config.llm_model
CHUNKS_IN_PROMPT = mve_config.chunks_in_prompt
LOG_PATH = mve_config.log_path()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class rag:    
    def __init__(self):
        # 1. Load Embedding Model
        logger.info(f"Loading: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE, local_files_only=True)
        
        # 2. Init Classifier (passando il modello per risparmiare memoria)
        self.classifier = SectorClassifier(embedding_model=self.embedding_model)
        
        # 3. Load Vector DB
        logger.info(f"Connecting at DB in: {DB_PATH}")
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        try:
            self.collection = self.client.get_collection(COLLECTION)
        except chromadb.errors.NotFoundError as e:
            logging.error(f"{COLLECTION} was not found! {e}")
            raise e
        
        self.llm_model = LLM_MODEL
    
    def retrieve(self, query, n_results=5):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
        return results['documents'][0], results['metadatas'][0]
    
    def consult(self, company_profile, question, scope, interview_history=None):
        # 1. Recupero contesto intervista
        interview_text = ""
        if interview_history:
            interview_text = " ".join([f"Q: {q} A: {a}" for q, a in interview_history])
        
        # 2. Classificazione Settoriale (Prima del retrieval, utile per filtrare o arricchire)
        sector_name, sector_profile = self.classifier.classify(company_profile['activity'])
        
        # 3. Retrieval
        # Arricchiamo la query con il nome del settore VSME identificato
        query_enriched = f"{company_profile['activity']} ({sector_name}) {question} {interview_text}"
        chunks, _ = self.retrieve(query_enriched, n_results=CHUNKS_IN_PROMPT)
        
        # 4. Prompt Generation
        prompt = self.generate_prompt(company_profile, question, chunks, scope, 
                                      sector_name, sector_profile, interview_history)
        
        # 5. Generation
        response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    
    def generate_prompt(self, company_profile, question, context_chunks, scope, 
                        sector_name, sector_profile, interview_history=None):
        
        context = "\n\n".join([f"[VSME Standard Ref #{i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        interview_section = ""
        if interview_history:
            interview_str = "\n".join([f"- Q: {item[0]}\n  A: {item[1]}" for item in interview_history])
            interview_section = f"INTERVIEW DATA:\n{interview_str}\n"

        # Costruzione dinamica della sezione VSME Guidance basata sul PDF
        vsme_guidance = f"""
VSME SECTOR CLASSIFICATION: {sector_name}
- Type: {sector_profile['VSME_Sector_Type']}
- Focus Modules: {', '.join(sector_profile['Priority_Modules'])}
- Key VSME Metrics: {', '.join(sector_profile['Key_Metrics'])}
- STRATEGIC HINT: {sector_profile['Hint']}
"""

        prompt = f"""You are an expert sustainability consultant specializing in the 'Voluntary Standard for non-listed SMEs' (VSME).

COMPANY PROFILE:
- Activity: {company_profile['activity']}
- Employees: {company_profile['num_employees']} (Determines Micro/Small/Medium status)
{vsme_guidance}

{interview_section}

OFFICIAL VSME STANDARD CONTEXT:
{context}

USER QUESTION:
{question}

TASK:
Provide a response strictly aligned with the VSME Standard.
1. Identify if the company is Micro, Small, or Medium based on employee count (Micro < 10, Small < 50, Medium < 250).
2. If the sector is 'Services/Office', explicitly mention that Pollution (B4) and Mass-flow (B7) metrics are likely not applicable unless specific circumstances apply.
3. If 'High Climate Impact', emphasize the need for Transition Plans (C3).
4. Provide a concrete Action Plan based on the 'Key VSME Metrics' identified above.

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