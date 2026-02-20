import logging
import os
from pathlib import Path
import sys
os.environ["ANONYMIZED_TELEMETRY"] = "False" # offline usage
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
 
# root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import mve_config

from RAG.sector_classifier3 import SectorClassifier

from RAG.prompt_ESG2 import promptESG
from RAG.prompt_VSME import promptVSME
from RAG.prompt_GreenFinance import promptGreenFinance

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
            device=DEVICE,
            local_files_only=True
        )
        
        # Load vectorial database
        logger.info(f"Connecting at DB in: {DB_PATH}")
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        
        try:
            self.collection = self.client.get_collection(COLLECTION)
        except chromadb.errors.NotFoundError as e:
            logging.error(f"{COLLECTION} was not found! {e}")
            raise e
        
        self.llm_model = LLM_MODEL
        self.classifier = SectorClassifier(self.embedding_model)
    
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
        
        interview_text = ""
        if interview_history:
            interview_text = " ".join([f"Q: {q} A: {a}" for q, a in interview_history])
        
        # Enrich prompt with interview's questions&answers if present 
        query_enriched = f"{company_profile['activity']} {question} {interview_text}"
        
        # 1. Retrieval
        chunks, _ = self.retrieve(query_enriched, n_results=CHUNKS_IN_PROMPT)
        
        # 2. LLM Augmentation
        prompt = self.generate_prompt(company_profile, question, chunks, scope, interview_history)
        
        # 3. Generation
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}] # TODO streaming?
        )
        
        return response['message']['content']
    
    def generate_prompt(self, company_profile, question, context_chunks, scope, interview_history=None):
        
        sector, hints, score = self.classifier.classify(company_profile['activity'])
            
        context = "\n\n".join([f"[reference #{i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        interview_section = ""
        if interview_history:
            interview_str = "\n".join([f"- Q: {item[0]}\n  A: {item[1]}" for item in interview_history])
            interview_section = f"""
COMPANY DEEP-DIVE (INTERVIEW DATA):
The user has provided specific details about their operations:
{interview_str}
"""

        if scope == "REPORTING_COMPLIANCE":
            prompt = promptVSME(sector, company_profile, interview_section, context, hints, question)
        if scope == "FINANCE_ALIGNMENT":
            prompt = promptGreenFinance(sector, company_profile, interview_section, context, hints, question)
        if scope == "GENERAL_ADVICE":
            #promptESG3
            # rec = self.classifier.recommend_reporting(sector_label=sector, employee_count=company_profile['num_employees'])
            # prompt = promptESG(sector, company_profile, interview_section, context, hints, question, vsme_recommendations=rec)
            #promptESG2
            prompt = promptESG(sector=sector, company_profile=company_profile, interview_section=interview_section, context=context, hints=hints, question=question)
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
