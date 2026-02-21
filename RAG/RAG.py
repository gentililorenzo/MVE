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

from RAG.prompt_General import promptGeneral
from RAG.prompt_Customized import promptCustomized
from RAG.prompt_VSME import promptVSME

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
    
    def consult(self, user_question, company_profile=None, vsme_question=None, customized_response=False):
        """ 
        Chat with prompted LLM.
        """
        user_q = user_question or ""
        vsme_q = vsme_question or ""
        activity = ""
        if company_profile and isinstance(company_profile, dict):
            activity = company_profile.get('activity') or ""
    
        # Enrich prompt with interview's questions&answers if present 
        query_enriched = f"{activity} {user_q} {vsme_q}"
        
        prompt = promptGeneral(user_question=user_q)
        
        # Personalize response based on user's company details TODO non ci interessa quando trattiamo le metriche del VSME, giusto?
        if customized_response or vsme_q != "":
            # 1. Retrieval
            chunks, _ = self.retrieve(query_enriched, n_results=CHUNKS_IN_PROMPT)
                                 
            vsme_context = "\n\n".join([f"[vsme reference #{i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
            
            # 2. Augmentation
            
            # vsme context used or in sustainability awareness or in the guided reporting
            if vsme_question is not None:
                # sector = self.classifier.classify(company_profile['activity']) TODO omit????
                prompt = promptVSME(user_question=user_q, vsme_chunks=vsme_context, vsme_question=vsme_q)
            else:
                prompt = promptCustomized(user_question=user_q, companyProfile=company_profile,
                                      vsme_chunks=vsme_context)
            
        log_prompt(prompt=prompt)
        
        # 3. Generation
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}] # TODO streaming?    
        )
        
        return response['message']['content']
         
def log_prompt(prompt):
    try:
        LOG_PATH.mkdir(exist_ok=True, parents=True)
        with open(LOG_PATH / "prompt_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n\nPrompt with {CHUNKS_IN_PROMPT} chunks\n")
            f.write("="*60 + "\n")
            f.write(prompt)
                
    except IOError as e:
        logger.error(f"❌ Error saving the .txt log file: {e}")
