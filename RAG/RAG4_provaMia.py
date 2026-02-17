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
        
        sector, hints, _ = self.classifier.classify(company_profile['activity'])
        
        mode = "GENERAL_ADVICE" # Default
        if "report" in str(scope).lower() or "vsme" in str(scope).lower():
            mode = "REPORTING_COMPLIANCE"
        elif "finance" in str(scope).lower() or "bank" in str(scope).lower():
            mode = "FINANCE_ALIGNMENT"
            
        system_instruction = _get_system_instruction(mode, sector)
        context = "\n\n".join([f"[reference #{i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Formattazione dell'intervista per il prompt TODO finire sector classifier e referenze VSME 
        interview_section = ""
        if interview_history:
            interview_str = "\n".join([f"- Q: {item[0]}\n  A: {item[1]}" for item in interview_history])
            interview_section = f"""
COMPANY DEEP-DIVE (INTERVIEW DATA):
The user has provided specific details about their operations:
{interview_str}
"""

        prompt = f"""
# ROLE
{system_instruction}
Your tone is professional, encouraging, and highly specific to the user's industry.

# INPUT DATA

<company_profile>
Sector: {sector}
Size: {company_profile['num_employees']} employees
Activity: {company_profile['activity']}
</company_profile>

<interview_context>
{interview_section}
</interview_context>

<sector_hints_vsme>
{hints['Hint'] if hints else 'No specific sector hints provided.'}
</sector_hints_vsme>

<vsme_standards_context>
{context}
</vsme_standards_context>

<current_objective>
The user is specifically looking for help with: "{scope}"
</current_objective>

<user_request>
{question}
</user_request>

# INSTRUCTIONS
Analyze the provided data to answer the user request. You must align your advice with the <current_objective>.

1. **Grounding:** Use the <vsme_standards_context> as your primary source of truth for compliance or reporting questions. If the user asks about something not in the standards, use general best practices for the {sector}.
2. **personalization:** Do not give generic advice. Reference specific details from the <interview_context> to show you understand their business.
3. **Gap Analysis:** Identify where their current activity (from the interview) fails to meet the standards or best practices.

# OUTPUT FORMAT
Provide your response in the following structure:

### 1. 🔍 DIAGNOSIS
*Briefly summarize their current status based on the interview. Highlight 1-2 critical gaps related to the {sector}.*

### 2. 📝 ACTION PLAN
*Provide 3-5 concrete, step-by-step actions. Start each action with a verb. Mark actions that are specifically required by VSME standards with a [VSME] tag.*

### 3. 📊 METRICS TO MONITOR
*List 2-3 specific KPIs they should track. If available, cite the specific VSME metric ID/Name from the context.*

### 4. 📄 DOCUMENTATION NEEDED
*List the specific documents, policies, or data points they need to collect to achieve the goal.*

---
**Constraint:** Keep the response concise and strictly relevant to a company of {company_profile['num_employees']} employees (avoid enterprise-level complexity)."""
        log_prompt(prompt)
        return prompt
    
def _get_system_instruction(mode, sector):
    """
    Docstring for _get_system_instruction
    
    :param mode: Assistant mode selected by the user
    :param sector: Specialize LLM in a spercific, defined, sector
    """
    if mode == "REPORTING_COMPLIANCE":
        return f"""You are a VSME Audit Expert. 
        GOAL: Help the user fill out the VSME report correctly.
        1. Use <vsme_knowledge_base> as strict LAW. 
        2. Focus on metrics, data points, and disclosure requirements for the {sector} sector.
        3. If the user asks for actions, frame them as 'data collection' actions."""
        
    elif mode == "FINANCE_ALIGNMENT":
        return f"""You are a Sustainable Finance Advisor.
        GOAL: Help the user get a loan/investment.
        1. Use <vsme_knowledge_base> to suggest standardized metrics that banks trust.
        2. Explain WHY a bank cares about specific ESG risks in the {sector} sector.
        3. Frame your advice in terms of 'risk reduction' and 'creditworthiness'."""
        
    else: # GENERAL_ADVICE
        return f"""You are an Operational Sustainability Consultant.
        GOAL: Help the user improve their business efficiency and reduce impact.
        1. Use <vsme_knowledge_base> ONLY as a background reference for definitions, but DO NOT obsess over reporting codes.
        2. Focus on practical, low-cost operational changes for a {sector}.
        3. Suggest tangible actions (e.g., machinery upgrades, waste reduction processes) rather than paperwork."""
        
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