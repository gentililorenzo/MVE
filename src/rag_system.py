import os
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

BASE_DIR = os.path.dirname(__file__)   
DEFAULT_DB_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'database', 'chroma_db'))

class VSMERagSystem:
    """Sistema RAG per consigli sostenibilità VSME"""
    
    def __init__(self, db_path=DEFAULT_DB_PATH, model_name="gemma3:4b"):
        print("🚀 Inizializzazione sistema RAG...")
        
        # Carica embedding model (offline)
        self.embedding_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # Carica database vettoriale
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection("vsme_standard")
        except chromadb.errors.NotFoundError:
            print("⚠️ Collection 'vsme_standard' non trovata — ne creo una nuova.")
            self.collection = self.client.create_collection("vsme_standard")
        
        # Nome modello LLM
        self.llm_model = model_name
        
        print(f"✅ Sistema pronto")
        print(f"   📚 Chunks disponibili: {self.collection.count()}")
        print(f"   🤖 LLM: {model_name}")
    
    def retrieve(self, query, n_results=5):
        """Recupera chunk rilevanti"""
        # Embedding della query
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Cerca nel database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results['documents'][0], results['metadatas'][0]
    
    def generate_response(self, query, context_chunks):
        """Genera risposta con LLM 
        TODO definire col prof se SOLO con il contesto locale (standard vsme) oppure accedendo anche online 
        (la privacy non viene meno... Non dò in pasto le bollette ad un AI cloud, non uso API di OpenAI ecc... 
        Però sarebbe a questo punto da fare attenzione a non far fare a gemma3:4b 
        chiamate API verso il web(?) contenenti documenti personali... )
        """
        # Costruisci contesto
        context = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Prompt base
        prompt = f"""Sei un consulente esperto di sostenibilità per PMI secondo lo standard VSME EFRAG.

CONTESTO NORMATIVO (Standard VSME):
{context}

DOMANDA AZIENDA:
{query}

Fornisci una risposta pratica e concreta basata SOLO sulle informazioni del contesto normativo.
Struttura la risposta con:
- Metriche VSME rilevanti (es. B3, B7, ecc.)
- Azioni concrete da intraprendere
- Documenti necessari

RISPOSTA:"""

        # Chiamata a Ollama (offline)
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    def query(self, user_query, n_chunks=5):
        """Pipeline completa RAG"""
        print(f"\n🔍 Query: {user_query}")
        
        # 1. Retrieval --> recupera chunks rilevanti partendo dalla query dell'utente: User's info --> chunks retrieval
        print(f"📚 Recupero {n_chunks} chunk rilevanti...")
        chunks, metadatas = self.retrieve(user_query, n_results=n_chunks)
        
        # 2. Generation
        print(f"🤖 Generazione risposta con {self.llm_model}...")
        response = self.generate_response(user_query, chunks)
        
        return {
            'response': response,
            'chunks_used': chunks,
            'metadatas': metadatas
        }

# Test rapido TODO se la risposta non va bene/non trova niente --> Formattare una risposta non valida, non riuscita
if __name__ == "__main__":
    rag = VSMERagSystem()
    
    result = rag.query(
        "Siamo una falegnameria di 15 persone. Come possiamo ridurre i consumi energetici?"
    )
    
    print("\n" + "="*60)
    print("RISPOSTA:")
    print("="*60)
    print(result['response'])