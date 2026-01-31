import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)   
DEFAULT_DB_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'database', 'chroma_db'))

class VSMERagSystem:    
    def __init__(self, db_path=DEFAULT_DB_PATH, model_name="gemma3:4b"):
        
        # Load embedding model (offline)
        self.embedding_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # Load vectorial database
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection("vsme_standard")
        except chromadb.errors.NotFoundError:
            print("⚠️ Collection 'vsme_standard' not found — creating a new one.")
            self.collection = self.client.create_collection("vsme_standard")
        
        self.llm_model = model_name
    
    def retrieve(self, query, n_results=5):
        """
        Retrieve relevant chunks
        """
        # Query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Database search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results['documents'][0], results['metadatas'][0]