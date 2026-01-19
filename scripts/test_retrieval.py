import chromadb
from sentence_transformers import SentenceTransformer

# Carica modello e database
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = chromadb.PersistentClient(path="../database/chroma_db")
collection = client.get_collection("vsme_standard")

def test_query(query_text, n_results=3):
    """Test una query"""
    print(f"\n🔍 Query: '{query_text}'")
    
    # Genera embedding della query
    query_embedding = embedding_model.encode([query_text]).tolist()
    
    # Cerca nel database
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    # Stampa risultati
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n--- Risultato {i+1} ---")
        print(f"Sezione: {metadata.get('section', 'N/A')}")
        print(f"Preview: {doc[:200]}...")

# Test queries
test_query("consumo energia elettrica")
test_query("rifiuti e scarti produzione")
test_query("sicurezza lavoratori")