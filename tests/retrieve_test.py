import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", device="cpu")

# 1. Connect to the DB
client = chromadb.PersistentClient(path="../database/chroma_db")
collection = client.get_collection("docling_docs")

# Test query
query = "What are the key points of the VSME standard?"

# 2. Retrieval
print(f"Asking: '{query}'...")
results = collection.query(
    query_embeddings=model.encode([query]).tolist(),
    n_results=3
)

# 3. Results
for i, doc in enumerate(results['documents'][0]):
    print(f"\n--- RESULT {i+1} ---")
    print(doc)