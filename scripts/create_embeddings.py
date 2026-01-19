from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from tqdm import tqdm

# Carica modello embedding (OFFLINE, scaricato al primo uso)
print("🧠 Caricamento modello embedding...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Modello caricato (supporta italiano)")

# Carica chunks dal passo precedente
print("\n📂 Caricamento chunks...")
from ingest_pdf import extract_pdf, chunk_text
text = extract_pdf("../data/vsme_standard.pdf")
chunks = chunk_text(text)

# Crea database vettoriale (OFFLINE, su disco)
print("\n🗄️  Inizializzazione ChromaDB...")
client = chromadb.PersistentClient(path="../database/chroma_db")

# Crea o ottieni collezione
collection = client.get_or_create_collection(
    name="vsme_standard",
    metadata={"description": "EFRAG VSME Standard per PMI"}
)

print(f"\n⚙️  Generazione embeddings per {len(chunks)} chunks...")

# Genera embeddings in batch
batch_size = 32
for i in tqdm(range(0, len(chunks), batch_size)):
    batch = chunks[i:i+batch_size]
    
    # Genera embeddings
    embeddings = embedding_model.encode(batch).tolist()
    
    # IDs univoci
    ids = [f"chunk_{j}" for j in range(i, i+len(batch))]
    
    # Metadati (per filtrare dopo)
    metadatas = []
    for j, chunk in enumerate(batch):
        # Classifica automaticamente il chunk
        metadata = {
            "chunk_id": i+j,
            "length": len(chunk)
        }
        
        # Identifica sezione VSME
        if "B3" in chunk or "Energy" in chunk:
            metadata["section"] = "B3_Energy"
        elif "B7" in chunk or "waste" in chunk.lower():
            metadata["section"] = "B7_Waste"
        elif "B9" in chunk or "Health and safety" in chunk:
            metadata["section"] = "B9_Safety"
        else:
            metadata["section"] = "General"
        
        metadatas.append(metadata)
    
    # Aggiungi al database
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=batch,
        metadatas=metadatas
    )

print(f"\n✅ Database creato con {collection.count()} documenti")
print(f"📍 Path: ../database/chroma_db")