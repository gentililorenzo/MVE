import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from tqdm import tqdm
import sys
import torch  # Importante per la gestione della memoria GPU se necessario

sys.path.append('..')

# Importa la funzione aggiornata
from scripts.chunking import extract_all_pdfs, chunk_documents

################
# --- 1. FORZA MODALITÀ OFFLINE ---
# Queste variabili impediscono qualsiasi tentativo di connessione a Hugging Face
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Funzioni helper
def classify_vsme_section(text):
    """Classifica più accuratamente la sezione VSME"""
    text_lower = text.lower()
    
    # Cerca pattern specifici
    patterns = {
        "B1_Basis": ["b1", "basis for preparation"],
        "B2_Policies": ["b2", "practices, policies"],
        "B3_Energy": ["b3", "energy", "ghg", "greenhouse"],
        "B4_Pollution": ["b4", "pollution", "emissions to"],
        "B5_Biodiversity": ["b5", "biodiversity", "ecosystems"],
        "B6_Water": ["b6", "water consumption", "water withdrawal"],
        "B7_Waste": ["b7", "waste", "circular economy", "resource use"],
        "B8_Workforce": ["b8", "workforce", "employees"],
        "B9_Safety": ["b9", "health and safety", "accidents"],
        "B10_Remuneration": ["b10", "remuneration", "pay gap", "training"],
        "B11_Corruption": ["b11", "corruption", "bribery"],
        "C1_Strategy": ["c1", "business model"],
        "C2_Description": ["c2", "description of practices"],
        "C3_Targets": ["c3", "ghg reduction", "targets"],
        "C4_Climate": ["c4", "climate risks"],
        "C5_Additional": ["c5", "additional workforce"],
        "C6_Human": ["c6", "human rights"],
        "C7_Incidents": ["c7", "incidents"],
        "C8_Revenues": ["c8", "revenues from"],
        "C9_Diversity": ["c9", "gender diversity"]
    }
    
    for section, keywords in patterns.items():
        if any(kw in text_lower for kw in keywords):
            return section
    
    # Fallback: cerca "Appendix", "Guidance", ecc.
    if "appendix" in text_lower:
        return "Appendix"
    if "guidance" in text_lower:
        return "Guidance"
    if "comprehensive module" in text_lower:
        return "Comprehensive"
    if "basic module" in text_lower:
        return "Basic"
    
    return "General"

def get_document_type(filename):
    """Identifica tipo di documento dal nome file"""
    filename_lower = filename.lower()
    
    if "standard" in filename_lower:
        return "standard"
    elif "basis" in filename_lower or "conclusions" in filename_lower:
        return "basis_for_conclusions"
    elif "guide" in filename_lower or "implementation" in filename_lower:
        return "implementation_guide"
    elif "appendix" in filename_lower:
        return "appendix"
    else:
        return "other"


# --- CONFIGURAZIONE MODELLO OFFLINE ---
print("🧠 Caricamento modello embedding (Locale - Stella v5)...")

# Percorso locale dove hai scaricato il modello nella Fase 1
local_model_path = "./models/stella_en_400M_v5"

# Verifica che il modello esista
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"❌ Modello non trovato in {local_model_path}. Esegui prima lo script di download online.")

model_kwargs = {
    "device_map": "auto",
    "load_in_8bit": True
}

# NOTA: Passiamo il path locale invece del repo ID
embedding_model = SentenceTransformer(
    local_model_path,       
    trust_remote_code=True, # Necessario anche offline per eseguire il codice custom locale
    model_kwargs=model_kwargs,
    local_files_only=True   # Ulteriore sicurezza per evitare chiamate di rete
)
print("✅ Modello caricato in modalità OFFLINE")
# -----------------------------------------

# Estrai TUTTI i PDF
print("\n📂 Caricamento documenti...")
documents = extract_all_pdfs("../data")
chunks = chunk_documents(documents)

print(f"\n📊 STATISTICHE:")
print(f"   Documenti: {len(documents)}")
print(f"   Chunks totali: {len(chunks)}")
for doc in documents:
    doc_chunks = [c for c in chunks if c['source'] == doc['filename']]
    print(f"   - {doc['filename']}: {len(doc_chunks)} chunks")

# Crea database
print("\n🗄️  Inizializzazione ChromaDB...")
client = chromadb.PersistentClient(path="../database/chroma_db")

# Elimina collezione esistente se presente (fresh start)
try:
    client.delete_collection("vsme_standard")
    print("   🗑️  Collezione esistente eliminata")
except:
    pass

# Crea nuova collezione
# Nota: Stella v5 ha dimensioni diverse (1024) rispetto a MiniLM (384).
# ChromaDB rileverà automaticamente la dimensione al primo inserimento.
collection = client.create_collection(
    name="vsme_standard",
    metadata={"description": "EFRAG VSME Standard - Stella v5 Embeddings"}
)

print(f"\n⚙️  Generazione embeddings per {len(chunks)} chunks...")

# Genera embeddings in batch
# Ridotto batch_size se necessario per la memoria GPU con il modello più grande
batch_size = 32 
for i in tqdm(range(0, len(chunks), batch_size)):
    batch_chunks = chunks[i:i+batch_size]
    
    # Estrai solo il testo
    batch_texts = [c['text'] for c in batch_chunks]
    
    # Genera embeddings
    # Stella v5 non richiede prompt specifici per l'indicizzazione dei passaggi
    # (per le query si usa solitamente "s2p_query")
    embeddings = embedding_model.encode(batch_texts).tolist()
    
    # IDs univoci
    ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
    
    # Metadati arricchiti
    metadatas = []
    for j, chunk_data in enumerate(batch_chunks):
        # Classifica sezione VSME (miglioriamo la logica)
        chunk_text = chunk_data['text']
        section = classify_vsme_section(chunk_text)
        
        metadata = {
            # Info fonte
            "source": chunk_data['source'],
            "source_path": chunk_data['source_path'],
            
            # Info chunk
            "chunk_index": chunk_data['chunk_index'],
            "chunk_id": i + j,
            "length": len(chunk_text),
            
            # Classificazione contenuto
            "section": section,
            
            # Per filtraggio
            "document_type": get_document_type(chunk_data['source'])
        }
        
        metadatas.append(metadata)
    
    # Aggiungi al database
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=batch_texts,
        metadatas=metadatas
    )

print(f"\n✅ Database creato con {collection.count()} documenti")
print(f"📍 Path: ../database/chroma_db")

# Salva statistiche database
db_stats = {
    'total_chunks': collection.count(),
    'model': 'NovaSearch/stella_en_400M_v5',
    'sources': {}
}

for doc in documents:
    doc_name = doc['filename']
    doc_chunks = [c for c in chunks if c['source'] == doc_name]
    db_stats['sources'][doc_name] = {
        'chunks': len(doc_chunks),
        'pages': doc['num_pages']
    }

with open("../database/db_stats.json", "w") as f:
    json.dump(db_stats, f, indent=2)

print(f"💾 Statistiche DB salvate in: database/db_stats.json")