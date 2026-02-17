import json
import logging
import sys
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# --- CONFIGURATION ---
from config import mve_config
INPUT_FILE = mve_config.data_path() / "full_chunks.json"
DB_PATH = mve_config.db_path()
COLLECTION = mve_config.collection
EMBEDDING_MODEL = mve_config.embedding_model_path()
BATCH_SIZE = mve_config.batch_size
DEVICE = mve_config.device

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_chunks(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    # ChromaDB need flat metadatas (no dictionaries)
    meta = {
        "source": chunk.get("source", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "page_numbers": str(chunk.get("page_numbers", [])).strip("[]")
    }
    context = chunk.get("structural_context", {})
    for key, value in context.items():
        meta[f"header_{key}"] = value
    return meta

def main():

    # 1. Data (chunks) loading
    if not INPUT_FILE.exists():
        logger.error(f"❌ File not found: {INPUT_FILE}")
        return
    chunks = load_chunks(INPUT_FILE)
    
    # 2. Model loading
    logger.info("🧠 Loading model (may take some minutes the first time)...")
    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )
    except Exception as e:
        logger.error(f"Embedding model error: {e}")
        return

    # 3. Create ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION)
    
    logger.info(f"📂 Vectorialization of {len(chunks)} chunks.")

    # 4. Elaborazione
    batch_size = BATCH_SIZE
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing"):
        batch = chunks[i : i + batch_size]
        
        batch_ids = [f"{item['source']}_{item['chunk_index']}" for item in batch]
        batch_texts = [item['text'] for item in batch]
        batch_metas = [flatten_metadata(item) for item in batch]

        # Generate embeddings (using only CPU)
        embeddings = embedding_model.encode(batch_texts, convert_to_numpy=True).tolist()

        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas
        )

if __name__ == "__main__":
    main()