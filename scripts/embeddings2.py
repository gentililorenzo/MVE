"""
Generazione embeddings offline con Stella v5 per VSME Standard
Versione migliorata con gestione memoria, error handling e logging avanzato
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Setup path
sys.path.append('..')
from scripts.chunking import extract_all_pdfs, chunk_documents

# ============================================================================
# CONFIGURAZIONE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODALITÀ OFFLINE FORZATA
# ============================================================================

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evita warning con multiprocessing

logger.info("🔒 Modalità offline attivata")

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

class Config:
    """Configurazione centralizzata"""
    
    # Percorsi
    MODEL_PATH = "./models/stella_en_400M_v5"
    DATA_DIR = "../data"
    DB_PATH = "../database/chroma_db"
    DB_STATS_PATH = "../database/db_stats.json"
    LOGS_DIR = "../logs"
    
    # Parametri modello
    BATCH_SIZE = 32
    MAX_SEQ_LENGTH = 512
    LOAD_IN_8BIT = True
    
    # ChromaDB
    COLLECTION_NAME = "vsme_standard"
    
    # Controllo memoria
    ENABLE_MEMORY_MONITORING = True
    CLEAR_CACHE_EVERY_N_BATCHES = 10

# ============================================================================
# CLASSIFICAZIONE CONTENUTI
# ============================================================================

class VSMEClassifier:
    """Classificatore sezioni VSME migliorato"""
    
    SECTION_PATTERNS = {
        # Moduli Base
        "B1_Basis": ["b1", "basis for preparation", "preparation of financial statements"],
        "B2_Policies": ["b2", "practices, policies", "accounting policies"],
        "B3_Energy": ["b3", "energy consumption", "ghg emissions", "greenhouse gas"],
        "B4_Pollution": ["b4", "pollution", "emissions to water", "emissions to air"],
        "B5_Biodiversity": ["b5", "biodiversity", "ecosystems", "natural habitats"],
        "B6_Water": ["b6", "water consumption", "water withdrawal", "water discharge"],
        "B7_Waste": ["b7", "waste generation", "circular economy", "resource use"],
        "B8_Workforce": ["b8", "workforce", "employees", "own workforce"],
        "B9_Safety": ["b9", "health and safety", "work-related accidents", "occupational"],
        "B10_Remuneration": ["b10", "remuneration", "pay gap", "training and development"],
        "B11_Corruption": ["b11", "corruption", "bribery", "anti-corruption"],
        
        # Moduli Comprensivi
        "C1_Strategy": ["c1", "business model", "strategy and business model"],
        "C2_Description": ["c2", "description of practices", "sustainability practices"],
        "C3_Targets": ["c3", "ghg reduction", "emission reduction targets"],
        "C4_Climate": ["c4", "climate risks", "climate-related risks"],
        "C5_Workforce": ["c5", "additional workforce", "workforce disclosures"],
        "C6_Human_Rights": ["c6", "human rights", "rights of workers"],
        "C7_Incidents": ["c7", "incidents", "non-compliance incidents"],
        "C8_Revenues": ["c8", "revenues from", "sustainable revenues"],
        "C9_Diversity": ["c9", "gender diversity", "diversity in governance"],
        
        # Altri
        "Appendix": ["appendix", "annex"],
        "Guidance": ["guidance", "implementation guidance"],
        "Glossary": ["glossary", "definitions"],
    }
    
    @classmethod
    def classify(cls, text: str) -> str:
        """Classifica il testo nella sezione VSME appropriata"""
        text_lower = text.lower()
        
        # Cerca pattern specifici
        for section, keywords in cls.SECTION_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                return section
        
        # Identifica tipo modulo
        if "comprehensive module" in text_lower:
            return "Comprehensive_Module"
        if "basic module" in text_lower:
            return "Basic_Module"
        
        return "General"

class DocumentTypeClassifier:
    """Classificatore tipo documento"""
    
    @staticmethod
    def classify(filename: str) -> str:
        """Identifica tipo documento dal nome file"""
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

# ============================================================================
# GESTIONE MEMORIA
# ============================================================================

class MemoryManager:
    """Gestione memoria GPU/CPU"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Ottieni informazioni memoria"""
        info = {}
        
        if torch.cuda.is_available():
            info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            info['gpu_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                   torch.cuda.memory_allocated()) / 1e9
        
        return info
    
    @staticmethod
    def clear_cache():
        """Pulisci cache GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("🧹 Cache GPU pulita")

# ============================================================================
# GENERATORE EMBEDDINGS
# ============================================================================

class EmbeddingGenerator:
    """Generatore embeddings con Stella v5"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.memory_manager = MemoryManager()
        
    def load_model(self):
        """Carica modello embedding"""
        logger.info(f"🧠 Caricamento modello da: {self.config.MODEL_PATH}")
        
        # Verifica esistenza
        if not os.path.exists(self.config.MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Modello non trovato in {self.config.MODEL_PATH}\n"
                f"Esegui prima lo script di download."
            )
        
        # Configurazione modello
        model_kwargs = {
            "device_map": "auto",
            "load_in_8bit": self.config.LOAD_IN_8BIT
        }
        
        # Carica modello
        try:
            self.model = SentenceTransformer(
                self.config.MODEL_PATH,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
                local_files_only=True
            )
            
            # Imposta lunghezza massima sequenza
            self.model.max_seq_length = self.config.MAX_SEQ_LENGTH
            
            logger.info("✅ Modello caricato in modalità OFFLINE")
            
            # Log info memoria
            if self.config.ENABLE_MEMORY_MONITORING:
                mem_info = self.memory_manager.get_memory_info()
                if mem_info:
                    logger.info(f"📊 Memoria GPU: {mem_info['gpu_allocated_gb']:.2f} GB allocati")
                    
        except Exception as e:
            logger.error(f"❌ Errore caricamento modello: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings per lista di testi"""
        if self.model is None:
            raise RuntimeError("Modello non caricato. Chiama load_model() prima.")
        
        try:
            # Genera embeddings
            # Stella v5 non richiede prompt prefix per document encoding
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"❌ Errore generazione embeddings: {e}")
            raise

# ============================================================================
# GESTORE DATABASE
# ============================================================================

class ChromaDBManager:
    """Gestore database ChromaDB"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.collection = None
        
    def initialize(self, reset: bool = True):
        """Inizializza database"""
        logger.info(f"🗄️ Inizializzazione ChromaDB: {self.config.DB_PATH}")
        
        # Crea directory se non esiste
        os.makedirs(self.config.DB_PATH, exist_ok=True)
        
        # Crea client
        self.client = chromadb.PersistentClient(path=self.config.DB_PATH)
        
        # Reset collezione se richiesto
        if reset:
            try:
                self.client.delete_collection(self.config.COLLECTION_NAME)
                logger.info("   🗑️ Collezione esistente eliminata")
            except:
                pass
        
        # Crea collezione
        self.collection = self.client.create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={
                "description": "EFRAG VSME Standard - Stella v5 Embeddings",
                "model": "NovaSearch/stella_en_400M_v5",
                "created": datetime.now().isoformat()
            }
        )
        
        logger.info("✅ Database inizializzato")
    
    def add_batch(self, 
                  ids: List[str],
                  embeddings: List[List[float]],
                  documents: List[str],
                  metadatas: List[Dict]):
        """Aggiungi batch al database"""
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"❌ Errore aggiunta batch: {e}")
            raise
    
    def get_count(self) -> int:
        """Ottieni numero documenti"""
        return self.collection.count()

# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================

class EmbeddingPipeline:
    """Pipeline completa generazione embeddings"""
    
    def __init__(self, config: Config):
        self.config = config
        self.generator = EmbeddingGenerator(config)
        self.db_manager = ChromaDBManager(config)
        self.memory_manager = MemoryManager()
        
    def prepare_metadata(self, chunk_data: Dict, chunk_global_index: int) -> Dict:
        """Prepara metadati per chunk"""
        chunk_text = chunk_data['text']
        
        return {
            # Info fonte
            "source": chunk_data['source'],
            "source_path": chunk_data['source_path'],
            
            # Info chunk
            "chunk_index": chunk_data['chunk_index'],
            "chunk_id": chunk_global_index,
            "length": len(chunk_text),
            
            # Classificazione contenuto
            "section": VSMEClassifier.classify(chunk_text),
            "document_type": DocumentTypeClassifier.classify(chunk_data['source']),
            
            # Timestamp
            "indexed_at": datetime.now().isoformat()
        }
    
    def run(self):
        """Esegui pipeline completa"""
        start_time = datetime.now()
        
        try:
            # 1. Carica modello
            self.generator.load_model()
            
            # 2. Carica documenti
            logger.info("\n📂 Caricamento documenti...")
            documents = extract_all_pdfs(self.config.DATA_DIR)
            chunks = chunk_documents(documents)
            
            # Statistiche
            logger.info(f"\n📊 STATISTICHE DOCUMENTI:")
            logger.info(f"   Documenti: {len(documents)}")
            logger.info(f"   Chunks totali: {len(chunks)}")
            for doc in documents:
                doc_chunks = [c for c in chunks if c['source'] == doc['filename']]
                logger.info(f"   - {doc['filename']}: {len(doc_chunks)} chunks")
            
            # 3. Inizializza database
            self.db_manager.initialize(reset=True)
            
            # 4. Genera embeddings in batch
            logger.info(f"\n⚙️ Generazione embeddings per {len(chunks)} chunks...")
            
            batch_size = self.config.BATCH_SIZE
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            with tqdm(total=len(chunks), desc="Embeddings") as pbar:
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(chunks))
                    batch_chunks = chunks[start_idx:end_idx]
                    
                    # Estrai testi
                    batch_texts = [c['text'] for c in batch_chunks]
                    
                    # Genera embeddings
                    embeddings = self.generator.generate_embeddings(batch_texts)
                    
                    # Prepara dati per DB
                    ids = [f"chunk_{i}" for i in range(start_idx, end_idx)]
                    metadatas = [
                        self.prepare_metadata(chunk, start_idx + i)
                        for i, chunk in enumerate(batch_chunks)
                    ]
                    
                    # Aggiungi al database
                    self.db_manager.add_batch(
                        ids=ids,
                        embeddings=embeddings,
                        documents=batch_texts,
                        metadatas=metadatas
                    )
                    
                    pbar.update(len(batch_chunks))
                    
                    # Gestione memoria periodica
                    if (batch_num + 1) % self.config.CLEAR_CACHE_EVERY_N_BATCHES == 0:
                        self.memory_manager.clear_cache()
                        
                        if self.config.ENABLE_MEMORY_MONITORING:
                            mem_info = self.memory_manager.get_memory_info()
                            if mem_info:
                                logger.debug(
                                    f"Batch {batch_num + 1}/{total_batches} - "
                                    f"GPU: {mem_info['gpu_allocated_gb']:.2f} GB"
                                )
            
            # 5. Salva statistiche
            self.save_statistics(documents, chunks)
            
            # 6. Riepilogo finale
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ COMPLETATO CON SUCCESSO")
            logger.info(f"{'='*60}")
            logger.info(f"📦 Chunks processati: {self.db_manager.get_count()}")
            logger.info(f"⏱️  Tempo totale: {elapsed:.2f} secondi")
            logger.info(f"📈 Velocità: {len(chunks)/elapsed:.1f} chunks/sec")
            logger.info(f"📁 Database: {self.config.DB_PATH}")
            logger.info(f"📊 Statistiche: {self.config.DB_STATS_PATH}")
            logger.info(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"❌ Errore pipeline: {e}", exc_info=True)
            raise
    
    def save_statistics(self, documents: List[Dict], chunks: List[Dict]):
        """Salva statistiche database"""
        logger.info("\n💾 Salvataggio statistiche...")
        
        db_stats = {
            'metadata': {
                'total_chunks': self.db_manager.get_count(),
                'model': 'NovaSearch/stella_en_400M_v5',
                'model_path': self.config.MODEL_PATH,
                'batch_size': self.config.BATCH_SIZE,
                'max_seq_length': self.config.MAX_SEQ_LENGTH,
                'created_at': datetime.now().isoformat()
            },
            'sources': {}
        }
        
        for doc in documents:
            doc_name = doc['filename']
            doc_chunks = [c for c in chunks if c['source'] == doc_name]
            
            # Conta sezioni
            sections = {}
            for chunk in doc_chunks:
                section = VSMEClassifier.classify(chunk['text'])
                sections[section] = sections.get(section, 0) + 1
            
            db_stats['sources'][doc_name] = {
                'chunks': len(doc_chunks),
                'pages': doc['num_pages'],
                'sections': sections
            }
        
        # Salva
        os.makedirs(os.path.dirname(self.config.DB_STATS_PATH), exist_ok=True)
        with open(self.config.DB_STATS_PATH, "w", encoding='utf-8') as f:
            json.dump(db_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Statistiche salvate in: {self.config.DB_STATS_PATH}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point"""
    
    # Crea directory logs se non esiste
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("VSME EMBEDDINGS GENERATOR - OFFLINE MODE")
    logger.info("=" * 60)
    
    # Crea ed esegui pipeline
    pipeline = EmbeddingPipeline(Config)
    pipeline.run()

if __name__ == "__main__":
    main()