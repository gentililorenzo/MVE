import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from docling.document_converter import DocumentConverter, ConversionResult
from docling.chunking import HybridChunker
from docling.datamodel.document import DoclingDocument

import sys
from pathlib import Path

# root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import mve_config

"""
Uses the powerful HybridChunker of docling to extract chunks from PDFs.
"""
# TODO fix page numbers --> come estrapolare il numero della pagina? --> Serve veramente?
# https://github.com/docling-project/docling/discussions/1012


# --- CONFIGURATIONS ---
DATA_PATH = mve_config.data_path()
STATS_FILE = DATA_PATH / "chunking_stats.json"
OUTPUT_JSON = DATA_PATH / "full_chunks.json"
OUTPUT_TXT = DATA_PATH / "full_chunks.txt"
MAX_TOKENS = mve_config.max_tokens

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def extract_pdf_document(pdf_path: Path) -> Optional[DoclingDocument]:
    """
    Analyzes a single PDF with Docling and returns the structured Document object.
    """
    logger.info(f"📄 Converting: {pdf_path.name}")
    
    try:
        converter = DocumentConverter()
        result: ConversionResult = converter.convert(pdf_path)
        
        doc_object = result.document
        num_pages = len(doc_object.pages)
        logger.info(f"   ✅ Conversion completed: {num_pages} pages")
        return doc_object
        
    except Exception as e:
        logger.error(f"❌ Error while converting {pdf_path.name}: {e}")
        return None


def load_documents(data_folder: Path) -> List[Dict[str, Any]]:
    """
    Scan the folder and convert all PDFs found.
    """
    if not data_folder.exists():
        logger.error(f"Folder not found: {data_folder}")
        return []

    pdf_files = list(data_folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️ No PDF found at {data_folder}")
        return []

    logger.info(f"📚 Found {len(pdf_files)} PDF.")
    
    documents = []
    for pdf_path in pdf_files:
        doc_object = extract_pdf_document(pdf_path)
        
        if doc_object:
            documents.append({
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'doc_object': doc_object,
                'num_pages': len(doc_object.pages)
            })
    
    logger.info(f"✅ Loaded {len(documents)} successfully.")
    return documents


def _extract_page_numbers(chunk: Any) -> List[int]:
    """
    Extracts page numbers from a chunk by managing various internal structures of Docling.
    """
    pages = set()
    
    # 1. Try via provenance (provenance items)
    provs = getattr(chunk, 'prov', None) or getattr(chunk, 'provenance', None)
    
    if provs:
        # Normalize the list even if it is a single object
        if not isinstance(provs, (list, tuple)):
            provs = [provs]
            
        for p in provs:
            # Search for common attributes for the page number
            pn = getattr(p, 'page_no', None) or getattr(p, 'page', None)
            if pn is not None:
                pages.add(int(pn))

    # 2. Try via metadata
    meta = getattr(chunk, 'meta', None)
    if meta:
        doc_pages = getattr(meta, 'doc_items', None) # v2 structure is common
        if doc_pages:
             for item in doc_pages:
                 pn = getattr(item, 'page_numbers', None)
                 if pn: pages.add(int(pn))

    return sorted(list(pages)) if pages else []


def generate_chunks(documents: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    """
    Divide documents with docling HybridChunker.
    """
    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    all_chunks = []

    for doc in documents:
        logger.info(f"🔨 Chunking: {doc['filename']}")
        chunk_iter = chunker.chunk(doc['doc_object'])
        
        for i, chunk in enumerate(chunk_iter):
            # Extracting context (breadcrumb headers)
            headers = chunk.meta.headings if chunk.meta.headings else []
            header_metadata = {f"H{idx+1}": h for idx, h in enumerate(headers)}
            
            chunk_data = {
                'text': getattr(chunk, 'text', '').strip(),
                'source': doc['filename'],
                'source_path': doc['filepath'],
                'chunk_index': i,
                'structural_context': header_metadata,
                'page_numbers': _extract_page_numbers(chunk)
            }
            all_chunks.append(chunk_data)

    if all_chunks:
        avg_len = sum(len(c['text']) for c in all_chunks) / len(all_chunks)
        logger.info(f"✅ Generated {len(all_chunks)} chunks. {avg_len:.0f} characters is the medium length of chunks.")
    
    return all_chunks


def export_chunks_to_txt(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks in a .txt readable format."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{len(chunks)} CHUNKS EXPORT\n")
            f.write("="*60 + "\n\n")

            for chunk in chunks:
                context = " > ".join(chunk['structural_context'].values()) or "No header"
                pages = chunk['page_numbers'] if chunk['page_numbers'] else "N/A"
                
                f.write(f"📄 FILE: {chunk['source']} | ID: {chunk['chunk_index']}\n")
                f.write(f"📍 Page: {pages} | 📌 CONTEXT: {context}\n")
                f.write("-" * 30 + " CONTENT " + "-" * 30 + "\n")
                f.write(f"{chunk['text']}\n")
                f.write("\n" + "="*60 + "\n\n")
                
        logger.info(f"💾 Export .txt saved in: {output_path}")
    except IOError as e:
        logger.error(f"❌ Error saving the .txt output file: {e}")


def save_statistics(documents: List[Dict], chunks: List[Dict]):
    """Calculate and save statistics"""
    
    # 1. General statistics
    stats = {
        'total_documents': len(documents),
        'total_chunks': len(chunks),
        'documents': []
    }

    for doc in documents:
        doc_chunks_count = sum(1 for c in chunks if c['source'] == doc['filename'])
        stats['documents'].append({
            'filename': doc['filename'],
            'pages': doc['num_pages'],
            'chunks': doc_chunks_count
        })

    # 2. Save as .json
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Statistics saved in: {STATS_FILE}")
    except IOError as e:
        logger.error(f"❌ Error saving the .json file: {e}")


def main(): 
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    # 1. Load PDFs
    documents = load_documents(DATA_PATH)
    
    if not documents:
        logger.error(f"No documents found at {DATA_PATH}")
        return

    # 2. Chunking
    chunks = generate_chunks(documents, max_tokens=MAX_TOKENS)
    
    if chunks:
        # 3. Save .txt chunks
        export_chunks_to_txt(chunks, OUTPUT_TXT)
        # 3. Save .json chunks
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"💾 All chunks saved in {OUTPUT_JSON} ready for embeddings")
        
        # 4. Save .json statistics
        save_statistics(documents, chunks)

if __name__ == "__main__":
    main()