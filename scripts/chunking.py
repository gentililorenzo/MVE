from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os
from pathlib import Path

def extract_pdf(pdf_path):
    """Estrae testo da un singolo PDF"""
    print(f"📄 Caricamento: {os.path.basename(pdf_path)}")
    
    reader = PdfReader(pdf_path)
    text = ""
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Pagina {i+1} ---\n{page_text}"
    
    print(f"   ✅ {len(reader.pages)} pagine estratte")
    
    return text, len(reader.pages)

def extract_all_pdfs(data_folder="../data"):
    """Estrae testo da TUTTI i PDF nella cartella"""
    print(f"🗂️  Scansione cartella: {data_folder}")
    
    # Trova tutti i PDF
    pdf_files = list(Path(data_folder).glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"Nessun PDF trovato in {data_folder}")
    
    print(f"📚 Trovati {len(pdf_files)} PDF:\n")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    print("\n" + "="*60)
    
    # Estrai tutti i documenti
    documents = []
    
    for pdf_path in pdf_files:
        text, num_pages = extract_pdf(pdf_path)
        
        documents.append({
            'filename': pdf_path.name,
            'filepath': str(pdf_path),
            'text': text,
            'num_pages': num_pages
        })
    
    print(f"\n✅ Totale documenti processati: {len(documents)}")
    total_pages = sum(doc['num_pages'] for doc in documents)
    print(f"📄 Totale pagine: {total_pages}")
    
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Divide i documenti in chunk mantenendo la fonte"""
    print(f"\n✂️  Chunking con size={chunk_size}, overlap={chunk_overlap}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    
    for doc in documents:
        print(f"\n   Chunking: {doc['filename']}")
        
        # Crea chunks del testo
        text_chunks = splitter.split_text(doc['text'])
        
        # Aggiungi metadati a ogni chunk
        for i, chunk_text in enumerate(text_chunks):
            all_chunks.append({
                'text': chunk_text,
                'source': doc['filename'],
                'source_path': doc['filepath'],
                'chunk_index': i,
                'total_chunks_in_doc': len(text_chunks)
            })
        
        print(f"      → {len(text_chunks)} chunks creati")
    
    print(f"\n✅ Totale chunks: {len(all_chunks)}")
    avg_length = sum(len(c['text']) for c in all_chunks) / len(all_chunks)
    print(f"📏 Lunghezza media chunk: {avg_length:.0f} caratteri")
    
    return all_chunks

if __name__ == "__main__":
    # Estrai tutti i PDF
    documents = extract_all_pdfs("../data")
    
    # Crea chunks con metadati
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
    
    # Salva statistiche
    stats = {
        'total_documents': len(documents),
        'total_chunks': len(chunks),
        'documents': [
            {
                'filename': doc['filename'],
                'pages': doc['num_pages'],
                'chunks': sum(1 for c in chunks if c['source'] == doc['filename'])
            }
            for doc in documents
        ]
    }
    
    with open("../data/ingestion_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Salva preview chunks (primi 5 di ogni documento)
    preview = {}
    for doc in documents:
        doc_chunks = [c for c in chunks if c['source'] == doc['filename']][:5]
        preview[doc['filename']] = [c['text'][:200] + "..." for c in doc_chunks]
    
    with open("../data/chunks_preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Statistiche salvate in: data/ingestion_stats.json")
    print(f"💾 Preview chunks salvata in: data/chunks_preview.json")