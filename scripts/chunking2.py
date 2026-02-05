import json
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def extract_pdf_with_docling(pdf_path):
    """
    Estrae testo strutturato (Markdown) usando Docling.
    Mantiene tabelle, headers e formattazione.
    """
    print(f"📄 Elaborazione con Docling: {os.path.basename(pdf_path)}")
    
    # Inizializza il converter (puoi configurarlo per usare OCR se necessario)
    converter = DocumentConverter()
    
    # Converte il documento
    result = converter.convert(pdf_path)
    
    # Esporta in Markdown (formato ideale per RAG/LLM)
    markdown_text = result.document.export_to_markdown()
    
    # Recupera il numero di pagine dai metadati di Docling
    # Nota: Docling gestisce il doc come un flusso unico, ma tiene traccia delle pagine internamente
    num_pages = len(result.document.pages)
    
    print(f"   ✅ Conversione completata ({num_pages} pagine originali)")
    
    return markdown_text, num_pages

def extract_all_pdfs(data_folder="../data"):
    """Estrae testo da TUTTI i PDF nella cartella"""
    print(f"🗂️  Scansione cartella: {data_folder}")
    
    pdf_files = list(Path(data_folder).glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"Nessun PDF trovato in {data_folder}")
    
    print(f"📚 Trovati {len(pdf_files)} PDF:\n")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    print("\n" + "="*60)
    
    documents = []
    
    for pdf_path in pdf_files:
        try:
            text, num_pages = extract_pdf_with_docling(pdf_path)
            
            documents.append({
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'text': text, # Ora è Markdown, non testo semplice
                'num_pages': num_pages
            })
        except Exception as e:
            print(f"❌ Errore processando {pdf_path.name}: {e}")
    
    print(f"\n✅ Totale documenti processati: {len(documents)}")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Divide i documenti Markdown.
    Usa una strategia ibrida: Prima split per Header, poi per Caratteri.
    """
    print(f"\n✂️  Chunking Strategico (Markdown + Recursive)")
    
    all_chunks = []
    
    # 1. Definiamo gli header su cui splittare prima (preserva il contesto semantico)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 2. Definiamo lo splitter ricorsivo per i blocchi che sono ancora troppo grandi
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for doc in documents:
        print(f"\n   Chunking: {doc['filename']}")
        
        # Fase 1: Split logico basato sulla struttura Markdown (Header)
        md_header_splits = markdown_splitter.split_text(doc['text'])
        
        final_chunks = []
        # Fase 2: Split ulteriore se i blocchi sono troppo lunghi
        for split in md_header_splits:
            # Recursive splitter preserva i metadati degli header trovati
            splits = text_splitter.split_text(split.page_content)
            
            # Combina il contenuto con il contesto degli header (utile per il retrieval)
            header_metadata = split.metadata # es: {'Header 1': 'Intro', 'Header 2': 'Dettagli'}
            
            for s in splits:
                # Creiamo un testo arricchito (opzionale: prepende il contesto al chunk)
                # context_str = " > ".join(header_metadata.values()) + "\n"
                
                final_chunks.append({
                    'text': s,
                    'metadata': header_metadata
                })

        # Aggiungi al listone totale con i metadati del file
        for i, chunk_data in enumerate(final_chunks):
            all_chunks.append({
                'text': chunk_data['text'],
                'source': doc['filename'],
                'source_path': doc['filepath'],
                'chunk_index': i,
                'total_chunks_in_doc': len(final_chunks),
                'structural_context': chunk_data['metadata'] # Info extra sugli headers
            })
        
        print(f"      → {len(final_chunks)} chunks creati")
    
    print(f"\n✅ Totale chunks: {len(all_chunks)}")
    if all_chunks:
        avg_length = sum(len(c['text']) for c in all_chunks) / len(all_chunks)
        print(f"📏 Lunghezza media chunk: {avg_length:.0f} caratteri")
    
    return all_chunks

if __name__ == "__main__":
    # Assicurati che le cartelle esistano
    os.makedirs("../data", exist_ok=True)
    
    # Estrai tutti i PDF con Docling
    try:
        documents = extract_all_pdfs("../data")
        
        if documents:
            # Crea chunks
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
            
            # Salva preview
            preview = {}
            for doc in documents:
                doc_chunks = [c for c in chunks if c['source'] == doc['filename']][:5]
                preview[doc['filename']] = []
                for c in doc_chunks:
                    # Mostra anche il contesto strutturale nella preview
                    context = str(c.get('structural_context', {}))
                    preview[doc['filename']].append(f"[{context}] {c['text'][:200]}...")
            
            with open("../data/chunks_preview.json", "w", encoding="utf-8") as f:
                json.dump(preview, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Dati salvati in: data/ingestion_stats.json e data/chunks_preview.json")
        else:
            print("⚠️ Nessun documento processato.")
            
    except Exception as main_e:
        print(f"\n❌ Errore critico: {main_e}")
        print("Assicurati di avere dei PDF nella cartella '../data'")