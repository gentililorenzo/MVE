import json
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# Opzionale: Importa il tokenizer se vuoi controllo preciso sui token (default usa quello standard)
# from docling.chunking import HybridChunker

def extract_pdf_with_docling(pdf_path):
    """
    Analizza il PDF con Docling e restituisce l'oggetto Document strutturato.
    Non esportiamo subito in Markdown perché il HybridChunker lavora sull'oggetto nativo.
    """
    print(f"📄 Elaborazione con Docling: {os.path.basename(pdf_path)}")
    
    # Inizializza il converter
    converter = DocumentConverter()
    
    # Converte il documento
    result = converter.convert(pdf_path)
    
    # L'oggetto 'document' contiene tutta la struttura (paragrafi, tabelle, layout)
    doc_object = result.document
    
    # Recupera il numero di pagine
    num_pages = len(doc_object.pages)
    
    print(f"   ✅ Conversione completata ({num_pages} pagine)")
    
    return doc_object, num_pages

def extract_all_pdfs(data_folder="../data"):
    """Processa TUTTI i PDF nella cartella e restituisce gli oggetti Docling"""
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
            # Nota: Ora salviamo l'oggetto docling nativo ('doc_object'), non la stringa text
            doc_object, num_pages = extract_pdf_with_docling(pdf_path)
            
            documents.append({
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'doc_object': doc_object, # Oggetto ricco di Docling
                'num_pages': num_pages
            })
        except Exception as e:
            print(f"❌ Errore processando {pdf_path.name}: {e}")
    
    print(f"\n✅ Totale documenti processati: {len(documents)}")
    return documents

def chunk_documents(documents, max_tokens=512):
    """
    Divide i documenti usando il Hybrid Chunking nativo di Docling.
    Sfrutta la struttura del layout (Header, Liste, Tabelle) per creare chunk semantici.
    """
    print(f"\n✂️  Chunking Strategico (Docling Native Hybrid)")
    
    # Inizializza il HybridChunker
    # max_tokens: Definisce la dimensione target del chunk in token
    # merge_peers: Cerca di unire frasi/paragrafi contigui se stanno nel limite
    chunker = HybridChunker(
        max_tokens=max_tokens, 
        merge_peers=True
    )
    
    all_chunks = []

    for doc in documents:
        print(f"\n   Chunking: {doc['filename']}")
        
        doc_object = doc['doc_object']
        
        # Docling genera un iteratore di chunk
        chunk_iter = chunker.chunk(doc_object)
        
        doc_chunks = []
        
        for i, chunk in enumerate(chunk_iter):
            # Serializziamo il chunk per l'output JSON
            # Estrazione contesto (Header gerarchici)
            # chunk.meta.headings restituisce una lista ['Header 1', 'Header 2']
            headers = chunk.meta.headings if chunk.meta.headings else []
            
            # Creiamo un dizionario di header per compatibilità con visualizzazioni precedenti
            # Es: {"H1": "Titolo", "H2": "Sottotitolo"}
            header_metadata = {f"H{idx+1}": h for idx, h in enumerate(headers)}
            
            chunk_data = {
                'text': getattr(chunk, 'text', '') or '',
                'source': doc['filename'],
                'source_path': doc['filepath'],
                'chunk_index': i,
                'structural_context': header_metadata,
                'page_numbers': extract_page_numbers_from_chunk(chunk) 
            }
            
            doc_chunks.append(chunk_data)
            all_chunks.extend(doc_chunks)
        
        print(f"      → {len(doc_chunks)} chunks creati")
        # Aggiorniamo il totale chunks nel documento originale per le statistiche (opzionale)
        # (Lo calcoleremo dopo nel main per pulizia)

    print(f"\n✅ Totale chunks: {len(all_chunks)}")
    if all_chunks:
        # Nota: Qui calcoliamo caratteri, ma il limite era impostato sui token
        avg_length = sum(len(c['text']) for c in all_chunks) / len(all_chunks)
        print(f"📏 Lunghezza media chunk: {avg_length:.0f} caratteri")
    
    return all_chunks

def extract_page_numbers_from_chunk(chunk):
    """
    Prova più strategie per estrarre i numeri di pagina da un chunk in modo robusto.
    Restituisce una lista di interi (potrebbe essere vuota).
    """
    # 1) check attributi tipici (prov / provenance / provs ...)
    for attr in ('prov', 'provenance', 'provs', 'provenances'):
        provs = getattr(chunk, attr, None)
        if provs:
            # provs può essere iterabile di oggetti con .page_no oppure un singolo oggetto
            try:
                pages = []
                # se è un iterabile
                for p in provs:
                    pn = getattr(p, 'page_no', None) or getattr(p, 'page', None)
                    if pn is not None:
                        pages.append(int(pn))
                if pages:
                    return sorted(set(pages))
            except TypeError:
                # non iterabile: prova come singolo oggetto
                pn = getattr(provs, 'page_no', None) or getattr(provs, 'page', None)
                if pn is not None:
                    return [int(pn)]

    # 2) fallback su attributi dentro chunk.meta
    meta = getattr(chunk, 'meta', None)
    if meta:
        for attr in ('page_numbers', 'pages', 'page_no'):
            val = getattr(meta, attr, None)
            if val:
                if isinstance(val, (list, tuple)):
                    return [int(x) for x in val]
                try:
                    return [int(val)]
                except Exception:
                    pass

    # 3) attributo diretto del chunk (es. chunk.page_no)
    pn = getattr(chunk, 'page_no', None) or getattr(chunk, 'page', None)
    if pn is not None:
        return [int(pn)]

    # 4) niente trovato
    return []


if __name__ == "__main__":
    # Assicurati che le cartelle esistano
    os.makedirs("../data", exist_ok=True)
    
    # Estrai tutti i PDF con Docling
    try:
        # Step 1: Ingestion
        documents = extract_all_pdfs("../data")
        
        if documents:
            # Step 2: Chunking
            # Usiamo max_tokens invece di chunk_size in caratteri
            # 512 token ~= 1500-2000 caratteri, buon bilanciamento per embedding models
            chunks = chunk_documents(documents, max_tokens=600)
            
            # Step 3: Calcolo Statistiche
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
            
            # Salvataggio statistiche
            with open("../data/ingestion_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            # Salvataggio preview
            preview = {}
            # Creiamo un set di filename unici dai chunks per iterare
            unique_files = set(c['source'] for c in chunks)
            
            for filename in unique_files:
                doc_chunks = [c for c in chunks if c['source'] == filename][:5]
                preview[filename] = []
                for c in doc_chunks:
                    # Formatta contesto per la preview
                    context = " > ".join(c['structural_context'].values())
                    preview[filename].append(f"[{context}] {c['text'][:200]}...")
            
            with open("../data/chunks_preview.json", "w", encoding="utf-8") as f:
                json.dump(preview, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Dati salvati in: data/ingestion_stats.json e data/chunks_preview.json")
        else:
            print("⚠️ Nessun documento processato.")
            
    except Exception as main_e:
        import traceback
        print(f"\n❌ Errore critico: {main_e}")
        traceback.print_exc()
        print("Assicurati di avere dei PDF nella cartella '../data'")