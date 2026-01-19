from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def extract_pdf(pdf_path):
    """Estrae testo dal PDF VSME"""
    print(f"📄 Caricamento PDF: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text = ""
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Pagina {i+1} ---\n{page_text}"
    
    print(f"✅ Estratte {len(reader.pages)} pagine")
    print(f"📊 Caratteri totali: {len(text)}")
    
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Divide il testo in chunk"""
    print(f"\n✂️  Chunking con size={chunk_size}, overlap={chunk_overlap}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    print(f"✅ Creati {len(chunks)} chunks")
    print(f"📏 Chunk medio: {sum(len(c) for c in chunks) / len(chunks):.0f} caratteri")
    
    return chunks

if __name__ == "__main__":
    # Estrai PDF
    text = extract_pdf("../data/VSME Standard.pdf")
    
    # Crea chunks
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    
    # Salva per ispezione
    with open("../data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks[:10], f, indent=2, ensure_ascii=False)  # Prime 10 per test
    
    print(f"\n💾 Primi 10 chunks salvati in data/chunks.json")
    print(f"📝 Preview primo chunk:\n{chunks[0][:200]}...")