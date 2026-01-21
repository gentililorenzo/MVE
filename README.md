# Minimum Viable ESG - Offline sustainability assistant
Based only on VSME Knowledge base (TODO aggiungere paper EU) generate sustainability responses.
Company's data will maintain privacy by the offline usage. (TODO aggiungere inserimento documenti aziendali (bollette, procedimenti, fatture d'acquisto macchinari/strumenti sostenibili)

TODO(?) Utente aggiunge normative di riferimento o news settoriali (ad esempio un pacchetto di PDF, news/studi accademici, su come risparmiare energia?? O magari una nuova versione legislativa, Omnibus I o II?)

## Requirements 
First, download [Ollama](https://ollama.com/download) model gemma3:4b.

In the project folder, create the virtual environment and install the libraries needed.
```bash
python venv venv
pip install -r "requirements.txt"
```

## How to run
Activate the virtual environment (venv) 
```bash
.\venv\Scripts\activate
```

1) Create the knowledge base
```bash
python scripts/ingest_pdf.py
```
This will extract PDFs text and create chunks from them. Sample data for chunking is in `data/chunks.json`

2) Create the embeddings and the vector DB (ChromaDB) for indexing.
```bash
python scripts/create_embeddings.py
```

Then get back to the main folder and

3) Run the Streamlit application
```bash
streamlit run app.py
```

The MVE application will start on localhost:8501
