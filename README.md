# Minimum Viable ESG - Offline sustainability assistant
An offline sustainability assistant with a knowledge base focused on the [Voluntary Sustainability Reporting Standard for non-listed SMEs (VSME)](https://www.efrag.org/sites/default/files/sites/webpublishing/SiteAssets/VSME%20Standard.pdf) by [EFRAG](https://www.efrag.org/en).

Company's data will maintain privacy by the offline usage. 

## Requirements 
First, download [Ollama](https://ollama.com/download).
```bash
irm https://ollama.com/install.ps1 | iex
```

Then download the model (for the offline usage)
```bash
 ollama pull qwen2.5:7b
```
In the project folder, create the virtual environment (python version 3.10) and install the libraries needed.
```bash
py -3.10 -m venv venv
pip install -r "requirements.txt"
```

## How to run
First of all, activate the virtual environment (venv) 
```bash
.\venv\Scripts\activate
```

Then, download the embedding model
```bash
python .\download_embedding_model.py
```

Here we can proceed with creating our knowledge base initially by chunking PDFs documents (in `data` folder) by EFRAG
```bash
python .\chunking.py
```

This will extract PDFs text and create chunks from them. Full chunks at `data/full_chunks.json` should be created.
Now let's initialize the vector DB, let's create embeddings
```bash
python .\embeddings.py
```

Then get back to the main folder and run the application
```bash
streamlit run app.py
```
Finally, your MVE application will start locally at localhost:8501.

> [!NOTE]  
> This project is built with an HP laptop with no more than 16 GB or RAM and no GPU is provided. Any further implementation can consider this hardware limitations.