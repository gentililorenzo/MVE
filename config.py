
from pathlib import Path

class MVEConfigs():    
    max_tokens: int = 512 # tokens per chunk (512 is the limit)
    embedding_model: str = "BAAI/bge-m3"
    collection: str = "docling_docs"
    batch_size: int = 16
    device: str = "cpu"
    llm_model: str = "Qwen2.5:7b"
    chunks_in_prompt: int = 5 # 5 as default --> 20 chunks VERY slow (response in >30 minutes)

    # return directly the "Path" object
    @staticmethod
    def data_path():
        return Path("C:/Users/39389/Desktop/Tesi/provaTesiLLM/data/")
    
    @staticmethod
    def embedding_model_path():
        return "C:/Users/39389/Desktop/Tesi/provaTesiLLM/ingestion/models/bge-m3"

    @staticmethod
    def db_path():
        return Path("C:/Users/39389/Desktop/Tesi/provaTesiLLM/database/chromadb")
    
    @staticmethod
    def log_path():
        return Path("C:/Users/39389/Desktop/Tesi/provaTesiLLM/log/")
    
mve_config = MVEConfigs()
