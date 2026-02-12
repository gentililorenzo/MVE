
from pathlib import Path

class MVEConfigs():    
    max_tokens: int = 512 # tokens per chunk (512 is the limit)
    embedding_model: str = "BAAI/bge-m3"
    collection: str = "docling_docs"
    batch_size: int = 16
    device: str = "cpu"
    llm_model: str = "Qwen2.5:7b"
    chunks_in_prompt: int = 20 # 5 as default

    # return directly the "Path" object
    @staticmethod
    def data_path():
        return Path("C:/Users/39389/Desktop/Tesi/provaTesiLLM/data/")

    @staticmethod
    def db_path():
        return Path("C:/Users/39389/Desktop/Tesi/provaTesiLLM/database/chromadb")
    
    @staticmethod
    def log_path():
        return Path("C:/Users/39389/Desktop/Tesi/provaTesiLLM/log/")
    
mve_config = MVEConfigs()
