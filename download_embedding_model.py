from sentence_transformers import SentenceTransformer

MODEL_PATH = "./models/bge-m3"

print(f"📥 Downloading BAAI/bge-m3 in: {MODEL_PATH}")
model = SentenceTransformer('BAAI/bge-m3')
model.save(MODEL_PATH)

print("✅ Done.")