from sentence_transformers import SentenceTransformer
import os

# Definiamo dove salvarlo
MODEL_PATH = "./models/bge-m3"

print(f"📥 Scaricando BAAI/bge-m3 in: {MODEL_PATH}")
# Questo scarica i file reali (.json, .safetensors, ecc.) che Python sa leggere
model = SentenceTransformer('BAAI/bge-m3')
model.save(MODEL_PATH)

print("✅ Fatto! Ora vedrai la cartella piena.")