import torch
import transformers
from sentence_transformers import SentenceTransformer

print(f"Torch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer loaded successfully.")
    embeddings = model.encode(["test document"])
    print(f"Embeddings generated: {embeddings.shape}")
except Exception as e:
    print(f"Error: {e}")
