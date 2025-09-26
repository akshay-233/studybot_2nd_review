from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model once (fast & small)
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    return _embedder.encode(chunks, convert_to_tensor=False)

def embed_query(query):
    return _embedder.encode([query]).astype("float32")

def embed_texts(texts):
    arr = _embedder.encode(texts, convert_to_tensor=False)
    return np.array(arr).astype("float32")
