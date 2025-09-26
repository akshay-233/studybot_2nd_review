import re
import faiss
import numpy as np
import pickle
from embeddings import embed_query, embed_texts

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def search_best_sentences(query, index, chunks, k_chunks=3, k_sentences=3):
    q = embed_query(query)
    _, idxs = index.search(q, k_chunks)
    candidate_chunks = [chunks[i] for i in idxs[0]]

    sentences = []
    for ch in candidate_chunks:
        ss = [s.strip() for s in split_into_sentences(ch) if s.strip()]
        sentences.extend(ss)

    if not sentences:
        return []

    sent_vecs = embed_texts(sentences)
    dim = sent_vecs.shape[1]
    temp = faiss.IndexFlatL2(dim)
    temp.add(sent_vecs)
    _, sidx = temp.search(q, min(k_sentences, len(sentences)))
    best = [sentences[i] for i in sidx[0]]
    return best

# --- Cache helpers ---
def save_index(index, path):
    faiss.write_index(index, path)

def load_index(path):
    return faiss.read_index(path)

def save_chunks(chunks, path):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(path):
    with open(path, "rb") as f:
        return pickle.load(f)
