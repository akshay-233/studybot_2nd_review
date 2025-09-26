def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits text into chunks of 'chunk_size' words with 'overlap' words repeating between chunks.
    Default is tuned for medium-sized study PDFs (10–30 pages).
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move window with overlap

    return chunks

import re

def split_into_sentences(text):
    """
    Split text into clean sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]  # drop very short ones


def adaptive_chunking(text, num_pages):
    """
    Chooses chunk size & overlap based on PDF size.
    """
    if num_pages <= 5:  # small
        return chunk_text(text, chunk_size=200, overlap=30)
    elif num_pages <= 30:  # medium
        return chunk_text(text, chunk_size=400, overlap=50)
    else:  # large
        return chunk_text(text, chunk_size=600, overlap=100)
