from data_ingestion import extract_text_from_pdf
from text_processing import chunk_text
from embeddings import generate_embeddings
from vector_store import build_faiss_index, search_index

def run_pipeline(pdf_path, query):
    # Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk text
    chunks = chunk_text(text)

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 5: Search
    result = search_index(query, index, chunks)

    return result
