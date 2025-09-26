from data_ingestion import extract_text_from_pdf, count_pdf_pages
from text_processing import adaptive_chunking
from embeddings import generate_embeddings
from vector_store import (
    build_faiss_index, search_best_sentences,
    save_index, load_index, save_chunks, load_chunks
)
from answer_refiner import refine_answer   # bullet-point answers
from quiz_generator import generate_mcq, generate_short_question
from student_tracking import log_qa, log_quiz, get_progress, init_db


class StudyAssistant:
    def __init__(self, student_id="default"):
        self.chunks = None
        self.index = None
        self.student_id = student_id
        init_db()  # initialize DB when assistant starts

    def build_from_pdf(self, pdf_path: str, cache_base: str | None = None):
        """
        Build chunks + FAISS index from a PDF.
        Optionally cache to disk (cache_base without extension).
        """
        text = extract_text_from_pdf(pdf_path)
        num_pages = count_pdf_pages(pdf_path)
        self.chunks = adaptive_chunking(text, num_pages)

        emb = generate_embeddings(self.chunks)
        self.index = build_faiss_index(emb)

        # optional caching
        if cache_base:
            save_index(self.index, f"{cache_base}.faiss")
            save_chunks(self.chunks, f"{cache_base}.pkl")

    def load_from_cache(self, cache_base: str):
        """Load a previously cached index + chunks."""
        self.index = load_index(f"{cache_base}.faiss")
        self.chunks = load_chunks(f"{cache_base}.pkl")

    def answer(self, question: str, k_chunks=3, k_sentences=3) -> str:
        """
        Multi-step retrieval + bullet-point LLM refinement.
        Logs Q&A into database.
        """
        assert self.index is not None and self.chunks is not None, \
            "Index/chunks not ready. Call build_from_pdf(...) or load_from_cache(...)."

        top_sents = search_best_sentences(
            question, self.index, self.chunks,
            k_chunks=k_chunks, k_sentences=k_sentences
        )

        answer = refine_answer(question, top_sents)

        # Log Q&A
        log_qa(self.student_id, question, answer)

        return answer

    def rag_answer(self, question: str, top_k=5) -> str:
        """
        Retrieval-Augmented Generation (RAG):
        Retrieves top_k chunks and passes them directly to the LLM.
        Logs Q&A into database.
        """
        assert self.index is not None and self.chunks is not None, \
            "Index/chunks not ready. Call build_from_pdf(...) or load_from_cache(...)."

        # Step 1: Retrieve top-k chunks
        top_sents = search_best_sentences(
            question, self.index, self.chunks,
            k_chunks=top_k, k_sentences=3
        )

        # Step 2: Combine into a context passage
        context = " ".join(top_sents)

        # Step 3: Pass context + question to Flan-T5
        from answer_refiner import flan  # reuse loaded model

        prompt = f"""
        You are a study assistant.
        Question: {question}
        Context: {context}

        Write a clear, student-friendly answer in 4–6 bullet points.
        - Each bullet point should explain one important idea.
        - Avoid repetition.
        - Stay precise and grounded in the context.
        """

        output = flan(
            prompt,
            max_length=250,
            min_length=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=2.0
        )

        answer = output[0]['generated_text']

        # Log Q&A
        log_qa(self.student_id, question, answer)

        return answer

    def generate_quiz(self, question: str, top_k=3):
        """
        Generate quiz questions (MCQ + short answer) based on retrieved context.
        Logs quiz attempt (with placeholder correctness).
        """
        assert self.index is not None and self.chunks is not None, \
            "Index/chunks not ready. Call build_from_pdf(...) or load_from_cache(...)."

        # Step 1: Retrieve top chunks
        top_sents = search_best_sentences(
            question, self.index, self.chunks,
            k_chunks=top_k, k_sentences=3
        )

        context = " ".join(top_sents)

        # Step 2: Generate MCQ + Short Question
        mcq = generate_mcq(context)
        short_q = generate_short_question(context)

        # Log quiz attempt (set correct=False for now, to be updated after student answers)
        log_quiz(self.student_id, question, correct=False)

        return {"mcq": mcq, "short_question": short_q}

    def track_progress(self):
        """Get summary of student's progress."""
        return get_progress(self.student_id)





'''from data_ingestion import extract_text_from_pdf, count_pdf_pages
from text_processing import adaptive_chunking
from embeddings import generate_embeddings
from vector_store import (
    build_faiss_index, search_best_sentences,
    save_index, load_index, save_chunks, load_chunks
)
from answer_refiner import refine_answer   # bullet-point answers
from quiz_generator import generate_mcq, generate_short_question

class StudyAssistant:
    def __init__(self):
        self.chunks = None
        self.index = None

    def build_from_pdf(self, pdf_path: str, cache_base: str | None = None):
        """
        Build chunks + FAISS index from a PDF.
        Optionally cache to disk (cache_base without extension).
        """
        text = extract_text_from_pdf(pdf_path)
        num_pages = count_pdf_pages(pdf_path)
        self.chunks = adaptive_chunking(text, num_pages)

        emb = generate_embeddings(self.chunks)
        self.index = build_faiss_index(emb)

        # optional caching
        if cache_base:
            save_index(self.index, f"{cache_base}.faiss")
            save_chunks(self.chunks, f"{cache_base}.pkl")

    def load_from_cache(self, cache_base: str):
        """
        Load a previously cached index + chunks.
        """
        self.index = load_index(f"{cache_base}.faiss")
        self.chunks = load_chunks(f"{cache_base}.pkl")

    def answer(self, question: str, k_chunks=3, k_sentences=3) -> str:
        """
        Multi-step retrieval + bullet-point LLM refinement.
        """
        assert self.index is not None and self.chunks is not None, \
            "Index/chunks not ready. Call build_from_pdf(...) or load_from_cache(...)."

        top_sents = search_best_sentences(
            question, self.index, self.chunks,
            k_chunks=k_chunks, k_sentences=k_sentences
        )

        # Always use bullet-point refinement
        return refine_answer(question, top_sents)
    
    def rag_answer(self, question: str, top_k=5) -> str:
        """
        Retrieval-Augmented Generation (RAG):
        Retrieves top_k chunks and passes them directly to the LLM for answer generation.
        """
        assert self.index is not None and self.chunks is not None, \
            "Index/chunks not ready. Call build_from_pdf(...) or load_from_cache(...)."

        # Step 1: Retrieve top-k chunks
        top_sents = search_best_sentences(
            question, self.index, self.chunks,
            k_chunks=top_k, k_sentences=3
        )

        # Step 2: Combine into a context passage
        context = " ".join(top_sents)

        # Step 3: Pass context + question to Flan-T5
        from answer_refiner import flan  # reuse loaded model

        prompt = f"""
        You are a study assistant.
        Question: {question}
        Context: {context}

        Write a clear, student-friendly answer in 4–6 bullet points.
        - Each bullet point should explain one important idea.
        - Avoid repetition.
        - Stay precise and grounded in the context.
        """

        output = flan(
            prompt,
            max_length=250,
            min_length=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=2.0
        )

        return output[0]['generated_text']


    def generate_quiz(self, question: str, top_k=3):
        """
        Generate quiz questions (MCQ + short answer) based on retrieved context.
        """
        assert self.index is not None and self.chunks is not None, \
            "Index/chunks not ready. Call build_from_pdf(...) or load_from_cache(...)."

        # Step 1: Retrieve top chunks
        top_sents = search_best_sentences(
            question, self.index, self.chunks,
            k_chunks=top_k, k_sentences=3
        )

        context = " ".join(top_sents)

        # Step 2: Generate MCQ + Short Question
        mcq = generate_mcq(context)
        short_q = generate_short_question(context)

        return {"mcq": mcq, "short_question": short_q}
    
'''
    






'''from data_ingestion import extract_text_from_pdf, count_pdf_pages
from text_processing import adaptive_chunking
from embeddings import generate_embeddings
from vector_store import (
    build_faiss_index, search_best_sentences,
    save_index, load_index, save_chunks, load_chunks
)
from answer_refiner import refine_answer

class StudyAssistant:
    def __init__(self):
        self.chunks = None
        self.index = None

    def build_from_pdf(self, pdf_path: str, cache_base: str | None = None):
        """
        Build chunks + FAISS index from a PDF.
        Optionally cache to disk (cache_base without extension).
        """
        print(f"📘 Loading PDF: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        num_pages = count_pdf_pages(pdf_path)
        print(f"📑 PDF has {num_pages} pages")

        # Adaptive chunking to avoid long input problems
        self.chunks = adaptive_chunking(text, num_pages)

        # Generate embeddings
        emb = generate_embeddings(self.chunks)
        self.index = build_faiss_index(emb)
        print(f"✅ Index built with {len(self.chunks)} chunks")

        # Optional caching
        if cache_base:
            save_index(self.index, f"{cache_base}.faiss")
            save_chunks(self.chunks, f"{cache_base}.pkl")
            print(f"💾 Cached index and chunks as {cache_base}.faiss / {cache_base}.pkl")

    def load_from_cache(self, cache_base: str):
        """
        Load a previously cached index + chunks.
        """
        self.index = load_index(f"{cache_base}.faiss")
        self.chunks = load_chunks(f"{cache_base}.pkl")
        print(f"✅ Loaded cached assistant from {cache_base}")

    def answer(self, question: str, k_chunks=3, k_sentences=3) -> str:
        """
        Multi-step retrieval → summarization → refinement → polishing.
        Always uses refine_answer from answer_refiner.py
        """
        assert self.index is not None and self.chunks is not None, \
            "Index/chunks not ready. Call build_from_pdf(...) or load_from_cache(...)."

        # Step 1: Retrieve best matching sentences
        top_sents = search_best_sentences(
            question, self.index, self.chunks,
            k_chunks=k_chunks, k_sentences=k_sentences
        )

        # Step 2: Pass through refine pipeline (summarize + polish)
        final_answer = refine_answer(question, top_sents)

        return final_answer

'''