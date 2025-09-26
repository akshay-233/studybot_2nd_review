from transformers import pipeline

# Load once
flan = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1   # CPU (-1), or 0 if you have GPU
)

def summarize_context(context: str, max_chars: int = 1200) -> str:
    """
    Summarizes long retrieved context into a shorter passage
    to fit Flan-T5 input size (≤ 512 tokens).
    """
    context = context[:3000]  # truncate to avoid overflow

    prompt = f"""
    Summarize the following text in a clear way using bullet points (4–6 bullets).
    - Each bullet point should be a complete sentence.
    - Avoid repetition.
    - Focus only on the key ideas.

    Text: {context}
    """

    summary = flan(
        prompt,
        max_length=200,
        min_length=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=2.0
    )[0]['generated_text']

    return summary


def refine_answer(question: str, sentences: list[str]) -> str:
    """
    Refines retrieved sentences into a student-friendly bullet-point answer.
    Uses summarization first to handle long contexts.
    """

    # Step 1: Build context
    raw_context = " ".join(sentences)

    # Step 2: Summarize if too long
    if len(raw_context) > 1200:
        context = summarize_context(raw_context)
    else:
        context = raw_context

    # Step 3: Final Q&A refinement in bullet points
    prompt = f"""
    You are a helpful study assistant.
    Question: {question}
    Context: {context}

    Write the answer as 4–6 bullet points:
    - Each bullet should be short and clear.
    - Do not repeat the same idea.
    - Keep it precise and relevant to the question.
    """

    output = flan(
        prompt,
        max_length=220,
        min_length=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=2.0
    )

    return output[0]['generated_text']




'''from transformers import pipeline

# Load a stronger model (change to flan-t5-large if you want better answers)
flan = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",   # try flan-t5-large if possible
    device=-1                      # CPU (-1), GPU (0) if available
)

def summarize_context(context: str, max_chars: int = 1200) -> str:
    """
    Summarizes long retrieved context into a shorter passage
    to fit Flan-T5 input size (≤ 512 tokens).
    """
    context = context[:3000]  # Hard truncate

    prompt = f"""
    Summarize the following text in a clear and concise way (5–6 sentences).
    Focus only on the important points and remove repetition.

    Text: {context}
    """

    summary = flan(
        prompt,
        max_length=180,
        min_length=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=2.0
    )[0]['generated_text']

    return summary


def clean_answer(answer: str) -> str:
    """
    Polishes the model's raw output into a concise, non-repetitive explanation.
    """
    prompt = f"""
    Rewrite the following answer so it is clear, concise, and suitable for students.
    Use exactly 4–5 complete sentences. 
    Avoid repetition and irrelevant details.

    Answer: {answer}
    """

    cleaned = flan(
        prompt,
        max_length=150,
        min_length=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=2.0
    )[0]['generated_text']

    return cleaned


def refine_answer(question: str, sentences: list[str]) -> str:
    """
    Refines retrieved sentences into a student-friendly answer.
    Skips summarization if context is short (to reduce hallucinations).
    """

    # Step 1: Build raw context
    raw_context = " ".join(sentences)

    # Step 2: Summarize only if very long
    if len(raw_context) > 1200:
        context = summarize_context(raw_context)
    else:
        context = raw_context

    # Step 3: Final Q&A refinement with grounding
    prompt = f"""
    You are a helpful study assistant.
    Question: {question}
    Context: {context}

    Write a clear, student-friendly answer in 4–5 sentences.
    Stay strictly grounded in the context. 
    Do NOT invent information. 
    If the context does not fully answer, say "The text does not provide enough detail."
    """

    output = flan(
        prompt,
        max_length=220,
        min_length=80,
        do_sample=True,
        temperature=0.5,   # lower temperature → less random
        top_p=0.9,
        top_k=50,
        repetition_penalty=2.0
    )

    return output[0]['generated_text']

'''
