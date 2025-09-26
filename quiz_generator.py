from transformers import pipeline

# Load Flan-T5 (reuse if already loaded elsewhere)
flan = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1   # CPU (-1), or 0 if you have GPU
)

def generate_mcq(context: str, n_options: int = 4) -> dict:
    """
    Generate a multiple-choice question with options and one marked correct.
    """
    prompt = f"""
    From the following text, create ONE multiple-choice question.
    - Give exactly {n_options} options.
    - Put ✅ after the correct option.
    - Format like this:

    Q: <question>
    a) <option1>
    b) <option2>
    c) <option3>
    d) <option4>

    Text: {context}
    """

    output = flan(prompt, max_length=250, temperature=0.7)[0]['generated_text']
    return {"mcq": output}


def generate_short_question(context: str) -> dict:
    """
    Generate one short descriptive question with its correct answer.
    """
    prompt = f"""
    From the following text, create ONE short descriptive question with its correct answer.
    Format like this:

    Q: <question>
    A: <answer>

    Text: {context}
    """

    output = flan(prompt, max_length=180, temperature=0.7)[0]['generated_text']
    return {"short_question": output}

