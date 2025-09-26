import streamlit as st
from assistant import StudyAssistant

st.set_page_config(page_title="Personalized Study Assistant", layout="wide")

st.title("📘 Personalized Study Assistant")

# Sidebar for PDF Upload
st.sidebar.header("Upload Study Materials")
uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Initialize session storage for multiple assistants
if "assistants" not in st.session_state:
    st.session_state.assistants = {}   # key: filename, value: StudyAssistant
if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None

# Process uploaded PDFs
if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        filename = pdf.name
        if filename not in st.session_state.assistants:
            with open(filename, "wb") as f:
                f.write(pdf.read())
            sa = StudyAssistant(student_id="student1")
            sa.build_from_pdf(filename, cache_base=filename.replace(".pdf", "_cache"))
            st.session_state.assistants[filename] = sa
    st.sidebar.success("✅ PDFs processed successfully!")

# Select active PDF
if st.session_state.assistants:
    st.session_state.active_pdf = st.sidebar.selectbox(
        "Choose active study material:",
        list(st.session_state.assistants.keys())
    )

active_assistant = (
    st.session_state.assistants[st.session_state.active_pdf]
    if st.session_state.active_pdf else None
)

# Tabs for features
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📝 Quizzes", "📊 Progress"])

with tab1:
    st.header("💬 Ask Your Questions")
    user_q = st.text_input("Enter your question:")
    if st.button("Get Answer") and user_q and active_assistant:
        ans = active_assistant.rag_answer(user_q)
        st.markdown("### 📘 Answer")
        st.markdown(ans)

with tab2:
    st.header("📝 Practice Quiz")
    quiz_topic = st.text_input("Enter a topic for quiz:")
    if st.button("Generate Quiz") and quiz_topic and active_assistant:
        quiz = active_assistant.generate_quiz(quiz_topic)

        # Show MCQ
        st.subheader("MCQ")
        mcq_text = quiz["mcq"]["mcq"]
        st.write(mcq_text)

        # Extract options
        options = []
        correct_answer = None
        for line in mcq_text.split("\n"):
            line = line.strip()
            if line.lower().startswith(("a)", "b)", "c)", "d)")):
                options.append(line)
            if "✅" in line:
                correct_answer = line.replace("✅", "").strip()
                options.append(correct_answer)

        if options:
            user_choice = st.radio("Choose your answer:", options)
            if st.button("Submit Answer"):
                if correct_answer and user_choice.strip() == correct_answer:
                    st.success("🎉 Correct!")
                else:
                    st.error("❌ Wrong! Correct answer: " + (correct_answer if correct_answer else "Not available"))

        # Short Question
        st.subheader("Short Question")
        st.write(quiz["short_question"]["short_question"])

with tab3:
    st.header("📊 Progress Dashboard")
    if active_assistant:
        progress = active_assistant.track_progress()
        st.metric("Total Q&A", progress["total_qa"])
        st.metric("Total Quiz Attempts", progress["total_quiz"])
        st.metric("Quiz Accuracy", f"{progress['accuracy']}%")
