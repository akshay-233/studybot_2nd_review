from assistant import StudyAssistant

pdf_path = "C:/Users/aksha/OneDrive/Documents/Desktop/personalized_studybot/resnet.pdf"
cache_base = "ml_types_cache"

# Initialize with a student_id (later can be roll number/login ID)
sa = StudyAssistant(student_id="student1")
sa.build_from_pdf(pdf_path, cache_base=cache_base)

print("✅ Assistant is ready. Commands: ask a question, type 'quiz', type 'progress', or 'exit' to quit.")

while True:
    q = input("\nYour command: ")
    if q.lower().strip() == "exit":
        print("👋 Exiting Study Assistant. Goodbye!")
        break
    elif q.lower().strip() == "quiz":
        quiz = sa.generate_quiz("ResNet Architecture")
        print("\n📝 Quiz Generated:\n")
        print("MCQ:", quiz["mcq"]["mcq"])
        print("\nShort Question:", quiz["short_question"]["short_question"])
    elif q.lower().strip() == "progress":
        progress = sa.track_progress()
        print("\n📊 Progress Report")
        print(f"- Total Q&A sessions: {progress['total_qa']}")
        print(f"- Total Quiz Attempts: {progress['total_quiz']}")
        print(f"- Quiz Accuracy: {progress['accuracy']}%")
    else:
        ans = sa.rag_answer(q)   # Uses RAG-enhanced answers
        print("\n📘 Question:", q)
        print("→ Answer:\n")
        print(ans)
        print("\n" + "-"*60)







'''from assistant import StudyAssistant

pdf_path = "C:/Users/aksha/OneDrive/Documents/Desktop/personalized_studybot/resnet.pdf"
cache_base = "ml_types_cache"

sa = StudyAssistant()
sa.build_from_pdf(pdf_path, cache_base=cache_base)

print("✅ Assistant is ready. Ask questions (type 'exit' to quit).")

while True:
    q = input("\nYour question: ")
    if q.lower() == "exit":
        break
    ans = sa.answer(q, k_chunks=5, k_sentences=5)
    print("\n→", ans, "\n")

'''



'''from assistant import StudyAssistant

if __name__ == "__main__":
    # 1) Build from a PDF once (and optionally cache)
    pdf_path = "C:/Users/aksha/OneDrive/Documents/Desktop/personalized_studybot/1808.02342v4.pdf"         # <-- your PDF
    cache_base = "ml_types_cache"     # will create ml_types_cache.faiss + ml_types_cache.pkl

    sa = StudyAssistant()
    sa.build_from_pdf(pdf_path, cache_base=cache_base)

    print("✅ Assistant is ready. Ask questions (type 'exit' to quit).")

    # 2) Multi-query loop (no rebuild needed)
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ans = sa.enhanced_search(q, top_k=5)

        print(f"\n→ {ans}")
'''