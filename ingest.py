from app.rag import RAGStore


if __name__ == "__main__":
    store = RAGStore()
    n = store.ingest_cases("data/exam_cases.json")
    print(f"Indexed {n} cases into Chroma.")
