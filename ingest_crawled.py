import json
from pathlib import Path

from app.rag import RAGStore


def load_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    docs = list(load_jsonl("data/crawled_docs.jsonl"))
    store = RAGStore()
    n = store.ingest_crawled_docs(docs)
    print(f"Indexed {n} crawled docs into Chroma.")
