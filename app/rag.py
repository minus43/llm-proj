import json
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import requests
from requests.exceptions import Timeout
from chromadb.config import Settings


OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "exam_cases"
EMBED_TIMEOUT_SEC = int(os.getenv("OLLAMA_EMBED_TIMEOUT_SEC", "90"))


class RAGStore:
    def __init__(self, persist_dir: str = "./.chroma") -> None:
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    @staticmethod
    def _embed(text: str) -> List[float]:
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=EMBED_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Timeout as exc:
            raise RuntimeError(
                f"Ollama 임베딩 시간 초과({EMBED_TIMEOUT_SEC}초). "
                "모델 로딩 상태를 확인하세요."
            ) from exc

    def ingest_cases(self, json_path: str) -> int:
        items = json.loads(Path(json_path).read_text(encoding="utf-8"))
        ids, docs, metas, embs = [], [], [], []

        for item in items:
            doc = (
                f"시험:{item['exam']} | 레벨:{item['level']} | "
                f"권장기간:{item['recommended_weeks']}주 | 주간학습:{item['weekly_hours']}시간 | "
                f"메모:{item['notes']}"
            )
            ids.append(item["id"])
            docs.append(doc)
            metas.append(item)
            embs.append(self._embed(doc))

        # upsert for repeatable runs
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        return len(ids)

    def ingest_crawled_docs(self, docs: List[Dict[str, Any]]) -> int:
        ids, documents, metas, embs = [], [], [], []
        for d in docs:
            doc_text = (
                f"시험:{d.get('exam', '미분류')} | "
                f"제목:{d.get('title', '')} | "
                f"본문:{d.get('text', '')[:4000]}"
            )
            ids.append(d["id"])
            documents.append(doc_text)
            metas.append(
                {
                    "source_type": "crawl",
                    "exam": d.get("exam", "미분류"),
                    "url": d.get("url", ""),
                    "collected_at": d.get("collected_at", ""),
                    "title": d.get("title", ""),
                }
            )
            embs.append(self._embed(doc_text))

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metas,
                embeddings=embs,
            )
        return len(ids)

    def retrieve_similar(self, exam: str, baseline: float, target: float, top_k: int = 3) -> List[Dict[str, Any]]:
        query = f"시험:{exam} 현재:{baseline} 목표:{target} 유사 준비 사례"
        q_emb = self._embed(query)
        out = self.collection.query(query_embeddings=[q_emb], n_results=top_k)

        results = []
        for i in range(len(out["ids"][0])):
            results.append(
                {
                    "id": out["ids"][0][i],
                    "document": out["documents"][0][i],
                    "metadata": out["metadatas"][0][i],
                    "distance": out["distances"][0][i] if "distances" in out else None,
                }
            )
        return results
