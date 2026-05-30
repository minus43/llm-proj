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
EXAM_ALIASES = {
    "TOEIC": ["TOEIC", "토익"],
    "TOEIC Speaking": ["TOEIC Speaking", "토익스피킹", "토스"],
    "OPIc": ["OPIc", "오픽"],
    "TEPS": ["TEPS", "텝스"],
    "IELTS": ["IELTS", "아이엘츠"],
    "TOEFL": ["TOEFL", "토플"],
    "JLPT": ["JLPT", "JLPT N2", "일본어능력시험"],
    "JPT": ["JPT"],
    "HSK": ["HSK"],
    "한국사능력검정시험": ["한국사능력검정시험", "한국사능력검정", "한능검"],
    "컴퓨터활용능력 1급": ["컴퓨터활용능력 1급", "컴활 1급"],
    "컴퓨터활용능력 2급": ["컴퓨터활용능력 2급", "컴활 2급"],
    "정보처리기사": ["정보처리기사", "정처기"],
    "SQLD": ["SQLD"],
    "ADsP": ["ADsP"],
}


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
            sm = d.get("study_meta", {})
            signals = d.get("signals", {})
            doc_text = (
                f"시험:{d.get('exam', '미분류')} | "
                f"제목:{d.get('title', '')} | "
                f"요약:{d.get('summary', '')} | "
                f"기준점수:{sm.get('baseline_score')} 목표점수:{sm.get('target_score')} | "
                f"기간주:{sm.get('duration_weeks')} 하루시간:{sm.get('daily_hours')} 결과:{sm.get('result')} | "
                f"신호(점수/기간/시간/결과):"
                f"{signals.get('has_score')}/{signals.get('has_duration')}/"
                f"{signals.get('has_study_hours')}/{signals.get('has_result')} | "
                f"본문:{d.get('text', '')[:3000]}"
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
                    "quality_score": d.get("quality", {}).get("quality_score"),
                    "result": sm.get("result"),
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
        exam_candidates = self._exam_candidates(exam)
        out = None
        # Prefer exam-filtered retrieval first for more relevant results.
        try:
            out = self.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                where={"exam": {"$in": exam_candidates}},
            )
        except Exception:
            out = None

        # Fallback when filtered results are empty or filter is unsupported.
        if not out or not out.get("ids") or not out["ids"][0]:
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

    @staticmethod
    def _exam_candidates(exam: str) -> List[str]:
        normalized = exam.strip()
        cands = {normalized}
        for canonical, aliases in EXAM_ALIASES.items():
            lower_aliases = [a.lower() for a in aliases]
            if normalized.lower() in lower_aliases or canonical.lower() in normalized.lower():
                cands.add(canonical)
                cands.update(aliases)
        return list(cands)
