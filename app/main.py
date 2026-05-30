from datetime import date, datetime
import os
from pathlib import Path
from typing import Dict, Any

import requests
from requests.exceptions import Timeout, RequestException
from flask import Flask, render_template, request

from app.planner import PlannerInput, build_plan
from app.rag import RAGStore


OLLAMA_HOST = "http://localhost:11434"
GEN_MODEL = "llama3.1"
GEN_TIMEOUT_SEC = int(os.getenv("OLLAMA_GEN_TIMEOUT_SEC", "180"))

BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
store = RAGStore()


def llm_brief_advice(user_context: Dict[str, Any], retrieval_context: str) -> str:
    prompt = f"""
너는 '벼락치기 한계 계산기' 코치다.
과장 없이 짧고 실용적으로 답해라.

[사용자 상황]
{user_context}

[유사 사례]
{retrieval_context}

요구사항:
1) 최대 몇 일 미룰 수 있는지 숫자 기준으로 재확인
2) 미루기 리스크 2개
3) 오늘 당장 할 3가지 (각 1문장)
4) 8줄 이내, 한국어
""".strip()

    try:
        # Fast health check for clearer errors when Ollama is down.
        ping = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        ping.raise_for_status()

        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
            timeout=GEN_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json().get("response", "LLM 응답이 비어 있습니다.")
    except Timeout:
        return (
            f"LLM 조언 생성 시간 초과({GEN_TIMEOUT_SEC}초). "
            "Ollama가 바쁘거나 모델 로딩이 느릴 수 있습니다."
        )
    except RequestException as exc:
        return f"LLM 조언 생성 실패(연결/HTTP): {exc}"
    except Exception as exc:
        return f"LLM 조언 생성 실패(기타): {exc}"


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    exam_name = request.form["exam_name"]
    exam_date = datetime.strptime(request.form["exam_date"], "%Y-%m-%d").date()

    baseline_score = float(request.form["baseline_score"])
    target_score = float(request.form["target_score"])
    progress_pct = float(request.form["progress_pct"])
    weekday_hours = float(request.form["weekday_hours"])
    weekend_hours = float(request.form["weekend_hours"])
    cram_tolerance = int(request.form["cram_tolerance"])

    inp = PlannerInput(
        exam_name=exam_name,
        exam_date=exam_date,
        today=date.today(),
        baseline_score=baseline_score,
        target_score=target_score,
        progress_pct=progress_pct,
        weekday_hours=weekday_hours,
        weekend_hours=weekend_hours,
        cram_tolerance=cram_tolerance,
    )

    out = build_plan(inp)
    similar = store.retrieve_similar(exam_name, baseline_score, target_score, top_k=3)

    retrieval_text = "\n".join([f"- {x['document']}" for x in similar]) if similar else "유사 사례 없음"
    advice = llm_brief_advice(
        user_context={
            "exam_name": exam_name,
            "days_left": out.days_left,
            "max_delay_days": out.max_delay_days,
            "risk": out.risk_level,
            "progress": progress_pct,
        },
        retrieval_context=retrieval_text,
    )

    return render_template(
        "result.html",
        inp=inp,
        out=out,
        similar=similar,
        advice=advice,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
