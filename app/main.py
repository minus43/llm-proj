from datetime import date, datetime
import os
from pathlib import Path
import re
from typing import Dict, Any
from datetime import timedelta

import requests
import dateparser
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


def _first_number(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_exam_date(text: str, today: date) -> tuple[date, str]:
    t = (text or "").strip()
    low = t.lower()

    # Natural-language fast path
    if "오늘" in t:
        return today, "high"
    if "내일" in t:
        return today + timedelta(days=1), "high"
    if "모레" in t:
        return today + timedelta(days=2), "high"
    if "글피" in t:
        return today + timedelta(days=3), "high"

    try:
        return datetime.strptime(t, "%Y-%m-%d").date(), "high"
    except ValueError:
        pass

    num = _first_number(low)
    if num is not None:
        if "일" in low and "뒤" in low:
            return today + timedelta(days=int(num)), "high"
        if "주" in low and "뒤" in low:
            return today + timedelta(days=int(num * 7)), "high"
        if ("달" in low or "개월" in low) and "뒤" in low:
            return today + timedelta(days=int(num * 30)), "medium"

    if "다음주" in t:
        return today + timedelta(days=7), "medium"
    if "이번주" in t:
        return today + timedelta(days=4), "medium"
    if "다음달" in t:
        return today + timedelta(days=30), "medium"

    parsed = dateparser.parse(
        t,
        languages=["ko", "en"],
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.combine(today, datetime.min.time()),
        },
    )
    if parsed:
        return parsed.date(), "medium"

    # Safe default when date expression is unclear.
    return today + timedelta(days=30), "low"


def parse_baseline_score(text: str) -> float:
    num = _first_number(text)
    if num is not None:
        return num
    t = text.lower()
    if any(k in t for k in ["처음", "노베", "기초", "거의 몰라"]):
        return 35.0
    if any(k in t for k in ["중간", "보통", "무난", "기본은"]):
        return 55.0
    if any(k in t for k in ["잘해", "상위", "고수", "꽤"]):
        return 72.0
    return 50.0


def parse_target_score(text: str, baseline: float) -> float:
    num = _first_number(text)
    if num is not None:
        return num
    t = text.lower()
    if any(k in t for k in ["합격만", "합격권", "커트라인"]):
        return baseline + 15.0
    if any(k in t for k in ["무난", "안정권"]):
        return baseline + 25.0
    if any(k in t for k in ["고득점", "상위", "높게", "빡세게"]):
        return baseline + 35.0
    if any(k in t for k in ["만점", "최고점"]):
        return baseline + 45.0
    return baseline + 25.0


def parse_progress_pct(text: str) -> float:
    num = _first_number(text)
    if num is not None:
        return max(0.0, min(100.0, num))
    t = text.lower()
    if any(k in t for k in ["아직", "거의 안", "시작 전"]):
        return 10.0
    if any(k in t for k in ["조금", "초반", "1회독 전"]):
        return 25.0
    if any(k in t for k in ["절반", "반", "중간"]):
        return 50.0
    if any(k in t for k in ["거의 끝", "막바지", "2회독"]):
        return 75.0
    return 35.0


def parse_daily_hours(text: str, weekend: bool = False) -> float:
    num = _first_number(text)
    if num is not None:
        return max(0.5, num)
    t = text.lower()
    if any(k in t for k in ["거의 못", "바빠", "시간 없음"]):
        return 0.8 if not weekend else 1.5
    if any(k in t for k in ["짧게", "1~2", "조금"]):
        return 1.5 if not weekend else 2.5
    if any(k in t for k in ["보통", "2~3", "꾸준"]):
        return 2.5 if not weekend else 4.0
    if any(k in t for k in ["많이", "집중", "몰아서", "빡세게"]):
        return 4.0 if not weekend else 6.0
    return 2.0 if not weekend else 3.5


def parse_cram_tolerance(text: str) -> int:
    num = _first_number(text)
    if num is not None:
        return int(max(1, min(5, round(num))))
    t = text.lower()
    if any(k in t for k in ["체력 약", "금방 지쳐", "집중 안", "야근 많아"]):
        return 2
    if any(k in t for k in ["보통", "무난", "적당"]):
        return 3
    if any(k in t for k in ["버틸 수", "빡세게 가능", "몰입 잘", "강행"]):
        return 4
    return 3


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    exam_name = request.form["exam_name"]
    today = date.today()
    exam_date, date_confidence = parse_exam_date(request.form["exam_date"], today)

    baseline_score = parse_baseline_score(request.form["baseline_score"])
    target_score = parse_target_score(request.form["target_score"], baseline_score)
    progress_pct = parse_progress_pct(request.form["progress_pct"])
    weekday_hours = parse_daily_hours(request.form["weekday_hours"], weekend=False)
    weekend_hours = parse_daily_hours(request.form["weekend_hours"], weekend=True)
    cram_tolerance = parse_cram_tolerance(request.form["cram_tolerance"])

    inp = PlannerInput(
        exam_name=exam_name,
        exam_date=exam_date,
        today=today,
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
        raw_exam_date_text=request.form["exam_date"],
        date_confidence=date_confidence,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
