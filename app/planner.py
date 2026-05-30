from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import List, Dict


@dataclass
class PlannerInput:
    exam_name: str
    exam_date: date
    today: date
    baseline_score: float
    target_score: float
    progress_pct: float
    weekday_hours: float
    weekend_hours: float
    cram_tolerance: int  # 1..5


@dataclass
class PlannerOutput:
    days_left: int
    required_work_units: float
    daily_capacity_units: float
    buffer_days: int
    max_delay_days: int
    delay_status_text: str
    urgency_mode: str
    risk_level: str
    plan: List[Dict[str, str]]
    pass_probability_band: str


def _score_gap_units(baseline: float, target: float) -> float:
    gap = max(target - baseline, 0)
    return 80 + gap * 1.4


def _progress_discount(progress_pct: float) -> float:
    p = min(max(progress_pct, 0), 100)
    return 1.0 - (p / 100.0) * 0.45


def _daily_capacity_units(weekday_hours: float, weekend_hours: float, cram_tolerance: int) -> float:
    weekly_hours = weekday_hours * 5 + weekend_hours * 2
    base_per_day = weekly_hours / 7.0

    # 1~5 => 0.85~1.15 multipliers
    tolerance_mult = 0.85 + (min(max(cram_tolerance, 1), 5) - 1) * 0.075
    return base_per_day * 6.0 * tolerance_mult


def _buffer_days(days_left: int, cram_tolerance: int) -> int:
    # lower tolerance => more protection
    safety = 0.20 if cram_tolerance <= 2 else (0.16 if cram_tolerance == 3 else 0.12)
    return max(2, math.ceil(days_left * safety))


def _risk(max_delay_days: int, days_left: int) -> str:
    if max_delay_days < 0:
        return "위험"
    if max_delay_days <= max(2, int(days_left * 0.08)):
        return "주의"
    return "안전"


def _pass_band(risk_level: str, progress_pct: float) -> str:
    if risk_level == "위험":
        return "35~55%"
    if risk_level == "주의":
        return "55~72%"
    if progress_pct >= 60:
        return "78~90%"
    return "68~85%"


def build_plan(inp: PlannerInput) -> PlannerOutput:
    days_left = max((inp.exam_date - inp.today).days, 0)

    base_units = _score_gap_units(inp.baseline_score, inp.target_score)
    required_units = base_units * _progress_discount(inp.progress_pct)

    cap = max(_daily_capacity_units(inp.weekday_hours, inp.weekend_hours, inp.cram_tolerance), 1.0)
    b_days = _buffer_days(days_left, inp.cram_tolerance)

    needed_days = math.ceil(required_units / cap)
    max_delay = days_left - needed_days - b_days
    safe_delay_days = max(0, max_delay)

    risk = _risk(max_delay, days_left)
    band = _pass_band(risk, inp.progress_pct)

    if days_left <= 3:
        urgency_mode = "응급 벼락치기 모드"
        plan = [
            {"day": "오늘", "todo": "점수 영향 큰 파트 1개만 선정해 90분 집중"},
            {"day": "오늘", "todo": "기출/모의 1회 실전 시간으로 풀고 오답만 정리"},
            {"day": "시험 전날", "todo": "암기 체크리스트 최종 점검 후 수면 6.5시간 확보"},
            {"day": "시험 당일", "todo": "새로운 내용 금지, 시간배분 전략만 실행"},
        ]
    else:
        urgency_mode = "일반 벼락치기 모드"
        plan = [
            {"day": "D-7 ~ D-5", "todo": "핵심 개념 압축 + 취약 파트 진단 1회"},
            {"day": "D-4 ~ D-3", "todo": "기출/모의 2회 + 오답노트 재회독"},
            {"day": "D-2", "todo": "빈출/암기 파트 집중 + 시간배분 리허설"},
            {"day": "D-1", "todo": "가벼운 총정리, 수면 확보(최소 6.5h)"},
        ]

    if max_delay >= 0:
        delay_status_text = f"최대 {safe_delay_days}일 추가로 미뤄도 계산상 가능"
    else:
        delay_status_text = f"이미 지연 한계 {abs(max_delay)}일 초과 (오늘 시작 권장)"

    return PlannerOutput(
        days_left=days_left,
        required_work_units=round(required_units, 1),
        daily_capacity_units=round(cap, 1),
        buffer_days=b_days,
        max_delay_days=max_delay,
        delay_status_text=delay_status_text,
        urgency_mode=urgency_mode,
        risk_level=risk,
        plan=plan,
        pass_probability_band=band,
    )
