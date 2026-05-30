from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "DelayPlannerBot/0.2 (+local-rag-project)"
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT_SEC = 20
SLEEP_SEC = 0.6
MAX_PAGES_PER_SEED = 12
MIN_TEXT_LEN = 250

# Heuristic filters for community-style 후기 data
AD_KEYWORDS = [
    "협찬", "광고", "체험단", "파트너스", "수수료", "이 포스팅은",
    "홍보", "이벤트 참여", "원고료",
]
REVIEW_HINTS = [
    "합격", "불합격", "후기", "공부법", "점수", "모의고사", "준비기간", "하루",
    "n수", "독학", "인강", "벼락치기", "오답",
]
RESULT_PASS_HINTS = ["합격", "붙", "통과"]
RESULT_FAIL_HINTS = ["불합격", "떨어", "탈락"]


@dataclass
class Seed:
    exam: str
    urls: List[str]


def load_seeds(path: str) -> List[Seed]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Seed(exam=x["exam"], urls=x["urls"]) for x in items]


def robots_parser(url: str) -> RobotFileParser:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        pass
    return rp


def allowed(url: str, rp: RobotFileParser) -> bool:
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return False


def same_host(base: str, href: str) -> bool:
    return urlparse(base).netloc == urlparse(href).netloc


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        bad.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
    body = normalize_text(" ".join(x for x in paragraphs if x))

    if title and title not in body:
        return normalize_text(f"{title}. {body}")
    return body


def discover_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.startswith("http") and same_host(base_url, href):
            links.append(href.split("#")[0])
    return list(dict.fromkeys(links))


def stable_id(exam: str, url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"crawl::{exam}::{digest}"


def is_ad_like(text: str) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in AD_KEYWORDS)


def review_relevance_score(text: str) -> float:
    t = text.lower()
    hits = sum(1 for k in REVIEW_HINTS if k.lower() in t)
    score = hits / max(len(REVIEW_HINTS), 1)
    return min(score * 2.2, 1.0)


def extract_study_meta(text: str) -> Dict[str, Any]:
    t = text

    score_nums = re.findall(r"\b(\d{2,3})\s*점", t)
    week_nums = re.findall(r"(\d{1,2})\s*주", t)
    hour_nums = re.findall(r"하루\s*(\d{1,2}(?:\.\d)?)\s*시간", t)

    result = "unknown"
    if any(k in t for k in RESULT_PASS_HINTS):
        result = "pass"
    elif any(k in t for k in RESULT_FAIL_HINTS):
        result = "fail"

    baseline = float(score_nums[0]) if len(score_nums) >= 1 else None
    target = float(score_nums[1]) if len(score_nums) >= 2 else None
    duration_weeks = float(week_nums[0]) if week_nums else None
    daily_hours = float(hour_nums[0]) if hour_nums else None

    return {
        "baseline_score": baseline,
        "target_score": target,
        "duration_weeks": duration_weeks,
        "daily_hours": daily_hours,
        "result": result,
    }


def build_summary(text: str, max_len: int = 260) -> str:
    clipped = text[:max_len]
    if len(text) > max_len:
        clipped += "..."
    return clipped


def classify_source_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if any(x in host for x in ["blog", "tistory", "brunch", "velog"]):
        return "blog"
    if any(x in host for x in ["dcinside", "ppomppu", "instiz", "theqoo", "reddit"]):
        return "community"
    if any(x in host for x in ["go.kr", "or.kr", "ac.kr"]):
        return "official"
    return "news"


def to_case(seed: Seed, url: str, title: str, text: str) -> Dict[str, Any]:
    meta = extract_study_meta(text)
    quality_score = review_relevance_score(text)
    ad_like = is_ad_like(text)

    return {
        "id": stable_id(seed.exam, url),
        "exam": seed.exam,
        "source_type": classify_source_type(url),
        "title": title,
        "url": url,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "summary": build_summary(text),
        "signals": {
            "has_score": meta["baseline_score"] is not None or meta["target_score"] is not None,
            "has_duration": meta["duration_weeks"] is not None,
            "has_study_hours": meta["daily_hours"] is not None,
            "has_result": meta["result"] != "unknown",
        },
        "study_meta": meta,
        "quality": {
            "is_ad_like": ad_like,
            "is_duplicate": False,
            "quality_score": quality_score,
        },
    }


def dedupe_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in cases:
        sig = hashlib.sha1((c["title"] + "|" + c["summary"]).encode("utf-8")).hexdigest()
        if sig in seen:
            c["quality"]["is_duplicate"] = True
            continue
        seen.add(sig)
        out.append(c)
    return out


def crawl_seed(seed: Seed) -> Iterable[Dict[str, Any]]:
    for seed_url in seed.urls:
        rp = robots_parser(seed_url)
        queue = [seed_url]
        visited = set()

        while queue and len(visited) < MAX_PAGES_PER_SEED:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if not allowed(url, rp):
                continue

            try:
                resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
                if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                    continue
            except requests.RequestException:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else ""
            text = extract_main_text(resp.text)

            if len(text) >= MIN_TEXT_LEN:
                case = to_case(seed, url, title, text)
                # Keep docs that look review-like and not ad-like
                if (not case["quality"]["is_ad_like"]) and case["quality"]["quality_score"] >= 0.12:
                    yield case

            for link in discover_links(url, resp.text)[:20]:
                if link not in visited and link not in queue:
                    queue.append(link)

            time.sleep(SLEEP_SEC)


def save_jsonl(rows: Iterable[Dict[str, Any]], out_path: str) -> int:
    row_list = list(rows)
    row_list = dedupe_cases(row_list)

    count = 0
    with Path(out_path).open("w", encoding="utf-8") as f:
        for row in row_list:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    seeds = load_seeds("data/crawl_seeds.json")
    all_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        all_rows.extend(list(crawl_seed(seed)))

    out = "data/crawled_docs.jsonl"
    n = save_jsonl(all_rows, out)
    print(f"Saved {n} crawled docs to {out}")
