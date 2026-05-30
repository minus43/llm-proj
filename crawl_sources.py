from __future__ import annotations

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

USER_AGENT = "DelayPlannerBot/0.1 (+local-rag-project)"
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT_SEC = 20
SLEEP_SEC = 0.6
MAX_PAGES_PER_SEED = 8
MIN_TEXT_LEN = 200


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
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

            text = extract_main_text(resp.text)
            if len(text) >= MIN_TEXT_LEN:
                yield {
                    "id": f"crawl::{seed.exam}::{abs(hash(url))}",
                    "exam": seed.exam,
                    "title": (BeautifulSoup(resp.text, "html.parser").title.get_text(strip=True)
                              if BeautifulSoup(resp.text, "html.parser").title else ""),
                    "url": url,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                }

            for link in discover_links(url, resp.text)[:20]:
                if link not in visited and link not in queue:
                    queue.append(link)

            time.sleep(SLEEP_SEC)


def save_jsonl(rows: Iterable[Dict[str, Any]], out_path: str) -> int:
    count = 0
    with Path(out_path).open("w", encoding="utf-8") as f:
        for row in rows:
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
