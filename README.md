# 벼락치기 한계 계산기 (Ollama + RAG MVP)

시험 준비에서 `얼마나 미룰 수 있는지`를 계산하고,
유사 사례(RAG) + LLM 코멘트로 현실적인 단기 계획을 제안하는 MVP입니다.

## 1) 요구사항
- Python 3.10+
- Ollama 실행 중
- 추천 모델
  - 생성: `llama3.1`
  - 임베딩: `nomic-embed-text`

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## 2) 설치
```bash
pip install -r requirements.txt
```

## 3) RAG 데이터 인덱싱
```bash
python ingest.py
```

## 3-1) 크롤링 데이터 수집 (선택)
시드 URL 파일: `data/crawl_seeds.json`

```bash
python3 crawl_sources.py
python3 ingest_crawled.py
```

생성 파일:
- `data/crawled_docs.jsonl` (정제 텍스트 + 출처 URL + 수집 시각)
- `data/community_case_schema.json` (후기형 데이터 스키마 가이드)

크롤링 품질 필터:
- 광고성 키워드 포함 문서 제외
- 후기성 키워드 점수(합격/불합격/점수/준비기간/하루시간 등) 낮은 문서 제외
- 제목+요약 기반 중복 제거

주의:
- 각 사이트 `robots.txt`와 이용약관을 반드시 준수하세요.
- 원문 재배포 대신 요약/인용 중심으로 사용하세요.

## 4) 실행
```bash
python -m app.main
```
브라우저: `http://localhost:8000`

### 타임아웃이 날 때
기본 타임아웃은 생성 180초, 임베딩 90초입니다.
더 늘리고 싶으면:

```bash
export OLLAMA_GEN_TIMEOUT_SEC=300
export OLLAMA_EMBED_TIMEOUT_SEC=180
python3 -m app.main
```

## 5) 계산 로직 핵심
- 총 필요 학습량 `W`
- 하루 처리량 `C`
- 남은 일수 `T`
- 버퍼 `B`
- 최대 미루기 가능 일수 = `T - ceil(W / C) - B`

## 6) 확장 아이디어
- 시험별 더 정밀한 난이도 함수
- 실제 모의고사 점수 추이를 반영한 동적 업데이트
- 공유 가능한 결과 카드 이미지 생성
