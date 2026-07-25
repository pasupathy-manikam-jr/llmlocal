# llmlocal — cp4 model inference service

Python inference service that cp4 (Laravel) calls over HTTP. See
[cp4-small-model-plan.md](cp4-small-model-plan.md) for the full plan.

Started with: **news sentiment (FinBERT, off-the-shelf, no training)**.
Next: XGBoost trader-scoring.

## Setup

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
```

First run downloads FinBERT (`ProsusAI/finbert`, ~440 MB) to the HF cache.

## Run

```bash
./venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8008
```

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | `{"status":"ok"}` |
| POST | `/sentiment` | `{"text": "..."}` | label + score + signed + probabilities |
| POST | `/sentiment/batch` | `{"texts": ["...", ...]}` | `{"results": [...]}` |

`signed` = P(positive) − P(negative), range [-1, 1] — a ready-made market-impact
feature for downstream ML.

### Example

```bash
curl -s -X POST http://127.0.0.1:8008/sentiment \
  -H 'Content-Type: application/json' \
  -d '{"text":"USD surges as Fed signals more hikes."}'
```

## Layout

```
app/
  main.py       FastAPI app + endpoints
  sentiment.py  FinBERT loader + score()
requirements.txt
```
