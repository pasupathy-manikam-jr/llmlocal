"""cp4 model inference service.

FastAPI service that cp4 (Laravel) calls over HTTP — same integration shape
cp4 uses for MT4/Sofinx/MAM. Start: news sentiment (FinBERT).
Later: XGBoost trader-scoring endpoints get added here.
"""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from .sentiment import score

app = FastAPI(title="cp4 model service", version="0.1.0")


class SentimentIn(BaseModel):
    text: str


class SentimentBatchIn(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sentiment")
def sentiment(body: SentimentIn):
    return score(body.text)


@app.post("/sentiment/batch")
def sentiment_batch(body: SentimentBatchIn):
    return {"results": [score(t) for t in body.texts]}
