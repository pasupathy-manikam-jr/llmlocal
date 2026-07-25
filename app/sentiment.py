"""FinBERT sentiment — off-the-shelf, no training.

Loads ProsusAI/finbert once (lazy) and scores finance/news text as
positive / negative / neutral with a confidence score.
"""
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]  # ProsusAI/finbert label order


@lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model


def score(text: str) -> dict:
    """Return {label, score, probabilities} for one piece of text."""
    tok, model = _load()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    # Signed impact: positive - negative, in [-1, 1]. Handy as an ML feature.
    p = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    signed = p["positive"] - p["negative"]
    return {
        "label": LABELS[idx],
        "score": float(probs[idx]),
        "signed": signed,
        "probabilities": p,
    }
