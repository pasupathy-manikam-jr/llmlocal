# Small / Local Model Plan for CP4

**Question:** Build a tiny LLM specialized for cp4 "forex" instead of using a huge general LLM.

**Short answer:** A from-scratch tiny LLM is the wrong tool. cp4 is not a price-prediction trading engine — it is an **affiliate / partner management platform** for forex brokers. Most valuable "AI" tasks here are **tabular ML** (numbers), not language. There is exactly **one** genuine LLM task (news sentiment). Below: what cp4 actually holds, which task needs which model, and the cheapest stack for each.

---

## 1. What CP4 Actually Is

- Laravel 13 + React. "AIMS Client React" — redo of cp2.
- Manages broker **downlines, commissions, copy-trading networks** — infrastructure, not a trader.
- Route → Middleware → Controller → Service → Model. 83 models, ~488 services, ~261 routes.
- **No existing AI/LLM.** No OpenAI/Claude/Gemini keys. No Python, no notebooks, no ML dir.

## 2. Data On Hand (the real asset)

| Data | Source | ML value |
|------|--------|----------|
| Closed trades — symbol, volume, profit, open/close price, timestamps | `MemberMT4Trade` (table is "huge") | High |
| Trade summaries per account | `MemberMT4TradeSum` | High |
| Forex symbols | `Symbol` | Feature |
| News — title, content, country, publish_date, type | `News` | LLM (sentiment) |
| Member accounts — balance, equity, credit, leverage | `MemberMT4` | High |
| Commission payouts per plan/member/period | `CommissionPlanPayout` | High |
| Copy-trade transactions | `MemberMT4TransVowtrade` (Sofinx) | High |

**Live integrations:** MT4/MT5 server API, Sofinx copy-trading API, MAM API.

**Gaps:** No OHLCV candles. No technical indicators. No stored trade signals. No sentiment scores. → Real price-forecasting would need external market feed (broker API / Alpha Vantage / IQFeed). Not present today.

## 3. Task → Right Model (the core point)

Most cp4 "AI" needs are **not language**. Do NOT use an LLM for numbers.

| Task | Data | Right tool | NOT |
|------|------|-----------|-----|
| Rate trader / predict win-rate | closed trades | **XGBoost / LightGBM** | LLM |
| Best copy-trader to follow | trades + summaries | **XGBoost + ranking** | LLM |
| Churn prediction | account balance/equity history | **LightGBM** | LLM |
| Account health / risk tier | account metadata | **Gradient boosting** | LLM |
| Fraud / unusual trade pattern | trade sequences | **Isolation Forest / autoencoder** | LLM |
| Commission / revenue forecast | payout time-series | **XGBoost or Prophet** | LLM |
| **News sentiment → market impact** | News text | **Small fine-tuned LLM** ✓ | — |

**Rule:** need *language understanding* → small LLM. Need *number → number/label* → tree model. cp4 is ~85% tree-model territory.

## 4. Recommended Stacks

### A. Tabular tasks (do these first — highest ROI)
- **Model:** XGBoost or LightGBM.
- **Runs on:** CPU. Trains in minutes. Model file KBs–MBs.
- **Pipeline:** Laravel exports features (trade aggregates, account stats) → CSV/Parquet → Python trains → export model → serve via small FastAPI microservice OR ONNX inference. Laravel calls it over HTTP, same pattern as MT4 API.
- **Cost:** ~free. Laptop / existing server.

### B. News sentiment (the one real LLM)
- **Don't train from scratch** (costs $1M+). Fine-tune a small open model.
- **Base model options:** Qwen2.5 0.5B/1.5B, Llama 3.2 1B/3B, Phi-3.5-mini (3.8B), Gemma 2 2B.
- **Method:** LoRA / QLoRA. Tools: `unsloth`, `axolotl`, HuggingFace `peft`.
- **Train cost:** ~$5–50 cloud GPU, few hours. Or one 24GB consumer GPU.
- **Run local:** quantize to GGUF q4 → `llama.cpp` / `ollama`. 1–3B model ≈ 1–2 GB RAM. Runs on small server. No API cost, private.
- **Even cheaper start:** classic sentiment models (FinBERT — finance-tuned BERT) may be enough before any fine-tune. Try FinBERT off-the-shelf first.

### C. Cost / resource comparison
| Approach | Train cost | Run resource | Use for |
|----------|-----------|--------------|---------|
| XGBoost / LightGBM | minutes, CPU | tiny (KB–MB) | numeric signals — MOST cp4 tasks |
| FinBERT (off-shelf) | none | ~500MB RAM | quick news sentiment |
| Fine-tune 1B LoRA | ~$5–50 | ~2GB RAM | custom text/sentiment |
| Train LLM from scratch | $1M+ | huge | never |

## 5. Suggested Order of Work

1. **Sentiment quick win** — run FinBERT on `News` content, store score column. No training. Immediate.
2. **Trader scoring** — XGBoost on `MemberMT4Trade` aggregates → win-rate / risk score. Powers copy-trader selection.
3. **Churn / account health** — LightGBM on account balance/equity trends.
4. **Commission forecast** — replace the current straight-line `LeaderboardService::forecast()` (earned/day × days) with XGBoost/Prophet.
5. Only if a **price feed** is added later: consider real signal/forecast models.

## 6. Serving Pattern (fits existing cp4)

```
Laravel Service  --HTTP-->  Python inference microservice (FastAPI)
   (feature export)            - XGBoost model  (tabular)
                               - GGUF LLM via llama.cpp (sentiment)
   <--JSON score--
```
Same integration shape cp4 already uses for MT4/Sofinx/MAM APIs. Keep model service separate from Laravel; deploy on same box or small container.

---

## 7. Hardware Check (this Mac)

- **Machine:** Apple M4, 16 GB RAM, ~450 GB free. Python 3.9.6 (system).
- **Verdict:** Handles 100% of *build-and-run* work.

| Task | RAM need | OK on 16GB? |
|------|----------|-------------|
| XGBoost / LightGBM (tasks #2–5) | 1–4 GB | ✅ trivial |
| FinBERT sentiment (task #1) | ~0.5–1.5 GB | ✅ easy |
| Run 1–3B LLM local (q4 GGUF) | 1.5–3 GB | ✅ fine (M4 Metal + unified mem) |
| **Fine-tune** LoRA on 1B+ | wants 24GB GPU | ❌ do in cloud (~$5–50), run result locally |

Disk usage total ≈ under 10 GB. 450 GB is overkill. Only *training* a custom LLM needs bigger hardware — a one-off cloud rental, not this Mac.

## 8. Where This Lives (project layout)

Keep the model service **separate from Laravel** (per §6). Two sibling projects:

```
/Users/oric/Sites/llmlocal/   ← Python: FinBERT + XGBoost + FastAPI service (THIS repo)
/Users/oric/Sites/cp4/        ← Laravel app (calls llmlocal over HTTP)

cp4 (Laravel)  --HTTP-->  llmlocal (Python inference service)
               <--JSON--
```

| Work | Where |
|------|-------|
| Python venv, FinBERT, XGBoost, FastAPI | **llmlocal** |
| Feature-export queries / DB schema ref | read cp4, build exporter in **llmlocal** |
| The `Http::post(...)` client call | **cp4**, added last |

cp4 stays untouched until the service works. Same integration shape cp4 already uses for MT4/Sofinx/MAM.

## 9. Concrete Start (order of build)

1. venv in llmlocal + install `transformers`, `torch`, `fastapi`, `uvicorn`.
2. FastAPI `/sentiment` endpoint running **FinBERT off-the-shelf** (no training).
3. Test on sample news strings.
4. Wire to cp4's real `News` table (needs cp4 DB access — read cp4 `.env` or an export endpoint).
5. Then add XGBoost trader-scoring (task #2).

---

**Bottom line:** Skip the custom tiny LLM. Use **XGBoost for the numbers** (most tasks) + **one small fine-tuned/off-shelf LLM for news sentiment**. Cheap, local, CPU-friendly, private. Start with FinBERT sentiment + trader-scoring XGBoost.
