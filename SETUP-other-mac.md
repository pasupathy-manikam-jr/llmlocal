# Handoff — Continue on Other Mac

Goal: build small/local models for **cp4** (forex-broker affiliate platform). Full analysis in `cp4-small-model-plan.md` (same folder). This file = environment bootstrap so a fresh session on the other Mac can start immediately.

## 0. Bring these across
- This whole `llmlocal` folder (both `.md` files).
- Access to `cp4` repo/data (for feature exports). If cp4 not on that Mac, we work with exported CSVs instead.

## 1. Check hardware (run first, paste output back to Claude)
```bash
sysctl -n machdep.cpu.brand_string      # CPU
sysctl -n hw.memsize | awk '{print $1/1073741824" GB RAM"}'
system_profiler SPDisplaysDataType | grep -i "chipset\|vram\|metal"   # GPU / Apple Silicon
uname -m                                 # arm64 = Apple Silicon (good for MLX/llama.cpp)
```
Apple Silicon (M-series) = can run local LLMs well via `mlx` or `llama.cpp` (Metal). Report chip + RAM.

## 2. Base toolchain
```bash
# Homebrew (if missing)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.12 uv git
brew install ollama            # local LLM runner
```

## 3. Python env for ML (tabular — the main work)
```bash
mkdir -p ~/Sites/llmlocal/ml && cd ~/Sites/llmlocal/ml
uv venv && source .venv/bin/activate
uv pip install xgboost lightgbm scikit-learn pandas pyarrow fastapi uvicorn
# sentiment / LLM path:
uv pip install transformers torch    # FinBERT off-the-shelf
```

## 4. Local LLM runtime (news sentiment path)
```bash
ollama pull qwen2.5:1.5b       # small, fast base
# OR Apple Silicon native (faster on M-chips):
uv pip install mlx-lm
```
For fine-tune later: `uv pip install unsloth` (needs CUDA — on Mac use `mlx-lm` LoRA instead).

## 5. First tasks (order)
1. **FinBERT sentiment** on cp4 `News` — no training, immediate.
2. **XGBoost trader-scoring** on `MemberMT4Trade` aggregates.
3. Serve via FastAPI microservice; Laravel calls over HTTP (same as MT4/Sofinx pattern).

## 6. When you open Claude on other Mac, say:
> "Read llmlocal/SETUP-other-mac.md and cp4-small-model-plan.md. Here's my hardware: [paste step-1 output]. Start task 1."

Claude picks up from here.

---
**Note:** `unsloth`/CUDA fine-tuning is x86+NVIDIA. On Mac Apple Silicon use `mlx-lm` for LoRA, or rent a cloud GPU (~$5–50) for heavier fine-tunes. Tabular (XGBoost) runs fully local on any Mac.
