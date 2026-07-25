"""Translate the assistant widget's UI strings into cp4's 9 non-English locales
using the local Ollama model, and merge them into lang/<code>.json.

Idempotent: only translates strings missing from each file; preserves existing
keys and their order (new keys appended at the end).

Run:  llmlocal/venv/bin/python -m scripts.translate_ui
"""
import json
from collections import OrderedDict
from pathlib import Path

import requests

LANG_DIR = Path("/Users/oric/Sites/cp4/lang")
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

# locale code -> language name for the translation prompt
LOCALES = {
    "ch": "Simplified Chinese",
    "wy": "Traditional Chinese",
    "jp": "Japanese",
    "kr": "Korean",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "pt": "Portuguese",
    "sp": "Spanish",
}

# Exact t() keys used in assistant-widget.tsx (must match verbatim), with a
# context hint for ambiguous strings so the model translates them correctly.
STRINGS = {
    "Ask the assistant": "tooltip on the button that opens the AI help assistant",
    "Portal Assistant": "title of the AI help assistant; 'Portal' means the client website, translate the whole phrase naturally",
    "New chat": "button that starts a new conversation",
    "Close": "button that closes the chat window",
    "Ask me anything about using the portal — deposits, withdrawals, MT4/MT5, commissions, and more.": "welcome line inside the assistant",
    "How do I make a deposit?": "example question a user can tap",
    "How do I withdraw my commission?": "example question a user can tap",
    "How do I open an MT4 account?": "example question a user can tap",
    "What is an IB?": "example question; IB = Introducing Broker, keep 'IB'",
    "From the manual:": "label shown before listing which handbook/guide sections the answer came from; 'manual' = user handbook/documentation, NOT manual-vs-automatic",
    "The assistant is temporarily unavailable. Please try again shortly, or use the live chat to reach our team.": "error shown when the AI service is down",
    "Sorry, something went wrong. Please try again.": "generic error message",
    "Thinking…": "status shown while the assistant prepares its reply; translate the word, keep the ellipsis",
    "Type your question…": "placeholder in the chat input box",
    "Send": "button that sends the message",
}


def translate(text: str, language: str, context: str) -> str:
    prompt = (
        f"You are localizing UI text for a forex trading client portal into {language}.\n"
        f"Translate the TEXT below into natural {language}.\n"
        f"Context (do not translate the context): {context}.\n"
        f"Rules: reply with ONLY the translation — no quotes, no notes, no romanization, "
        f"no English words except the product names MT4, MT5, MAM, PAMM, IB, A-Plan. "
        f"Preserve any trailing ellipsis (…).\n\nTEXT: {text}"
    )
    r = requests.post(
        OLLAMA,
        json={"model": MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.1}},
        timeout=60,
    )
    r.raise_for_status()
    out = r.json()["response"].strip()
    # Strip stray surrounding quotes the model sometimes adds.
    if len(out) >= 2 and out[0] in "\"'“「" and out[-1] in "\"'”」":
        out = out[1:-1].strip()
    return out


def main():
    for code, language in LOCALES.items():
        path = LANG_DIR / f"{code}.json"
        if not path.is_file():
            print(f"skip {code}: {path.name} not found")
            continue

        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
        added = 0
        for s, ctx in STRINGS.items():
            try:
                out = translate(s, language, ctx)
                if out and out != s:          # keep only a real translation
                    data[s] = out
                    added += 1
            except Exception as e:
                print(f"  ! {code} '{s[:30]}': {e}")

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
        )
        print(f"[{code}] +{added} strings -> {path.name}")


if __name__ == "__main__":
    main()
