"""Ingest cp4's PDF user guides into the assistant knowledge base.

Downloads every row in `user_guide` (PDFs on S3), extracts text, and for guides
with enough real text (many are image-only screenshots), chunks + embeds them
into `knowledge_chunks` alongside the manual. Language is auto-detected and
mapped to cp4 locale codes so retrieval stays language-scoped.

Embeddings use the same local Ollama model (nomic-embed-text) as the PHP
ingester, so the vectors are compatible.

Run:  llmlocal/venv/bin/python -m scripts.ingest_pdfs
"""
import io
import ssl
import urllib.request

import pymysql
import requests
from langdetect import DetectorFactory, detect
from pypdf import PdfReader

DetectorFactory.seed = 0  # deterministic detection

DB = dict(host="127.0.0.1", port=3306, user="root", password="root", database="aimsfx_db3")
OLLAMA = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

MIN_TEXT_CHARS = 500      # below this a PDF is treated as image-only (skip text)
CHUNK_CHARS = 1200        # target chunk size
SOURCE_PREFIX = "guide:"  # marks PDF-derived chunks (vs manual)

# langdetect ISO code -> cp4 locale code
LANG_MAP = {
    "en": "en", "zh-cn": "ch", "zh-tw": "ch", "ja": "jp", "ko": "kr",
    "id": "id", "vi": "vi", "th": "th", "pt": "pt", "es": "sp",
}

_ssl = ssl.create_default_context()
_ssl.check_hostname = False
_ssl.verify_mode = ssl.CERT_NONE


def fetch_pdf_text(url: str) -> tuple[str, int]:
    data = urllib.request.urlopen(url, timeout=45, context=_ssl).read()
    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages), len(reader.pages)


def detect_locale(text: str) -> str:
    try:
        return LANG_MAP.get(detect(text[:2000]), "en")
    except Exception:
        return "en"


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    # Paragraph-aware packing into ~size-char chunks.
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 > size and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += p + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def embed(text: str) -> list[float]:
    r = requests.post(OLLAMA, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def main():
    conn = pymysql.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT ID, name, file_name FROM user_guide ORDER BY ID")
    guides = cur.fetchall()

    # Idempotent: clear previously ingested PDF chunks.
    cur.execute("DELETE FROM knowledge_chunks WHERE source LIKE %s", (SOURCE_PREFIX + "%",))
    conn.commit()

    ingested, skipped, chunk_total = 0, 0, 0
    for gid, name, url in guides:
        name = (name or f"guide {gid}").strip()
        try:
            text, npages = fetch_pdf_text(url)
        except Exception as e:
            print(f"  ! ID {gid} download/parse failed: {e}")
            skipped += 1
            continue

        if len(text) < MIN_TEXT_CHARS:
            print(f"  - skip (image-only, {len(text)}c/{npages}pg): {name[:45]}")
            skipped += 1
            continue

        locale = detect_locale(text)
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks, 1):
            vec = embed(f"{name}\n\n{ch}")
            cur.execute(
                "INSERT INTO knowledge_chunks (locale, source, section, content, embedding, dims, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,NOW(),NOW())",
                (locale, f"{SOURCE_PREFIX}{name}"[:191], f"{name} (part {i})"[:500],
                 ch, __import__("json").dumps(vec), len(vec)),
            )
        conn.commit()
        chunk_total += len(chunks)
        ingested += 1
        print(f"  + [{locale}] {len(chunks):2d} chunks  {name[:50]}")

    print(f"\nDone. {ingested} guides ingested ({chunk_total} chunks), {skipped} skipped (image-only/failed).")
    print(f"Total guides processed: {len(guides)}.")


if __name__ == "__main__":
    main()
