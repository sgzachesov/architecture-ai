"""Runs sample queries against the FAISS index.

Run: python tools/search_index.py
     python tools/search_index.py "your question" --top-k 5

Prints the retrieved chunks with their score, source file and position so that
the bot can cite them. The output of the default queries is also saved to
index/search_examples.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding import BgeM3, model_dir_from_env

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"

DEFAULT_QUERIES = [
    "Who was Xarn Velgor and how did he turn to the null side of the Synth Flux?",
    "What is the Synth Flux and how do Fluxadepts use it?",
    "What happened to Sorrhas during the Vat Wars?",
    "How does a fluxblade work and what crystal does it use?",
]


def load_index(index_dir: Path) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(index_dir / "faiss.index"))
    payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    return index, payload["chunks"]


def search(encoder: BgeM3, index: faiss.Index, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    vector = encoder.encode([query])
    scores, ids = index.search(vector, top_k)
    hits = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id < 0:
            continue
        chunk = chunks[int(chunk_id)]
        hits.append(
            {
                "score": round(float(score), 4),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "title": chunk["title"],
                "heading": chunk["heading"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "words": chunk["words"],
                "text": chunk["text"],
            }
        )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the FAISS knowledge base index.")
    parser.add_argument("query", nargs="*", help="search query; defaults to the built-in examples")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of text")
    parser.add_argument("--save", type=Path, help="write the results to a Markdown file")
    args = parser.parse_args()

    index, chunks = load_index(INDEX_DIR)
    encoder = BgeM3(model_dir_from_env())

    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    results = [(query, search(encoder, index, chunks, query, args.top_k)) for query in queries]

    if args.json:
        print(json.dumps([{"query": q, "hits": hits} for q, hits in results], ensure_ascii=False, indent=2))
        return 0

    blocks: list[str] = []
    for query, hits in results:
        print(f"\nQuery: {query}")
        lines = [f"### {query}", ""]
        for hit in hits:
            print(f"  {hit['score']:.4f}  {hit['chunk_id']}  ({hit['words']} words)")
            print(f"    {hit['text'][:220].replace(chr(10), ' ')}...")
            lines.append(
                f"- **{hit['chunk_id']}** — score `{hit['score']}`, "
                f"source `{hit['source']}`, chars {hit['start_char']}–{hit['end_char']}, "
                f"section “{hit['heading'] or hit['title']}”"
            )
            lines.append("")
            lines.append(f"  > {hit['text'][:400].replace(chr(10), ' ')}...")
            lines.append("")
        blocks.append("\n".join(lines))

    if args.save:
        args.save.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
        print(f"\nsaved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
