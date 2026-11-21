"""Builds the FAISS index for the fictional knowledge base.

Run: python tools/build_index.py

The script reads knowledge_base/*.md (except README.md), splits documents into
chunks of up to 300 words, encodes them with BAAI/bge-m3 (int8 ONNX) and writes
index/faiss.index plus index/chunks.json with citation metadata.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding import DIMENSION, MODEL_NAME, BgeM3, ensure_model, model_dir_from_env

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "index"

MIN_WORDS = 100
MAX_WORDS = 300


def words(text: str) -> int:
    return len(text.split())


def parse_blocks(text: str) -> list[dict]:
    """Splits a Markdown document into paragraphs with their heading and offsets."""
    blocks: list[dict] = []
    heading = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = offset
        offset += len(line)
        if not stripped:
            continue
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            continue
        if stripped.startswith("# "):
            continue
        if stripped.startswith("_(excerpt truncated)_"):
            continue
        blocks.append({"start": start, "end": start + len(stripped), "heading": heading, "text": stripped})
    return blocks


def split_block(block: dict, max_words: int = MAX_WORDS) -> list[dict]:
    """Splits an oversized paragraph by sentences, then by words as a fallback."""
    if words(block["text"]) <= max_words:
        return [block]

    pieces: list[dict] = []
    text = block["text"]
    cursor = 0
    current: list[tuple[int, str]] = []

    def flush() -> None:
        if not current:
            return
        start = block["start"] + current[0][0]
        end = block["start"] + current[-1][0] + len(current[-1][1])
        pieces.append(
            {
                "start": start,
                "end": end,
                "heading": block["heading"],
                "text": " ".join(sentence for _, sentence in current),
            }
        )
        current.clear()

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        position = text.find(sentence, cursor)
        if position < 0:
            position = cursor
        cursor = position + len(sentence)

        if words(sentence) > max_words:
            flush()
            offset = 0
            for part in range(0, len(sentence), max_words * 6):
                piece = sentence[part : part + max_words * 6]
                pieces.append(
                    {
                        "start": block["start"] + position + offset,
                        "end": block["start"] + position + offset + len(piece),
                        "heading": block["heading"],
                        "text": piece,
                    }
                )
                offset += len(piece)
            continue

        if current and sum(words(item) for _, item in current) + words(sentence) > max_words:
            flush()
        current.append((position, sentence))

    flush()
    return pieces


def group_words(group: list[dict]) -> int:
    return sum(words(piece["text"]) for piece in group)


def split_evenly(group: list[dict], parts: int) -> list[list[dict]]:
    """Splits a group into `parts` groups of roughly equal word count."""
    target = group_words(group) / parts
    limit = max(MIN_WORDS, int(target))

    pieces: list[dict] = []
    for piece in group:
        pieces.extend(split_block(piece, limit))

    total = group_words(pieces)
    result: list[list[dict]] = []
    current: list[dict] = []
    for piece in pieces:
        if current:
            would = group_words(current) + words(piece["text"])
            bins_left = parts - len(result) - 1
            rest = total - would
            too_big = would > MAX_WORDS
            fair_cut = (
                bins_left > 0
                and group_words(current) >= MIN_WORDS
                and would > target
                and rest >= bins_left * MIN_WORDS
            )
            if too_big or fair_cut:
                result.append(current)
                current = []
        current.append(piece)
    if current:
        result.append(current)

    # Move whole pieces from the previous chunk when the last one is too short.
    while (
        len(result) > 1
        and group_words(result[-1]) < MIN_WORDS
        and len(result[-2]) > 1
    ):
        result[-1].insert(0, result[-2].pop())

    return result


def balance(groups: list[list[dict]]) -> list[list[dict]]:
    """Merges short groups with a neighbour and re-splits them evenly.

    A group below MIN_WORDS is joined with the closest group, then the pair is
    divided again so that every resulting chunk stays within MIN_WORDS..MAX_WORDS.
    """
    index = 0
    while index < len(groups) and len(groups) > 1:
        if group_words(groups[index]) >= MIN_WORDS:
            index += 1
            continue

        left = index - 1 if index > 0 else None
        right = index + 1 if index < len(groups) - 1 else None
        if left is None:
            partner = right
        elif right is None:
            partner = left
        else:
            partner = left if group_words(groups[left]) <= group_words(groups[right]) else right

        low, high = min(index, partner), max(index, partner)
        combined = groups[low] + groups[high]
        total = group_words(combined)
        parts = max(1, -(-total // MAX_WORDS))
        replacement = [combined] if parts == 1 else split_evenly(combined, parts)

        if len(replacement) == 2 and any(group_words(item) < MIN_WORDS for item in replacement):
            index += 1  # balancing cannot improve this pair, keep it as is
            continue

        del groups[high]
        del groups[low]
        groups[low:low] = replacement
        index = max(0, low - 1)
    return groups


def build_chunks(text: str, title: str, source: str) -> list[dict]:
    pieces: list[dict] = []
    for block in parse_blocks(text):
        pieces.extend(split_block(block))

    groups: list[list[dict]] = []
    current: list[dict] = []
    for piece in pieces:
        if current and group_words(current) + words(piece["text"]) > MAX_WORDS:
            groups.append(current)
            current = []
        current.append(piece)
    if current:
        groups.append(current)
    groups = balance(groups)

    chunks: list[dict] = []
    for number, group in enumerate(groups, start=1):
        body = "\n\n".join(item["text"] for item in group)
        heading = group[0]["heading"]
        chunks.append(
            {
                "id": len(chunks),
                "chunk_id": f"{Path(source).stem}-{number:03d}",
                "source": source,
                "title": title,
                "heading": heading,
                "start_char": group[0]["start"],
                "end_char": group[-1]["end"],
                "words": words(body),
                "text": body,
            }
        )
    return chunks


def main() -> None:
    model_dir = ensure_model(model_dir_from_env())
    encoder = BgeM3(model_dir)

    documents = sorted(path for path in KB_DIR.glob("*.md") if path.name != "README.md")
    chunks: list[dict] = []
    for path in documents:
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].removeprefix("# ").strip()
        chunks.extend(build_chunks(text, title, f"knowledge_base/{path.name}"))

    print(f"documents: {len(documents)}, chunks: {len(chunks)}")

    # The title and section heading give short chunks enough context for retrieval.
    to_embed = [
        f"{chunk['title']}. {chunk['heading']}. {chunk['text']}"
        if chunk["heading"]
        else f"{chunk['title']}. {chunk['text']}"
        for chunk in chunks
    ]

    started = time.perf_counter()
    vectors = encoder.encode(to_embed)
    elapsed = time.perf_counter() - started

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIMENSION))
    index.add_with_ids(vectors, np.arange(len(chunks), dtype=np.int64))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    payload = {
        "model": MODEL_NAME,
        "dimension": DIMENSION,
        "chunking": {"unit": "words", "min": MIN_WORDS, "max": MAX_WORDS},
        "count": len(chunks),
        "embedding_seconds": round(elapsed, 1),
        "chunks": chunks,
    }
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    sizes = [chunk["words"] for chunk in chunks]
    print(f"embedding: {elapsed:.1f}s, words per chunk: {min(sizes)}-{max(sizes)}")
    print(f"index written to {INDEX_DIR / 'faiss.index'}")


if __name__ == "__main__":
    main()
