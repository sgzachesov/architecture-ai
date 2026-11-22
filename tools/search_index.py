"""Ищет по готовому FAISS-индексу базы знаний.

Запуск: python tools/search_index.py
        python tools/search_index.py "ваш вопрос" --top-k 5

Скрипт печатает найденные чанки с оценкой близости, файлом-источником
и позицией в нём, чтобы бот мог ссылаться на них в ответе.
Результаты запросов по умолчанию можно сохранить в index/search_examples.md.
"""

# Позволяет использовать современные аннотации типов.
from __future__ import annotations

# argparse — разбор аргументов командной строки.
import argparse
# json — вывод результата в машиночитаемом виде.
import json
# sys — добавляем папку tools в путь импорта.
import sys
# Path — работаем с путями к файлам.
from pathlib import Path

# faiss — чтение индекса и поиск по нему.
import faiss
# numpy — не используется напрямую, но нужен faiss для матрицы запроса.
import numpy as np

# Чтобы `from embedding import ...` работал при запуске из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Кодировщик, который превращает запрос в вектор.
from embedding import BgeM3, model_dir_from_env

# Корень проекта (файл лежит в tools/).
ROOT = Path(__file__).resolve().parent.parent
# Папка с индексом и метаданными.
INDEX_DIR = ROOT / "index"

# Примеры запросов, которые прогоняются, если пользователь не передал свой.
DEFAULT_QUERIES = [
    "Who was Xarn Velgor and how did he turn to the null side of the Synth Flux?",
    "What is the Synth Flux and how do Fluxadepts use it?",
    "What happened to Sorrhas during the Vat Wars?",
    "How does a fluxblade work and what crystal does it use?",
]


def load_index(index_dir: Path) -> tuple[faiss.Index, list[dict]]:
    """Читает с диска индекс FAISS и метаданные чанков."""
    # Загружаем бинарный индекс.
    index = faiss.read_index(str(index_dir / "faiss.index"))
    # Читаем JSON с чанками и достаём список.
    payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    # Возвращаем индекс и чанки: id в индексе совпадает с позицией в списке.
    return index, payload["chunks"]


def search(encoder: BgeM3, index: faiss.Index, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    """Кодирует запрос и возвращает top_k ближайших чанков."""
    # Превращаем запрос в вектор той же моделью, что кодировала чанки.
    vector = encoder.encode([query])
    # Ищем top_k ближайших векторов: получаем оценки и их id.
    scores, ids = index.search(vector, top_k)
    # Сюда складываем найденные чанки.
    hits = []
    # Проходим по результатам первого (и единственного) запроса.
    for score, chunk_id in zip(scores[0], ids[0]):
        # Отрицательный id означает пустой результат — пропускаем.
        if chunk_id < 0:
            continue
        # Достаём чанк по id.
        chunk = chunks[int(chunk_id)]
        # Собираем всё нужное для ответа и цитирования.
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
    # Возвращаем найденные чанки в порядке убывания близости.
    return hits


def main() -> int:
    """Разбирает аргументы, выполняет запросы и печатает результат."""
    # Описываем параметры командной строки.
    parser = argparse.ArgumentParser(description="Query the FAISS knowledge base index.")
    # Сам запрос: можно не указывать, тогда берутся примеры по умолчанию.
    parser.add_argument("query", nargs="*", help="search query; defaults to the built-in examples")
    # Сколько чанков показывать на один запрос.
    parser.add_argument("--top-k", type=int, default=3)
    # Вывести результат в JSON вместо текста.
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of text")
    # Сохранить результат в Markdown-файл.
    parser.add_argument("--save", type=Path, help="write the results to a Markdown file")
    # Читаем аргументы.
    args = parser.parse_args()

    # Загружаем индекс и метаданные чанков.
    index, chunks = load_index(INDEX_DIR)
    # Готовим кодировщик запросов.
    encoder = BgeM3(model_dir_from_env())

    # Если запрос передан — используем его, иначе прогоняем примеры.
    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    # Выполняем поиск по каждому запросу.
    results = [(query, search(encoder, index, chunks, query, args.top_k)) for query in queries]

    # Режим JSON: печатаем всё одной структурой и выходим.
    if args.json:
        print(json.dumps([{"query": q, "hits": hits} for q, hits in results], ensure_ascii=False, indent=2))
        return 0

    # Здесь накапливаем Markdown-блоки для сохранения.
    blocks: list[str] = []
    # Печатаем результаты по каждому запросу.
    for query, hits in results:
        # Заголовок запроса в консоли.
        print(f"\nQuery: {query}")
        # Начало Markdown-блока.
        lines = [f"### {query}", ""]
        # Перебираем найденные чанки.
        for hit in hits:
            # Печатаем оценку, id и размер чанка.
            print(f"  {hit['score']:.4f}  {hit['chunk_id']}  ({hit['words']} words)")
            # Печатаем начало текста чанка одной строкой.
            print(f"    {hit['text'][:220].replace(chr(10), ' ')}...")
            # В Markdown пишем источник, позицию и раздел для цитирования.
            lines.append(
                f"- **{hit['chunk_id']}** — score `{hit['score']}`, "
                f"source `{hit['source']}`, chars {hit['start_char']}–{hit['end_char']}, "
                f"section “{hit['heading'] or hit['title']}”"
            )
            # Пустая строка перед цитатой.
            lines.append("")
            # Цитата — начало текста чанка.
            lines.append(f"  > {hit['text'][:400].replace(chr(10), ' ')}...")
            # Пустая строка после цитаты.
            lines.append("")
        # Сохраняем блок запроса.
        blocks.append("\n".join(lines))

    # Если просили сохранить — пишем Markdown-файл.
    if args.save:
        args.save.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
        # Сообщаем, куда сохранили.
        print(f"\nsaved to {args.save}")
    # Успешное завершение.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
