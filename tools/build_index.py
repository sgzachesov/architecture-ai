"""Собирает FAISS-индекс по базе знаний.

Запуск: python tools/build_index.py

Скрипт читает knowledge_base/*.md (кроме README.md), режет документы на чанки
100–300 слов с помощью RecursiveCharacterTextSplitter (langchain-text-splitters),
кодирует их моделью BAAI/bge-m3 (ONNX int8) и сохраняет index/faiss.index вместе
с index/chunks.json, где лежат метаданные для цитирования.
"""

# Позволяет использовать современные аннотации типов.
from __future__ import annotations

# json — сохраняем метаданные чанков в chunks.json.
import json
# sys — добавляем папку tools в путь импорта.
import sys
# time — измеряем, сколько заняла генерация эмбеддингов.
import time
# Path — работаем с путями к файлам и папкам.
from pathlib import Path

# faiss — векторный индекс.
import faiss
# numpy — матрица векторов и массив id.
import numpy as np
# RecursiveCharacterTextSplitter — готовая нарезка текста по разделителям.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Чтобы `from embedding import ...` работал при запуске из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Общий кодировщик и константы модели.
from embedding import DIMENSION, MODEL_NAME, BgeM3, ensure_model, model_dir_from_env

# Корень проекта (файл лежит в tools/).
ROOT = Path(__file__).resolve().parent.parent
# Откуда берём документы.
KB_DIR = ROOT / "knowledge_base"
# Куда пишем индекс.
INDEX_DIR = ROOT / "index"

# Нижняя граница размера чанка в словах.
MIN_WORDS = 100
# Верхняя граница размера чанка в словах.
MAX_WORDS = 300

# Готовая функция из langchain-text-splitters: режет текст по разделителям
# от абзаца к строке, предложению и слову, пока чанк не наберёт MAX_WORDS.
# Длина считается в словах (length_function), перекрытие между чанками не нужно,
# а add_start_index сохраняет позицию чанка в исходном тексте.
SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=MAX_WORDS,
    chunk_overlap=0,
    length_function=lambda text: len(text.split()),
    add_start_index=True,
)


def words(text: str) -> int:
    """Считает слова в строке простым разбиением по пробелам."""
    return len(text.split())


def prepare(text: str) -> tuple[str, list[tuple[int, str]], list[tuple[int, int]]]:
    """Готовит текст для сплиттера и карту для восстановления позиций.

    Заголовки документов и разделов («# …», «## …» ) убираются из текста:
    сплиттер работает только с телом документа. Вместо заголовка остаётся
    перевод строки, поэтому общие длины строк не меняются. Возвращается:
    текст для сплиттера, список заголовков (позиция, название) и таблица
    поправок «маскированная позиция → сдвиг к исходной».
    """
    # Сюда складываем строки тела документа.
    out: list[str] = []
    # Заголовки разделов: (позиция в исходном тексте, название).
    headings: list[tuple[int, str]] = []
    # Поправки: (маскированная позиция, на сколько исходная больше маскированной).
    adjust: list[tuple[int, int]] = []
    # Текущая позиция в исходном тексте.
    orig_offset = 0
    # Текущая позиция в маскированном тексте.
    masked_offset = 0
    # Идём по строкам, сохраняя переводы строк.
    for line in text.splitlines(keepends=True):
        # Строка без пробелов по краям.
        stripped = line.strip()
        # Это заголовок документа или раздела?
        is_heading = stripped.startswith("## ") or stripped.startswith("# ")
        # Заголовок раздела запоминаем как метаданные.
        if is_heading and stripped.startswith("## "):
            headings.append((orig_offset, stripped[3:].strip()))
        # Вместо заголовка оставляем один перевод строки, иначе — исходную строку.
        placeholder = "\n" if is_heading else line
        out.append(placeholder)
        # Сохраняем поправку для текущей точки.
        adjust.append((masked_offset, orig_offset - masked_offset))
        # Сдвигаем обе позиции на длину строки.
        orig_offset += len(line)
        masked_offset += len(placeholder)
    # Возвращаем текст, заголовки и таблицу поправок.
    return "".join(out), headings, adjust


def to_original(masked_pos: int, adjust: list[tuple[int, int]]) -> int:
    """Переводит позицию в маскированном тексте в позицию исходного файла."""
    # Сдвиг — последняя поправка, начинающаяся не позже нашей позиции.
    delta = 0
    for position, shift in adjust:
        if position <= masked_pos:
            delta = shift
        else:
            break
    # Прибавляем поправку к маскированной позиции.
    return masked_pos + delta


def heading_at(headings: list[tuple[int, str]], orig_pos: int) -> str:
    """Возвращает последний заголовок раздела, начавшийся раньше orig_pos."""
    # Заголовок по умолчанию пустой (документ без разделов).
    heading = ""
    for position, name in headings:
        # Заголовок раньше нашего фрагмента — запоминаем.
        if position < orig_pos:
            heading = name
        else:
            # Заголовок уже внутри фрагмента — дальше не идём.
            break
    # Возвращаем найденный заголовок.
    return heading


def merge_small(pieces: list[dict]) -> list[dict]:
    """Склеивает слишком короткие чанки с соседями, пока помещаются в MAX_WORDS."""
    # Сюда складываем итоговые чанки.
    merged: list[dict] = []
    # Индекс текущего чанка в исходном списке.
    index = 0
    # Идём по всем чанкам.
    while index < len(pieces):
        piece = pieces[index]
        # Первый проход: склеиваем со следующими, пока чанк короче MIN_WORDS
        # и следующий помещается в лимит.
        while (
            index + 1 < len(pieces)
            and words(piece["text"]) < MIN_WORDS
            and words(piece["text"]) + words(pieces[index + 1]["text"]) <= MAX_WORDS
        ):
            # Берём следующий чанк.
            following = pieces[index + 1]
            # Объединяем их: границы и заголовок берём от объединения.
            piece = {
                "start": piece["start"],
                "end": following["end"],
                "heading": piece["heading"] or following["heading"],
                "text": piece["text"] + "\n\n" + following["text"],
            }
            # Следующий чанк поглощён — сдвигаемся.
            index += 1
        # Второй проход: если чанк всё ещё короткий, а в предыдущий он помещается —
        # приклеиваем его к концу предыдущего чанка.
        while (
            merged
            and words(piece["text"]) < MIN_WORDS
            and words(merged[-1]["text"]) + words(piece["text"]) <= MAX_WORDS
        ):
            previous = merged.pop()
            piece = {
                "start": previous["start"],
                "end": piece["end"],
                "heading": previous["heading"] or piece["heading"],
                "text": previous["text"] + "\n\n" + piece["text"],
            }
        # Добавляем чанк в результат.
        merged.append(piece)
        # Переходим к следующему чанку.
        index += 1
    # Возвращаем список без слишком коротких чанков.
    return merged


def build_chunks(text: str, title: str, source: str) -> list[dict]:
    """Превращает документ в список чанков с метаданными."""
    # Маскируем заголовки и готовим карту позиций.
    masked, headings, adjust = prepare(text)
    # Режем документ готовым сплиттером: каждый фрагмент с позицией start_index.
    documents = SPLITTER.create_documents([masked])
    # Сюда складываем чанки с восстановленными позициями.
    pieces: list[dict] = []
    # Обрабатываем каждый фрагмент, выданный сплиттером.
    for document in documents:
        # Извлекаем текст фрагмента и убираем случайный «.» в начале.
        body_text = document.page_content
        if body_text.startswith(". "):
            body_text = body_text[2:]
        # Пустые фрагменты и служебную пометку об обрезке отбрасываем.
        if not body_text or not body_text.strip() or body_text.startswith("_(excerpt truncated)_"):
            continue
        # Сколько пробелов убрать в начале, чтобы текст начинался с первого слова.
        leading = len(body_text) - len(body_text.lstrip())
        # Чистый текст фрагмента.
        body = body_text.strip()
        # Переводим позицию начала из маскированного текста в исходный.
        start = to_original(document.metadata.get("start_index", 0), adjust) + leading
        # Складываем словами и метаданными: позиция, раздел, сам текст.
        pieces.append(
            {
                "start": start,
                "end": start + len(body),
                "heading": heading_at(headings, start),
                "text": body,
            }
        )
    # Склеиваем слишком короткие чанки.
    pieces = merge_small(pieces)

    # Формируем итоговые чанки.
    chunks: list[dict] = []
    for number, piece in enumerate(pieces, start=1):
        # Текст чанка.
        body = piece["text"]
        # Складываем чанк со всеми метаданными для цитирования.
        chunks.append(
            {
                "id": len(chunks),
                "chunk_id": f"{Path(source).stem}-{number:03d}",
                "source": source,
                "title": title,
                "heading": piece["heading"],
                "start_char": piece["start"],
                "end_char": piece["end"],
                "words": words(body),
                "text": body,
            }
        )
    # Возвращаем чанки документа.
    return chunks


def main() -> None:
    """Собирает индекс по всем документам базы знаний."""
    # Скачиваем веса модели, если их ещё нет, и запоминаем папку.
    model_dir = ensure_model(model_dir_from_env())
    # Создаём кодировщик.
    encoder = BgeM3(model_dir)

    # Берём все документы базы знаний, кроме README.
    documents = sorted(path for path in KB_DIR.glob("*.md") if path.name != "README.md")
    # Сюда складываем чанки всех документов.
    chunks: list[dict] = []
    # Обрабатываем каждый документ.
    for path in documents:
        # Читаем текст файла.
        text = path.read_text(encoding="utf-8")
        # Заголовок документа — первая строка без "# ".
        title = text.splitlines()[0].removeprefix("# ").strip()
        # Режем документ на чанки и добавляем их к общему списку.
        chunks.extend(build_chunks(text, title, f"knowledge_base/{path.name}"))

    # Печатаем, сколько документов и чанков получилось.
    print(f"documents: {len(documents)}, chunks: {len(chunks)}")

    # Заголовок и раздел дают коротким чанкам контекст для поиска.
    to_embed = [
        f"{chunk['title']}. {chunk['heading']}. {chunk['text']}"
        if chunk["heading"]
        else f"{chunk['title']}. {chunk['text']}"
        for chunk in chunks
    ]

    # Засекаем время генерации эмбеддингов.
    started = time.perf_counter()
    # Кодируем все тексты: получаем матрицу векторов.
    vectors = encoder.encode(to_embed)
    # Считаем, сколько это заняло.
    elapsed = time.perf_counter() - started

    # Создаём индекс: id-обёртка над плоским индексом скалярных произведений.
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIMENSION))
    # Добавляем векторы, используя их порядковый номер как id.
    index.add_with_ids(vectors, np.arange(len(chunks), dtype=np.int64))

    # Создаём папку index/, если её нет.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Сохраняем индекс на диск.
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # Готовим метаданные: модель, размерность, параметры нарезки и сами чанки.
    payload = {
        "model": MODEL_NAME,
        "dimension": DIMENSION,
        "chunking": {"unit": "words", "min": MIN_WORDS, "max": MAX_WORDS},
        "count": len(chunks),
        "embedding_seconds": round(elapsed, 1),
        "chunks": chunks,
    }
    # Записываем chunks.json рядом с индексом.
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Собираем размеры чанков для отчёта.
    sizes = [chunk["words"] for chunk in chunks]
    # Печатаем время генерации и диапазон размеров.
    print(f"embedding: {elapsed:.1f}s, words per chunk: {min(sizes)}-{max(sizes)}")
    # Сообщаем, куда записан индекс.
    print(f"index written to {INDEX_DIR / 'faiss.index'}")


if __name__ == "__main__":
    main()