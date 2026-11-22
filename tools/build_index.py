"""Собирает FAISS-индекс по базе знаний.

Запуск: python tools/build_index.py

Скрипт читает knowledge_base/*.md (кроме README.md), режет документы на чанки
до 300 слов, кодирует их моделью BAAI/bge-m3 (ONNX int8) и сохраняет
index/faiss.index вместе с index/chunks.json, где лежат метаданные для цитирования.
"""

# Позволяет использовать современные аннотации типов.
from __future__ import annotations

# json — сохраняем метаданные чанков в chunks.json.
import json
# re — режем текст по предложениям и чистим заголовки.
import re
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


def words(text: str) -> int:
    """Считает слова в строке простым разбиением по пробелам."""
    return len(text.split())


def parse_blocks(text: str) -> list[dict]:
    """Разбирает Markdown на абзацы с заголовком раздела и позицией в файле."""
    # Сюда складываем найденные абзацы.
    blocks: list[dict] = []
    # Текущий раздел документа (заголовок ##).
    heading = ""
    # Смещение от начала файла в символах.
    offset = 0
    # Идём по строкам, сохраняя переводы строк, чтобы точно считать позиции.
    for line in text.splitlines(keepends=True):
        # Строка без пробелов по краям.
        stripped = line.strip()
        # Начало строки в исходном файле.
        start = offset
        # Сдвигаем смещение на длину строки.
        offset += len(line)
        # Пустые строки пропускаем.
        if not stripped:
            continue
        # Заголовок раздела запоминаем и в текст абзаца не включаем.
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            continue
        # Заголовок документа (# ...) тоже не нужен: он хранится в title.
        if stripped.startswith("# "):
            continue
        # Служебную пометку об обрезке документа пропускаем.
        if stripped.startswith("_(excerpt truncated)_"):
            continue
        # Обычная строка становится абзацем с точными границами.
        blocks.append({"start": start, "end": start + len(stripped), "heading": heading, "text": stripped})
    # Возвращаем абзацы в порядке следования.
    return blocks


def split_block(block: dict, max_words: int = MAX_WORDS) -> list[dict]:
    """Делит слишком длинный абзац по предложениям, а если надо — по словам."""
    # Короткий абзац уже годится как один кусок.
    if words(block["text"]) <= max_words:
        return [block]

    # Сюда складываем получившиеся куски.
    pieces: list[dict] = []
    # Исходный текст абзаца.
    text = block["text"]
    # Позиция, с которой продолжается поиск предложений.
    cursor = 0
    # Предложения, накопленные для текущего куска: (позиция, текст).
    current: list[tuple[int, str]] = []

    def flush() -> None:
        """Сохраняет накопленные предложения как отдельный кусок."""
        # Пустой буфер сохранять нечего.
        if not current:
            return
        # Границы куска: от первого предложения до конца последнего.
        start = block["start"] + current[0][0]
        end = block["start"] + current[-1][0] + len(current[-1][1])
        # Добавляем кусок с тем же заголовком раздела.
        pieces.append(
            {
                "start": start,
                "end": end,
                "heading": block["heading"],
                "text": " ".join(sentence for _, sentence in current),
            }
        )
        # Очищаем буфер под следующий кусок.
        current.clear()

    # Режем абзац на предложения по точкам, восклицательным и вопросительным знакам.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        # Ищем, где предложение начинается в исходном тексте.
        position = text.find(sentence, cursor)
        # Если не нашли (редкий случай), считаем началом текущую позицию.
        if position < 0:
            position = cursor
        # Сдвигаем курсор за найденное предложение.
        cursor = position + len(sentence)

        # Если одно предложение само длиннее лимита — режем его по словам.
        if words(sentence) > max_words:
            # Сначала сохраняем всё, что накопили до него.
            flush()
            # Смещение внутри предложения.
            offset = 0
            # Идём по предложению кусками примерно по max_words слов.
            for part in range(0, len(sentence), max_words * 6):
                # Берём очередной кусок текста.
                piece = sentence[part : part + max_words * 6]
                # Сохраняем его как самостоятельный кусок с точными границами.
                pieces.append(
                    {
                        "start": block["start"] + position + offset,
                        "end": block["start"] + position + offset + len(piece),
                        "heading": block["heading"],
                        "text": piece,
                    }
                )
                # Сдвигаем смещение на длину куска.
                offset += len(piece)
            # Переходим к следующему предложению.
            continue

        # Если добавление предложения переполнит лимит — закрываем текущий кусок.
        if current and sum(words(item) for _, item in current) + words(sentence) > max_words:
            flush()
        # Добавляем предложение в текущий кусок.
        current.append((position, sentence))

    # Сохраняем последний накопленный кусок.
    flush()
    # Возвращаем все куски абзаца.
    return pieces


def group_words(group: list[dict]) -> int:
    """Считает суммарное число слов в группе кусков."""
    return sum(words(piece["text"]) for piece in group)


def split_evenly(group: list[dict], parts: int) -> list[list[dict]]:
    """Делит группу на `parts` групп примерно одинакового размера."""
    # Целевой размер одной группы в словах.
    target = group_words(group) / parts
    # Лимит для дробления отдельных кусков: не меньше MIN_WORDS.
    limit = max(MIN_WORDS, int(target))

    # Сначала дробим куски, которые сами больше целевого размера.
    pieces: list[dict] = []
    for piece in group:
        pieces.extend(split_block(piece, limit))

    # Общее число слов во всех кусках.
    total = group_words(pieces)
    # Готовые группы.
    result: list[list[dict]] = []
    # Текущая набираемая группа.
    current: list[dict] = []
    # Раскладываем куски по группам.
    for piece in pieces:
        if current:
            # Сколько слов будет, если добавить этот кусок.
            would = group_words(current) + words(piece["text"])
            # Сколько групп ещё останется после текущей.
            bins_left = parts - len(result) - 1
            # Сколько слов останется после добавления куска.
            rest = total - would
            # Превышение жёсткого лимита — группа обязательно закрывается.
            too_big = would > MAX_WORDS
            # Мягкое условие: группа уже набрала минимум, превысила цель,
            # и оставшихся слов хватит на все последующие группы.
            fair_cut = (
                bins_left > 0
                and group_words(current) >= MIN_WORDS
                and would > target
                and rest >= bins_left * MIN_WORDS
            )
            # Закрываем группу, если сработало любое из условий.
            if too_big or fair_cut:
                result.append(current)
                current = []
        # Добавляем кусок в текущую группу.
        current.append(piece)
    # Сохраняем последнюю группу.
    if current:
        result.append(current)

    # Если последняя группа слишком короткая, переносим в неё куски из предыдущей.
    while (
        len(result) > 1
        and group_words(result[-1]) < MIN_WORDS
        and len(result[-2]) > 1
    ):
        # Забираем последний кусок у предыдущей группы.
        result[-1].insert(0, result[-2].pop())

    # Возвращаем выровненные группы.
    return result


def balance(groups: list[list[dict]]) -> list[list[dict]]:
    """Склеивает короткие группы с соседними и делит их заново.

    Группа короче MIN_WORDS присоединяется к ближайшей соседней, после чего
    объединение делится так, чтобы каждый чанк попал в диапазон MIN_WORDS..MAX_WORDS.
    """
    # Индекс текущей проверяемой группы.
    index = 0
    # Идём по списку, пока в нём больше одной группы.
    while index < len(groups) and len(groups) > 1:
        # Группа достаточно большая — переходим к следующей.
        if group_words(groups[index]) >= MIN_WORDS:
            index += 1
            continue

        # Ищем соседа слева и справа.
        left = index - 1 if index > 0 else None
        right = index + 1 if index < len(groups) - 1 else None
        # Если слева никого нет — берём правого.
        if left is None:
            partner = right
        # Если справа никого нет — берём левого.
        elif right is None:
            partner = left
        # Иначе берём того, кто меньше по числу слов.
        else:
            partner = left if group_words(groups[left]) <= group_words(groups[right]) else right

        # Упорядочиваем индексы пары.
        low, high = min(index, partner), max(index, partner)
        # Объединяем куски обеих групп.
        combined = groups[low] + groups[high]
        # Считаем общий размер объединения.
        total = group_words(combined)
        # На сколько групп его нужно разделить (округление вверх).
        parts = max(1, -(-total // MAX_WORDS))
        # Одна группа — оставляем как есть, иначе делим равномерно.
        replacement = [combined] if parts == 1 else split_evenly(combined, parts)

        # Если после деления всё равно осталась слишком короткая группа —
        # пара не поддаётся выравниванию, оставляем её как есть.
        if len(replacement) == 2 and any(group_words(item) < MIN_WORDS for item in replacement):
            index += 1  # выравнивание эту пару не улучшит
            continue

        # Заменяем две группы на их переработанную версию.
        del groups[high]
        del groups[low]
        groups[low:low] = replacement
        # Возвращаемся на шаг назад: соседи изменились.
        index = max(0, low - 1)
    # Возвращаем выровненный список групп.
    return groups


def build_chunks(text: str, title: str, source: str) -> list[dict]:
    """Превращает документ в список чанков с метаданными."""
    # Разбираем документ на абзацы и дробим слишком длинные.
    pieces: list[dict] = []
    for block in parse_blocks(text):
        pieces.extend(split_block(block))

    # Собираем куски в группы не больше MAX_WORDS.
    groups: list[list[dict]] = []
    current: list[dict] = []
    for piece in pieces:
        # Если добавление куска переполнит лимит — начинаем новую группу.
        if current and group_words(current) + words(piece["text"]) > MAX_WORDS:
            groups.append(current)
            current = []
        # Добавляем кусок в текущую группу.
        current.append(piece)
    # Сохраняем последнюю группу.
    if current:
        groups.append(current)
    # Выравниваем размеры групп.
    groups = balance(groups)

    # Формируем итоговые чанки.
    chunks: list[dict] = []
    for number, group in enumerate(groups, start=1):
        # Текст чанка: куски через пустую строку.
        body = "\n\n".join(item["text"] for item in group)
        # Раздел берём у первого куска группы.
        heading = group[0]["heading"]
        # Складываем чанк со всеми метаданными для цитирования.
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
