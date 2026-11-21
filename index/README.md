# Векторный индекс

Индекс собран по базе знаний из `knowledge_base/` — 38 документов о вымышленной вселенной Aetherium Concord.

## Модель эмбеддингов

| | |
|---|---|
| Модель | `BAAI/bge-m3` |
| Репозиторий | https://huggingface.co/BAAI/bge-m3 |
| Размерность | 1024 |
| Формат | ONNX int8 (`Xenova/bge-m3`), инференс через `onnxruntime`, пулинг по CLS-токену, L2-нормализация |

Модель эмбеддингов — не LLM: она только превращает текст в вектор, генерация ответов в индексации не участвует.

## Чанки

Документ режется по абзацам, длинные абзацы — по предложениям, чтобы чанк не превышал 300 слов. Короткие хвосты склеиваются с соседним чанком и при необходимости делятся заново, поэтому почти все чанки попадают в диапазон 100–300 слов.

| | |
|---|---|
| Чанков в индексе | 228 |
| Слов в чанке | 91–300, медиана 224 |
| Время генерации эмбеддингов | ~70 с на CPU (228 чанков) |

Каждый чанк хранит метаданные для цитирования: `chunk_id`, `source` (путь к файлу), `title`, `heading` (раздел), `start_char` / `end_char` (позиция в исходном файле), `words`.

## Файлы

- `faiss.index` — индекс FAISS (`IndexIDMap2` + `IndexFlatIP`, косинусная близость на нормализованных векторах).
- `chunks.json` — метаданные чанков и текст, по индексу совпадает с id в FAISS.
- `search_examples.md` — примеры запросов и найденные чанки.

## Запуск

```bash
pip install numpy onnxruntime tokenizers faiss-cpu

python tools/build_index.py            # собрать индекс
python tools/search_index.py           # примеры запросов
python tools/search_index.py "Who was Xarn Velgor?" --top-k 5
```

При первом запуске `build_index.py` скачивает веса модели (~570 МБ) в `models/bge-m3/`; папка `models/` в git не попадает.

## Пример запроса

```
Query: How does a fluxblade work and what crystal does it use?

  0.6764  fluxblade-001  (179 words)
    The fluxblade, also referred to as a laser sword by those who were unfamiliar with it,
    was a weapon usually used by the Fluxadept, Voidsworn, and other Flux-sensitives...

  0.6288  fluxblade-002  (219 words)
    Fluxblades were generally used for both offense and defense. A fluxblade could cut
    through virtually anything, from flesh to blast doors...
```

Полный список примеров — в `search_examples.md`.
