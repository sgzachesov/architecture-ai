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

## Скрипты в `tools/`

| Скрипт | Роль |
|---|---|
| `build_index.py` | собирает индекс: читает документы, режет их на чанки, кодирует и сохраняет `index/faiss.index` и `index/chunks.json`. Аргументов не принимает. |
| `embedding.py` | общий модуль, отдельно не запускается. Скачивает веса bge-m3 в `models/bge-m3/`, держит ONNX-сессию и метод `encode()`, которым пользуются оба скрипта. |
| `search_index.py` | поиск по индексу: кодирует запрос тем же энкодером и печатает топ-k чанков с метаданными. |
| `build_knowledge_base.py` | из задания 2: скачивает статьи и заменяет термины. Нужен только для пересборки самой базы знаний, к индексу отношения не имеет. |

Как они связаны:

```
сборка:   knowledge_base/*.md → чанки → embedding.py → index/faiss.index + index/chunks.json
поиск:    запрос → embedding.py → поиск в faiss.index → топ-k чанков с метаданными
```

`search_index.py` принимает:

| Аргумент | Что делает |
|---|---|
| `"запрос"` | поисковый запрос; если не указан, берутся четыре примера из `DEFAULT_QUERIES` |
| `--top-k N` | сколько чанков вернуть, по умолчанию 3 |
| `--json` | вывести результат в JSON вместо текста |
| `--save FILE` | сохранить результаты в Markdown-файл |

## Запуск

Python на компьютере не нужен — всё считается в контейнере. Достаточно Docker Desktop.

### 1. Запустить Docker

Открыть Docker Desktop и дождаться, пока движок начнёт отвечать:

```powershell
docker info --format "{{.ServerVersion}}"
```

### 2. Собрать образ

Из корня проекта, там где лежит `Dockerfile`:

```powershell
cd C:\Project\lern\ya-architecture\architecture-ai
docker build -t quantumforge-rag .
```

Сборка занимает 2–3 минуты: скачивает зависимости из `tools/requirements.txt`. Повторять нужно только после изменения этого файла.

### 3. Запускать скрипты

Папку проекта надо смонтировать в `/app`, иначе контейнер не увидит базу знаний, индекс и веса модели:

```powershell
docker run --rm -v "C:\Project\lern\ya-architecture\architecture-ai:/app" quantumforge-rag tools/search_index.py
docker run --rm -v "C:\Project\lern\ya-architecture\architecture-ai:/app" quantumforge-rag tools/search_index.py "Who was Xarn Velgor?" --top-k 5
docker run --rm -v "C:\Project\lern\ya-architecture\architecture-ai:/app" quantumforge-rag tools/build_index.py
```

В bash тот же путь короче:

```bash
docker run --rm -v "$(pwd):/app" quantumforge-rag tools/search_index.py
```

Файлы, которые создаёт контейнер (`index/faiss.index`, `index/chunks.json`, `index/search_examples.md`), появляются сразу в папке проекта.

### Первый запуск

`build_index.py` скачивает веса модели (~570 МБ) в `models/bge-m3/` — через смонтированную папку, то есть на диск хоста. При следующем запуске они уже на месте. Папка `models/` в git не попадает.

### Без Docker

Если Python 3.12 уже установлен:

```bash
python -m pip install -r tools/requirements.txt
python tools/build_index.py
python tools/search_index.py
```

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
