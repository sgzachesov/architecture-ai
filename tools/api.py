"""REST-сервис RAG-бота (FastAPI) для задания 4.

Оборачивает ядро `rag_bot` в HTTP-API:
    GET  /health  — проверка живости и текущий режим генерации;
    GET  /        — краткая информация о сервисе;
    POST /chat    — вопрос пользователя → ответ с источниками.

Запуск через docker-compose (см. docker-compose.yml в корне проекта):
    docker compose up --build
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"query": "Кто такой Арэн Велмор?"}'
"""

# Позволяет использовать современные аннотации типов.
from __future__ import annotations

# sys — добавляем папку tools в путь импорта.
import sys
# Path — определяем папку, где лежит этот файл.
from pathlib import Path

# FastAPI — веб-фреймворк; HTTPException — понятные ошибки клиенту.
from fastapi import FastAPI, HTTPException
# BaseModel и Field — схемы запроса/ответа с валидацией.
from pydantic import BaseModel, Field

# Чтобы `import rag_bot` работал при запуске из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ядро RAG: поиск, промптинг и генерация ответа.
import rag_bot  # noqa: E402

# Приложение FastAPI с описанием для встроенной документации /docs.
app = FastAPI(
    title="QuantumForge RAG Bot",
    description="RAG-бот по базе знаний: FAISS + bge-m3 + few-shot и Chain-of-Thought.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """Запрос пользователя к боту."""

    # Вопрос обязателен и не может быть пустым.
    query: str = Field(..., min_length=1, description="Вопрос пользователя")
    # Сколько чанков искать; по умолчанию берётся TOP_K из окружения.
    top_k: int | None = Field(None, ge=1, le=20, description="Число фрагментов для поиска")
    # Порог релевантности; ниже него бот отвечает «Я не знаю».
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Порог косинусной близости")


class Source(BaseModel):
    """Источник, на который опирался ответ."""

    # Идентификатор чанка вида <slug>-<номер>.
    chunk_id: str
    # Путь к исходному документу базы знаний.
    source: str
    # Заголовок документа.
    title: str
    # Раздел документа.
    heading: str
    # Позиция чанка в исходном файле.
    start_char: int
    end_char: int
    # Оценка близости к запросу.
    score: float


class ChatResponse(BaseModel):
    """Ответ бота."""

    # Исходный вопрос.
    query: str
    # Текст ответа.
    answer: str
    # True, если ответ собран демо-режимом без вызова LLM.
    demo: bool
    # Источники для цитирования.
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    """Проверка живости сервиса и текущего режима генерации."""
    # demo=True означает, что OPENAI_API_KEY не задан и работает демо-режим.
    return {"status": "ok", "demo": not bool(rag_bot.OPENAI_API_KEY), "model": rag_bot.LLM_MODEL}


@app.get("/")
def root() -> dict:
    """Краткая информация о сервисе и доступных эндпоинтах."""
    # Отдаём подсказку, куда обращаться.
    return {
        "service": "QuantumForge RAG Bot",
        "endpoints": {"chat": "POST /chat", "health": "GET /health", "docs": "GET /docs"},
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Принимает вопрос, прогоняет цепочку RAG и возвращает ответ с источниками."""
    # Подставляем значения по умолчанию из окружения, если клиент их не передал.
    top_k = request.top_k or rag_bot.TOP_K
    threshold = rag_bot.SCORE_THRESHOLD if request.threshold is None else request.threshold

    try:
        # Запускаем полную цепочку: поиск → промпт → генерация.
        result = rag_bot.answer(request.query, top_k=top_k, threshold=threshold)
    except Exception as error:  # noqa: BLE001
        # Ошибки внешнего LLM (ключ, лимиты, сеть) отдаём как 502 Bad Gateway.
        raise HTTPException(status_code=502, detail=f"LLM request failed: {error}") from error

    # Собираем компактный список источников и возвращаем ответ.
    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        demo=result["demo"],
        sources=[Source(**item) for item in rag_bot.sources(result["hits"])],
    )
