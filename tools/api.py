"""REST-сервис RAG-бота (FastAPI) для задания 4.

Оборачивает ядро `rag_bot` в HTTP-API:
    GET  /health  — проверка живости и текущий режим генерации;
    GET  /        — краткая информация о сервисе;
    POST /chat    — вопрос пользователя → ответ с источниками.

Встроенная документация Swagger UI доступна по /docs, схема OpenAPI — по
/openapi.json, альтернативная документация ReDoc — по /redoc.

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
# BaseModel — схемы запроса/ответа; ConfigDict — примеры для Swagger;
# Field — валидация и описания отдельных полей.
from pydantic import BaseModel, ConfigDict, Field

# Чтобы `import rag_bot` работал при запуске из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ядро RAG: поиск, промптинг и генерация ответа.
import rag_bot  # noqa: E402


# Разделы Swagger UI: какие эндпоинты к какой группе относятся.
TAGS_METADATA = [
    {
        "name": "bot",
        "description": "Вопросы к базе знаний: поиск по индексу, промптинг, ответ.",
    },
    {
        "name": "service",
        "description": "Служебные эндпоинты: живость сервиса и справка.",
    },
]

# Описание сервиса на главной странице Swagger UI (поддерживается Markdown).
DESCRIPTION = """
RAG-бот по базе знаний вымышленной вселенной QuantumForge Software.

**Цепочка:** вопрос → эмбеддинг bge-m3 → поиск `top-k` чанков в FAISS →
LCEL-цепочка (`RunnableParallel → ChatPromptTemplate → LLM → StrOutputParser`)
с few-shot и Chain-of-Thought → ответ LLM → источники.

| Режим | Когда включается | Что делает |
|---|---|---|
| LLM | задан `OPENAI_API_KEY` | промпт уходит в OpenAI-совместимый `/chat/completions` |
| Демо | ключа нет | ответ детерминированно собирается из найденного чанка |

**Как проверить:** откройте `POST /chat` → «Try it out» → введите вопрос →
«Execute». Готовые примеры есть в поле `query` схемы запроса.
"""

# Приложение FastAPI с описанием для встроенной документации /docs.
app = FastAPI(
    title="QuantumForge RAG Bot",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    # Явно закрепляем адреса документации: Swagger UI, схема, ReDoc.
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc",
)


class ChatRequest(BaseModel):
    """Запрос пользователя к боту."""

    # Готовые примеры показываются в Swagger UI в выпадающем списке.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "Кто такой Арэн Велмор?"},
                {"query": "Как работает флюксблейд и какой в нём кристалл?", "top_k": 5},
                {"query": "Как приготовить борщ?"},
            ]
        }
    )

    # Вопрос обязателен и не может быть пустым.
    query: str = Field(
        ...,
        min_length=1,
        description="Вопрос пользователя на русском языке",
        examples=["Кто такой Арэн Велмор?"],
    )
    # Сколько чанков искать; по умолчанию берётся TOP_K из окружения.
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Сколько фрагментов искать и передать в промпт (по умолчанию TOP_K=3)",
        examples=[3],
    )
    # Порог релевантности; ниже него бот отвечает «Я не знаю».
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Порог косинусной близости; ниже него бот отвечает «Я не знаю» (по умолчанию 0.45)",
        examples=[0.45],
    )


class Source(BaseModel):
    """Источник, на который опирался ответ."""

    # Идентификатор чанка вида <slug>-<номер>.
    chunk_id: str = Field(..., description="Идентификатор чанка в индексе", examples=["aran-velmor-001"])
    # Путь к исходному документу базы знаний.
    source: str = Field(..., description="Файл базы знаний", examples=["knowledge_base/aran-velmor.md"])
    # Заголовок документа.
    title: str = Field(..., description="Заголовок документа", examples=["Aran Velmor"])
    # Раздел документа.
    heading: str = Field("", description="Раздел документа (может быть пустым)")
    # Позиция чанка в исходном файле.
    start_char: int = Field(..., description="Начало фрагмента в файле, символы", examples=[15])
    end_char: int = Field(..., description="Конец фрагмента в файле, символы", examples=[557])
    # Оценка близости к запросу.
    score: float = Field(..., description="Косинусная близость запроса и чанка, 0–1", examples=[0.552])


class ChatResponse(BaseModel):
    """Ответ бота."""

    # Пример полного ответа для Swagger UI.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Кто такой Арэн Велмор?",
                    "answer": (
                        "Размышление:\n"
                        "1. На вопрос отвечает фрагмент «aran-velmor-001» "
                        "(оценка близости 0.552, раздел «Aran Velmor»).\n"
                        "2. Ключевой факт из фрагмента: Aran Velmor was a legendary "
                        "Flux-sensitive human male...\n\n"
                        "Ответ на основе базы знаний: Aran Velmor was a legendary "
                        "Flux-sensitive human male..."
                    ),
                    "demo": True,
                    "sources": [
                        {
                            "chunk_id": "aran-velmor-001",
                            "source": "knowledge_base/aran-velmor.md",
                            "title": "Aran Velmor",
                            "heading": "Aran Velmor",
                            "start_char": 15,
                            "end_char": 557,
                            "score": 0.552,
                        }
                    ],
                }
            ]
        }
    )

    # Исходный вопрос.
    query: str = Field(..., description="Исходный вопрос")
    # Текст ответа.
    answer: str = Field(..., description="Ответ бота (в демо-режиме — с шагами CoT)")
    # True, если ответ собран демо-режимом без вызова LLM.
    demo: bool = Field(..., description="True, если LLM не вызывалась (демо-режим)")
    # Источники для цитирования.
    sources: list[Source] = Field(..., description="Найденные фрагменты, на которые опирался ответ")


class HealthResponse(BaseModel):
    """Состояние сервиса."""

    status: str = Field(..., description="Всегда ok, если сервис отвечает", examples=["ok"])
    demo: bool = Field(..., description="True, если OPENAI_API_KEY не задан")
    model: str = Field(..., description="Имя модели генератора", examples=["gpt-4o-mini"])


class ServiceInfo(BaseModel):
    """Краткая справка о сервисе."""

    service: str = Field(..., description="Название сервиса")
    endpoints: dict[str, str] = Field(..., description="Доступные эндпоинты")


@app.get(
    "/health",
    tags=["service"],
    summary="Проверка живости",
    description="Возвращает статус сервиса и текущий режим генерации (LLM или демо).",
    response_model=HealthResponse,
    response_description="Сервис доступен",
)
def health() -> HealthResponse:
    """Проверка живости сервиса и текущего режима генерации."""
    # demo=True означает, что OPENAI_API_KEY не задан и работает демо-режим.
    return HealthResponse(
        status="ok",
        demo=not bool(rag_bot.OPENAI_API_KEY),
        model=rag_bot.LLM_MODEL,
    )


@app.get(
    "/",
    tags=["service"],
    summary="Информация о сервисе",
    description="Краткая справка: название сервиса и список эндпоинтов.",
    response_model=ServiceInfo,
    response_description="Справка о сервисе",
)
def root() -> ServiceInfo:
    """Краткая информация о сервисе и доступных эндпоинтах."""
    # Отдаём подсказку, куда обращаться.
    return ServiceInfo(
        service="QuantumForge RAG Bot",
        endpoints={"chat": "POST /chat", "health": "GET /health", "docs": "GET /docs"},
    )


@app.post(
    "/chat",
    tags=["bot"],
    summary="Задать вопрос базе знаний",
    description=(
        "Прогоняет полную цепочку RAG: эмбеддинг вопроса → поиск в FAISS → "
        "LCEL-цепочка (промпт с few-shot и Chain-of-Thought) → "
        "ответ LLM (или демо-ответ).\n\n"
        "Если лучший чанк ниже порога `threshold`, бот отвечает «Я не знаю» "
        "без вызова LLM."
    ),
    response_model=ChatResponse,
    response_description="Ответ бота и источники",
    responses={
        422: {"description": "Некорректный запрос: пустой query, top_k вне 1–20 или threshold вне 0–1"},
        502: {"description": "Ошибка внешней LLM: неверный ключ, лимит запросов, сеть"},
    },
)
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
