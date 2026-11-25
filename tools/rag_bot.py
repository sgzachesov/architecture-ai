"""Ядро RAG-бота для задания 4: поиск по индексу + промптинг (few-shot, CoT).

Цепочка: запрос пользователя → эмбеддинг (bge-m3, тот же энкодер, что и при
построении индекса) → поиск ближайших чанков в FAISS → готовая цепочка
LangChain `RetrievalQA` (chain_type="stuff") собирает промпт с few-shot
примерами и Chain-of-Thought, вызывает LLM и возвращает ответ с источниками.

Это библиотека: её использует REST-сервис `tools/api.py`. Напрямую не
запускается.

Если задана переменная окружения OPENAI_API_KEY, ответ генерирует реальная
модель через OpenAI-совместимый API (подходит OpenAI, YandexGPT и локальные
серверы с таким API). Без ключа включается демо-режим: ответ собирается из
найденных чанков по шаблону, чтобы цепочку можно было проверить без сети.
Цепочка RetrievalQA при этом та же, подменяется только LLM.

Техники промптинга:
- Few-shot: два примера Q/A из этой же базы знаний (ответы сверены с чанками).
- Chain-of-Thought: в system-промпте модель просят сначала размышлять по шагам,
  а потом отвечать, и опираться только на предоставленные фрагменты.
- «Я не знаю»: если лучший чанк слишком далёк от запроса (ниже SCORE_THRESHOLD),
  бот отвечает отказом ещё до вызова LLM; та же инструкция есть и в промпте.
"""

# Позволяет использовать современные аннотации типов.
from __future__ import annotations

# os — переменные окружения: ключ API, адрес эндпоинта, имя модели.
import os
# sys — добавляем папку tools в путь импорта.
import sys
# Path — работаем с путями к файлам.
from pathlib import Path

# requests — HTTP-запрос к OpenAI-совместимому /chat/completions.
import requests
# Готовая цепочка «поиск + вопрос → ответ» из LangChain.
from langchain_classic.chains import RetrievalQA
# Document — единица контекста для цепочки.
from langchain_core.documents import Document
# BaseChatModel — базовый класс для своей chat-модели.
from langchain_core.language_models.chat_models import BaseChatModel
# AIMessage — сообщение модели; ChatGeneration/ChatResult — формат её ответа.
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
# ChatPromptTemplate — шаблон промпта с ролями system/human.
from langchain_core.prompts import ChatPromptTemplate
# BaseRetriever — базовый класс ретривера для RetrievalQA.
from langchain_core.retrievers import BaseRetriever
# Field — поля pydantic-моделей LangChain.
from pydantic import Field

# Чтобы `from search_index import ...` работал при запуске из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Готовые функции поиска из задания 3: чтение индекса и поиск ближайших чанков.
from search_index import load_index, search  # noqa: E402

# Общий кодировщик bge-m3: превращает запрос в вектор.
from embedding import BgeM3, model_dir_from_env  # noqa: E402

# Корень проекта (файл лежит в tools/).
ROOT = Path(__file__).resolve().parent.parent
# Папка с индексом и метаданными.
INDEX_DIR = ROOT / "index"

# Сколько чанков брать в выдачу и отдавать модели в промпт.
TOP_K = int(os.environ.get("TOP_K", "3"))
# Ниже этого скоринга (косинусная близость) чанки считаем нерелевантными.
# Порог подобран эмпирически по корпусу: релевантные вопросы дают >=0.465,
# а явно посторонние («погода», «борщ») — <=0.432. Это эвристика: на
# кросс-языковых запросах распределения частично пересекаются, поэтому в
# реальном режиме дополнительно работает инструкция LLM «Я не знаю».
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.45"))
# Адрес OpenAI-совместимого эндпоинта; по умолчанию OpenAI.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
# Имя модели генератора (для OpenAI — например, gpt-4o-mini).
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
# Ключ API берём из окружения, в код он не попадает.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Ответ, когда в базе нет достаточно близкого материала.
REFUSAL = "Я не знаю: в базе знаний не нашлось достаточно близкого материала по этому вопросу."

# System-промпт: роль ассистента, Chain-of-Thought и правила честности.
SYSTEM_PROMPT = (
    "Ты — справочный ассистент корпоративной базы знаний QuantumForge Software. "
    "Отвечай на русском языке.\n\n"
    "Прежде чем ответить, размышляй по шагам (Chain-of-Thought):\n"
    "1) Определи, какой из предоставленных фрагментов отвечает на вопрос.\n"
    "2) Выпиши из него ключевые факты, относящиеся к вопросу.\n"
    "3) Сделай вывод и сформулируй итоговый ответ.\n\n"
    "Правила:\n"
    "- Опирайся ТОЛЬКО на предоставленные фрагменты базы знаний. Не выдумывай факты.\n"
    "- Если среди фрагментов нет ответа на вопрос, ответь ровно: «Я не знаю».\n"
    "- Ответ должен быть кратким и по делу, без лишней воды."
)

# Few-shot примеры. Ответы сверены с реальными чанками индекса
# (aran-velmor-001, the-synth-flux-001) — это не выдуманные сценарии.
FEW_SHOT = [
    {
        "q": "Кто такой Арэн Велмор?",
        "a": "Арэн Велмор — легендарный человек, чувствительный к Синто-Флюксу, "
             "Рыцарь Флюкса Аэтериум-Конкорда и Предсказанный Единства Флюкс-Ордена, "
             "которому суждено принести равновесие Синто-Флюксу. В детстве его звали "
             "«Ари», а за подвиги в Войнах Ват он получил прозвище «Герой без страха». "
             "Его альтер-эго — Ксарн Велгор, Нулевой Лорд Пустотных.",
    },
    {
        "q": "Что такое Синто-Флюкс?",
        "a": "Синто-Флюкс — энергетическое поле, создаваемое всей жизнью и связывающее "
             "вселенную воедино. На протяжении галактической истории его называли "
             "по-разному: Луна у Вокков, Жизненный Поток у Синдрали, Прилив у Левелан "
             "и другими именами.",
    },
]

# Блок few-shot в том виде, в каком он попадает в промпт.
_EXAMPLES = "\n\n".join(f"Вопрос: {example['q']}\nОтвет: {example['a']}" for example in FEW_SHOT)

# Шаблон промпта для RetrievalQA: system (CoT) + примеры + {context} + {question}.
# Плейсхолдеры {context} и {question} заполняет сама цепочка.
PROMPT = ChatPromptTemplate.from_messages(
    [
        # Роль и правила: CoT-порядок действий и запрет выдумывать.
        ("system", SYSTEM_PROMPT),
        # Примеры, фрагменты из базы и вопрос пользователя.
        (
            "human",
            "Примеры правильных ответов (few-shot):\n"
            + _EXAMPLES
            + "\n\nФрагменты из базы знаний:\n{context}\n\nВопрос: {question}\n\nОтвет:",
        ),
    ]
)


def answer(query: str, top_k: int = TOP_K, threshold: float = SCORE_THRESHOLD, demo: bool = False) -> dict:
    """Полная цепочка RAG: поиск → RetrievalQA (промпт + LLM) → ответ с источниками."""
    # Шаг 1–3: ищем ближайшие чанки в векторной базе (с оценками близости).
    hits = retrieve(query, top_k)

    # Реальный LLM вызываем, только если есть ключ и не просили демо.
    use_demo = demo or not OPENAI_API_KEY

    # Шаг 4a: нет релевантных фрагментов — отвечаем отказом до вызова LLM.
    if not hits or hits[0]["score"] < threshold:
        return {"query": query, "answer": REFUSAL, "hits": hits, "demo": use_demo}

    # Шаг 4b: выбираем модель. Без ключа — заглушка с детерминированным ответом.
    llm = DemoChat(text=demo_answer(query, hits)) if use_demo else _chat_model()

    # Шаг 5: готовая цепочка RetrievalQA собирает промпт, вызывает модель
    # и возвращает ответ и документы, на которые она опиралась.
    result = _chain(llm, hits).invoke({"query": query})

    # Возвращаем вопрос, ответ, найденные чанки и режим генерации.
    return {
        "query": query,
        "answer": result["result"],
        "hits": _hits_from_documents(result["source_documents"]),
        "demo": use_demo,
    }


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Ищет ближайшие чанки к запросу тем же энкодером, что строил индекс."""
    # Загружаем индекс и метаданные чанков (кэшируются на уровне модуля).
    index, chunks = _load_resources()
    # Превращаем запрос в вектор и ищем top_k ближайших чанков.
    return search(_encoder(), index, chunks, query, top_k)


def build_prompt(query: str, hits: list[dict]) -> list[dict]:
    """Показывает, какой промпт получит модель (для отладки и логов).

    Используется тот же PROMPT, что и в RetrievalQA, поэтому результат
    побайтово совпадает с тем, что увидит LLM.
    """
    # Подставляем фрагменты и вопрос в общий шаблон.
    messages = PROMPT.format_messages(context=_context(hits), question=query)
    # Переводим сообщения LangChain в привычный формат OpenAI chat.
    return [{"role": _openai_role(message), "content": message.content} for message in messages]


def generate(messages: list[dict]) -> str:
    """Отправляет готовые сообщения в OpenAI-совместимый API (для отладки)."""
    # Используем те же настройки окружения, что и chat-модель в цепочке.
    return _post_chat(LLM_MODEL, messages, OPENAI_BASE_URL, OPENAI_API_KEY, 0.2)


def demo_answer(query: str, hits: list[dict]) -> str:
    """Собирает ответ из найденных чанков по шаблону, без обращения к API.

    Демо-режим нужен, чтобы проверить всю цепочку RAG без ключа LLM: он
    воспроизводит CoT-структуру (шаги рассуждения) детерминированно, беря
    факты из реальных фрагментов базы. При заданном OPENAI_API_KEY этот путь
    заменяется генерацией настоящей моделью.
    """
    # Самый близкий к запросу фрагмент.
    top = hits[0]
    # Первое предложение фрагмента — ключевой факт для ответа.
    first_sentence = top["text"].split(". ")[0].strip() + "."
    # Раздел документа, откуда взят факт.
    section = top["heading"] or top["title"]
    # Детерминированный ответ: шаги CoT + факт из базы + источник.
    return (
        f"Размышление:\n"
        f"1. На вопрос отвечает фрагмент «{top['chunk_id']}» "
        f"(оценка близости {top['score']:.3f}, раздел «{section}»).\n"
        f"2. Ключевой факт из фрагмента: {first_sentence}\n\n"
        f"Ответ на основе базы знаний: {first_sentence}\n"
        f"(демо-режим: ответ собран из фрагмента, LLM не вызывалась)"
    )


def sources(hits: list[dict]) -> list[dict]:
    """Превращает найденные чанки в компактные источники для ответа API."""
    # Для каждого чанка оставляем только то, что нужно клиенту для цитирования.
    return [
        {
            "chunk_id": hit["chunk_id"],
            "source": hit["source"],
            "title": hit["title"],
            "heading": hit["heading"],
            "start_char": hit["start_char"],
            "end_char": hit["end_char"],
            "score": hit["score"],
        }
        for hit in hits
    ]


# --- Внутреннее состояние (детали внизу) --------------------------------------

# Кэш загруженных ресурсов, чтобы не читать их заново на каждый запрос.
_resources_cache: dict = {}


class FoundFragmentsRetriever(BaseRetriever):
    """Ретривер для RetrievalQA, отдающий уже найденные фрагменты.

    Поиск в FAISS выполняется до цепочки: так отказ по порогу срабатывает
    раньше вызова LLM. Цепочка получает готовые документы и занимается
    промптингом и генерацией.
    """

    # Документы, найденные поиском для текущего запроса.
    documents: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """Возвращает документы, подготовленные для этого запроса."""
        return self.documents


class OpenAICompatibleChat(BaseChatModel):
    """Chat-модель LangChain поверх OpenAI-совместимого /chat/completions.

    Своя реализация вместо пакета `openai`, чтобы ключ и адрес брались из
    окружения, а запрос уходил через уже используемый `requests`.
    """

    # Имя модели и адрес эндпоинта.
    model: str
    base_url: str
    api_key: str
    # Низкая температура — ответы ближе к фактам.
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        """Тип модели для LangChain."""
        return "openai-compatible"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """Превращает сообщения LangChain в запрос к API и возвращает ответ."""
        # Переводим сообщения в формат OpenAI chat и отправляем их в LLM.
        text = _post_chat(
            self.model,
            [_openai_message(message) for message in messages],
            self.base_url,
            self.api_key,
            self.temperature,
        )
        # Оборачиваем текст в стандартный результат chat-модели.
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class DemoChat(BaseChatModel):
    """Заглушка LLM для демо-режима: возвращает заранее собранный ответ.

    Она нужна, чтобы цепочка RetrievalQA выполнялась целиком без ключа:
    промпт собирается по-настоящему, а генерация подменяется детерминированным
    текстом из найденного фрагмента.
    """

    # Готовый текст ответа, собранный demo_answer().
    text: str

    @property
    def _llm_type(self) -> str:
        """Тип модели для LangChain."""
        return "demo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """Возвращает заранее подготовленный демо-ответ."""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])


def _chain(llm: BaseChatModel, hits: list[dict]) -> RetrievalQA:
    """Собирает готовую цепочку RetrievalQA: prompt + документы + LLM."""
    # Превращаем найденные чанки в документы с метаданными для цитирования.
    documents = [_document(hit, number) for number, hit in enumerate(hits, start=1)]
    # chain_type="stuff" — все фрагменты кладутся в один промпт.
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=FoundFragmentsRetriever(documents=documents),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )


def _chat_model() -> OpenAICompatibleChat:
    """Создаёт chat-модель из настроек окружения."""
    return OpenAICompatibleChat(
        model=LLM_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )


def _post_chat(model: str, messages: list[dict], base_url: str, api_key: str, temperature: float) -> str:
    """Отправляет сообщения в OpenAI-совместимый API и возвращает текст."""
    # Запрос к эндпоинту чат-завершений.
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=120,
    )
    # Ошибка API (неверный ключ, лимит) всплывает с понятным текстом.
    response.raise_for_status()
    # Текст первого сгенерированного сообщения.
    return response.json()["choices"][0]["message"]["content"]


def _document(hit: dict, number: int) -> Document:
    """Превращает найденный чанк в документ с заголовком и метаданными."""
    # Раздел документа: заголовок раздела, а если его нет — заголовок статьи.
    section = hit["heading"] or hit["title"]
    # Текст документа начинается с источника — так модель видит, откуда факт.
    return Document(
        page_content=(
            f"Фрагмент {number} (источник: {hit['source']}, раздел «{section}»):\n{hit['text']}"
        ),
        metadata={
            "chunk_id": hit["chunk_id"],
            "source": hit["source"],
            "title": hit["title"],
            "heading": hit["heading"],
            "start_char": hit["start_char"],
            "end_char": hit["end_char"],
            "score": hit["score"],
            "text": hit["text"],
        },
    )


def _context(hits: list[dict]) -> str:
    """Собирает блок фрагментов так же, как это делает RetrievalQA."""
    return "\n\n".join(_document(hit, number).page_content for number, hit in enumerate(hits, start=1))


def _hits_from_documents(documents: list[Document]) -> list[dict]:
    """Возвращает найденные чанки в формате hits по документам цепочки."""
    return [dict(document.metadata) for document in documents]


def _openai_message(message: BaseMessage) -> dict:
    """Переводит сообщение LangChain в сообщение формата OpenAI chat."""
    return {"role": _openai_role(message), "content": message.content}


def _openai_role(message: BaseMessage) -> str:
    """Сопоставляет роль LangChain с ролью OpenAI API."""
    return {"human": "user", "ai": "assistant", "system": "system"}[message.type]


def _load_resources() -> tuple:
    """Загружает индекс и метаданные чанков один раз и кэширует их."""
    # Если ещё не загружали — читаем с диска.
    if "pair" not in _resources_cache:
        _resources_cache["pair"] = load_index(INDEX_DIR)
    # Возвращаем пару (индекс FAISS, список чанков).
    return _resources_cache["pair"]


def _encoder() -> BgeM3:
    """Создаёт кодировщик bge-m3 один раз и использует его далее."""
    # Если кодировщик ещё не создан — загружаем модель из папки models/.
    if "encoder" not in _resources_cache:
        _resources_cache["encoder"] = BgeM3(model_dir_from_env())
    # Возвращаем общий экземпляр.
    return _resources_cache["encoder"]
