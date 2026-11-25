# Примеры диалогов RAG-бота (задание 4)

Сервис: FastAPI, эндпоинт `POST /chat`. Цепочка: вопрос → эмбеддинг bge-m3 →
поиск в FAISS → LCEL-цепочка на Runnable-ах собирает промпт с few-shot и
Chain-of-Thought → ответ с источниками.

> **Режим примеров: демо.** `OPENAI_API_KEY` не задан, поэтому ответ собирается
> из найденного фрагмента детерминированно (LLM не вызывается). LCEL-цепочка
> при этом выполняется целиком, но вместо модели стоит заглушка `DemoChat`.
> Фрагменты базы знаний написаны на английском, поэтому в демо-ответе факт
> цитируется как есть. При заданном `OPENAI_API_KEY` тот же промпт уходит в
> LLM (OpenAI-совместимый API), и ответ генерируется на русском по инструкции
> Chain-of-Thought.

## Как запустить

```bash
docker compose up --build
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Кто такой Арэн Велмор?"}'
```

С ключом LLM:

```bash
OPENAI_API_KEY=sk-... docker compose up --build
# или совместимый эндпоинт:
OPENAI_API_KEY=... OPENAI_BASE_URL=https://... LLM_MODEL=... docker compose up --build
```

### Через Swagger UI

1. Откройте `http://localhost:8000/docs`.
2. Раскройте `POST /chat` → **Try it out**.
3. Выберите готовый пример из выпадающего списка (например, «Кто такой Арэн
   Велмор?») или введите вопрос.
4. Нажмите **Execute** — в блоке Response появится тот же JSON, что в примерах
   ниже.

В схеме размечены описания полей, три примера запроса, пример ответа с
источниками и коды `200`/`422`/`502`. Схема OpenAPI доступна по
`/openapi.json`, альтернативная документация — по `/redoc`.

## Техники промптинга

### Few-shot (2 примера из этой же базы)

Ответы сверены с реальными чанками `aran-velmor-001` и `the-synth-flux-001`.

```
Вопрос: Кто такой Арэн Велмор?
Ответ: Арэн Велмор — легендарный человек, чувствительный к Синто-Флюксу,
Рыцарь Флюкса Аэтериум-Конкорда и Предсказанный Единства Флюкс-Ордена,
которому суждено принести равновесие Синто-Флюксу. В детстве его звали «Ари»,
а за подвиги в Войнах Ват он получил прозвище «Герой без страха». Его
альтер-эго — Ксарн Велгор, Нулевой Лорд Пустотных.

Вопрос: Что такое Синто-Флюкс?
Ответ: Синто-Флюкс — энергетическое поле, создаваемое всей жизнью и
связывающее вселенную воедино. На протяжении галактической истории его
называли по-разному: Луна у Вокков, Жизненный Поток у Синдрали, Прилив у
Левелан и другими именами.
```

### Chain-of-Thought (system-промпт)

```
Ты — справочный ассистент корпоративной базы знаний QuantumForge Software.
Отвечай на русском языке.

Прежде чем ответить, размышляй по шагам (Chain-of-Thought):
1) Определи, какой из предоставленных фрагментов отвечает на вопрос.
2) Выпиши из него ключевые факты, относящиеся к вопросу.
3) Сделай вывод и сформулируй итоговый ответ.

Правила:
- Опирайся ТОЛЬКО на предоставленные фрагменты базы знаний. Не выдумывай факты.
- Если среди фрагментов нет ответа на вопрос, ответь ровно: «Я не знаю».
- Ответ должен быть кратким и по делу, без лишней воды.
```

### «Я не знаю»

Два уровня защиты:

1. **До LLM** — если лучший чанк ниже порога `SCORE_THRESHOLD` (по умолчанию 0.45),
   бот сразу отвечает «Я не знаю».
2. **В промпте** — модели запрещено выдумывать; если во фрагментах нет ответа,
   она должна ответить ровно «Я не знаю».

Порог подобран по корпусу: релевантные вопросы дают ≥ 0.465, явно посторонние
(«погода», «борщ») — ≤ 0.432. На кросс-языковых запросах распределения частично
пересекаются, поэтому в реальном режиме решает ещё и инструкция LLM.

---

## Успешные диалоги

### 1. Кто такой Арэн Велмор?

**Запрос:** `Кто такой Арэн Велмор?`

**Ответ бота:**

> Размышление:
> 1. На вопрос отвечает фрагмент «aran-velmor-001» (оценка близости 0.552, раздел «Aran Velmor»).
> 2. Ключевой факт из фрагмента: Aran Velmor was a legendary Flux-sensitive human male who was a Flux Knight of the Aetherium Concord and the prophesied Foretold One of the Flux Order, destined to bring balance to the Synth Flux.
>
> Ответ на основе базы знаний: Aran Velmor was a legendary Flux-sensitive human male who was a Flux Knight of the Aetherium Concord and the prophesied Foretold One of the Flux Order, destined to bring balance to the Synth Flux.
> (демо-режим: ответ собран из фрагмента, LLM не вызывалась)

**Источники:** `aran-velmor-001` (0.552, `knowledge_base/aran-velmor.md`, chars 15–557), `aran-velmor-002` (0.534), `aran-velmor-005` (0.530).

### 2. Что такое Синто-Флюкс и как его используют Флюкс-адепты?

**Запрос:** `Что такое Синто-Флюкс и как его используют Флюкс-адепты?`

**Ответ бота:**

> Размышление:
> 1. На вопрос отвечает фрагмент «the-synth-flux-001» (оценка близости 0.531, раздел «The Synth Flux»).
> 2. Ключевой факт из фрагмента: The Synth Flux was an energy field created by all life that bound everything in the universe together.
>
> Ответ на основе базы знаний: The Synth Flux was an energy field created by all life that bound everything in the universe together.
> (демо-режим: ответ собран из фрагмента, LLM не вызывалась)

**Источники:** `the-synth-flux-001` (0.531, `knowledge_base/the-synth-flux.md`, chars 18–1701), `the-synth-flux-006` (0.501), `fluxadept-001` (0.500).

### 3. Как работает флюксблейд и какой в нём кристалл?

**Запрос:** `Как работает флюксблейд и какой в нём кристалл?`

**Ответ бота:**

> Размышление:
> 1. На вопрос отвечает фрагмент «fluxblade-001» (оценка близости 0.557, раздел «Fluxblade»).
> 2. Ключевой факт из фрагмента: The fluxblade, also referred to as a laser sword by those who were unfamiliar with it, was a weapon usually used by the Fluxadept, Voidsworn, and other Flux-sensitives.
>
> Ответ на основе базы знаний: The fluxblade, also referred to as a laser sword by those who were unfamiliar with it, was a weapon usually used by the Fluxadept, Voidsworn, and other Flux-sensitives.
> (демо-режим: ответ собран из фрагмента, LLM не вызывалась)

**Источники:** `fluxblade-001` (0.557, `knowledge_base/fluxblade.md`, chars 13–1077), `fluxblade-002` (0.555), `fluxadept-005` (0.528).

### 4. Что случилось с Сорхой во время Войн Ват?

**Запрос:** `Что случилось с Сорхой во время Войн Ват?`

**Ответ бота:**

> Размышление:
> 1. На вопрос отвечает фрагмент «sorrhas-002» (оценка близости 0.496, раздел «Sorrhas»).
> 2. Ключевой факт из фрагмента: Following the death of Ollag and the withdrawal of Dominion forces after the Battle of Sylvaneth, an enormous power vacuum occurred on the world, with numerous crime syndicates like the Red Key Raiders attempting to build legitimacy as mining companies.
>
> Ответ на основе базы знаний: Following the death of Ollag and the withdrawal of Dominion forces after the Battle of Sylvaneth, an enormous power vacuum occurred on the world, with numerous crime syndicates like the Red Key Raiders attempting to build legitimacy as mining companies.
> (демо-режим: ответ собран из фрагмента, LLM не вызывалась)

**Источники:** `sorrhas-002` (0.496, `knowledge_base/sorrhas.md`, chars 1662–3028), `sorrhas-006` (0.491), `sorel-vand-002` (0.475).

### 5. Кто такая Кейлит Рау?

**Запрос:** `Кто такая Кейлит Рау?`

**Ответ бота:**

> Размышление:
> 1. На вопрос отвечает фрагмент «kaelith-005» (оценка близости 0.465, раздел «Early life and apprenticeship»).
> 2. Ключевой факт из фрагмента: The origins of Kaelith Rau were closely guarded secrets that he took to his grave more than once.
>
> Ответ на основе базы знаний: The origins of Kaelith Rau were closely guarded secrets that he took to his grave more than once.
> (демо-режим: ответ собран из фрагмента, LLM не вызывалась)

**Источники:** `kaelith-005` (0.465, `knowledge_base/kaelith.md`, chars 3968–5414), `kaelith-002` (0.458), `kaelith-001` (0.444).

---

## Случаи «Я не знаю»

### 6. Вопрос вне базы знаний: погода

**Запрос:** `Какая сегодня погода в Москве?`

**Ответ бота:**

> Я не знаю: в базе знаний не нашлось достаточно близкого материала по этому вопросу.

**Почему:** лучший чанк (`meridian-prime-006`) дал 0.432 — ниже порога 0.45. Отказ сработал до вызова LLM.

### 7. Вопрос вне базы знаний: кулинария

**Запрос:** `Как приготовить борщ?`

**Ответ бота:**

> Я не знаю: в базе знаний не нашлось достаточно близкого материала по этому вопросу.

**Почему:** лучший чанк (`gorran-002`) дал 0.354 — ниже порога 0.45.

---

## Формат ответа API

```json
{
  "query": "Кто такой Арэн Велмор?",
  "answer": "Размышление: ...",
  "demo": true,
  "sources": [
    {
      "chunk_id": "aran-velmor-001",
      "source": "knowledge_base/aran-velmor.md",
      "title": "Aran Velmor",
      "heading": "",
      "start_char": 15,
      "end_char": 557,
      "score": 0.552
    }
  ]
}
```

Поля `source`, `heading`, `start_char`, `end_char` дают боту всё для точной
цитаты: файл, раздел и границы фрагмента в исходном документе.
