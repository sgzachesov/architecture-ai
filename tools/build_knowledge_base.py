"""Собирает вымышленную базу знаний для задания по RAG.

1. Скачивает 38 статей с starwars.fandom.com через MediaWiki API.
2. Очищает wikitext до простого Markdown: без шаблонов, сносок, таблиц
   и служебных разделов.
3. Заменяет все узнаваемые термины по словарю knowledge_base/terms_map.json.
4. Записывает knowledge_base/<slug>.md.

Запуск: python tools/build_knowledge_base.py

Используется только стандартная библиотека Python, ставить ничего не нужно.
"""

# Позволяет использовать современные аннотации типов.
from __future__ import annotations

# html — раскодирует HTML-сущности вроде &nbsp; и &amp;.
import html
# json — читаем словарь замен и разбираем ответ API.
import json
# re — регулярные выражения для очистки разметки.
import re
# time — пауза между запросами к API и задержки при повторах.
import time
# unicodedata — нормализация заголовка перед превращением в имя файла.
import unicodedata
# Path — работаем с путями к файлам и папкам.
from pathlib import Path
# quote — кодирует название статьи для подстановки в URL.
from urllib.parse import quote
# Request и urlopen — выполняем HTTP-запросы к API.
from urllib.request import Request, urlopen

# Корень проекта: файл лежит в tools/, поэтому поднимаемся на уровень выше.
ROOT = Path(__file__).resolve().parent.parent
# Папка, куда складываются готовые документы и откуда читается словарь замен.
KB_DIR = ROOT / "knowledge_base"
# Адрес MediaWiki API вики по Star Wars.
API = "https://starwars.fandom.com/api.php"
# Максимальная длина документа в символах: дальше текст обрезается.
MAX_CHARS = 9000

# Статьи, которые скачиваются: название в вики и категория для индекса.
PAGES = [
    ("Anakin Skywalker", "character"),
    ("Luke Skywalker", "character"),
    ("Leia Organa", "character"),
    ("Han Solo", "character"),
    ("Obi-Wan Kenobi", "character"),
    ("Yoda", "character"),
    ("Palpatine", "character"),
    ("Darth Maul", "character"),
    ("Ahsoka Tano", "character"),
    ("Boba Fett", "character"),
    ("Padmé Amidala", "character"),
    ("Lando Calrissian", "character"),
    ("Chewbacca", "character"),
    ("Grand Moff Tarkin", "character"),
    ("Tatooine", "planet"),
    ("Coruscant", "planet"),
    ("Naboo", "planet"),
    ("Hoth", "planet"),
    ("Endor", "planet"),
    ("Dagobah", "planet"),
    ("Alderaan", "planet"),
    ("Mustafar", "planet"),
    ("Death Star", "technology"),
    ("Lightsaber", "technology"),
    ("Millennium Falcon", "technology"),
    ("X-wing starfighter", "technology"),
    ("TIE series", "technology"),
    ("Star Destroyer", "technology"),
    ("Blaster", "technology"),
    ("Carbonite", "technology"),
    ("The Force", "concept"),
    ("Jedi", "organization"),
    ("Sith", "organization"),
    ("Galactic Empire", "organization"),
    ("Alliance to Restore the Republic", "organization"),
    ("Clone trooper", "organization"),
    ("Stormtrooper", "organization"),
    ("Wookiee", "species"),
]

# Признаки «внешнего» текста: упоминания франшизы, фильмов и служебных разделов.
OUT_OF_UNIVERSE = re.compile(
    r"\b(Star Wars|Lucasfilm|George Lucas|franchise|non-canon|canon|film|movie|novel|"
    r"comic book|comics|video game|TV series|television series|animated series|"
    r"Episode (I|II|III|IV|V|VI|VII|VIII|IX)|Behind the scenes)\b",
    re.IGNORECASE,
)
# Заголовки разделов, с которых начинается служебная часть статьи (== Раздел ==).
DROP_SECTIONS = re.compile(
    r"^==+\s*(Behind the scenes|Appearances|Sources|Notes and references|External links|"
    r"Bibliography|Gallery|Credits|Merchandise|Video games|Notes|References|Further reading)\s*==+",
    re.IGNORECASE | re.MULTILINE,
)
# Те же разделы уже после конвертации в Markdown (## Раздел).
DROP_HEADINGS = re.compile(
    r"^##\s+(Sources|External links|Notes and references|Appearances|Behind the scenes|"
    r"References|Further reading|Gallery)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Дубли слов, которые появляются после удаления разметки: «the the».
REPEATED_WORD = re.compile(r"\b([A-Za-z]+)\s+\1\b", re.IGNORECASE)


def main() -> None:
    """Скачивает статьи, очищает их и сохраняет готовые документы."""
    # Читаем словарь замен.
    terms_map = json.loads((KB_DIR / "terms_map.json").read_text(encoding="utf-8"))
    # Готовим функцию замены.
    replace = build_replacer(terms_map)
    # Создаём папку базы знаний, если её нет.
    KB_DIR.mkdir(parents=True, exist_ok=True)

    # Счётчик записанных документов.
    written = 0
    # Уже занятые имена файлов, чтобы не перезаписать документ.
    used_slugs: set[str] = set()
    # Обрабатываем каждую статью из списка.
    for title, _category in PAGES:
        # Скачиваем wikitext.
        _, content = fetch_wikitext(title)
        # Чистим разметку.
        cleaned = clean_wikitext(content)
        # Слишком короткий результат пропускаем: вероятно, статья не подходит.
        if len(cleaned) < 500:
            print(f"skip {title}: only {len(cleaned)} chars after cleaning")
            continue

        # Заменяем термины и убираем дубли слов, появившиеся после чистки.
        replaced = REPEATED_WORD.sub(r"\1", replace(cleaned))
        # Заголовок документа — тоже с заменой терминов.
        heading = replace(title)
        # Имя файла из заголовка.
        slug = slugify(heading)
        # Если такое имя уже занято — добавляем суффикс.
        if slug in used_slugs:
            slug = f"{slug}-2"
        # Запоминаем имя.
        used_slugs.add(slug)

        # Убираем первый дублирующий заголовок и обрезаем текст до лимита.
        body = truncate(re.sub(r"^##\s+.+\n+", "", replaced, count=1), MAX_CHARS)
        # Собираем документ: заголовок и текст.
        document = f"# {heading}\n\n{body}\n"

        # Записываем файл в UTF-8 с переводами строк \n.
        (KB_DIR / f"{slug}.md").write_text(document, encoding="utf-8", newline="\n")
        # Увеличиваем счётчик.
        written += 1
        # Сообщаем о готовом файле.
        print(f"ok {slug}.md ({len(document)} chars)")
        # Небольшая пауза, чтобы не нагружать вики.
        time.sleep(0.25)

    # Итог: сколько документов записано.
    print(f"\n{written} documents written to knowledge_base/")


def fetch_wikitext(title: str) -> tuple[str, str]:
    """Скачивает исходный wikitext статьи через MediaWiki API."""
    # Собираем URL запроса: получаем содержимое страницы по её названию.
    url = (
        f"{API}?action=query&format=json&formatversion=2&redirects=1"
        f"&prop=revisions&rvprop=content&rvslots=main&titles={quote(title)}"
    )
    # Готовим запрос с User-Agent, как требует API.
    request = Request(url, headers={"User-Agent": "quantumforge-rag-kb/1.0 (educational knowledge base)"})

    # Делаем до трёх попыток при сетевых сбоях.
    for attempt in range(1, 4):
        try:
            # Открываем соединение и разбираем JSON-ответ.
            with urlopen(request, timeout=30) as response:
                data = json.load(response)
            # Берём первую (и единственную) страницу из ответа.
            page = data["query"]["pages"][0]
            # Если страницы нет — сообщаем об ошибке.
            if page.get("missing"):
                raise ValueError(f"page not found: {title}")
            # Достаём содержимое основной версии статьи.
            content = page["revisions"][0]["slots"]["main"]["content"]
            # Пустая статья тоже считается ошибкой.
            if not content:
                raise ValueError(f"empty content: {title}")
            # Возвращаем реальное название и текст.
            return page["title"], content
        except Exception:
            # На третьей попытке пробрасываем ошибку дальше.
            if attempt == 3:
                raise
            # Перед повтором ждём всё дольше.
            time.sleep(0.8 * attempt)

    # Сюда дойти не должны, но на всякий случай.
    raise RuntimeError(f"unreachable: {title}")


def clean_wikitext(raw: str) -> str:
    """Полностью очищает статью: от вики-разметки до готового Markdown."""
    # Приводим переводы строк к \n.
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Ищем начало служебного раздела и обрезаем всё после него.
    section_start = DROP_SECTIONS.search(text)
    if section_start:
        text = text[: section_start.start()]

    # Удаляем теги, шаблоны и таблицы.
    text = strip_tags(text)
    text = strip_templates(text)
    text = strip_tables(text)
    # Преобразуем ссылки и заголовки в Markdown.
    text = convert_markup(text)
    # Раскодируем HTML-сущности и заменяем неразрывный пробел.
    text = html.unescape(text).replace("\xa0", " ")

    # Ещё раз обрезаем по служебному заголовку, уже в виде ##.
    heading_cut = DROP_HEADINGS.search(text)
    if heading_cut:
        text = text[: heading_cut.start()]

    # Убираем оставшиеся «внешние» строки и пустые обрывки.
    lines = []
    for line in text.split("\n"):
        # Убираем пробелы по краям.
        line = line.strip()
        # Пустые строки не нужны.
        if not line:
            continue
        # Строки про франшизу и служебные разделы выбрасываем.
        if OUT_OF_UNIVERSE.search(line):
            continue
        # Строки из одних символов разметки выбрасываем.
        if re.fullmatch(r"[|!{}]+", line):
            continue
        # Пустые скобки выбрасываем.
        if re.fullmatch(r"\(?\)?", line):
            continue
        # Остальное сохраняем.
        lines.append(line)

    # Схлопываем подряд идущие заголовки: второй обычно пустой раздел.
    kept: list[str] = []
    for line in lines:
        # Это заголовок раздела?
        is_heading = line.startswith("## ")
        # Предыдущая сохранённая строка тоже заголовок?
        previous_is_heading = bool(kept) and kept[-1].startswith("## ")
        # Если два заголовка подряд — убираем предыдущий.
        if is_heading and previous_is_heading:
            kept.pop()
        # Добавляем строку.
        kept.append(line)

    # Склеиваем абзацы, убираем лишние пустые строки и обрезаем края.
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(kept)).strip()


def strip_tags(text: str) -> str:
    """Удаляет комментарии, сноски и прочие HTML-теги."""
    # Убираем HTML-комментарии.
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    # Убираем самозакрывающиеся сноски <ref name="..." />.
    text = re.sub(r"<ref[^>]*/>", "", text, flags=re.IGNORECASE)
    # Убираем парные сноски вместе с содержимым.
    text = re.sub(r"<ref[^>]*>[\s\S]*?</ref>", "", text, flags=re.IGNORECASE)
    # Убираем незакрытую сноску до конца текста.
    text = re.sub(r"<ref[^>]*>[\s\S]*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    # Убираем галереи изображений целиком.
    text = re.sub(r"<gallery[\s\S]*?</gallery>", "", text, flags=re.IGNORECASE)
    # Убираем оставшиеся теги, сохраняя их содержимое.
    return re.sub(r"<[^>]+>", "", text)


def strip_templates(text: str) -> str:
    """Удаляет шаблоны {{...}}, включая вложенные."""
    # Начинаем с исходного текста.
    current = text
    # До 15 проходов: каждый убирает самый внутренний уровень вложенности.
    for _ in range(15):
        # Удаляем шаблоны без вложенных фигурных скобок.
        following = re.sub(r"\{\{[^{}]*\}\}", "", current)
        # Если ничего не изменилось — вложенность закончилась.
        if following == current:
            break
        # Переходим к следующему проходу.
        current = following
    # Добиваем то, что осталось (например, незакрытые шаблоны).
    return re.sub(r"\{\{[\s\S]*?\}\}", "", current)


def strip_tables(text: str) -> str:
    """Удаляет вики-таблицы и строки таблиц."""
    # Вырезаем таблицу целиком от {| до |}.
    text = re.sub(r"\{\|[\s\S]*?\|\}", "", text)
    # Выбрасываем оставшиеся строки, начинающиеся с | или !.
    return "\n".join(line for line in text.split("\n") if not re.match(r"^\s*[|!]", line))


def convert_markup(text: str) -> str:
    """Превращает вики-разметку в обычный Markdown."""
    # Убираем ссылки на файлы, изображения и видео.
    text = re.sub(r"\[\[(?:File|Image|Video|Media):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Убираем ссылки на категории.
    text = re.sub(r"\[\[(?:Category|Cat):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Убираем интервики-ссылки на другие языки.
    text = re.sub(r"\[\[[a-z]{2,3}:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Из [[ссылка|подпись]] оставляем подпись.
    text = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", text)
    # Из [[ссылка]] оставляем текст ссылки.
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    # Из внешней ссылки [url подпись] оставляем подпись.
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]", r"\1", text)
    # Внешние ссылки без подписи удаляем.
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    # Убираем вики-разметку курсива и жирного ('' и ''').
    text = re.sub(r"'{2,5}", "", text)
    # Заголовки == Раздел == превращаем в Markdown ## Раздел.
    text = re.sub(r"^=+\s*(.+?)\s*=+\s*$", r"## \1", text, flags=re.MULTILINE)
    # Элементы списка * превращаем в -.
    text = re.sub(r"^\*\s+", "- ", text, flags=re.MULTILINE)
    # Нумерованные элементы # превращаем в 1.
    text = re.sub(r"^#\s+", "1. ", text, flags=re.MULTILINE)
    # Убираем служебные слова вида __TOC__.
    text = re.sub(r"__\w+__", "", text)
    # Схлопываем повторяющиеся пробелы и табы.
    text = re.sub(r"[ \t]+", " ", text)
    # Убираем пробелы перед переводом строки.
    return re.sub(r" +\n", "\n", text)


def build_replacer(terms_map: dict):
    """Собирает функцию замены терминов по словарю."""
    # Слова, которые заменяются только при точном совпадении регистра.
    sensitive = set(terms_map.get("case_sensitive", []))
    # Здесь будут все пары «исходный термин → замена».
    entries: dict[str, str] = {}
    # Словарь разбит на категории — сливаем их в один.
    for group in terms_map["categories"].values():
        entries.update(group)

    # Сортируем термины от длинных к коротким: «Darth Vader» раньше «Vader».
    keys = sorted(entries, key=len, reverse=True)
    # Одно регулярное выражение на все термины, с границами слова.
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(key) for key in keys) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    # Индекс «в нижнем регистре → исходный термин» для поиска без учёта регистра.
    by_lower: dict[str, str] = {}
    for key in keys:
        by_lower.setdefault(key.lower(), key)

    def replace(text: str) -> str:
        """Заменяет все термины в тексте за один проход."""

        def substitute(match: re.Match[str]) -> str:
            # Найденный фрагмент.
            found = match.group(0)
            # Ищем, какому термину он соответствует.
            key = by_lower.get(found.lower())
            # Не нашли — оставляем как есть.
            if key is None:
                return found
            # Чувствительный термин с другим регистром не трогаем.
            if key in sensitive and found != key:
                return found
            # Иначе подставляем замену с сохранением регистра.
            return apply_case(found, entries[key])

        # Применяем подстановку ко всем совпадениям.
        return pattern.sub(substitute, text)

    # Возвращаем готовую функцию.
    return replace


def apply_case(source: str, target: str) -> str:
    """Переносит регистр исходного слова на замену."""
    # Аббревиатуры вроде TIE остаются заглавными.
    if len(source) > 1 and source.isupper() and " " not in target and len(target) <= 4:
        return target.upper()
    # Слово с заглавной буквы — замена тоже с заглавной.
    if source[0] == source[0].upper():
        return target[0].upper() + target[1:]
    # Иначе — со строчной.
    return target[0].lower() + target[1:]


def slugify(title: str) -> str:
    """Превращает заголовок в имя файла: латиница, цифры и дефисы."""
    # Разлагаем символы с диакритикой на базовые (é → e).
    normalized = unicodedata.normalize("NFKD", title)
    # Оставляем только буквы, цифры, пробелы и дефисы.
    cleaned = re.sub(r"[^\w\s-]", "", normalized)
    # Пробелы и подчёркивания меняем на дефис, всё приводим к нижнему регистру.
    slug = re.sub(r"[\s_]+", "-", cleaned.strip().lower())
    # Схлопываем повторяющиеся дефисы.
    return re.sub(r"-+", "-", slug)


def truncate(text: str, limit: int) -> str:
    """Обрезает текст до limit символов по границе абзаца."""
    # Короткий текст не трогаем.
    if len(text) <= limit:
        return text
    # Берём первые limit символов.
    cut = text[:limit]
    # Ищем последний абзацный разрыв.
    boundary = cut.rfind("\n\n")
    # Обрезаем по абзацу, если он не слишком далеко от лимита.
    body = cut[:boundary] if boundary > limit * 0.5 else cut
    # Добавляем пометку об обрезке.
    return body + "\n\n_(excerpt truncated)_"


if __name__ == "__main__":
    # Запускаем сборку только при прямом вызове скрипта.
    main()
