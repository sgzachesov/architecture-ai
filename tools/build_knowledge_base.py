"""Builds the fictional knowledge base for the RAG assignment.

1. Downloads 38 in-universe articles from starwars.fandom.com (MediaWiki API).
2. Cleans wikitext into plain Markdown (no templates, refs, tables, out-of-universe sections).
3. Replaces every key term using knowledge_base/terms_map.json.
4. Writes knowledge_base/<slug>.md.

Run: python tools/build_knowledge_base.py

Only the Python standard library is used, so no extra packages are required.
"""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
API = "https://starwars.fandom.com/api.php"
MAX_CHARS = 9000

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

OUT_OF_UNIVERSE = re.compile(
    r"\b(Star Wars|Lucasfilm|George Lucas|franchise|non-canon|canon|film|movie|novel|"
    r"comic book|comics|video game|TV series|television series|animated series|"
    r"Episode (I|II|III|IV|V|VI|VII|VIII|IX)|Behind the scenes)\b",
    re.IGNORECASE,
)
DROP_SECTIONS = re.compile(
    r"^==+\s*(Behind the scenes|Appearances|Sources|Notes and references|External links|"
    r"Bibliography|Gallery|Credits|Merchandise|Video games|Notes|References|Further reading)\s*==+",
    re.IGNORECASE | re.MULTILINE,
)
DROP_HEADINGS = re.compile(
    r"^##\s+(Sources|External links|Notes and references|Appearances|Behind the scenes|"
    r"References|Further reading|Gallery)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

REPEATED_WORD = re.compile(r"\b([A-Za-z]+)\s+\1\b", re.IGNORECASE)


def strip_tags(text: str) -> str:
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    text = re.sub(r"<ref[^>]*/>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref[^>]*>[\s\S]*?</ref>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref[^>]*>[\s\S]*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"<gallery[\s\S]*?</gallery>", "", text, flags=re.IGNORECASE)
    return re.sub(r"<[^>]+>", "", text)


def strip_templates(text: str) -> str:
    current = text
    for _ in range(15):
        following = re.sub(r"\{\{[^{}]*\}\}", "", current)
        if following == current:
            break
        current = following
    return re.sub(r"\{\{[\s\S]*?\}\}", "", current)


def strip_tables(text: str) -> str:
    text = re.sub(r"\{\|[\s\S]*?\|\}", "", text)
    return "\n".join(line for line in text.split("\n") if not re.match(r"^\s*[|!]", line))


def convert_markup(text: str) -> str:
    text = re.sub(r"\[\[(?:File|Image|Video|Media):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:Category|Cat):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[[a-z]{2,3}:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"^=+\s*(.+?)\s*=+\s*$", r"## \1", text, flags=re.MULTILINE)
    text = re.sub(r"^\*\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+", "1. ", text, flags=re.MULTILINE)
    text = re.sub(r"__\w+__", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r" +\n", "\n", text)


def clean_wikitext(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    section_start = DROP_SECTIONS.search(text)
    if section_start:
        text = text[: section_start.start()]

    text = strip_tags(text)
    text = strip_templates(text)
    text = strip_tables(text)
    text = convert_markup(text)
    text = html.unescape(text).replace("\xa0", " ")

    heading_cut = DROP_HEADINGS.search(text)
    if heading_cut:
        text = text[: heading_cut.start()]

    # Drop leftover out-of-universe lines and empty artefacts.
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if OUT_OF_UNIVERSE.search(line):
            continue
        if re.fullmatch(r"[|!{}]+", line):
            continue
        if re.fullmatch(r"\(?\)?", line):
            continue
        lines.append(line)

    kept: list[str] = []
    for line in lines:
        is_heading = line.startswith("## ")
        previous_is_heading = bool(kept) and kept[-1].startswith("## ")
        if is_heading and previous_is_heading:
            kept.pop()
        kept.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(kept)).strip()


def apply_case(source: str, target: str) -> str:
    if len(source) > 1 and source.isupper() and " " not in target and len(target) <= 4:
        return target.upper()
    if source[0] == source[0].upper():
        return target[0].upper() + target[1:]
    return target[0].lower() + target[1:]


def build_replacer(terms_map: dict):
    sensitive = set(terms_map.get("case_sensitive", []))
    entries: dict[str, str] = {}
    for group in terms_map["categories"].values():
        entries.update(group)

    keys = sorted(entries, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(key) for key in keys) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    by_lower: dict[str, str] = {}
    for key in keys:
        by_lower.setdefault(key.lower(), key)

    def replace(text: str) -> str:
        def substitute(match: re.Match[str]) -> str:
            found = match.group(0)
            key = by_lower.get(found.lower())
            if key is None:
                return found
            if key in sensitive and found != key:
                return found
            return apply_case(found, entries[key])

        return pattern.sub(substitute, text)

    return replace


def slugify(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    cleaned = re.sub(r"[^\w\s-]", "", normalized)
    slug = re.sub(r"[\s_]+", "-", cleaned.strip().lower())
    return re.sub(r"-+", "-", slug)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind("\n\n")
    body = cut[:boundary] if boundary > limit * 0.5 else cut
    return body + "\n\n_(excerpt truncated)_"


def fetch_wikitext(title: str) -> tuple[str, str]:
    url = (
        f"{API}?action=query&format=json&formatversion=2&redirects=1"
        f"&prop=revisions&rvprop=content&rvslots=main&titles={quote(title)}"
    )
    request = Request(url, headers={"User-Agent": "quantumforge-rag-kb/1.0 (educational knowledge base)"})

    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=30) as response:
                data = json.load(response)
            page = data["query"]["pages"][0]
            if page.get("missing"):
                raise ValueError(f"page not found: {title}")
            content = page["revisions"][0]["slots"]["main"]["content"]
            if not content:
                raise ValueError(f"empty content: {title}")
            return page["title"], content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.8 * attempt)

    raise RuntimeError(f"unreachable: {title}")


def main() -> None:
    terms_map = json.loads((KB_DIR / "terms_map.json").read_text(encoding="utf-8"))
    replace = build_replacer(terms_map)
    KB_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    used_slugs: set[str] = set()
    for title, _category in PAGES:
        _, content = fetch_wikitext(title)
        cleaned = clean_wikitext(content)
        if len(cleaned) < 500:
            print(f"skip {title}: only {len(cleaned)} chars after cleaning")
            continue

        replaced = REPEATED_WORD.sub(r"\1", replace(cleaned))
        heading = replace(title)
        slug = slugify(heading)
        if slug in used_slugs:
            slug = f"{slug}-2"
        used_slugs.add(slug)

        body = truncate(re.sub(r"^##\s+.+\n+", "", replaced, count=1), MAX_CHARS)
        document = f"# {heading}\n\n{body}\n"

        (KB_DIR / f"{slug}.md").write_text(document, encoding="utf-8", newline="\n")
        written += 1
        print(f"ok {slug}.md ({len(document)} chars)")
        time.sleep(0.25)

    print(f"\n{written} documents written to knowledge_base/")


if __name__ == "__main__":
    main()
