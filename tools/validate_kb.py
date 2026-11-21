"""Checks that no source-universe term leaked into the generated knowledge base.

Run: python tools/validate_kb.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

# Names that cannot be confused with ordinary English words.
STRICT = [
    "Star Wars", "Lucasfilm", "Lucas", "Skywalker", "Vader", "Anakin", "Leia",
    "Kenobi", "Obi-Wan", "Yoda", "Palpatine", "Sidious", "Maul", "Dooku",
    "Grievous", "Ahsoka", "Tano", "Fett", "Padmé", "Amidala", "Calrissian",
    "Chewbacca", "Tarkin", "Windu", "Mothma", "Ackbar", "Jabba",
    "Tatooine", "Coruscant", "Naboo", "Hoth", "Endor", "Dagobah", "Alderaan", "Mustafar",
    "Geonosis", "Kamino", "Bespin", "Yavin", "Corellia", "Kashyyyk", "Ryloth", "Dantooine",
    "Jedi", "Sith", "lightsaber", "Wookiee", "Ewok", "Mandalore",
    "Tusken", "Gungan", "Twi'lek", "Rodian", "Stormtrooper", "Death Star",
    "Millennium Falcon", "X-wing", "Star Destroyer",
    "carbonite", "holocron", "kyber", "beskar", "Kylo", "Snoke", "Grogu",
    "Djarin", "Hux", "Phasma", "Sienar", "Rancor", "Bantha", "Tauntaun", "Sarlacc",
]

# Words that are ordinary English in lowercase but Star Wars terms when capitalised.
AMBIGUOUS = [
    "Force", "Resistance", "Empire", "Imperial", "Republic", "Rebel", "Rebellion",
    "Alliance", "Clone", "Droid", "Blaster", "Hutt", "Jawa", "Lando", "Han",
]

strict_pattern = re.compile(r"\b(" + "|".join(map(re.escape, STRICT)) + r")\b", re.IGNORECASE)
ambiguous_pattern = re.compile(r"\b(" + "|".join(map(re.escape, AMBIGUOUS)) + r")\b")


def main() -> int:
    files = sorted(path for path in KB_DIR.glob("*.md") if path.name != "README.md")
    problems = 0
    shortest = sys.maxsize

    for path in files:
        text = path.read_text(encoding="utf-8")
        shortest = min(shortest, len(text))
        hits = list(strict_pattern.finditer(text)) + list(ambiguous_pattern.finditer(text))
        if not hits:
            continue

        problems += len(hits)
        unique = sorted({match.group(1) for match in hits})
        print(f"{path.name}: {len(hits)} hit(s) — {', '.join(unique)}")
        for match in hits[:3]:
            start = max(0, match.start() - 60)
            snippet = text[start : match.end() + 60].replace("\n", " ")
            print(f"    ...{snippet}...")

    print(f"\nDocuments: {len(files)}, shortest: {shortest} chars, blocklist hits: {problems}")
    if len(files) < 30:
        print("FAIL: fewer than 30 documents", file=sys.stderr)
        return 1
    if problems > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
