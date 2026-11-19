// Checks that no source-universe term leaked into the generated knowledge base.
// Run: node tools/validate_kb.mjs

import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

const KB_DIR = "knowledge_base";

// Names that cannot be confused with ordinary English words.
const STRICT = [
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
];

// Words that are ordinary English in lowercase but Star Wars terms when capitalised.
const AMBIGUOUS = [
  "Force", "Resistance", "Empire", "Imperial", "Republic", "Rebel", "Rebellion",
  "Alliance", "Clone", "Droid", "Blaster", "Hutt", "Jawa", "Lando", "Han",
];

const escape = (term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const strictPattern = new RegExp(`\\b(${STRICT.map(escape).join("|")})\\b`, "gi");
const ambiguousPattern = new RegExp(`\\b(${AMBIGUOUS.map(escape).join("|")})\\b`, "g");

const files = readdirSync(KB_DIR).filter((file) => file.endsWith(".md") && file !== "README.md");
let problems = 0;
let shortest = Infinity;

for (const file of files) {
  const text = readFileSync(path.join(KB_DIR, file), "utf8");
  shortest = Math.min(shortest, text.length);
  const hits = [...text.matchAll(strictPattern), ...text.matchAll(ambiguousPattern)];
  if (hits.length === 0) continue;
  problems += hits.length;
  const unique = [...new Set(hits.map((match) => match[1]))];
  console.log(`${file}: ${hits.length} hit(s) — ${unique.join(", ")}`);
  for (const match of hits.slice(0, 3)) {
    const start = Math.max(0, match.index - 60);
    console.log(`    ...${text.slice(start, match.index + match[0].length + 60).replace(/\n/g, " ")}...`);
  }
}

console.log(`\nDocuments: ${files.length}, shortest: ${shortest} chars, blocklist hits: ${problems}`);
if (files.length < 30) {
  console.error("FAIL: fewer than 30 documents");
  process.exitCode = 1;
}
if (problems > 0) process.exitCode = 1;
