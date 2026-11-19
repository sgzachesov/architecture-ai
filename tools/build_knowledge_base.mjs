// Builds the fictional knowledge base for the RAG assignment.
//
// 1. Downloads 38 in-universe articles from starwars.fandom.com (MediaWiki API).
// 2. Cleans wikitext into plain Markdown (no templates, refs, tables, out-of-universe sections).
// 3. Replaces every key term using knowledge_base/terms_map.json.
// 4. Writes knowledge_base/<slug>.md and knowledge_base/index.json.
//
// Run: node tools/build_knowledge_base.mjs
//
// Python is not installed on this machine, so the script is written in Node.js.
// The replacement logic itself is language-agnostic and described in knowledge_base/README.md.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const KB_DIR = path.join(ROOT, "knowledge_base");
const API = "https://starwars.fandom.com/api.php";
const MAX_CHARS = 9000;

const PAGES = [
  { title: "Anakin Skywalker", category: "character" },
  { title: "Luke Skywalker", category: "character" },
  { title: "Leia Organa", category: "character" },
  { title: "Han Solo", category: "character" },
  { title: "Obi-Wan Kenobi", category: "character" },
  { title: "Yoda", category: "character" },
  { title: "Palpatine", category: "character" },
  { title: "Darth Maul", category: "character" },
  { title: "Ahsoka Tano", category: "character" },
  { title: "Boba Fett", category: "character" },
  { title: "Padmé Amidala", category: "character" },
  { title: "Lando Calrissian", category: "character" },
  { title: "Chewbacca", category: "character" },
  { title: "Grand Moff Tarkin", category: "character" },
  { title: "Tatooine", category: "planet" },
  { title: "Coruscant", category: "planet" },
  { title: "Naboo", category: "planet" },
  { title: "Hoth", category: "planet" },
  { title: "Endor", category: "planet" },
  { title: "Dagobah", category: "planet" },
  { title: "Alderaan", category: "planet" },
  { title: "Mustafar", category: "planet" },
  { title: "Death Star", category: "technology" },
  { title: "Lightsaber", category: "technology" },
  { title: "Millennium Falcon", category: "technology" },
  { title: "X-wing starfighter", category: "technology" },
  { title: "TIE series", category: "technology" },
  { title: "Star Destroyer", category: "technology" },
  { title: "Blaster", category: "technology" },
  { title: "Carbonite", category: "technology" },
  { title: "The Force", category: "concept" },
  { title: "Jedi", category: "organization" },
  { title: "Sith", category: "organization" },
  { title: "Galactic Empire", category: "organization" },
  { title: "Alliance to Restore the Republic", category: "organization" },
  { title: "Clone trooper", category: "organization" },
  { title: "Stormtrooper", category: "organization" },
  { title: "Wookiee", category: "species" },
];

const OUT_OF_UNIVERSE = /\b(Star Wars|Lucasfilm|George Lucas|franchise|non-canon|canon|film|movie|novel|comic book|comics|video game|TV series|television series|animated series|Episode (I|II|III|IV|V|VI|VII|VIII|IX)|Behind the scenes)\b/i;
const DROP_SECTIONS = /^==+\s*(Behind the scenes|Appearances|Sources|Notes and references|External links|Bibliography|Gallery|Credits|Merchandise|Video games|Notes|References|Further reading)\s*==+/im;
const DROP_HEADINGS = /^##\s+(Sources|External links|Notes and references|Appearances|Behind the scenes|References|Further reading|Gallery)\b.*$/im;

function decodeEntities(text) {
  const map = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&lt;": "<",
    "&gt;": ">",
    "&mdash;": "—",
    "&ndash;": "–",
    "&hellip;": "…",
    "&rsquo;": "’",
    "&lsquo;": "‘",
    "&ldquo;": "“",
    "&rdquo;": "”",
  };
  return text.replace(/&[a-z#0-9]+;/gi, (entity) => map[entity.toLowerCase()] ?? entity);
}

function stripTags(text) {
  return text
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<ref[^>]*\/>/gi, "")
    .replace(/<ref[^>]*>[\s\S]*?<\/ref>/gi, "")
    .replace(/<ref[^>]*>[\s\S]*$/gim, "")
    .replace(/<gallery[\s\S]*?<\/gallery>/gi, "")
    .replace(/<[^>]+>/g, "");
}

function stripTemplates(text) {
  let current = text;
  for (let i = 0; i < 15; i += 1) {
    const next = current.replace(/\{\{[^{}]*\}\}/g, "");
    if (next === current) break;
    current = next;
  }
  return current.replace(/\{\{[\s\S]*?\}\}/g, "");
}

function stripTables(text) {
  return text
    .replace(/\{\|[\s\S]*?\|\}/g, "")
    .split("\n")
    .filter((line) => !/^\s*[|!]/.test(line))
    .join("\n");
}

function convertMarkup(text) {
  return text
    .replace(/\[\[(?:File|Image|Video|Media):[^\]]*\]\]/gi, "")
    .replace(/\[\[(?:Category|Cat):[^\]]*\]\]/gi, "")
    .replace(/\[\[[a-z]{2,3}:[^\]]*\]\]/gi, "")
    .replace(/\[\[([^\]|]*)\|([^\]]*)\]\]/g, "$2")
    .replace(/\[\[([^\]]*)\]\]/g, "$1")
    .replace(/\[https?:\/\/[^\s\]]+\s+([^\]]*)\]/g, "$1")
    .replace(/\[https?:\/\/[^\]]+\]/g, "")
    .replace(/'{2,5}/g, "")
    .replace(/^=+\s*(.+?)\s*=+\s*$/gm, (_, heading) => `## ${heading.trim()}`)
    .replace(/^\*\s+/gm, "- ")
    .replace(/^#\s+/gm, "1. ")
    .replace(/__\w+__/g, "")
    .replace(/[ \t]+/g, " ")
    .replace(/ +\n/g, "\n");
}

function cleanWikitext(raw) {
  let text = raw.replace(/\r\n?/g, "\n");
  const sectionStart = text.search(DROP_SECTIONS);
  if (sectionStart !== -1) text = text.slice(0, sectionStart);
  text = stripTags(text);
  text = stripTemplates(text);
  text = stripTables(text);
  text = convertMarkup(text);
  text = decodeEntities(text);

  const headingCut = text.search(DROP_HEADINGS);
  if (headingCut !== -1) text = text.slice(0, headingCut);

  // Drop leftover out-of-universe lines and empty artefacts.
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .filter((line) => !OUT_OF_UNIVERSE.test(line))
    .filter((line) => !/^[|!{}]+$/.test(line))
    .filter((line) => !/^\(?\)?$/.test(line));

  const kept = [];
  for (const line of lines) {
    const isHeading = line.startsWith("## ");
    const previousIsHeading = kept.length > 0 && kept[kept.length - 1].startsWith("## ");
    if (isHeading && previousIsHeading) kept.pop();
    kept.push(line);
  }

  return kept
    .join("\n\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function buildReplacer(termsMap) {
  const sensitive = new Set(termsMap.case_sensitive ?? []);
  const entries = new Map();
  for (const group of Object.values(termsMap.categories)) {
    for (const [from, to] of Object.entries(group)) entries.set(from, to);
  }
  const keys = [...entries.keys()].sort((a, b) => b.length - a.length);
  const escape = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(
    `(?<![A-Za-z0-9])(?:${keys.map(escape).join("|")})(?![A-Za-z0-9])`,
    "gi",
  );

  const applyCase = (source, target) => {
    if (source.length > 1 && source === source.toUpperCase() && !target.includes(" ") && target.length <= 4) {
      return target.toUpperCase();
    }
    if (source[0] === source[0].toUpperCase()) return target[0].toUpperCase() + target.slice(1);
    return target[0].toLowerCase() + target.slice(1);
  };

  return (text) =>
    text.replace(pattern, (match) => {
      const key = keys.find((candidate) => candidate.toLowerCase() === match.toLowerCase());
      if (!key) return match;
      if (sensitive.has(key) && match !== key) return match;
      return applyCase(match, entries.get(key));
    });
}

function slugify(title) {
  return title
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, "-")
    .replace(/-+/g, "-");
}

function truncate(text, limit) {
  if (text.length <= limit) return text;
  const cut = text.slice(0, limit);
  const boundary = cut.lastIndexOf("\n\n");
  return `${boundary > limit * 0.5 ? cut.slice(0, boundary) : cut}\n\n_(excerpt truncated)_`;
}

async function fetchWikitext(title) {
  const url = `${API}?action=query&format=json&formatversion=2&redirects=1&prop=revisions&rvprop=content&rvslots=main&titles=${encodeURIComponent(title)}`;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { "User-Agent": "quantumforge-rag-kb/1.0 (educational knowledge base)" },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const page = data.query?.pages?.[0];
      if (!page || page.missing) throw new Error(`page not found: ${title}`);
      const content = page.revisions?.[0]?.slots?.main?.content;
      if (!content) throw new Error(`empty content: ${title}`);
      return { canonicalTitle: page.title, content };
    } catch (error) {
      if (attempt === 3) throw error;
      await new Promise((resolve) => setTimeout(resolve, 800 * attempt));
    }
  }
}

async function main() {
  const termsMap = JSON.parse(await readFile(path.join(KB_DIR, "terms_map.json"), "utf8"));
  const replace = buildReplacer(termsMap);
  await mkdir(KB_DIR, { recursive: true });

  const index = [];
  const usedSlugs = new Set();

  for (const { title, category } of PAGES) {
    const { canonicalTitle, content } = await fetchWikitext(title);
    const cleaned = cleanWikitext(content);
    if (cleaned.length < 500) {
      console.warn(`skip ${title}: only ${cleaned.length} chars after cleaning`);
      continue;
    }
    const replaced = replace(cleaned).replace(/\b([A-Za-z]+)\s+\1\b/gi, "$1");
    const heading = replace(title);
    const slug = slugify(heading);
    const uniqueSlug = usedSlugs.has(slug) ? `${slug}-2` : slug;
    usedSlugs.add(uniqueSlug);

    const body = truncate(replaced.replace(/^##\s+.+\n+/, ""), MAX_CHARS);
    const document = `# ${heading}\n\n${body}\n`;

    await writeFile(path.join(KB_DIR, `${uniqueSlug}.md`), document, "utf8");
    index.push({ file: `${uniqueSlug}.md`, title: heading, category });
    console.log(`ok ${uniqueSlug}.md (${document.length} chars)`);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }

  await writeFile(
    path.join(KB_DIR, "index.json"),
    `${JSON.stringify({ documents: index }, null, 2)}\n`,
    "utf8",
  );
  console.log(`\n${index.length} documents written to knowledge_base/`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
