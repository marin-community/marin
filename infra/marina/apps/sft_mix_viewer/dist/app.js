// Browse sampled documents of the 10.05 SFT mixture: by source, or drawn in proportion to mix share.

const SPECIAL_TOKEN = /(<\|[^|<>\n]{1,48}\|>)/;
const OMITTED = /(\n\n\[… [\d,]+ characters omitted …\]\n\n)/;

const elements = {
  summary: document.getElementById("summary"),
  sources: document.getElementById("sources"),
  head: document.getElementById("source-head"),
  pager: document.getElementById("pager"),
  text: document.getElementById("text"),
  draw: document.getElementById("draw"),
  kind: document.getElementById("kind"),
  filter: document.getElementById("filter"),
};

let sources = [];
let byName = new Map();
let request = 0;

function percent(share) {
  const value = share * 100;
  return (value >= 1 ? value.toFixed(1) : value >= 0.01 ? value.toFixed(2) : value.toFixed(4)) + "%";
}

function compact(count) {
  if (count >= 1e6) return (count / 1e6).toFixed(1) + "M";
  if (count >= 1e3) return (count / 1e3).toFixed(1) + "k";
  return String(count);
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function visibleSources() {
  const kind = elements.kind.value;
  const needle = elements.filter.value.trim().toLowerCase();
  return sources.filter(
    (source) => (kind === "all" || source.kind === kind) && (!needle || source.name.toLowerCase().includes(needle)),
  );
}

function selection() {
  const parameters = new URLSearchParams(location.hash.slice(1));
  const source = byName.get(parameters.get("source"));
  if (!source) return null;
  const docIndex = Number(parameters.get("doc"));
  const sample = source.samples.find((candidate) => candidate.doc_index === docIndex) || source.samples[0];
  return { source, sample };
}

function select(source, sample) {
  location.hash = new URLSearchParams({ source: source.name, doc: String(sample.doc_index) }).toString();
}

function renderSources() {
  const selected = selection();
  const families = new Map();
  for (const source of visibleSources()) {
    if (!families.has(source.family)) families.set(source.family, []);
    families.get(source.family).push(source);
  }
  const ordered = [...families.entries()]
    .map(([family, members]) => ({ family, members, share: members.reduce((sum, member) => sum + member.share, 0) }))
    .sort((left, right) => right.share - left.share);
  elements.sources.replaceChildren();
  if (!ordered.length) elements.sources.append(node("div", "empty", "No sources match."));
  for (const { family, members, share } of ordered) {
    const heading = node("div", "family");
    heading.append(node("span", "", family), node("span", "share", percent(share)));
    elements.sources.append(heading);
    for (const source of members.sort((left, right) => right.share - left.share)) {
      const row = node("button", "source" + (selected && selected.source === source ? " selected" : ""));
      row.type = "button";
      const label = source.name.startsWith(family + "/") ? source.name.slice(family.length + 1) : source.name;
      row.append(node("span", "name", label), node("span", "share", percent(source.share)));
      row.addEventListener("click", () => select(source, source.samples[0]));
      elements.sources.append(row);
    }
  }
  const selectedRow = elements.sources.querySelector(".source.selected");
  if (selectedRow) selectedRow.scrollIntoView({ block: "nearest" });
}

function renderText(text) {
  elements.text.replaceChildren();
  for (const [index, section] of text.split(OMITTED).entries()) {
    if (index % 2) {
      elements.text.append(node("span", "gap", section.trim()));
      continue;
    }
    for (const [part, piece] of section.split(SPECIAL_TOKEN).entries()) {
      if (piece) elements.text.append(part % 2 ? node("span", "token", piece) : document.createTextNode(piece));
    }
  }
  elements.text.scrollTop = 0;
}

async function renderSelection() {
  renderSources();
  const selected = selection();
  if (!selected) {
    elements.head.replaceChildren(node("div", "empty", "Pick a source, or draw a document by mix share."));
    elements.pager.replaceChildren();
    elements.text.replaceChildren();
    return;
  }
  const { source, sample } = selected;
  const facts = node("div", "facts");
  facts.append(
    node("span", "kind " + source.kind, source.kind === "sft" ? "SFT" : "Replay"),
    document.createTextNode(
      ` · ${percent(source.share)} of training positions · ${source.documents.toLocaleString()} documents · `,
    ),
    node("code", "", source.cache_path),
  );
  elements.head.replaceChildren(node("h2", "", source.name), facts);
  elements.pager.replaceChildren(
    ...source.samples.map((candidate) => {
      const label = `#${candidate.doc_index.toLocaleString()} · ${compact(candidate.tokens)} tok`;
      const button = node("button", candidate === sample ? "selected" : "", label);
      button.type = "button";
      button.addEventListener("click", () => select(source, candidate));
      return button;
    }),
  );
  const current = ++request;
  elements.text.textContent = "Loading…";
  const query = new URLSearchParams({ source: source.name, doc_index: String(sample.doc_index) });
  const response = await fetch("api/sample?" + query);
  if (current !== request) return;
  if (!response.ok) {
    elements.text.textContent = `Could not load this document (${response.status}).`;
    return;
  }
  renderText((await response.json()).text);
}

function draw() {
  const candidates = visibleSources();
  const total = candidates.reduce((sum, source) => sum + source.share, 0);
  if (!total) return;
  let remaining = Math.random() * total;
  let chosen = candidates[candidates.length - 1];
  for (const source of candidates) {
    remaining -= source.share;
    if (remaining < 0) {
      chosen = source;
      break;
    }
  }
  select(chosen, chosen.samples[Math.floor(Math.random() * chosen.samples.length)]);
}

async function start() {
  const response = await fetch("api/sources");
  if (!response.ok) {
    elements.summary.textContent = `Could not load sources (${response.status}).`;
    return;
  }
  sources = await response.json();
  byName = new Map(sources.map((source) => [source.name, source]));
  if (!sources.length) {
    elements.summary.textContent = "No snapshot is loaded yet.";
    return;
  }
  const parts = ["sft", "replay"].map((kind) => {
    const members = sources.filter((source) => source.kind === kind);
    const share = members.reduce((sum, source) => sum + source.share, 0);
    const samples = members.reduce((sum, source) => sum + source.samples.length, 0);
    const label = kind === "sft" ? "SFT stores" : "replay caches";
    return `${members.length} ${label} (${percent(share)}, ${samples.toLocaleString()} documents sampled)`;
  });
  elements.summary.textContent = "10.05 Grug SFT mixture · " + parts.join(" · ");
  elements.draw.addEventListener("click", draw);
  elements.kind.addEventListener("change", renderSources);
  elements.filter.addEventListener("input", renderSources);
  window.addEventListener("hashchange", renderSelection);
  await renderSelection();
}

start();
