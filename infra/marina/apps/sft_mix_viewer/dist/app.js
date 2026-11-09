// Read sampled documents of the 10.05 SFT mixture: by dataset, or at random weighted by mix share.
// Each document can be shown as decoded text, as its tokens, or as token ids.

const SPECIAL_TOKEN = /(<\|[^|<>\n]{1,48}\|>)/;
const OMITTED = /(\n\n\[… [\d,]+ tokens omitted …\]\n\n)/;
const VIEWS = ["text", "tokens", "ids"];

const elements = {
  summary: document.getElementById("summary"),
  sources: document.getElementById("sources"),
  head: document.getElementById("source-head"),
  pager: document.getElementById("pager"),
  views: document.getElementById("views"),
  text: document.getElementById("text"),
  random: document.getElementById("random"),
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

function hashParameters() {
  return new URLSearchParams(location.hash.slice(1));
}

function currentView() {
  const view = hashParameters().get("view");
  return VIEWS.includes(view) ? view : "text";
}

function selection() {
  const parameters = hashParameters();
  const source = byName.get(parameters.get("source"));
  if (!source) return null;
  const docIndex = Number(parameters.get("doc"));
  const sample = source.samples.find((candidate) => candidate.doc_index === docIndex) || source.samples[0];
  return { source, sample };
}

function select(source, sample, view = currentView()) {
  location.hash = new URLSearchParams({ source: source.name, doc: String(sample.doc_index), view }).toString();
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
  if (!ordered.length) elements.sources.append(node("div", "empty", "No datasets match."));
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
  for (const [index, section] of text.split(OMITTED).entries()) {
    if (index % 2) {
      elements.text.append(node("span", "gap", section.trim()));
      continue;
    }
    for (const [part, piece] of section.split(SPECIAL_TOKEN).entries()) {
      if (piece) elements.text.append(part % 2 ? node("span", "token", piece) : document.createTextNode(piece));
    }
  }
}

// One chip per token. Tokens view shows each token's text with line breaks marked; ids view shows the number.
function renderTokens(sample, tokens, view) {
  const omitted = sample.tokens - tokens.token_ids.length;
  tokens.token_ids.forEach((id, position) => {
    if (omitted && position === tokens.head_tokens) {
      elements.text.append(node("span", "gap", `[… ${omitted.toLocaleString()} tokens omitted …]`));
    }
    const piece = tokens.pieces[position];
    const special = SPECIAL_TOKEN.test(piece);
    const chip = node("span", "tok " + (special ? "token" : position % 2 ? "odd" : "even"));
    if (view === "ids") {
      chip.textContent = String(id);
      chip.title = JSON.stringify(piece);
      elements.text.append(chip, document.createTextNode(" "));
    } else {
      chip.textContent = piece.replaceAll("\n", "↵\n");
      chip.title = `id ${id}`;
      elements.text.append(chip);
    }
  });
}

function renderViews(source, sample, view) {
  const labels = { text: "Text", tokens: "Tokens", ids: "Token IDs" };
  elements.views.replaceChildren(
    ...VIEWS.map((candidate) => {
      const button = node("button", candidate === view ? "selected" : "", labels[candidate]);
      button.type = "button";
      button.addEventListener("click", () => select(source, sample, candidate));
      return button;
    }),
  );
}

async function renderSelection() {
  renderSources();
  const selected = selection();
  if (!selected) {
    elements.head.replaceChildren(
      node("div", "empty", "Pick a dataset on the left to read documents sampled from it, or press Random document."),
    );
    elements.pager.replaceChildren();
    elements.views.replaceChildren();
    elements.text.replaceChildren();
    return;
  }
  const { source, sample } = selected;
  const view = currentView();
  const facts = node("div", "facts");
  facts.append(
    node("span", "kind " + source.kind, source.kind === "sft" ? "SFT" : "Replay"),
    document.createTextNode(
      ` · ${percent(source.share)} of training positions · ${source.documents.toLocaleString()} documents · `,
    ),
    node("code", "", source.cache_path),
  );
  elements.head.replaceChildren(node("h2", "", source.name), facts);
  const sampledOf = `${source.samples.length} sampled of ${source.documents.toLocaleString()}:`;
  elements.pager.replaceChildren(
    node("span", "label", sampledOf),
    ...source.samples.map((candidate) => {
      const label = `#${candidate.doc_index.toLocaleString()} · ${compact(candidate.tokens)} tok`;
      const button = node("button", candidate === sample ? "selected" : "", label);
      button.type = "button";
      button.title = `Document ${candidate.doc_index} of this dataset, ${candidate.tokens.toLocaleString()} tokens`;
      button.addEventListener("click", () => select(source, candidate));
      return button;
    }),
  );
  renderViews(source, sample, view);

  const current = ++request;
  elements.text.textContent = "Loading…";
  elements.text.className = view;
  const query = new URLSearchParams({ source: source.name, doc_index: String(sample.doc_index) });
  const response = await fetch((view === "text" ? "api/sample?" : "api/tokens?") + query);
  if (current !== request) return;
  if (!response.ok) {
    elements.text.textContent = `Could not load this document (${response.status}).`;
    return;
  }
  const body = await response.json();
  elements.text.replaceChildren();
  if (view === "text") renderText(body.text);
  else renderTokens(sample, body, view);
  elements.text.scrollTop = 0;
}

// Pick a dataset with probability equal to its share of training positions, then one of its samples.
function randomDocument() {
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
    elements.summary.textContent = `Could not load datasets (${response.status}).`;
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
    const label = kind === "sft" ? "SFT datasets" : "pretraining replay caches";
    return `${members.length} ${label} (${percent(share)} of training)`;
  });
  elements.summary.textContent = "What the 10.05 Grug SFT run trained on: " + parts.join(" and ") + ".";
  elements.random.addEventListener("click", randomDocument);
  elements.kind.addEventListener("change", renderSources);
  elements.filter.addEventListener("input", renderSources);
  window.addEventListener("hashchange", renderSelection);
  await renderSelection();
}

start();
