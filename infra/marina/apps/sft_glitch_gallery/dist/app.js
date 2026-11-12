// Browse flawed training documents by kind of flaw. Each example opens scrolled to its highlighted passage.

const SPECIAL_TOKEN = /(<\|[^|<>\n]{1,48}\|>)/;
const OMITTED = /(\n*\[… [^\]\n]{1,80} omitted …\]\n*)/;
const SEVERITY_LABEL = { high: "High", medium: "Medium", low: "Low" };

const elements = {
  categories: document.getElementById("categories"),
  categoryHead: document.getElementById("category-head"),
  examples: document.getElementById("examples"),
  exampleHead: document.getElementById("example-head"),
  text: document.getElementById("text"),
};

let categories = [];
let request = 0;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function compact(count) {
  if (count >= 1e6) return (count / 1e6).toFixed(1) + "M";
  if (count >= 1e3) return (count / 1e3).toFixed(1) + "k";
  return String(count);
}

function selection() {
  const parameters = new URLSearchParams(location.hash.slice(1));
  const category = categories.find((candidate) => candidate.key === parameters.get("kind")) || categories[0];
  if (!category) return null;
  const docIndex = Number(parameters.get("doc"));
  const example =
    category.examples.find(
      (candidate) => candidate.source === parameters.get("source") && candidate.doc_index === docIndex,
    ) || category.examples[0];
  return { category, example };
}

function select(category, example) {
  const parameters = { kind: category.key };
  if (example) Object.assign(parameters, { source: example.source, doc: String(example.doc_index) });
  location.hash = new URLSearchParams(parameters).toString();
}

function renderCategories(selected) {
  elements.categories.replaceChildren(
    ...categories.map((category) => {
      const row = node("button", "category" + (category === selected ? " selected" : ""));
      row.type = "button";
      row.append(
        node("span", "dot " + category.severity),
        node("span", "name", category.title),
        node("span", "count", String(category.examples.length)),
      );
      row.title = SEVERITY_LABEL[category.severity] + " severity";
      row.addEventListener("click", () => select(category, category.examples[0]));
      return row;
    }),
  );
}

function renderExamples(category, selected) {
  const severity = node("span", "severity " + category.severity, SEVERITY_LABEL[category.severity] + " severity");
  const prevalence = node("p", "prevalence");
  prevalence.append(node("b", "", "In the review: "), document.createTextNode(category.prevalence));
  elements.categoryHead.replaceChildren(
    node("h2", "", category.title),
    severity,
    node("p", "", category.description),
    prevalence,
  );
  elements.examples.replaceChildren(
    ...category.examples.map((example) => {
      const row = node("button", "example" + (example === selected ? " selected" : ""));
      row.type = "button";
      row.append(
        node("span", "source", example.source),
        node("span", "meta", `#${example.doc_index.toLocaleString()} · ${compact(example.tokens)} tokens`),
      );
      row.addEventListener("click", () => select(category, example));
      return row;
    }),
  );
  if (!category.examples.length) elements.examples.append(node("div", "empty", "No examples loaded for this kind."));
  const selectedRow = elements.examples.querySelector(".example.selected");
  if (selectedRow) selectedRow.scrollIntoView({ block: "nearest" });
}

// Plain text with template tokens and omission markers set apart.
function appendText(parent, text) {
  for (const [index, section] of text.split(OMITTED).entries()) {
    if (index % 2) {
      parent.append(node("span", "gap", section.trim()));
      continue;
    }
    for (const [part, piece] of section.split(SPECIAL_TOKEN).entries()) {
      if (piece) parent.append(part % 2 ? node("span", "token", piece) : document.createTextNode(piece));
    }
  }
}

function renderDocument(example) {
  const start = example.match_start;
  const end = start + example.match_length;
  const mark = node("mark");
  elements.text.replaceChildren();
  appendText(elements.text, example.text.slice(0, start));
  appendText(mark, example.text.slice(start, end));
  elements.text.append(mark);
  appendText(elements.text, example.text.slice(end));
  mark.scrollIntoView({ block: "center" });
}

async function render() {
  const selected = selection();
  if (!selected) return;
  const { category, example } = selected;
  renderCategories(category);
  renderExamples(category, example);
  if (!example) {
    elements.exampleHead.replaceChildren();
    elements.text.replaceChildren();
    return;
  }
  const position = category.examples.indexOf(example) + 1;
  const facts = node("div", "facts");
  facts.append(
    node("span", "", `Example ${position} of ${category.examples.length} · document #${example.doc_index.toLocaleString()}`),
    node("span", "", ` · ${example.tokens.toLocaleString()} tokens · highlighted: ${example.note}`),
  );
  elements.exampleHead.replaceChildren(node("h2", "", example.source), facts);

  const current = ++request;
  elements.text.textContent = "Loading…";
  const query = new URLSearchParams({
    category: category.key,
    source: example.source,
    doc_index: String(example.doc_index),
  });
  const response = await fetch("api/example?" + query);
  if (current !== request) return;
  if (!response.ok) {
    elements.text.textContent = `Could not load this example (${response.status}).`;
    return;
  }
  renderDocument(await response.json());
}

// Up and down arrows (or k and j) step through the examples of the current kind.
function step(event) {
  const delta = { ArrowDown: 1, j: 1, ArrowUp: -1, k: -1 }[event.key];
  const selected = selection();
  if (!delta || !selected || !selected.example || event.metaKey || event.ctrlKey || event.altKey) return;
  const examples = selected.category.examples;
  const next = examples[examples.indexOf(selected.example) + delta];
  if (!next) return;
  event.preventDefault();
  select(selected.category, next);
}

async function start() {
  const response = await fetch("api/categories");
  if (!response.ok) {
    elements.categoryHead.textContent = `Could not load the gallery (${response.status}).`;
    return;
  }
  categories = await response.json();
  if (!categories.length) {
    elements.categoryHead.textContent = "No examples are loaded yet.";
    return;
  }
  window.addEventListener("hashchange", render);
  window.addEventListener("keydown", step);
  await render();
}

start();
