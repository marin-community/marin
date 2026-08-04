"use strict";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const PAGE_LIMIT = 200;
const SEGMENTS = ["begin", "middle", "end"];
const RANGE_KEYS = ["begin", "middle", "end", "overall"];
const RANGE_LABELS = {
  begin: "begin segment",
  middle: "middle segment",
  end: "end segment",
  overall: "worst segment",
};
const VERSION_KEY = "sampleBrowser.scoreVersion";
const DEFAULT_VERSION = "v1";
const STEP = 0.1;
// A quick "= N" button selects everything that rounds to N. For the integer oracle
// versions that is exactly N; for the continuous model score it is N's bucket.
const EXACT_HALF_WIDTH = 0.5;

const state = {
  hits: [],
  total: 0,
  index: -1,
  doc: null,
  pdf: null,
  pdfPage: 1,
  openSegment: null,
  syncing: false,
  version: DEFAULT_VERSION,
  versions: [],
  scoreRange: { min: 0, max: 5 },
  // key -> {min, max}; null bounds mean unfiltered on that side.
  ranges: Object.fromEntries(RANGE_KEYS.map((k) => [k, { min: null, max: null }])),
  openRange: null,
};

function otherVersions(v) {
  return state.versions.map((x) => x.id).filter((id) => id !== v);
}

function fmtScore(v) {
  if (v === null || v === undefined) return null;
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

function versionInfo(v) {
  return state.versions.find((x) => x.id === v) || { id: v, label: v, available: false };
}

const el = (id) => document.getElementById(id);

function scoreClass(v) {
  return v === null || v === undefined || v < 0 ? "s-none" : "s" + Math.round(v);
}

function rangeText(r) {
  const { min, max } = r;
  if (min === null && max === null) return "any";
  if (min !== null && max !== null) {
    if (Math.abs(max - min - 2 * EXACT_HALF_WIDTH) < 1e-9) return "=" + fmtScore(min + EXACT_HALF_WIDTH);
    if (Math.abs(max - min) < 1e-9) return "=" + fmtScore(min);
    return `${fmtScore(min)}–${fmtScore(max)}`;
  }
  return min !== null ? `${fmtScore(min)}+` : `≤${fmtScore(max)}`;
}

function renderRangeButtons() {
  for (const node of document.querySelectorAll(".range-btn[data-key]")) {
    const r = state.ranges[node.dataset.key];
    const active = r.min !== null || r.max !== null;
    node.querySelector(".range-tag").textContent = rangeText(r);
    node.classList.toggle("active", active);
  }
}

function queryParams() {
  const p = new URLSearchParams();
  p.set("limit", PAGE_LIMIT);
  const q = el("q").value.trim();
  if (q) {
    p.set("q", q);
    p.set("q_field", el("qField").value);
  }
  for (const key of RANGE_KEYS) {
    const { min, max } = state.ranges[key];
    if (min !== null) p.set("min_" + key, min);
    if (max !== null) p.set("max_" + key, max);
  }
  if (el("status").value) p.set("status", el("status").value);
  p.set("score_version", state.version);
  return p;
}

function syncRangeInputs() {
  const r = state.ranges[state.openRange];
  const lo = r.min === null ? state.scoreRange.min : r.min;
  const hi = r.max === null ? state.scoreRange.max : r.max;
  el("rangeMin").value = lo;
  el("rangeMax").value = hi;
  el("rangeMinVal").textContent = fmtScore(lo);
  el("rangeMaxVal").textContent = fmtScore(hi);
  el("rangePopValue").textContent = rangeText(r);
}

function setRange(key, min, max) {
  state.ranges[key] = { min, max };
  renderRangeButtons();
  if (state.openRange === key) syncRangeInputs();
  search(false);
}

function openRangePop(key, button) {
  state.openRange = key;
  const pop = el("rangePop");
  el("rangePopTitle").textContent = RANGE_LABELS[key];
  for (const input of [el("rangeMin"), el("rangeMax")]) {
    input.min = state.scoreRange.min;
    input.max = state.scoreRange.max;
    input.step = STEP;
  }
  syncRangeInputs();
  const rect = button.getBoundingClientRect();
  pop.style.left = Math.max(4, Math.min(rect.left, window.innerWidth - 260)) + "px";
  pop.style.top = rect.bottom + 4 + "px";
  pop.classList.remove("hidden");
}

function closeRangePop() {
  state.openRange = null;
  el("rangePop").classList.add("hidden");
}

function onRangeSlide(which) {
  const key = state.openRange;
  if (!key) return;
  let min = Number(el("rangeMin").value);
  let max = Number(el("rangeMax").value);
  // Keep the handles ordered: dragging one past the other pushes the other along.
  if (min > max) {
    if (which === "min") max = min;
    else min = max;
  }
  setRange(key, min, max);
}

function renderRangeQuick() {
  const buttons = ['<button data-exact="">any</button>'];
  for (let i = state.scoreRange.min; i <= state.scoreRange.max; i++) {
    buttons.push(`<button data-exact="${i}">=${i}</button>`);
  }
  el("rangeQuick").innerHTML = buttons.join("");
  for (const node of el("rangeQuick").querySelectorAll("button[data-exact]")) {
    node.onclick = () => {
      const raw = node.dataset.exact;
      if (raw === "") return setRange(state.openRange, null, null);
      const n = Number(raw);
      const lo = Math.max(state.scoreRange.min, n - EXACT_HALF_WIDTH);
      const hi = Math.min(state.scoreRange.max, n + EXACT_HALF_WIDTH);
      setRange(state.openRange, lo, hi);
    };
  }
}

async function search(keepSelection) {
  const res = await fetch("/api/docs?" + queryParams().toString());
  const data = await res.json();
  state.hits = data.docs;
  state.total = data.total;
  el("count").textContent =
    `${data.total} match${data.total === 1 ? "" : "es"}` +
    (data.total > data.docs.length ? ` (showing ${data.docs.length})` : "");
  renderDrawer();
  if (!keepSelection) {
    state.index = -1;
    if (state.hits.length) selectIndex(0);
    else clearDoc();
  }
}

function renderDrawer() {
  const drawer = el("drawer");
  if (!state.hits.length) {
    drawer.innerHTML = '<div class="hit dim">no matches</div>';
    return;
  }
  const others = otherVersions(state.version);
  drawer.innerHTML = state.hits
    .map((d, i) => {
      const chips = SEGMENTS.map((s) => {
        const v = d.scores[state.version][s];
        const altText = others.map((o) => fmtScore(d.scores[o][s]) ?? "–").join("/");
        return (
          `<span class="chip ${scoreClass(v)}">${fmtScore(v) ?? "–"}` +
          `<sub class="alt">${altText}</sub></span>`
        );
      }).join("");
      return `<div class="hit ${i === state.index ? "active" : ""}" data-i="${i}">
        <span class="name" title="${escapeAttr(d.url)}">${escapeHtml(d.title)}</span>
        <span class="dim">${d.num_pages}p ${d.extraction_status}</span>
        <span class="chips">${chips}</span></div>`;
    })
    .join("");
  for (const node of drawer.querySelectorAll(".hit[data-i]")) {
    node.onclick = () => selectIndex(Number(node.dataset.i));
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}

function updatePosition() {
  el("position").textContent =
    state.index >= 0 ? `${state.index + 1}/${state.hits.length}` : "";
  for (const node of el("drawer").querySelectorAll(".hit[data-i]")) {
    node.classList.toggle("active", Number(node.dataset.i) === state.index);
  }
}

function clearDoc() {
  state.doc = null;
  state.pdf = null;
  el("pdfTitle").textContent = "no document selected";
  el("pdfMeta").textContent = "";
  el("metaTable").innerHTML = "";
  el("textArea").innerHTML = "";
  el("chips").innerHTML = "";
  el("pdfPlaceholder").classList.remove("hidden");
  el("pdfPlaceholder").textContent = "select a document";
  el("pdfCanvas").classList.add("hidden");
  updatePosition();
}

async function selectIndex(i) {
  if (i < 0 || i >= state.hits.length) return;
  state.index = i;
  updatePosition();
  await loadDoc(state.hits[i].id);
}

const META_ORDER = [
  "url",
  "source",
  "source_id",
  "content_digest",
  "warc_filename",
  "warc_record_offset",
  "num_pages",
  "extraction_status",
  "extraction_error",
  "pages_ocred",
  "pages_failed",
  "pages_truncated",
  "pages_unrendered",
  "pages_below_legibility_floor",
  "boilerplate_lines_removed",
  "mean_render_dpi",
  "completion_tokens",
];

async function loadDoc(id) {
  const res = await fetch("/api/doc/" + id);
  const doc = await res.json();
  state.doc = doc;
  state.openSegment = null;
  el("reasoning").classList.add("hidden");

  const m = doc.metadata;
  el("pdfTitle").textContent = doc.title;
  el("pdfTitle").title = m.url;
  el("pdfMeta").textContent = `${m.num_pages} pages · ${m.extraction_status}`;

  const keys = META_ORDER.filter((k) => k in m).concat(
    Object.keys(m).filter((k) => !META_ORDER.includes(k) && k !== "page_offsets" && k !== "id")
  );
  el("metaTable").innerHTML =
    `<dt>id</dt><dd>${escapeHtml(doc.id)}</dd>` +
    keys
      .map((k) => {
        const v = m[k];
        const shown =
          k === "url"
            ? `<a href="${escapeAttr(v)}" target="_blank" rel="noreferrer">${escapeHtml(v)}</a>`
            : escapeHtml(v === null ? "—" : v);
        return `<dt>${k}</dt><dd>${shown}</dd>`;
      })
      .join("");

  renderChips(doc);
  renderPages(doc);
  await loadPdf(doc.id);
}

function renderChips(doc) {
  const others = otherVersions(state.version);
  el("chips").innerHTML = SEGMENTS.map((s) => {
    const v = doc.scores[state.version][s].score;
    const label = s[0].toUpperCase();
    const text = fmtScore(v) ?? "…";
    const altScores = others.map((o) => fmtScore(doc.scores[o][s].score) ?? "…");
    const tip = [`${s}: ${state.version}=${text}`]
      .concat(others.map((o, i) => `${o}=${altScores[i]}`))
      .join(", ");
    return (
      `<span class="chip ${scoreClass(v)}" data-seg="${s}" title="${escapeAttr(tip)}">` +
      `${label} ${text}<sub class="alt">${altScores.join("/")}</sub></span>`
    );
  }).join("");
  for (const node of el("chips").querySelectorAll(".chip[data-seg]")) {
    node.onclick = () => toggleReasoning(node.dataset.seg);
  }
  const info = versionInfo(state.version);
  el("versionNote").textContent = info.available ? info.label : info.label + " pending";
}

function reasoningBlock(doc, version, segment) {
  const s = doc.scores[version][segment];
  const head = `[${version} · ${versionInfo(version).label} · ${segment} · score ${fmtScore(s.score) ?? "pending"}]`;
  if (s.reasoning) return `${head}\n\n${s.reasoning}`;
  // A scorer that emits a number and nothing else is not "pending" — say so.
  const info = versionInfo(version);
  const why = info.has_reasoning ? "reasoning pending — no scoring row yet." : "no reasoning: this score comes from a trained model, not a rubric prompt.";
  return `${head}\n\n${why}`;
}

function toggleReasoning(segment) {
  const box = el("reasoning");
  if (state.openSegment === segment) {
    state.openSegment = null;
    box.classList.add("hidden");
  } else {
    state.openSegment = segment;
    const ordered = [state.version, ...otherVersions(state.version)];
    box.textContent = ordered
      .map((v) => reasoningBlock(state.doc, v, segment))
      .join("\n\n" + "─".repeat(30) + "\n\n");
    box.classList.remove("hidden");
  }
  for (const node of el("chips").querySelectorAll(".chip[data-seg]")) {
    node.classList.toggle("sel", node.dataset.seg === state.openSegment);
  }
}

function renderPages(doc) {
  const area = el("textArea");
  area.innerHTML = doc.pages
    .map(
      (text, i) =>
        `<div class="page-divider" id="pdiv-${i + 1}">page ${i + 1} of ${doc.pages.length}</div>` +
        (text
          ? `<div class="page" id="page-${i + 1}">${escapeHtml(text)}</div>`
          : `<div class="page page-empty" id="page-${i + 1}">(no OCR output — blank or failed page)</div>`)
    )
    .join("");
  area.scrollTop = 0;
}

async function loadPdf(id) {
  state.pdf = null;
  state.pdfPage = 1;
  const canvas = el("pdfCanvas");
  const placeholder = el("pdfPlaceholder");
  canvas.classList.add("hidden");
  placeholder.classList.remove("hidden");
  placeholder.textContent = "loading PDF…";
  el("pdfPageLabel").textContent = "– / –";

  const res = await fetch("/api/pdf/" + id);
  if (!res.ok) {
    let reason = "PDF not yet available";
    try {
      reason = (await res.json()).error || reason;
    } catch (e) {
      /* non-JSON body */
    }
    placeholder.textContent = "PDF not yet available — " + reason;
    return;
  }
  const buf = await res.arrayBuffer();
  try {
    state.pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  } catch (e) {
    placeholder.textContent = "failed to parse PDF: " + e.message;
    return;
  }
  placeholder.classList.add("hidden");
  canvas.classList.remove("hidden");
  await showPdfPage(1, false);
}

async function showPdfPage(n, sync) {
  if (!state.pdf) return;
  n = Math.min(Math.max(1, n), state.pdf.numPages);
  state.pdfPage = n;
  const page = await state.pdf.getPage(n);
  const canvas = el("pdfCanvas");
  const width = el("pdfArea").clientWidth - 24;
  const base = page.getViewport({ scale: 1 });
  const viewport = page.getViewport({ scale: Math.max(0.2, width / base.width) });
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
  el("pdfPageLabel").textContent = `${n} / ${state.pdf.numPages}`;
  if (sync !== false) scrollTextToPage(n);
  markCurrentDivider(n);
}

function markCurrentDivider(n) {
  for (const node of el("textArea").querySelectorAll(".page-divider")) {
    node.classList.toggle("current", node.id === "pdiv-" + n);
  }
}

function scrollTextToPage(n) {
  const target = document.getElementById("pdiv-" + n);
  if (!target) return;
  state.syncing = true;
  target.scrollIntoView({ block: "start" });
  setTimeout(() => (state.syncing = false), 250);
}

function textScrollHandler() {
  if (state.syncing || !state.doc) return;
  const area = el("textArea");
  const top = area.getBoundingClientRect().top;
  let current = 1;
  for (const node of area.querySelectorAll(".page-divider")) {
    if (node.getBoundingClientRect().top - top <= 12) current = Number(node.id.slice(5));
  }
  markCurrentDivider(current);
  if (state.pdf && current !== state.pdfPage) showPdfPage(current, false);
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function renderVersionSelect() {
  const sel = el("scoreVersion");
  sel.innerHTML = state.versions
    .map((v) => `<option value="${v.id}">${v.label}${v.available ? "" : " (pending)"}</option>`)
    .join("");
  sel.value = state.version;
}

function setVersion(version) {
  state.version = version;
  localStorage.setItem(VERSION_KEY, version);
  renderVersionSelect();
  if (state.doc) {
    renderChips(state.doc);
    if (state.openSegment) {
      const segment = state.openSegment;
      state.openSegment = null;
      toggleReasoning(segment);
    }
  }
  search(true).then(() => {
    // Filters are version-scoped, so the hit set can change under the selection.
    if (state.index >= state.hits.length) state.index = state.hits.length - 1;
    if (state.index < 0 && state.hits.length) selectIndex(0);
    else updatePosition();
  });
}

async function loadVersions() {
  const res = await fetch("/api/schema");
  const schema = await res.json();
  state.versions = schema.versions || [{ id: "v1", label: "v1", available: true }];
  if (schema.score_range) state.scoreRange = schema.score_range;
  const saved = localStorage.getItem(VERSION_KEY);
  state.version = state.versions.some((v) => v.id === saved) ? saved : DEFAULT_VERSION;
  renderVersionSelect();
  renderRangeQuick();
  renderRangeButtons();
  document.title = `PDF OCR sample browser (${schema.num_docs} docs)`;
}

function init() {
  el("q").oninput = debounce(() => search(false), 250);
  el("qField").onchange = () => search(false);
  el("status").onchange = () => search(false);

  for (const node of document.querySelectorAll(".range-btn[data-key]")) {
    node.onclick = (e) => {
      e.stopPropagation();
      if (state.openRange === node.dataset.key) closeRangePop();
      else openRangePop(node.dataset.key, node);
    };
  }
  el("rangeMin").oninput = () => onRangeSlide("min");
  el("rangeMax").oninput = () => onRangeSlide("max");
  el("rangePop").onclick = (e) => e.stopPropagation();
  el("clearRanges").onclick = () => {
    for (const key of RANGE_KEYS) state.ranges[key] = { min: null, max: null };
    renderRangeButtons();
    if (state.openRange) syncRangeInputs();
    search(false);
  };
  document.addEventListener("click", () => {
    if (state.openRange) closeRangePop();
  });
  el("prevDoc").onclick = () => selectIndex(state.index - 1);
  el("nextDoc").onclick = () => selectIndex(state.index + 1);
  el("toggleList").onclick = () => el("drawer").classList.toggle("hidden");
  el("metaToggle").onclick = () => el("metaTable").classList.toggle("hidden");
  el("pdfPrev").onclick = () => showPdfPage(state.pdfPage - 1, true);
  el("pdfNext").onclick = () => showPdfPage(state.pdfPage + 1, true);
  el("textArea").addEventListener("scroll", debounce(textScrollHandler, 120));

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.openRange) return closeRangePop();
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") selectIndex(state.index - 1);
    if (e.key === "ArrowRight") selectIndex(state.index + 1);
  });

  el("scoreVersion").onchange = (e) => setVersion(e.target.value);

  loadVersions().then(() => search(false));
}

init();
