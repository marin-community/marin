// DOM building, formatting, markdown, clipboard, tooltips and small visual atoms.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "html") el.innerHTML = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  append(el, children);
  return el;
}

export function fill(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

// ---------- formatting ----------

export function duration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "–";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export function compact(n) {
  if (n === null || n === undefined) return "–";
  // Counts here are a few thousand at most: print them in full, which reads better than "4.0k".
  if (Math.abs(n) < 1e5) return Math.round(n).toLocaleString("en-US");
  if (Math.abs(n) < 1e6) return `${(n / 1000).toFixed(0)}k`;
  return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
}

export function tokens(n) {
  return n ? `${compact(n)} tokens` : "no tokens";
}

// A reward or score: at most three decimals, trailing zeros dropped.
export function num(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "–";
  return String(Number(Number(x).toFixed(3)));
}

export function percent(x) {
  return x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`;
}

export function fraction(a, b) {
  return `${a} of ${b}`;
}

export function when(unixSeconds) {
  if (!unixSeconds) return "";
  return new Date(unixSeconds * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function clock(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function plural(n, word, many) {
  return `${typeof n === "number" ? n.toLocaleString("en-US") : n} ${n === 1 ? word : many || `${word}s`}`;
}

export function shortHash(digest) {
  return digest ? digest.slice(0, 8) : "";
}

export function humanize(id) {
  return String(id).replaceAll("_", " ").replaceAll("-", " ");
}

export function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

export function words(list) {
  if (list.length <= 1) return list.join("");
  return `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}`;
}

// ---------- markdown, code, front matter ----------

marked.setOptions({ gfm: true, breaks: true });

const TRACEBACK = /(^|\n)(Traceback \(most recent call last\):[\s\S]*?\n[\w.]*(?:Error|Exception|Failure|Interrupt|Exit)\b[^\n]*)/g;

// Python tracebacks pasted into prose lose their layout as markdown; fence them unless already fenced.
function fenceTracebacks(text) {
  if (!text || text.includes("```")) return text || "";
  return text.replace(TRACEBACK, (_, lead, tb) => `${lead}\n\`\`\`\n${tb}\n\`\`\`\n`);
}

export function markdown(text, cls = "prose") {
  const el = h("div", { class: cls });
  el.innerHTML = DOMPurify.sanitize(marked.parse(fenceTracebacks(text)));
  for (const block of el.querySelectorAll("pre code")) hljs.highlightElement(block);
  return el;
}

export function code(text, language, opts = {}) {
  const codeEl = h("code", { class: language ? `language-${language}` : "" });
  codeEl.textContent = text || "";
  if (language && hljs.getLanguage(language)) hljs.highlightElement(codeEl);
  const pre = h("pre", { style: opts.maxHeight ? { maxHeight: opts.maxHeight } : null }, codeEl);
  const head = opts.title || opts.copy
    ? h("div", { class: "cb-head" }, h("span", {}, opts.title || ""), copyButton(() => text, opts.copyLabel || "Copy"))
    : null;
  return h("div", { class: `codeblock ${opts.cls || ""}` }, head, pre);
}

export function frontMatter(text) {
  if (!text) return { header: null, body: "" };
  const match = text.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { header: null, body: text };
  try {
    return { header: jsyaml.load(match[1]), body: match[2] };
  } catch {
    return { header: null, body: match[2] };
  }
}

// Sections of a markdown body keyed by "## " heading.
export function sections(body) {
  const out = {};
  let current = null;
  for (const line of (body || "").split("\n")) {
    const heading = line.match(/^##\s+(.*)/);
    if (heading) {
      current = heading[1].trim();
      out[current] = [];
    } else if (current) out[current].push(line);
  }
  return Object.fromEntries(Object.entries(out).map(([key, lines]) => [key, lines.join("\n").trim()]));
}

export function firstSentence(text, limit = 160) {
  const plain = (text || "").replace(/[*_`#>]/g, "").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/\s+/g, " ").trim();
  const match = plain.match(/^(.{20,}?[.!?])\s/);
  const sentence = match ? match[1] : plain;
  return sentence.length > limit ? `${sentence.slice(0, limit - 1)}…` : sentence;
}

export function tryJson(text) {
  if (typeof text !== "string") return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// ---------- line diff (LCS over lines, collapsed context) ----------

export function lineDiff(before, after) {
  const a = (before || "").split("\n");
  const b = (after || "").split("\n");
  const n = a.length;
  const m = b.length;
  if (n * m > 4_000_000) return null;
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push(["same", a[i]]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push(["del", a[i++]]);
    else ops.push(["add", b[j++]]);
  }
  while (i < n) ops.push(["del", a[i++]]);
  while (j < m) ops.push(["add", b[j++]]);
  return ops;
}

export function diffView(before, after, context = 3) {
  const ops = lineDiff(before, after);
  if (!ops) return h("div", { class: "empty inline" }, "These versions are too large to compare line by line.");
  const changed = ops.map((op) => op[0] !== "same");
  const keep = ops.map((_, idx) => {
    for (let d = -context; d <= context; d++) if (changed[idx + d]) return true;
    return false;
  });
  const root = h("div", { class: "diff" });
  let hidden = [];
  const flush = () => {
    if (!hidden.length) return;
    const lines = hidden;
    const gap = h("div", { class: "gap" }, `${plural(lines.length, "unchanged line")} — show`);
    gap.addEventListener("click", () => gap.replaceWith(...lines.map((line) => h("div", {}, line || " "))));
    root.append(gap);
    hidden = [];
  };
  ops.forEach(([kind, line], idx) => {
    if (!keep[idx]) { hidden.push(line); return; }
    flush();
    root.append(h("div", { class: kind === "same" ? "" : kind }, line || " "));
  });
  flush();
  const adds = ops.filter((op) => op[0] === "add").length;
  const dels = ops.filter((op) => op[0] === "del").length;
  return h("div", {}, h("div", { class: "chips", style: { marginBottom: "8px" } }, h("span", { class: "chip good" }, `${adds} lines added`), h("span", { class: "chip bad" }, `${dels} lines removed`)), root);
}

// ---------- clipboard, toast, tooltip ----------

export function toast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.add("on");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("on"), 1600);
}

export async function copy(value, what = "data") {
  const resolved = typeof value === "function" ? await value() : value;
  const text = typeof resolved === "string" ? resolved : JSON.stringify(resolved, null, 2);
  await navigator.clipboard.writeText(text);
  toast(`Copied ${what} (${compact(text.length)} characters)`);
}

export function copyButton(value, label = "Copy data", what) {
  return h("button", { class: "ghost copy", title: "Copy the underlying record to the clipboard", onclick: (event) => { event.stopPropagation(); copy(value, what || label.replace(/^Copy /, "").toLowerCase()); } }, `⧉ ${label}`);
}

const tooltipEl = () => document.getElementById("tooltip");
export function tip(el, content) {
  el.addEventListener("mouseenter", () => {
    const box = tooltipEl();
    box.replaceChildren(typeof content === "function" ? content() : content);
    box.classList.add("on");
  });
  el.addEventListener("mousemove", (event) => {
    const box = tooltipEl();
    const x = Math.min(event.clientX + 14, window.innerWidth - box.offsetWidth - 8);
    const y = event.clientY + 16 + box.offsetHeight > window.innerHeight ? event.clientY - box.offsetHeight - 10 : event.clientY + 16;
    box.style.left = `${x}px`;
    box.style.top = `${y}px`;
  });
  el.addEventListener("mouseleave", () => tooltipEl().classList.remove("on"));
  return el;
}

export function tipBody(title, ...rows) {
  return h("div", {}, h("div", { class: "tt-title" }, title), rows.filter(Boolean).map((row) => h("div", { class: "tt-row" }, row)));
}

export function hideTip() {
  tooltipEl().classList.remove("on");
}

// ---------- small visual atoms ----------

export function chip(text, tone = "", attrs = {}) {
  return h("span", { class: `chip ${tone}`, ...attrs }, text);
}

export function meter(value, color = "var(--accent)", width) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100;
  return h("div", { class: "meter", style: width ? { width } : null }, h("i", { style: { width: `${pct}%`, background: color } }));
}

export function stackBar(parts, attrs = {}) {
  const total = parts.reduce((sum, part) => sum + part.value, 0) || 1;
  const bar = h("div", { class: "stackbar", ...attrs });
  for (const part of parts) {
    if (!part.value) continue;
    bar.append(tip(h("i", { style: { flex: `${part.value / total} 0 0`, background: part.color } }), part.tip || part.label));
  }
  return bar;
}

export function legend(entries) {
  return h("div", { class: "legend" }, entries.map(([label, color]) => h("span", { class: "key" }, h("span", { class: "sw", style: { background: color } }), label)));
}

export function fold(summary, body, open = false) {
  const el = h("details", { class: "fold", open }, h("summary", {}, summary), h("div", { class: "fold-body" }));
  let filled = false;
  const fill = () => {
    if (filled) return;
    filled = true;
    el.querySelector(".fold-body").append(typeof body === "function" ? body() : body);
  };
  if (open) fill();
  el.addEventListener("toggle", () => el.open && fill());
  return el;
}

export function tabs(entries, initial) {
  const bar = h("div", { class: "tabs" });
  const body = h("div", {});
  const select = (key) => {
    for (const button of bar.children) button.classList.toggle("on", button.dataset.key === key);
    const entry = entries.find((e) => e.key === key);
    body.replaceChildren(typeof entry.render === "function" ? entry.render() : entry.render);
  };
  for (const entry of entries) bar.append(h("button", { dataset: { key: entry.key }, onclick: () => select(entry.key) }, entry.label));
  if (entries.length) select(initial && entries.some((e) => e.key === initial) ? initial : entries[0].key);
  return h("div", {}, bar, body);
}

export function card(title, body, opts = {}) {
  return h("div", { class: `card ${opts.cls || ""}` },
    title !== null ? h("div", { class: "card-head" }, h("div", { class: "title" }, opts.icon || null, h("h3", {}, title), opts.badge || null), h("div", { class: "actions" }, opts.actions || [])) : null,
    body);
}

export function tile(label, value, note) {
  return h("div", { class: "tile" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), note ? h("div", { class: "note" }, note) : null);
}

export function bandGauge(rate, band) {
  const el = h("div", {});
  const gauge = h("div", { class: "gauge" }, h("div", { class: "track" }));
  if (band) gauge.append(h("div", { class: "bandfill", style: { left: `${band[0] * 100}%`, width: `${(band[1] - band[0]) * 100}%` } }));
  if (rate !== null && rate !== undefined) gauge.append(h("div", { class: "needle", style: { left: `${rate * 100}%` } }));
  el.append(gauge, h("div", { class: "gauge-labels" }, h("span", {}, "never solved"), band ? h("span", {}, `target band ${percent(band[0])} to ${percent(band[1])}`) : null, h("span", {}, "always solved")));
  return el;
}

// ---------- vocabulary ----------

// ---------- vocabulary of forge tasks ----------

export const STOP_REASONS = {
  stop: "ended its turn normally",
  length: "ran out of output budget",
  tool_calls: "stopped to call a tool",
  agent_timeout: "hit the agent deadline",
  total_turn_timeout: "hit the deadline for all of its turns",
  max_turns: "used every allowed turn",
  context: "filled its context window",
};

export const BANDS = {
  too_hard: { label: "Too hard", tone: "warn", color: "var(--serious)", note: "solved in fewer than 25% of trials" },
  in_band: { label: "In band", tone: "good", color: "var(--good)", note: "solved in 25% to 75% of trials" },
  too_easy: { label: "Too easy", tone: "info", color: "var(--accent)", note: "solved in more than 75% of trials" },
};

export const GRADERS = {
  code: { label: "Grader program", note: "a Python program the task ships checks the answer or the files the solver produced" },
  answer_in_box: { label: "Boxed answer", note: "the final answer is compared with the reference: exactly, numerically, symbolically or as a choice" },
};

export const ENVIRONMENTS = {
  reasoning: { label: "Answer only", note: "the solver reads the task and replies; there is no machine" },
  shellsim: { label: "Simulated shell", note: "the solver works in a simulated shell with the task's files and writes its answer there" },
  linux: { label: "Linux sandbox", note: "the solver works in a sandboxed Linux machine with Python and the task's files" },
};

export const PARTS = {
  instructions: "Instructions",
  environment: "Environment",
  grader: "Grader",
  reference: "Reference answer",
  controls: "Controls",
};

export const EFFECTS = {
  correct_rejected: { label: "Correct work rejected", note: "the grader failed an answer that was right" },
  wrong_accepted: { label: "Wrong work accepted", note: "the grader passed an answer that was wrong" },
  inconsistent_grading: { label: "Inconsistent grading", note: "equivalent answers were graded differently" },
  ambiguity_split: { label: "Ambiguous task", note: "the task allowed readings that the grader does not all accept" },
  solver_blocked: { label: "Solver blocked", note: "something in the task kept a capable solver from finishing" },
  solver_misled: { label: "Solver misled", note: "the task text pointed the solver at the wrong thing" },
  shortcut: { label: "Shortcut", note: "the task could be passed without doing the intended work" },
};

export const CONFIDENCE = { established: "established", probable: "probable", possible: "possible" };

export const ADVERSARIES = {
  false_accept: { label: "False-accept hunter", short: "False accept", note: "tries to make the grader credit a wrong answer" },
  false_reject: { label: "False-reject hunter", short: "False reject", note: "tries to find a correct answer the grader refuses" },
};

export const CONTROL_KINDS = {
  positive: "Correct answer",
  negative: "Wrong answer",
  malformed: "Malformed answer",
};
