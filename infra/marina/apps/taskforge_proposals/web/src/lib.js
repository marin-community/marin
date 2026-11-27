// Shared helpers: API calls, hash routing, formatting, Markdown, tooltip and toast state.
import DOMPurify from "dompurify";
import { marked } from "marked";
import { reactive } from "vue";

export const LIST_FILTERS = ["domain", "kind", "environment", "verdict", "status"];
export const RANGE_FILTERS = ["completion_tokens", "reasoning_tokens", "prompt_tokens"];
export const FILTER_KEYS = ["q", "capability", "cut_off", ...LIST_FILTERS, ...RANGE_FILTERS.flatMap((n) => [`${n}_min`, `${n}_max`])];
export const KIND_ORDER = ["answer_in_box", "code", "decision_model", "llm_as_judge", "composite"];
export const LABELS = {
  answer_in_box: "answer in box", code: "code", decision_model: "decision model", llm_as_judge: "LLM as judge",
  composite: "composite", reasoning: "reasoning", shellsim: "shellsim", container: "container",
  accept: "accept", reject: "reject", none: "no verdict", proposed: "proposed", cut_off: "cut off",
  failed: "failed", pending: "pending", completion_tokens: "completion tokens", reasoning_tokens: "reasoning tokens",
  prompt_tokens: "prompt tokens",
};
export const label = (key) => (key == null ? "—" : LABELS[key] || String(key));

// ---------- API ----------

export function queryString(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value == null || value === "") continue;
    for (const item of Array.isArray(value) ? value : [value]) if (item !== "" && item != null) search.append(key, item);
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export async function api(path, params) {
  const response = await fetch(`api/${path}${queryString(params)}`, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const encodeId = (id) => encodeURIComponent(id);

// ---------- routing ----------

export function parseRoute() {
  const raw = location.hash.replace(/^#/, "") || "/proposals";
  const [path, search = ""] = raw.split("?");
  const query = {};
  for (const [key, value] of new URLSearchParams(search)) {
    if (LIST_FILTERS.includes(key)) (query[key] ||= []).push(value);
    else query[key] = value;
  }
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  return { name: parts[0] || "proposals", parts: parts.slice(1), query };
}

export function href(path, query) {
  return `#${path}${queryString(query)}`;
}

export function go(path, query) {
  location.hash = href(path, query);
}

export function filtersOf(query) {
  const out = {};
  for (const key of FILTER_KEYS) if (query[key] != null && query[key] !== "") out[key] = query[key];
  return out;
}

// ---------- formatting ----------

export function num(value) {
  if (value == null) return "—";
  return Number(value).toLocaleString("en-US");
}

export function compact(value) {
  if (value == null) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

export function pct(part, whole, digits = 1) {
  if (!whole) return "—";
  return `${((100 * part) / whole).toFixed(digits)}%`;
}

export function duration(seconds) {
  if (seconds == null) return "—";
  const s = Math.round(seconds);
  if (s < 90) return `${s}s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export function verdictClass(row) {
  if (row.verdict === "accept") return "good";
  if (row.verdict === "reject") return "bad";
  if (row.cut_off || row.status === "cut_off") return "warn";
  return "";
}

export function verdictText(row) {
  if (row.verdict) return row.verdict;
  if (row.cut_off || row.status === "cut_off") return "cut off";
  return row.status || "—";
}

export function markdown(text) {
  return DOMPurify.sanitize(marked.parse(text || "", { gfm: true }));
}

export function pretty(value) {
  return JSON.stringify(value, null, 2);
}

// ---------- tooltip and toast ----------

export const ui = reactive({ tip: null, toast: "" });

export function showTip(event, title, rows = []) {
  ui.tip = { x: event.clientX, y: event.clientY, title, rows };
}

export function hideTip() {
  ui.tip = null;
}

let toastTimer = null;
export async function copyText(text, what = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
    ui.toast = what;
  } catch {
    ui.toast = "Copy failed: the clipboard is unavailable";
  }
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { ui.toast = ""; }, 1600);
}
