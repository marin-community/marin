// Shared task-list vocabulary: filters (as URL parameters), flags, and the compact task row.

import { h, chip, tip, tipBody, BANDS, GRADERS, ENVIRONMENTS, PARTS, EFFECTS, plural } from "../util.js";
import { trialDots } from "../charts.js";
import { prefetchTask } from "../data.js";

// What validation caught in a task before it was accepted. `trigger`: the flag sends a task back to its author.
export const FLAGS = [
  { key: "ctl", label: "A known answer was graded wrong", short: "controls", note: "one of the author's own reference answers did not get the grade it should", color: "var(--phase-controls)", trigger: true, test: (t) => t.ctl > 0 },
  { key: "fa", label: "Grader credited a wrong answer", short: "false accept", note: "the false-accept hunter found a wrong answer the grader passed", color: "var(--critical)", trigger: true, test: (t) => t.fa > 0 },
  { key: "fr", label: "Grader refused a right answer", short: "false reject", note: "the false-reject hunter found a correct answer the grader failed", color: "var(--serious)", trigger: true, test: (t) => t.fr > 0 },
  { key: "review", label: "Reviewer found a task flaw", short: "review", note: "the transcript reviewer found a flaw in the instructions, files, grader or reference", color: "var(--phase-adversary)", trigger: true, test: (t) => t.defects > 0 },
  { key: "band_round", label: "Difficulty was adjusted", short: "difficulty", note: "the task was too easy or too hard and its author got one round to change that", color: "var(--accent)", trigger: true, test: (t) => t.band_rounds > 0 },
  { key: "revised", label: "Revised at least once", short: "revised", note: "the task went back to its author at least once", color: "var(--serious)", test: (t) => t.revised },
  { key: "reverted", label: "Kept an earlier version", short: "reverted", note: "a later revision broke something, so the last good version was accepted", color: "var(--muted)", test: (t) => t.reverted },
];
export const FLAG = Object.fromEntries(FLAGS.map((f) => [f.key, f]));

export function flagCounts(tasks) {
  const counts = Object.fromEntries(FLAGS.map((f) => [f.key, 0]));
  counts.both = 0;
  for (const t of tasks) {
    for (const f of FLAGS) if (f.test(t)) counts[f.key] += 1;
    if (t.fa > 0 && t.fr > 0) counts.both += 1;
  }
  return counts;
}

export function bandParts(tasks) {
  return Object.entries(BANDS).map(([key, band]) => ({ key, label: band.label, color: band.color, value: tasks.filter((t) => t.band === key).length }));
}

export function tasksHref(params) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""));
  return `#/tasks${query.size ? `?${query}` : ""}`;
}

export function taskHref(task) {
  return `#/task/${task.id}`;
}

// Filters, each one URL parameter. `values` is what a select offers; `test` keeps a task.
export const FILTERS = {
  subject: { label: "Subject", test: (t, v) => t.subject === v },
  cap: { label: "Skill", test: (t, v) => t.capability.id === v },
  grader: { label: "Grading", test: (t, v) => t.grader === v, values: Object.entries(GRADERS).map(([k, g]) => [k, g.label]) },
  env: { label: "Environment", test: (t, v) => t.env === v, values: Object.entries(ENVIRONMENTS).map(([k, e]) => [k, e.label]) },
  band: { label: "Difficulty", test: (t, v) => t.band === v, values: Object.entries(BANDS).map(([k, b]) => [k, b.label]) },
  passes: { label: "Trials solved", test: (t, v) => t.passes === Number(v), values: Array.from({ length: 9 }, (_, k) => [String(k), `${k} of 8`]) },
  rounds: { label: "Revisions", test: (t, v) => t.rounds === Number(v), values: Array.from({ length: 7 }, (_, k) => [String(k), plural(k, "revision")]) },
  caught: { label: "Validation caught", test: (t, v) => FLAG[v]?.test(t), values: FLAGS.map((f) => [f.key, f.label]) },
  part: { label: "Flaw in", test: (t, v) => t.parts.includes(v), values: Object.entries(PARTS) },
  effect: { label: "Flaw", test: (t, v) => t.effects.includes(v) },
};

export function applyFilters(tasks, params) {
  const active = [...params.entries()].filter(([key, value]) => FILTERS[key] && value);
  const q = (params.get("q") || "").trim().toLowerCase();
  return tasks.filter((t) => active.every(([key, value]) => FILTERS[key].test(t, value)) && (!q || q.split(/\s+/).every((word) => t.search.includes(word))));
}

export function filterLabel(key, value, index) {
  if (key === "subject") return index.subjectById.get(value)?.name || value;
  if (key === "cap") return index.capById.get(value)?.name || value;
  if (key === "effect") {
    const [part, effect] = value.split(":");
    return `${PARTS[part] || part}: ${(EFFECTS[effect]?.label || effect).toLowerCase()}`;
  }
  const entry = (FILTERS[key].values || []).find(([k]) => k === value);
  return entry ? entry[1] : value;
}

export function bandChip(task) {
  const band = BANDS[task.band];
  return tip(chip(band.label, band.tone), tipBody(band.label, band.note));
}

export function caughtMarks(task) {
  const marks = FLAGS.filter((f) => f.trigger && f.test(task));
  if (!marks.length) return h("span", { class: "muted small" }, "—");
  return h("span", { class: "marks" }, marks.map((f) => tip(h("span", { class: "mark", style: { background: f.color } }), tipBody(f.label, f.note))));
}

export function solvedCell(task, meta) {
  return h("div", { class: "solved" }, trialDots(task.passes, meta.trials), h("span", { class: "small ink2" }, `${task.passes}/${meta.trials}`));
}

export function prefetchOnHover(el, meta, task) {
  let armed = false;
  const warm = () => {
    if (armed) return;
    armed = true;
    prefetchTask(meta, task);
  };
  el.addEventListener("mouseenter", warm);
  el.addEventListener("focusin", warm);
  el.addEventListener("touchstart", warm, { passive: true });
  return el;
}

// A compact card for one task, used on skill pages and in the task list's grid view.
export function taskCard(task, meta) {
  const el = h("a", { class: "task-card", href: taskHref(task) },
    h("div", { class: "tc-title" }, task.title),
    h("div", { class: "tc-meta" }, solvedCell(task, meta), bandChip(task), chip(GRADERS[task.grader].label, "outline"), task.env !== "reasoning" ? chip(ENVIRONMENTS[task.env].label, "outline") : null),
    h("div", { class: "tc-foot muted small" }, task.rounds ? plural(task.rounds, "revision") : "accepted as first written", " · ", caughtMarks(task)));
  return prefetchOnHover(el, meta, task);
}
