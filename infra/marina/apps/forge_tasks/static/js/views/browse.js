// The task list: every filter is a URL parameter, filtering and sorting run in memory over the shipped index, and the
// page updates in place (no refetch, no rebuild) as filters change.

import { h, fill, chip, compact, percent, tipBody, stackBar, legend, BANDS, GRADERS, ENVIRONMENTS } from "../util.js";
import { columns } from "../charts.js";
import { setCrumbs } from "../main.js";
import { FILTERS, applyFilters, filterLabel, tasksHref, taskHref, bandChip, caughtMarks, solvedCell, prefetchOnHover, bandParts, flagCounts, FLAGS } from "./common.js";

const PAGE = 100;
const SORTS = {
  title: { label: "Task", key: (t) => t.title.toLowerCase() },
  subject: { label: "Subject", key: (t) => `${t.subject} ${t.capability.name}` },
  passes: { label: "Solved", key: (t) => t.passes },
  rounds: { label: "Revisions", key: (t) => t.rounds },
  caught: { label: "Caught", key: (t) => (t.fa > 0) + (t.fr > 0) + (t.ctl > 0) + (t.defects > 0) },
};

export function renderBrowse(page, index, initialParams, app) {
  setCrumbs([["Overview", "#/"], ["All tasks", null]]);
  const { meta, tasks } = index;
  let params = new URLSearchParams(initialParams);
  let shown = PAGE;

  const search = h("input", { type: "search", class: "search", placeholder: "Search titles, skills and tags", value: params.get("q") || "" });
  let typing = null;
  search.addEventListener("input", () => {
    clearTimeout(typing);
    typing = setTimeout(() => update({ q: search.value || null }, true), 120);
  });
  const selects = Object.entries(FILTERS).filter(([key]) => !["cap", "effect"].includes(key)).map(([key, filter]) => {
    const options = key === "subject"
      ? index.subjects.filter((s) => s.tasks.length).map((s) => [s.id, `${s.id} · ${s.name}`])
      : filter.values;
    const select = h("select", { dataset: { key } }, h("option", { value: "" }, `${filter.label}: any`), options.map(([value, label]) => h("option", { value }, label)));
    select.addEventListener("change", () => update({ [key]: select.value || null }));
    return select;
  });
  const active = h("div", { class: "chips active-filters" });
  const summary = h("div", { class: "browse-summary" });
  const charts = h("div", { class: "grid two browse-charts" });
  const table = h("table", { class: "items tasks" });
  const more = h("div", { class: "more" });
  page.append(
    h("div", { class: "page-head" }, h("h1", {}, "All tasks"), h("p", { class: "lede" }, "Filter by subject, grading, difficulty or what validation caught. Every view is a link you can share.")),
    h("div", { class: "filters" }, search, selects),
    active, summary, charts, h("div", { class: "table-wrap" }, table), more,
  );

  function update(changes, replace = false) {
    for (const [key, value] of Object.entries(changes)) {
      if (value === null || value === undefined || value === "") params.delete(key);
      else params.set(key, value);
    }
    const href = tasksHref(Object.fromEntries(params));
    if (replace) {
      history.replaceState(null, "", href);
      refilter(params);
    } else location.hash = href;
  }

  function refilter(next) {
    params = new URLSearchParams(next);
    shown = PAGE;
    if (document.activeElement !== search) search.value = params.get("q") || "";
    for (const select of selects) select.value = params.get(select.dataset.key) || "";
    const filtered = applyFilters(tasks, params);
    const sortKey = SORTS[params.get("sort")] ? params.get("sort") : null;
    const direction = params.get("dir") === "desc" ? -1 : 1;
    const sorted = sortKey
      ? [...filtered].sort((a, b) => {
          const x = SORTS[sortKey].key(a);
          const y = SORTS[sortKey].key(b);
          return (x < y ? -1 : x > y ? 1 : 0) * direction;
        })
      : filtered;
    fill(active, [...params.entries()].filter(([key]) => FILTERS[key]).map(([key, value]) =>
      h("button", { class: "chip info removable", title: "Remove this filter", onclick: () => update({ [key]: null }) }, `${FILTERS[key].label}: ${filterLabel(key, value, index)} ✕`)),
      params.size && [...params.keys()].some((k) => FILTERS[k] || k === "q") ? h("button", { class: "ghost small", onclick: () => { location.hash = "#/tasks"; } }, "Clear all") : null);
    const caps = new Set(filtered.map((t) => t.capability.id)).size;
    fill(summary, h("strong", {}, compact(filtered.length)), ` of ${compact(tasks.length)} tasks · ${compact(caps)} skills`);
    renderCharts(filtered);
    renderTable(sorted, sortKey, direction);
  }

  function renderCharts(filtered) {
    if (!filtered.length) {
      fill(charts);
      return;
    }
    const byPasses = Array.from({ length: meta.trials + 1 }, (_, k) => filtered.filter((t) => t.passes === k).length);
    const counts = flagCounts(filtered);
    const current = params.get("passes");
    fill(charts,
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h3", {}, "Trials solved, of 8")),
        columns(byPasses.map((value, k) => ({
          label: `${k}`, value, active: current === String(k), dim: current !== null && current !== String(k),
          color: BANDS[k / meta.trials < meta.band[0] ? "too_hard" : k / meta.trials > meta.band[1] ? "too_easy" : "in_band"].color,
          tip: tipBody(`${k} of 8 trials solved`, `${compact(value)} tasks`), onClick: () => update({ passes: current === String(k) ? null : String(k) }),
        })), { height: 150, shade: [2, 6] })),
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h3", {}, "In this selection")),
        stackBar(bandParts(filtered).map((p) => ({ ...p, tip: tipBody(p.label, `${compact(p.value)} tasks (${percent(p.value / filtered.length)})`) })), { style: { height: "12px" } }),
        legend(Object.values(BANDS).map((b) => [b.label, b.color])),
        h("div", { class: "flag-list" }, FLAGS.filter((f) => f.trigger || f.key === "revised").map((f) => h("button", {
          class: `flag-row${params.get("caught") === f.key ? " on" : ""}`,
          onclick: () => update({ caught: params.get("caught") === f.key ? null : f.key }),
        }, h("span", { class: "mark", style: { background: f.color } }), h("span", {}, f.label), h("span", { class: "ink2" }, `${compact(counts[f.key])} · ${percent(counts[f.key] / filtered.length)}`))))),
    );
  }

  function renderTable(sorted, sortKey, direction) {
    const header = (key, label) => {
      const on = sortKey === key;
      return h("th", { class: `sortable${on ? " on" : ""}`, onclick: () => update({ sort: key, dir: on && direction === 1 ? "desc" : null }) }, label, on ? (direction === 1 ? " ↑" : " ↓") : "");
    };
    const body = h("tbody", {});
    const draw = () => {
      body.replaceChildren(...sorted.slice(0, shown).map((task) => row(task)));
      fill(more, sorted.length > shown
        ? h("button", { class: "ghost", onclick: () => { shown += PAGE * 4; draw(); } }, `Show more (${compact(sorted.length - shown)} left)`)
        : null);
    };
    fill(table,
      h("thead", {}, h("tr", {}, header("title", "Task"), header("subject", "Skill"), h("th", {}, "Kind"), header("passes", "Solved"), h("th", {}, "Difficulty"), header("rounds", "Revisions"), header("caught", "Caught"))),
      body);
    if (!sorted.length) fill(more, h("div", { class: "empty" }, "No task matches these filters."));
    else draw();
  }

  function row(task) {
    const tr = h("tr", { onclick: (event) => { if (!event.target.closest("a")) location.hash = taskHref(task); } },
      h("td", {}, h("a", { href: taskHref(task), class: "row-title" }, task.title), h("div", { class: "mono muted small" }, task.id)),
      h("td", {}, h("a", { href: `#/cap/${task.capability.id}`, class: "ink2" }, task.capability.name), h("div", { class: "muted small" }, task.subject)),
      h("td", {}, h("div", { class: "chips" }, chip(GRADERS[task.grader].label, "outline"), task.env !== "reasoning" ? chip(ENVIRONMENTS[task.env].label, "outline") : null)),
      h("td", {}, solvedCell(task, meta)),
      h("td", {}, bandChip(task)),
      h("td", { class: "num" }, String(task.rounds)),
      h("td", {}, caughtMarks(task)));
    return prefetchOnHover(tr, meta, task);
  }

  app.refilter = refilter;
  refilter(params);
}
