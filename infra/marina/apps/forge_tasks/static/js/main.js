// Hash router, theme and breadcrumbs. Every route renders from the in-memory index; task pages add one lazy fetch.

import { h, fill, hideTip } from "./util.js";
import { loadIndex } from "./data.js";
import { renderOverview } from "./views/overview.js";
import { renderSubjects, renderSubject } from "./views/subjects.js";
import { renderCapability } from "./views/capability.js";
import { renderBrowse } from "./views/browse.js";
import { renderTask } from "./views/task.js";
import { renderMethod } from "./views/method.js";

const app = document.getElementById("app");

export function parseRoute() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  return { parts, params: new URLSearchParams(query) };
}

export function setCrumbs(entries) {
  fill(document.getElementById("crumbs"), entries.map(([label, href], i) => [
    i ? h("span", { class: "sep" }, "›") : null,
    href ? h("a", { href }, label) : h("span", { class: "here" }, label),
  ]));
}

let renderToken = 0;
let lastPath = null;

async function render() {
  hideTip();
  const token = ++renderToken;
  const { parts, params } = parseRoute();
  const path = parts.join("/");
  for (const link of document.querySelectorAll("[data-nav]")) {
    const key = parts[0] || "overview";
    link.classList.toggle("on", link.dataset.nav === key || (key === "task" && link.dataset.nav === "tasks") || ((key === "subject" || key === "cap") && link.dataset.nav === "subjects"));
  }
  let index;
  try {
    index = await loadIndex();
  } catch (error) {
    fill(app, h("div", { class: "empty" }, `Could not load the task index: ${error.message}`));
    return;
  }
  if (token !== renderToken) return;
  // A query-only change on the browse page re-filters in place instead of rebuilding the page.
  if (parts[0] === "tasks" && lastPath === path && app.dataset.view === "tasks" && app.refilter) {
    app.refilter(params);
    return;
  }
  const changed = lastPath !== path;
  lastPath = path;
  app.refilter = null;
  app.dataset.view = parts[0] || "overview";
  const page = h("div", { class: "page" });
  fill(app, page);
  if (changed) window.scrollTo(0, 0);
  const isLive = () => token === renderToken;
  switch (parts[0]) {
    case undefined:
      return renderOverview(page, index);
    case "subjects":
      return renderSubjects(page, index);
    case "subject":
      return renderSubject(page, index, parts[1], isLive);
    case "cap":
      return renderCapability(page, index, parts[1], isLive);
    case "tasks":
      return renderBrowse(page, index, params, app);
    case "task":
      return renderTask(page, index, `${parts[1]}/${parts[2]}`, parts[3], isLive);
    case "method":
      return renderMethod(page, index);
    default:
      fill(page, h("div", { class: "empty" }, "No such page."), h("a", { href: "#/" }, "Back to the overview"));
  }
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === "dark" || (!theme && matchMedia("(prefers-color-scheme: dark)").matches);
  document.getElementById("hljs-light").disabled = dark;
  document.getElementById("hljs-dark").disabled = !dark;
}

document.getElementById("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = dark ? "light" : "dark";
  try { localStorage.setItem("forge-tasks-theme", next); } catch { /* storage unavailable */ }
  setTheme(next);
});

let saved = null;
try { saved = localStorage.getItem("forge-tasks-theme"); } catch { /* storage unavailable */ }
setTheme(saved);
window.addEventListener("hashchange", render);
render();
