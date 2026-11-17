// One skill: its catalog entry (what it covers, what it excludes, sample tasks) and every task built from it.

import { h, fill, chip, plural, card } from "../util.js";
import { setCrumbs } from "../main.js";
import { subjectDetail } from "../data.js";
import { section } from "./overview.js";
import { taskCard, tasksHref } from "./common.js";

export async function renderCapability(page, index, capId, isLive) {
  const cap = index.capById.get(capId);
  if (!cap) {
    fill(page, h("div", { class: "empty" }, `No skill ${capId} in this collection.`));
    return;
  }
  const subject = index.subjectById.get(cap.subject);
  setCrumbs([["Overview", "#/"], [subject.name, `#/subject/${subject.id}`], [cap.name, null]]);
  const about = h("div", {}, h("div", { class: "loading inline" }, "Loading the catalog entry…"));
  page.append(
    h("div", { class: "page-head" },
      h("div", { class: "eyebrow" }, h("a", { href: `#/subject/${subject.id}` }, subject.name), " · ", cap.group_name),
      h("h1", {}, cap.name),
      h("div", { class: "chips" }, chip(cap.id, "outline mono"), chip(plural(cap.tasks.length, "task"), "info"))),
    about,
    section("Tasks built from this skill", h("a", { href: tasksHref({ cap: cap.id }) }, "Open in the task list →"),
      h("div", { class: "task-cards" }, cap.tasks.map((task) => taskCard(task, index.meta)))),
  );
  try {
    const entry = (await subjectDetail(cap.subject)).find((d) => d.id === cap.id);
    if (!isLive() || !entry) return;
    fill(about, h("div", { class: "grid two" },
      card("What the skill is", h("div", {},
        entry.outcome ? h("p", { class: "prose" }, entry.outcome) : null,
        entry.includes.length ? h("div", {}, h("div", { class: "kv-title" }, "Includes"), h("ul", { class: "plain" }, entry.includes.map((i) => h("li", {}, i)))) : null,
        entry.excludes.length ? h("div", {}, h("div", { class: "kv-title" }, "Excludes"), h("ul", { class: "plain muted" }, entry.excludes.map((i) => h("li", {}, i)))) : null)),
      card("Sample tasks from the catalog", entry.sample_tasks.length
        ? h("div", {}, entry.sample_tasks.map((sample) => h("div", { class: "sample" }, chip(sample.kind, "outline"), h("p", {}, sample.instruction))),
          h("p", { class: "muted small" }, "These illustrate the skill; the tasks below were written fresh from it."))
        : h("div", { class: "empty inline" }, "The catalog lists no samples for this skill."))));
  } catch (error) {
    if (isLive()) fill(about, h("div", { class: "empty inline" }, `The catalog entry did not load: ${error.message}`));
  }
}
