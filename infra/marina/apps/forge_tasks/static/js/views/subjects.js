// Subjects: all 45 at a glance, and one subject's skill groups with every task as a row of trial dots.

import { h, fill, tile, compact, percent, plural, chip, stackBar, legend, tip, tipBody, BANDS } from "../util.js";
import { trialDots } from "../charts.js";
import { setCrumbs } from "../main.js";
import { subjectDetail } from "../data.js";
import { subjectGrid, section } from "./overview.js";
import { tasksHref, taskHref, bandParts, flagCounts, prefetchOnHover } from "./common.js";

export function renderSubjects(page, index) {
  setCrumbs([["Overview", "#/"], ["Subjects", null]]);
  const { subjects, tasks } = index;
  page.append(
    h("div", { class: "page-head" }, h("h1", {}, "Subjects"),
      h("p", { class: "lede" }, `The catalog groups its skills into ${subjects.length} subjects. Each tile shows how many tasks a subject has and how many of its skills they cover.`)),
    subjectGrid(subjects, tasks.length),
  );
}

export async function renderSubject(page, index, subjectId, isLive) {
  const subject = index.subjectById.get(subjectId);
  if (!subject) {
    fill(page, h("div", { class: "empty" }, `No subject ${subjectId}.`));
    return;
  }
  setCrumbs([["Overview", "#/"], ["Subjects", "#/subjects"], [subject.name, null]]);
  const { meta } = index;
  const counts = flagCounts(subject.tasks);
  page.append(
    h("div", { class: "page-head" },
      h("div", { class: "eyebrow mono" }, subject.id),
      h("h1", {}, subject.name),
      h("p", { class: "lede" }, `${compact(subject.tasks.length)} tasks across ${plural(subject.covered, "skill")} of the ${subject.capabilities} this subject has in the catalog.`)),
    h("div", { class: "grid tiles" },
      tile("Tasks", compact(subject.tasks.length), h("a", { href: tasksHref({ subject: subject.id }) }, "list them →")),
      tile("Skills covered", `${subject.covered} / ${subject.capabilities}`, percent(subject.covered / subject.capabilities)),
      tile("In band", percent(subject.tasks.filter((t) => t.band === "in_band").length / subject.tasks.length), "solved in 2 to 6 of 8 trials"),
      tile("Revised before acceptance", percent(counts.revised / subject.tasks.length), `${compact(counts.revised)} tasks`),
    ),
    h("div", { class: "card", style: { marginTop: "14px" } },
      stackBar(bandParts(subject.tasks).map((p) => ({ ...p, tip: tipBody(p.label, `${compact(p.value)} tasks`) })), { style: { height: "12px" } }),
      legend(Object.values(BANDS).map((b) => [b.label, b.color]))),
  );
  const groupsEl = h("div", {});
  page.append(section("Skills", "Grouped as in the catalog. Each dot row is one task: a filled dot is a solved trial.", groupsEl));
  const groups = new Map();
  for (const cap of subject.caps) {
    if (!groups.has(cap.group)) groups.set(cap.group, { name: cap.group_name, caps: [] });
    groups.get(cap.group).caps.push(cap);
  }
  const draw = (detail) => {
    const outcomes = new Map((detail || []).map((d) => [d.id, d]));
    fill(groupsEl, [...groups.values()].map((group) => h("div", { class: "group" },
      h("h3", { class: "group-name" }, group.name, h("span", { class: "muted small" }, ` · ${plural(group.caps.length, "skill")}`)),
      h("div", { class: "cap-list" }, group.caps.map((cap) => h("div", { class: "cap-row" },
        h("div", { class: "cap-text" },
          h("a", { href: `#/cap/${cap.id}`, class: "cap-name" }, cap.name),
          outcomes.get(cap.id)?.outcome ? h("div", { class: "muted small" }, outcomes.get(cap.id).outcome) : null),
        h("div", { class: "cap-tasks" }, cap.tasks.map((task) => prefetchOnHover(tip(h("a", { href: taskHref(task), class: "dotlink" }, trialDots(task.passes, meta.trials, { size: 7, gap: 2 })),
          tipBody(task.title, `${task.passes} of 8 trials solved · ${BANDS[task.band].label}`, task.rounds ? plural(task.rounds, "revision") : "accepted as first written")), meta, task)))))))));
  };
  draw(null);
  try {
    const detail = await subjectDetail(subject.id);
    if (isLive()) draw(detail);
  } catch {
    /* outcomes are optional decoration */
  }
}
