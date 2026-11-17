// The landing page: what the collection is, how a task gets accepted, how hard the tasks are, what validation
// caught, and how the tasks cover the catalog. Every chart filters the task list when clicked.

import { h, fill, tile, compact, percent, plural, chip, stackBar, legend, tip, tipBody, card, BANDS, GRADERS, ENVIRONMENTS, PARTS, EFFECTS } from "../util.js";
import { columns, hbars, heatmap, pipeline } from "../charts.js";
import { setCrumbs } from "../main.js";
import { tasksHref, flagCounts, FLAGS, bandParts } from "./common.js";

export function renderOverview(page, index) {
  setCrumbs([]);
  const { meta, tasks, caps, subjects } = index;
  const counts = flagCounts(tasks);
  const catalogTotal = subjects.reduce((sum, s) => sum + s.capabilities, 0);
  const go = (params) => { location.hash = tasksHref(params); };

  page.append(
    h("div", { class: "hero" },
      h("h1", {}, "Forge tasks"),
      h("p", { class: "lede" },
        `${compact(tasks.length)} tasks for training and evaluating language models. Each one was written by an AI author from a skill in a `,
        `${compact(catalogTotal)}-skill catalog spanning ${subjects.length} subjects, and accepted only after its grader passed known-answer checks, `,
        "two adversaries failed to fool it, a reviewer read every solver transcript for flaws, and GLM-5.3 attempted it eight times."),
      h("div", { class: "hero-meta muted small" }, `Snapshot ${meta.snapshot.replace(/T(\d\d)(\d\d)Z/, " $1:$2 UTC")} · catalog ${meta.catalog_version}`)),
    h("div", { class: "grid tiles" },
      tile("Tasks", compact(tasks.length), `${plural(caps.length, "skill")} covered`),
      tile("Subjects", subjects.length, `${subjects.filter((s) => s.covered).length} with tasks`),
      tile("Revised before acceptance", percent(counts.revised / tasks.length), `${compact(counts.revised)} tasks sent back to their author`),
      tile("Grader flaws caught by adversaries", compact(counts.fa + counts.fr - counts.both), "tasks whose grader was fooled, then fixed"),
      tile("Task flaws caught by review", compact(counts.review), "tasks fixed after a transcript review"),
      tile("Solved in band", percent(tasks.filter((t) => t.band === "in_band").length / tasks.length), "between 2 and 6 of 8 trials"),
    ),
  );

  page.append(section("How a task is accepted", "Every task goes through the same loop. Click a check to list the tasks it caught.",
    pipeline(
      { capabilities: caps.length, tasks: tasks.length, revised: counts.revised, rounds: tasks.reduce((a, t) => a + t.rounds, 0), fa: counts.fa, fr: counts.fr, ctl: counts.ctl, review: counts.review, band: tasks.reduce((a, t) => a + t.band_rounds, 0) },
      { onPick: (key) => go({ caught: key }) },
    )));

  // Difficulty: passes out of eight, coloured by band.
  const byPasses = Array.from({ length: meta.trials + 1 }, (_, k) => tasks.filter((t) => t.passes === k).length);
  const bandOf = (k) => (k / meta.trials < meta.band[0] ? "too_hard" : k / meta.trials > meta.band[1] ? "too_easy" : "in_band");
  const difficulty = columns(byPasses.map((value, k) => ({
    label: `${k}/8`,
    value,
    color: BANDS[bandOf(k)].color,
    tip: tipBody(`Solved in ${k} of 8 trials`, `${compact(value)} tasks (${percent(value / tasks.length)})`, BANDS[bandOf(k)].label),
    onClick: () => go({ passes: String(k) }),
  })), { xLabel: "trials solved by GLM-5.3, of 8", shade: [Math.ceil(meta.band[0] * meta.trials), Math.floor(meta.band[1] * meta.trials), "target band"] });
  const bandBar = h("div", {},
    stackBar(bandParts(tasks).map((p) => ({ ...p, tip: tipBody(p.label, `${compact(p.value)} tasks (${percent(p.value / tasks.length)})`) })), { style: { height: "14px" } }),
    legend(Object.entries(BANDS).map(([, b]) => [b.label, b.color])));
  page.append(section("How hard the tasks are", "Each task's eight confirmation trials. Tasks outside the band are kept and tagged: they still teach, or test, something.",
    h("div", { class: "grid two" },
      card("Solve rate over eight trials", difficulty),
      card("Difficulty tags", h("div", {}, bandBar,
        h("p", { class: "muted small", style: { marginTop: "12px" } },
          "A task's author gets one round to make a task that is too easy harder (or a too-hard one easier). Validation reruns, and the task is accepted at whatever difficulty it then lands on."),
        hbars(Object.entries(GRADERS).flatMap(([gk, g]) => Object.entries(ENVIRONMENTS).map(([ek, e]) => {
          const value = tasks.filter((t) => t.grader === gk && t.env === ek).length;
          return { label: `${g.label} · ${e.label}`, value, color: "var(--accent)", tip: tipBody(`${g.label} · ${e.label}`, g.note, e.note, `${compact(value)} tasks`), onClick: () => go({ grader: gk, env: ek }) };
        })).filter((r) => r.value), { width: 560, labelWidth: 190 }))),
    )));

  // What validation caught.
  const effectKeys = Object.keys(EFFECTS);
  const partKeys = Object.keys(PARTS);
  const cell = new Map();
  for (const t of tasks) for (const pe of t.effects) cell.set(pe, (cell.get(pe) || 0) + 1);
  const usedParts = partKeys.filter((p) => effectKeys.some((e) => cell.get(`${p}:${e}`)));
  const usedEffects = effectKeys.filter((e) => partKeys.some((p) => cell.get(`${p}:${e}`)));
  const matrix = heatmap(
    usedParts.map((key) => ({ key, label: PARTS[key] })),
    usedEffects.map((key) => ({ key, label: EFFECTS[key].label })),
    (p, e) => cell.get(`${p}:${e}`) || 0,
    {
      onClick: (p, e) => go({ effect: `${p}:${e}` }),
      tipFor: (r, c, v) => tipBody(`${r.label}: ${c.label}`, EFFECTS[c.key].note, `${compact(v)} tasks had this flaw found and fixed before acceptance`),
      colWidth: 104,
    });
  const triggers = hbars(FLAGS.filter((f) => f.trigger).map((f) => ({
    label: f.label, value: counts[f.key], color: f.color, onClick: () => go({ caught: f.key }),
    tip: tipBody(f.label, f.note, `${compact(counts[f.key])} tasks (${percent(counts[f.key] / tasks.length)})`),
  })), { width: 560, labelWidth: 190 });
  const rounds = Array.from({ length: Math.max(...tasks.map((t) => t.rounds)) + 1 }, (_, k) => tasks.filter((t) => t.rounds === k).length);
  const roundChart = columns(rounds.map((value, k) => ({
    label: String(k), value, color: k ? "var(--serious)" : "var(--good)",
    tip: tipBody(k ? `${plural(k, "revision")}` : "Accepted as first written", `${compact(value)} tasks (${percent(value / tasks.length)})`),
    onClick: () => go({ rounds: String(k) }),
  })), { xLabel: "times sent back to the author", height: 190 });
  page.append(section("What validation caught", "Flaws found in the tasks that were finally accepted, each fixed by a revision. Click a bar or cell to list those tasks.",
    h("div", { class: "grid two" },
      card("Why tasks were sent back", triggers),
      card("Revision rounds before acceptance", roundChart)),
    card("Flaws the transcript reviewer found, by where they were", usedParts.length ? matrix : h("div", { class: "empty inline" }, "No reviewer findings in this selection."), { cls: "heatmap-card" }),
  ));

  page.append(section("Coverage", `${compact(caps.length)} of ${compact(catalogTotal)} catalog skills have at least one task. Open a subject to browse its skills.`, subjectGrid(subjects, tasks.length)));
  page.append(section("How these tasks were chosen", null,
    h("p", { class: "prose" },
      `${compact(meta.pool.accepted)} tasks were accepted by the pipeline. ${compact(meta.pool.not_clean || 0)} were left out because a reviewer attached a minor note or some validation evidence was missing. `,
      `From the ${compact((meta.pool.selected || 0) + (meta.pool.below_coverage_tier || 0))} clean ones, ${compact(tasks.length)} were picked for coverage first: every skill's best task, then every skill's second best, and so on. `,
      h("a", { href: "#/method" }, "How tasks are made and chosen →"))));
}

export function subjectGrid(subjects, totalTasks) {
  const max = Math.max(...subjects.map((s) => s.tasks.length));
  return h("div", { class: "subject-grid" }, subjects.map((subject) => {
    const el = h("a", { class: "subject-tile", href: `#/subject/${subject.id}` },
      h("div", { class: "st-head" }, h("span", { class: "st-id mono" }, subject.id), h("span", { class: "st-count" }, compact(subject.tasks.length))),
      h("div", { class: "st-name" }, subject.name),
      h("div", { class: "st-bar" }, h("i", { style: { width: `${(subject.tasks.length / max) * 100}%` } })),
      h("div", { class: "muted small" }, `${subject.covered} of ${subject.capabilities} skills`));
    tip(el, tipBody(subject.name, `${compact(subject.tasks.length)} tasks (${percent(subject.tasks.length / totalTasks)})`, `${subject.covered} of ${subject.capabilities} catalog skills covered`));
    return el;
  }));
}

export function section(title, sub, ...body) {
  return h("section", { class: "section" }, h("div", { class: "section-head" }, h("h2", {}, title), sub ? h("div", { class: "sub" }, sub) : null), body);
}
