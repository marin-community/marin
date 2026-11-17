// How the tasks are made and how these were chosen, in plain language.

import { h, compact, card, chip, EFFECTS, PARTS, BANDS, ADVERSARIES } from "../util.js";
import { setCrumbs } from "../main.js";
import { section } from "./overview.js";

export function renderMethod(page, index) {
  setCrumbs([["Overview", "#/"], ["How tasks are made", null]]);
  const { meta, tasks } = index;
  const steps = [
    ["A skill", "Each task starts from one skill in a catalog of 1,999 skills across 45 subjects, from exact linear algebra to veterinary pharmacology. The catalog says what the skill includes and excludes, with sample tasks."],
    ["A proposal", "A model writes a plan for one task that exercises the skill: what the solver is asked, which files it gets, and how an answer will be graded."],
    ["The author", "An agent with a sandbox builds the task from the proposal: the instructions, any files, a grader (a program, or a reference answer checked exactly, numerically, symbolically or as a choice), and controls, which are answers with a known grade (correct ones must pass, wrong and malformed ones must fail)."],
    ["Validation", "Every version of the task is validated the same way. The controls are graded. GLM-5.3 attempts the task three times. Two adversaries attack the grader, and a reviewer reads every attempt."],
    ["Review", "A fixed rule, not a model, reads the evidence. Any failed control, any grader flaw an adversary reproduced, or any task flaw the reviewer established sends the task back to its author with the evidence, up to four times. A task that is too easy or too hard gets one round to adjust."],
    ["Confirmation", "Once the rule would accept a task, GLM-5.3 attempts it five more times and a final review reads all eight attempts. Its difficulty tag comes from those eight trials."],
  ];
  page.append(
    h("div", { class: "page-head" }, h("h1", {}, "How tasks are made"),
      h("p", { class: "lede" }, "Forge is a pipeline that turns a catalog of skills into verified tasks: tasks whose graders have been attacked, whose transcripts have been read, and whose difficulty has been measured.")),
    h("ol", { class: "steps" }, steps.map(([title, text], i) => h("li", {}, h("div", { class: "step-n" }, String(i + 1)), h("div", {}, h("h3", {}, title), h("p", {}, text))))),
    section("The two adversaries", "Both work in a sandbox with the task's files and the grader. A claim counts only if the grader reproduces it on a fresh machine.",
      h("div", { class: "grid two" }, Object.values(ADVERSARIES).map((a) => card(a.label, h("p", {}, `It ${a.note}. If it succeeds, the grader is wrong in that direction, and the task goes back to its author with the answer that fooled it.`))))),
    section("What the reviewer looks for", "The reviewer reads every attempt and the task's files and reports flaws in the task itself, each with a quote from a transcript as evidence. Only established flaws force a revision; probable and possible ones go to the author, who may decline them.",
      card(null, h("dl", { class: "kv" }, Object.values(EFFECTS).flatMap((e) => [h("dt", {}, e.label), h("dd", {}, e.note)]))),
      card("Where a flaw can be", h("div", { class: "chips" }, Object.values(PARTS).map((p) => chip(p, "outline"))))),
    section("Difficulty", null,
      card(null, h("dl", { class: "kv" }, Object.values(BANDS).flatMap((b) => [h("dt", {}, b.label), h("dd", {}, `${b.note}. `, b === BANDS.in_band ? "The target." : "Kept and tagged.")])))),
    section("How these tasks were chosen", null,
      h("div", { class: "prose" },
        h("p", {}, `The pipeline accepted ${compact(meta.pool.accepted)} tasks for the main split. ${compact(meta.pool.not_clean || 0)} were left out because their acceptance carried a minor reviewer note or lacked some evidence (for example, a review that did not finish).`),
        h("p", {}, `From the remaining clean tasks, ${compact(tasks.length)} were chosen for coverage first. Within each skill, tasks are ranked by a score; every skill's top task is taken, then every skill's second, and so on, until the target is reached.`),
        h("p", {}, "The score favours what is most useful for training: tasks solved in the target band, a solve rate near one half (where a trial says the most about the model), an accepted version that was not a fallback to an earlier one, and more controls."),
        h("details", { class: "fold" }, h("summary", {}, "The exact rule"), h("div", { class: "fold-body mono small" }, meta.rule)))),
    section("What is not shown", null,
      h("p", { class: "prose" }, "Tasks graded by a learned decision model are a separate split and are not included. Solver transcripts are from GLM-5.3; the author and adversaries' own conversations are not published here.")),
  );
}
