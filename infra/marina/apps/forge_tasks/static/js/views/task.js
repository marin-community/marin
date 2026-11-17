// One task. The header renders at once from the index; the record (prompt, grader, controls, validation history) and
// the trimmed transcripts are fetched in parallel; a trial's full transcript loads when it is opened.

import {
  h, fill, chip, tile, plural, percent, num, compact, markdown, code, fold, tip, tipBody, card, copyButton,
  BANDS, GRADERS, ENVIRONMENTS, PARTS, EFFECTS, ADVERSARIES, CONTROL_KINDS, STOP_REASONS,
} from "../util.js";
import { trialDots } from "../charts.js";
import { transcript } from "../transcript.js";
import { setCrumbs } from "../main.js";
import { taskDetail, capabilityTrials, fullTranscript, prefetchTask } from "../data.js";
import { bandChip, caughtMarks, taskHref, FLAGS } from "./common.js";

const TABS = [
  ["task", "The task"],
  ["grader", "Grader and files"],
  ["controls", "Controls"],
  ["history", "Validation history"],
  ["trials", "Solver trials"],
  ["proposal", "Proposal"],
];

export async function renderTask(page, index, id, tab, isLive) {
  const { meta } = index;
  const task = index.byId.get(id);
  if (!task) {
    fill(page, h("div", { class: "empty" }, `No task ${id} in this collection.`), h("a", { href: "#/tasks" }, "All tasks"));
    return;
  }
  const subject = index.subjectById.get(task.subject);
  setCrumbs([["Overview", "#/"], [subject.name, `#/subject/${subject.id}`], [task.capability.name, `#/cap/${task.capability.id}`], [`Task ${task.slot}`, null]]);
  const detailPromise = taskDetail(meta, id);
  const trialsPromise = capabilityTrials(meta, task.capability.id);
  const siblings = task.capability.tasks;
  const position = siblings.indexOf(task);
  for (const neighbour of [siblings[position + 1], siblings[position - 1]]) if (neighbour) prefetchTask(meta, neighbour);

  page.append(
    h("div", { class: "page-head task-head" },
      h("div", { class: "eyebrow" }, h("a", { href: `#/subject/${subject.id}` }, subject.name), " · ", h("a", { href: `#/cap/${task.capability.id}` }, task.capability.name)),
      h("h1", {}, task.title),
      h("div", { class: "chips" },
        bandChip(task),
        tip(chip(GRADERS[task.grader].label, "outline"), tipBody(GRADERS[task.grader].label, GRADERS[task.grader].note)),
        tip(chip(ENVIRONMENTS[task.env].label, "outline"), tipBody(ENVIRONMENTS[task.env].label, ENVIRONMENTS[task.env].note)),
        chip(task.id, "outline mono"),
        task.tags.slice(0, 6).map((t) => chip(t, ""))),
      siblings.length > 1 ? h("div", { class: "sibling-nav small" },
        position > 0 ? h("a", { href: taskHref(siblings[position - 1]) }, "← previous task of this skill") : null,
        h("span", { class: "muted" }, ` ${position + 1} of ${siblings.length} `),
        position < siblings.length - 1 ? h("a", { href: taskHref(siblings[position + 1]) }, "next task of this skill →") : null) : null),
    h("div", { class: "grid tiles" },
      tile("Solved", h("span", { class: "solved-big" }, trialDots(task.passes, meta.trials, { size: 12, gap: 4 }), ` ${task.passes} of ${meta.trials}`), `${percent(task.rate)} of GLM-5.3 trials`),
      tile("Revisions", String(task.rounds), task.rounds ? "rounds back to the author" : "accepted as first written"),
      tile("Controls", String(task.controls), "known answers the grader must score correctly"),
      tile("Caught and fixed", caughtMarks(task), FLAGS.filter((f) => f.trigger && f.test(task)).map((f) => f.short).join(", ") || "nothing"),
    ),
  );
  const journeyBody = h("div", {}, h("div", { class: "loading inline" }, "Loading the validation record…"));
  const journey = h("div", { class: "section" }, h("div", { class: "section-head" }, h("h2", {}, "How it was validated"), h("div", { class: "sub" }, "Each column is one validation of a version of the task. Click a column for its details.")), journeyBody);
  const tabBar = h("div", { class: "tabs" });
  const body = h("div", { class: "tab-body" }, h("div", { class: "loading inline" }, "Loading…"));
  page.append(journey, h("div", { class: "section" }, tabBar, body));

  let detail;
  try {
    detail = await detailPromise;
  } catch (error) {
    if (isLive()) fill(body, h("div", { class: "empty" }, `This task's record did not load: ${error.message}`));
    return;
  }
  if (!isLive()) return;
  const state = { tab: TABS.some(([k]) => k === tab) ? tab : "task", round: null, trial: null };
  const rounds = pairRounds(detail);

  const select = (key, extra = {}) => {
    Object.assign(state, { tab: key }, extra);
    history.replaceState(null, "", `${taskHref(task)}/${key}`);
    for (const button of tabBar.children) button.classList.toggle("on", button.dataset.key === key);
    fill(body, renderTab(key));
    drawJourney();
  };
  fill(tabBar, TABS.map(([key, label]) => h("button", { dataset: { key }, onclick: () => select(key) }, label)));

  const drawJourney = () => {
    fill(journeyBody, journeyMap(rounds, meta, state.tab === "history" ? state.round ?? rounds.length - 1 : null, (i) => select("history", { round: i })));
    const picked = journeyBody.querySelector(".jcol.on");
    if (picked) picked.parentNode.scrollLeft = picked.offsetLeft - picked.parentNode.offsetLeft - 40;
  };

  const renderTab = (key) => {
    switch (key) {
      case "task": return taskTab(detail);
      case "grader": return graderTab(detail);
      case "controls": return controlsTab(detail);
      case "history": return historyTab(rounds, state.round ?? rounds.length - 1, (i) => select("history", { round: i }));
      case "trials": return trialsTab(task, detail, trialsPromise, isLive, state);
      case "proposal": return proposalTab(detail);
      default: return h("div", {});
    }
  };
  select(state.tab);
}

// Validation records paired with the decision each one led to (retries carry no new evidence).
function pairRounds(detail) {
  const decisions = detail.decisions.filter((d) => !["retry", "incomplete"].includes(d.kind));
  return detail.rounds.map((round, i) => ({ ...round, index: i, decision: decisions[i] || null }));
}

function adversaryState(round, name) {
  const run = round.adversaries.find((a) => a.name === name);
  if (!run) return { mark: "·", tone: "none", text: "not run on this record" };
  const confirmed = run.findings.filter((f) => f.confirmed).length;
  if (confirmed) return { mark: "✕", tone: "bad", text: `fooled the grader ${plural(confirmed, "time")}` };
  if (!run.completed) return { mark: "?", tone: "warn", text: "did not finish its report" };
  return { mark: "✓", tone: "good", text: run.findings.length ? `${plural(run.findings.length, "attempt")}, none reproduced` : "found nothing" };
}

function decisionText(decision) {
  if (!decision) return { label: "—", tone: "" };
  if (decision.kind === "accept") return { label: decision.revert ? "Accept earlier version" : `Accept · ${BANDS[decision.band]?.label.toLowerCase() || decision.band}`, tone: "good" };
  if (decision.kind === "modify") return { label: decision.defects?.length || (decision.findings || []).some((f) => !/trials solved/.test(f)) ? "Revise" : "Adjust difficulty", tone: "warn" };
  if (decision.kind === "revise") return { label: "Revise after final review", tone: "warn" };
  return { label: decision.kind, tone: "" };
}

function journeyMap(rounds, meta, selected, onPick) {
  const cols = rounds.map((round) => {
    const passes = round.solver.filter((t) => t.passed).length;
    const controlsMet = round.controls.filter((c) => c.met).length;
    const review = round.review;
    const defects = review ? review.defects.length : 0;
    const decision = decisionText(round.decision);
    const fa = adversaryState(round, "false_accept");
    const fr = adversaryState(round, "false_reject");
    const title = round.confirmation ? "Confirmation" : `Validation ${round.index + 1}`;
    const el = h("button", { class: `jcol${selected === round.index ? " on" : ""}${round.confirmation ? " confirm" : ""}`, onclick: () => onPick(round.index) },
      h("div", { class: "jhead" }, h("span", {}, title), round.final && !round.confirmation ? tip(chip("accepted version", "good"), tipBody("This is the version that was accepted")) : null),
      h("div", { class: "jrow" }, h("span", { class: "jlabel" }, "Solver"), trialDots(passes, round.solver.length, { size: 9, gap: 3 }), h("span", { class: "small ink2" }, `${passes}/${round.solver.length}`)),
      round.controls.length ? h("div", { class: "jrow" }, h("span", { class: "jlabel" }, "Controls"), h("span", { class: controlsMet === round.controls.length ? "good-ink" : "bad-ink" }, `${controlsMet}/${round.controls.length} graded right`)) : null,
      round.adversaries.length ? [
        h("div", { class: "jrow" }, h("span", { class: "jlabel" }, "False accept"), h("span", { class: `jmark ${fa.tone}` }, fa.mark), h("span", { class: "small ink2" }, fa.text)),
        h("div", { class: "jrow" }, h("span", { class: "jlabel" }, "False reject"), h("span", { class: `jmark ${fr.tone}` }, fr.mark), h("span", { class: "small ink2" }, fr.text)),
      ] : null,
      review ? h("div", { class: "jrow" }, h("span", { class: "jlabel" }, round.confirmation ? "Final review" : "Review"),
        h("span", { class: `jmark ${defects ? "bad" : review.completed ? "good" : "warn"}` }, defects ? "✕" : review.completed ? "✓" : "?"),
        h("span", { class: "small ink2" }, defects ? `${plural(defects, "flaw")}: ${[...new Set(review.defects.map((d) => PARTS[d.part]?.toLowerCase()))].join(", ")}` : review.completed ? "no flaws found" : "did not finish")) : null,
      h("div", { class: "jfoot" }, chip(decision.label, decision.tone)));
    return el;
  });
  const out = h("div", { class: "journey" });
  cols.forEach((col, i) => {
    if (i) out.append(h("div", { class: `jarrow${["modify", "revise"].includes(rounds[i - 1].decision?.kind) ? " loop" : ""}` }, ["modify", "revise"].includes(rounds[i - 1].decision?.kind) ? "revised →" : "→"));
    out.append(col);
  });
  out.append(h("div", { class: "jarrow" }, "→"), h("div", { class: "jcol done" }, h("div", { class: "jhead" }, "Accepted"), h("div", { class: "small ink2" }, `${BANDS[rounds.at(-1)?.decision?.band]?.label || ""}`), h("div", { class: "small muted" }, `${meta.trials} graded trials`)));
  return out;
}

function taskTab(detail) {
  const env = detail.environment;
  return h("div", { class: "grid task-grid" },
    h("div", {},
      card("What the solver reads", h("div", {}, detail.prompt.map((message, i) => h("div", { class: "prompt-msg" },
        detail.prompt.length > 1 ? h("div", { class: "muted small" }, `${message.role} message ${i + 1}`) : null,
        markdown(message.content)))), { actions: [copyButton(() => detail.prompt.map((m) => m.content).join("\n\n"), "Copy prompt", "prompt")] }),
      detail.notes ? fold("Author's notes on the final version", () => markdown(detail.notes)) : null),
    h("div", {},
      card("How it is answered", h("dl", { class: "kv" },
        h("dt", {}, "Environment"), h("dd", {}, ENVIRONMENTS[env.kind]?.label || env.kind, h("div", { class: "muted small" }, ENVIRONMENTS[env.kind]?.note || "")),
        env.tools.length ? [h("dt", {}, "Tools"), h("dd", {}, env.tools.map((t) => chip(t, "outline mono")))] : null,
        env.workdir ? [h("dt", {}, "Working directory"), h("dd", { class: "mono" }, env.workdir)] : null,
        h("dt", {}, "Answer"), h("dd", {}, detail.answer.path ? h("span", { class: "mono" }, detail.answer.path) : "the final reply", detail.answer.format?.kind ? h("div", { class: "muted small" }, detail.answer.format.kind.replaceAll("_", " ")) : null),
        h("dt", {}, "Grading"), h("dd", {}, GRADERS[detail.verification]?.label || detail.verification, h("div", { class: "muted small" }, GRADERS[detail.verification]?.note || "")),
        h("dt", {}, "Files"), h("dd", {}, plural(detail.files.length, "file"), " · ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); document.querySelector('.tabs [data-key="grader"]').click(); } }, "see them")))),
      detail.tags.length ? card("Tags", h("div", { class: "chips" }, detail.tags.map((t) => chip(t, "outline")))) : null));
}

const FILE_GROUPS = {
  worker: { label: "Given to the solver", note: "placed in the solver's working directory" },
  all: { label: "Shared", note: "available to the solver and the grader" },
  verifier: { label: "Grader only", note: "hidden from the solver; used to grade" },
  oracle: { label: "Reference", note: "reference material for checking, hidden from the solver" },
};

function language(path) {
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".json")) return "json";
  if (path.endsWith(".sh")) return "bash";
  return "";
}

function graderTab(detail) {
  const grader = detail.grader;
  const facts = h("dl", { class: "kv" },
    h("dt", {}, "Kind"), h("dd", {}, grader.kind === "script" ? "a program run on the answer" : grader.kind === "verifyit" ? "an answer checker (exact, numeric, symbolic or choice)" : grader.kind),
    grader.argv ? [h("dt", {}, "Command"), h("dd", { class: "mono" }, grader.argv.join(" "))] : null,
    grader.timeout ? [h("dt", {}, "Time limit"), h("dd", {}, `${num(grader.timeout)} s`)] : null,
    grader.answer_path ? [h("dt", {}, "Reads"), h("dd", { class: "mono" }, grader.answer_path)] : null);
  const extra = Object.fromEntries(Object.entries(grader).filter(([k]) => !["kind", "argv", "timeout", "answer_path", "cwd", "env", "collect", "artifacts", "conversation_path", "image", "empty_output", "judge"].includes(k)));
  const groups = Object.keys(FILE_GROUPS).filter((g) => detail.files.some((f) => f.group === g));
  return h("div", {},
    card("Grader", h("div", {}, facts, Object.keys(extra).length ? fold("Grader settings", () => code(JSON.stringify(extra, null, 2), "json")) : null)),
    groups.map((group) => h("div", { class: "section" },
      h("div", { class: "section-head" }, h("h3", {}, FILE_GROUPS[group].label), h("div", { class: "sub" }, FILE_GROUPS[group].note)),
      detail.files.filter((f) => f.group === group).map((file) => fold(
        h("span", {}, h("span", { class: "mono" }, file.path), h("span", { class: "muted small" }, file.size !== null ? ` · ${compact(file.size)} bytes` : "")),
        () => file.text !== null ? code(file.text, language(file.path), { title: file.path, copy: true, maxHeight: "560px" }) : h("div", { class: "empty inline" }, "Binary or large file; not shown."),
        file.group === "verifier" && /grader|grade/.test(file.path))))),
    !detail.files.length ? h("div", { class: "empty inline" }, "This task ships no files.") : null);
}

function rangeText(expected) {
  if (expected.status === "submission_failure") return "rejected as malformed";
  const lo = expected.reward_min;
  const hi = expected.reward_max;
  if (lo !== null && lo === hi) return `score ${num(lo)}`;
  if (lo === null || lo === 0) return `score at most ${num(hi)}`;
  if (hi === null || hi === 1) return `score at least ${num(lo)}`;
  return `score ${num(lo)} to ${num(hi)}`;
}

function controlsTab(detail) {
  if (!detail.controls.length) return h("div", { class: "empty" }, "This task has no controls.");
  return h("div", {},
    h("p", { class: "muted" }, "Controls are answers the author wrote with a known grade: correct ones must pass, wrong or malformed ones must fail. They are replayed through the grader on every validation; these are the grades from the final one."),
    h("div", { class: "controls" }, detail.controls.map((control, i) => h("div", { class: "ctl" },
      h("div", { class: "ctl-head" },
        h("span", { class: `jmark ${control.met ? "good" : control.met === false ? "bad" : "none"}` }, control.met ? "✓" : control.met === false ? "✕" : "·"),
        h("strong", {}, `${CONTROL_KINDS[control.kind] || control.kind} ${i + 1}`),
        h("span", { class: "muted small" }, `expects ${rangeText(control.expected)}`),
        control.reward !== null ? chip(`got ${num(control.reward)}`, control.met ? "good" : "bad") : null),
      fold(h("span", { class: "small" }, `The answer (${compact(control.reply.length)} characters${control.files.length ? `, ${plural(control.files.length, "file")}` : ""})`), () => code(control.reply, "", { maxHeight: "320px" }))))));
}

function findingView(finding) {
  const adversary = ADVERSARIES[finding.defect] || null;
  return h("div", { class: `finding ${finding.confirmed ? "decisive" : "noted"}` },
    h("div", { class: "f-head" },
      chip(finding.confirmed ? "reproduced: grader fooled" : "not reproduced", finding.confirmed ? "bad" : ""),
      adversary ? chip(adversary.short, "outline") : null,
      finding.reward !== null ? h("span", { class: "muted small" }, `grader gave ${num(finding.reward)}${finding.deserved !== null ? `, deserved ${num(finding.deserved)}` : ""}`) : null),
    markdown(finding.claim),
    fold(h("span", { class: "small" }, "The answer it submitted"), () => code(finding.reply, "", { maxHeight: "300px" })));
}

function defectView(defect) {
  return h("div", { class: "finding decisive" },
    h("div", { class: "f-head" },
      chip(PARTS[defect.part] || defect.part, "outline"),
      tip(chip(EFFECTS[defect.effect]?.label || defect.effect, "bad"), tipBody(EFFECTS[defect.effect]?.label || defect.effect, EFFECTS[defect.effect]?.note || "")),
      chip(defect.confidence, defect.confidence === "established" ? "warn" : "")),
    markdown(defect.description),
    defect.fix ? h("div", { class: "fix" }, h("strong", {}, "Suggested fix: "), defect.fix) : null,
    defect.evidence.length ? fold(h("span", { class: "small" }, `Quoted evidence (${defect.evidence.length})`), () => h("div", {}, defect.evidence.map((e) => h("blockquote", { class: "quote" }, h("div", { class: "muted small" }, `trial ${e.trial + 1}`), e.excerpt)))) : null);
}

const REASON_TEXT = {
  "The one band-only round is used": "The task's one difficulty adjustment was already used, so it is accepted with its tag.",
};

// The review rule's notes are terse and keyed; say them in words.
function humanize(text) {
  return text
    .replace(/^(\w+) \((\w+)\): /, (_, name) => `${ADVERSARIES[name]?.label || name.replaceAll("_", " ")}: `)
    .replace(/\b(too_easy|too_hard|in_band)\b/g, (key) => BANDS[key].label.toLowerCase());
}

function reasonsOf(decision) {
  // "accepted_with: <band>" repeats the decision chip.
  return (decision?.reasons || []).filter((r) => !r.startsWith("accepted_with:")).map((r) => REASON_TEXT[r] || humanize(r));
}

function historyTab(rounds, selected, onPick) {
  const round = rounds[selected];
  const selector = h("div", { class: "round-picker" }, rounds.map((r) => h("button", { class: `ghost${r.index === selected ? " on" : ""}`, onclick: () => onPick(r.index) }, r.confirmation ? "Confirmation" : `Validation ${r.index + 1}`)));
  const decision = round.decision;
  const decided = decisionText(decision);
  const adversaries = round.adversaries.map((a) => card(ADVERSARIES[a.name]?.label || a.name,
    a.findings.length ? h("div", {}, a.findings.map(findingView)) : h("div", { class: "empty inline" }, a.completed ? "Found nothing to report." : "Did not finish its report."),
    { badge: chip(ADVERSARIES[a.name]?.note || "", "") }));
  const review = round.review;
  return h("div", {},
    selector,
    h("div", { class: "grid two" },
      card("Decision", h("div", {},
        h("div", { class: "decision-line" }, chip(decided.label, decided.tone)),
        decision?.findings?.length ? h("div", {}, h("div", { class: "kv-title" }, "Sent back to the author with"), h("ul", { class: "plain" }, decision.findings.map((f) => h("li", {}, markdown(humanize(f)))))) : null,
        reasonsOf(decision).length ? h("div", {}, h("div", { class: "kv-title" }, "Why"), h("ul", { class: "plain" }, reasonsOf(decision).map((r) => h("li", {}, markdown(r))))) : null,
        round.lowering_error ? h("div", {}, h("div", { class: "kv-title" }, "The task did not build"), code(round.lowering_error, "")) : null)),
      card("Solver trials", h("div", {},
        h("div", { class: "trial-strip" }, round.solver.map((t) => tip(h("span", { class: `tdot ${t.passed ? "pass" : t.cause ? "void" : "fail"}` }, String(t.trial + 1)),
          tipBody(`Trial ${t.trial + 1}`, t.passed ? "solved" : t.cause ? `no grade: ${t.cause.replaceAll("_", " ")}` : `not solved${t.reward !== null ? ` (score ${num(t.reward)})` : ""}`)))),
        review?.trials?.length ? h("div", {}, h("div", { class: "kv-title" }, "Why the failed trials failed, per the reviewer"),
          h("ul", { class: "plain" }, review.trials.map((t) => h("li", {}, chip(t.attribution === "solver" ? "solver's mistake" : t.attribution === "defect" ? "task flaw" : "uninformative", t.attribution === "defect" ? "bad" : ""), ` trial ${t.trial + 1}: `, t.reason)))) : null))),
    review ? h("div", { class: "section" }, h("div", { class: "section-head" }, h("h3", {}, round.confirmation ? "Final transcript review" : "Transcript review"), h("div", { class: "sub" }, "An agent read every solver transcript and the task's files, looking for flaws in the task itself.")),
      review.defects.length ? review.defects.map(defectView) : h("div", { class: "empty inline" }, review.completed ? "No flaws found." : "The review did not finish."),
      review.notes?.length ? fold(`Minor notes (${review.notes.length})`, () => h("div", {}, review.notes.map(defectView))) : null) : null,
    adversaries.length ? h("div", { class: "section" }, h("div", { class: "section-head" }, h("h3", {}, "Adversaries"), h("div", { class: "sub" }, "Each adversary submits answers designed to expose a grader flaw; a claim counts only if the grader reproduces it on a fresh machine.")), h("div", { class: "grid two" }, adversaries)) : null,
    round.controls.some((c) => !c.met) ? card("Controls graded wrong", h("ul", { class: "plain" }, round.controls.filter((c) => !c.met).map((c) => h("li", {}, `control ${c.index + 1}: got ${num(c.reward)}`)))) : null);
}

function trialsTab(task, detail, trialsPromise, isLive, state) {
  const root = h("div", {}, h("p", { class: "muted" }, "The eight confirmation trials of the accepted version, by GLM-5.3. Open one to read its transcript: a trimmed version shows at once and the full one loads behind it."));
  const grid = h("div", { class: "trial-grid" }, h("div", { class: "loading inline" }, "Loading trials…"));
  const viewer = h("div", { class: "trial-viewer" });
  root.append(grid, viewer);
  trialsPromise.then((byTask) => {
    if (!isLive()) return;
    const trials = byTask[task.id] || [];
    const open = (trial) => {
      state.trial = trial.trial;
      for (const el of grid.children) el.classList.toggle("on", Number(el.dataset.trial) === trial.trial);
      showTrial(task, trial, viewer, isLive);
    };
    fill(grid, trials.map((trial) => {
      const grade = trial.grade;
      const ok = grade?.passed;
      const el = h("button", { class: `trial-card${ok ? " pass" : ""}`, dataset: { trial: trial.trial }, onclick: () => open(trial) },
        h("div", { class: "t-head" }, h("strong", {}, `Trial ${trial.trial + 1}`), chip(ok ? "solved" : grade ? "not solved" : "no grade", ok ? "good" : grade ? "bad" : "warn")),
        h("div", { class: "small ink2" }, grade?.reward !== null && grade?.reward !== undefined ? `score ${num(grade.reward)}` : trial.cause?.replaceAll("_", " ") || ""),
        h("div", { class: "small muted" }, `${plural(trial.turns, "turn")}${trial.tool_calls ? ` · ${plural(trial.tool_calls, "tool call")}` : ""}`),
        trial.stop_reason ? h("div", { class: "small muted" }, STOP_REASONS[trial.stop_reason] || trial.stop_reason) : null);
      el.addEventListener("mouseenter", () => fullTranscript(task.id, trial.trial).catch(() => {}), { once: true });
      return el;
    }));
    const initial = trials.find((t) => t.trial === state.trial) || trials.find((t) => !t.grade?.passed) || trials[0];
    if (initial) open(initial);
  }).catch((error) => {
    if (isLive()) fill(grid, h("div", { class: "empty" }, `The trials did not load: ${error.message}`));
  });
  return root;
}

function graderOutput(grade) {
  if (!grade) return null;
  return fold("What the grader printed", () => h("div", {}, grade.stdout ? code(grade.stdout, "", { maxHeight: "300px" }) : h("div", { class: "muted small" }, "(nothing on stdout)"), grade.stderr ? code(grade.stderr, "", { cls: "traceback" }) : null));
}

function showTrial(task, trial, viewer, isLive) {
  const status = h("span", { class: "muted small" });
  const body = h("div", {});
  fill(viewer, card(`Trial ${trial.trial + 1}`, h("div", {}, graderOutput(trial.grade), body), { badge: status }));
  if (!trial.transcript) {
    fill(body, h("div", { class: "empty inline" }, "No transcript was recorded for this trial."));
    return;
  }
  const trimmed = trial.trimmed;
  fill(body, trial.omitted ? withGap(trial) : transcript(trial.transcript));
  status.textContent = trimmed ? "trimmed · loading the full transcript…" : "";
  if (!trimmed) return;
  fullTranscript(task.id, trial.trial).then((full) => {
    if (!isLive() || !viewer.isConnected || !body.isConnected) return;
    const messages = (full.conversations || []).flatMap((c) => c.messages);
    if (messages.length && messages[0].role === "user") messages[0] = { ...messages[0], content: "(the task prompt, shown on the Task tab)" };
    fill(body, transcript(messages));
    status.textContent = `full transcript · ${plural(messages.length, "message")}`;
  }).catch((error) => {
    status.textContent = `trimmed · the full transcript did not load (${error.message})`;
  });
}

function withGap(trial) {
  const head = trial.transcript.slice(0, trial.omitted_at);
  const tail = trial.transcript.slice(trial.omitted_at);
  return h("div", {}, transcript(head, { markFinal: false }), h("div", { class: "gap-note" }, `… ${plural(trial.omitted, "message")} omitted here; they appear when the full transcript loads …`), transcript(tail));
}

function proposalTab(detail) {
  const p = detail.proposal;
  return h("div", { class: "grid task-grid" },
    card("The proposal the author worked from", markdown(p.body)),
    card("Plan", h("div", {},
      h("dl", { class: "kv" },
        h("dt", {}, "Grounding"), h("dd", {}, p.grounding.replaceAll("_", " ")),
        h("dt", {}, "Research"), h("dd", {}, p.research_need.replaceAll("_", " "))),
      p.build.length ? h("div", {}, h("div", { class: "kv-title" }, "What to build"), h("ul", { class: "plain" }, p.build.map((b) => h("li", {}, h("strong", {}, b.name.replaceAll("_", " ")), ": ", b.description)))) : null)));
}
