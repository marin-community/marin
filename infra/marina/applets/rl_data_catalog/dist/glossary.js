/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
(() => {
  const terms = [
    ["Quality", "The source's summary rating, based on sampled task runs and review findings: Good, Some issues, Bad, or Unrated."],
    ["Task attempt / verifier", "The model tries a task; the source's own verifier scores its answer or actions. A score of 1 means the verifier accepted it."],
    ["Independent judge", "A fresh review session that sees the task and execution evidence, without seeing the other judges' opinions. Judges can share the same model's biases."],
    ["Synthesis", "A final review combining the task outcomes and judge opinions, while retaining their disagreements."],
    ["Verdict", "An individual task or source review's decision. Keep means usable; Reject means unsuitable; Conditional means usable with restrictions; Inconclusive means insufficient evidence; Unrated means no decision. Source reviews contribute to the Quality rating."],
    ["Issue / observation", "An issue describes a potential technical defect. An observation records other evidence or context. Red highlights mark issues."],
    ["Severity / confidence", "Severity measures the impact of a defect. Confidence describes how strongly the evidence supports a judgment."],
    ["Difficulty", "Solved tasks divided by attempts with a usable verifier result. Missing or failed executions and unusable verifier results are excluded from that rate and counted as unverified. The 95% interval describes sampling uncertainty."],
    ["Reviewer solve baseline", "The solver's outcomes on the small sample used for its quality review. Three judges evaluate each saved attempt; they do not count as three additional solver attempts. This starting sample can differ from the later paired difficulty sample."],
    ["Historical difficulty evidence", "Earlier measurements retained from Marin issue #8942. Their data releases, task populations, and rollout settings can differ from current Atlas runs, so their scores are kept separate."],
    ["Reasoning effort / output budget", "Effort controls the checkpoint's thinking setting, such as Low or Max. The output budget limits all generated tokens, including thinking. Thinking can use up this budget before a final answer appears. The saved trace records when that happened."],
    ["Generation setting follow-up", "An additional run on the same sampled tasks with a documented setting change. The original measurements and traces remain available alongside the new result."],
    ["Hosted model follow-up", "An additional run with a provider-hosted model using the original paired comparison's tasks and verifier code. Its model identity, available version, and any LLM verifier judge changes are recorded separately. Score differences do not isolate the effect of the reasoning effort setting."],
    ["Verifier judge configuration", "Some datasets' own verifiers call an LLM to score an answer. Changing that judge can affect the score even when the verifier code stays the same. Deterministic verifiers do not use these judges."],
    ["Evidence / traces", "Saved model requests, responses, environment events, verifier logs, and source code supporting a review."],
    ["Revision / SHA", "A commit or content hash identifying the exact dataset, model checkpoint, or verifier code. MSkyRL means MarinSkyRL."],
    ["Dataset content equivalence", "A saved proof that the reviewed data file and component selection are unchanged at a later repository revision. Historical judgments can still apply when only the README changed; their actual review date and executed revision remain unchanged. This proof does not establish verifier code applicability."],
    ["Execution implementation applicability", "A saved check of the task loader, worker launcher, native identity check, and their code dependencies against the original difficulty run. Whole-file hashes remain recorded even when unrelated quality-review code changes. This check covers execution code, not model judgments or changed model settings."],
    ["Canonical source / family", "Canonical source names the parent dataset or blend. Component rows describe its distinct task subsets. Family describes the task domain."],
    ["Environment / interaction", "Environment runs the task, such as Gym or Harbor. Interaction describes support for one response or multiple conversation turns."],
    ["RLVR / Alignment / Agentic", "RLVR uses verifiable rewards; Alignment uses preferences or behavior objectives; Agentic tasks involve actions or tools."],
    ["Tags / applicability", "Tags label review findings. Applicability records which dataset revision an imported review is known to describe."],
    ["Imported review / runtime review", "Task Trove is a catalog of tasks executed in Harbor. Imported reviews preserve its earlier audits. Runtime reviews include new task attempts and actual verifier execution."],
  ];
  const guide = document.createElement("aside");
  guide.className = "field-guide";
  guide.setAttribute("aria-label", "Field guide");
  const intro = document.createElement("p");
  intro.textContent = "Reading this page: Quality describes data reliability. Verifier scores describe task attempts. Red highlights identify reported technical issues.";
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Field guide · definitions";
  const list = document.createElement("dl");
  for (const [term, definition] of terms) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = term; dd.textContent = definition; list.append(dt, dd);
  }
  details.append(summary, list); guide.append(intro, details);
  document.querySelector("main").prepend(guide);
})();
