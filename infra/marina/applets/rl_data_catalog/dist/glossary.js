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
    ["Evidence / traces", "Saved model requests, responses, environment events, verifier logs, and source code supporting a review."],
    ["Revision / SHA", "A commit or content hash identifying the exact dataset, model checkpoint, or verifier code. MSkyRL means MarinSkyRL."],
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
