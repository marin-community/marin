/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
(() => {
  const terms = [
    ["Quality", "The source's summary rating, based on sampled task runs and review findings: Good, Some issues, Bad, or Unrated."],
    ["Confirmed verifier defect", "A reproduced scoring, task-import, or judge-output defect recorded in the review pool with a GitHub issue. It caps the source at Some issues until a validated fix and a new source review clear it; an existing Bad rating stays Bad. Earlier reviews and difficulty traces remain historical evidence. Only Good sources qualify for new difficulty checks."],
    ["Task attempt / verifier", "The model tries a task; the source's own verifier scores its answer or actions. A score of 1 means the verifier accepted it."],
    ["Independent judge", "A fresh review session that sees the task and execution evidence, without seeing the other judges' opinions. Judges can share the same model's biases."],
    ["Synthesis", "A source or task conclusion combining existing reviews and available evidence, while retaining disagreements and coverage limits. It can reuse historical audits without another solver run or three-judge panel. The original reviewers, dates, and revision uncertainty remain recorded."],
    ["Superseded opinion", "An earlier conclusion retained after a later review corrects or replaces it. The current source synthesis appears first; the earlier opinion remains available for context."],
    ["MarinSkyRL reference / execution commit", "A reference commit identifies the codebase used to inspect or interpret evidence. An execution commit identifies native code used in a recorded task run. A reference commit alone does not claim that tasks were executed."],
    ["Verdict", "An individual task or source review's decision. Keep means usable; Reject means unsuitable; Conditional means usable with restrictions; Inconclusive means insufficient evidence; Unrated means no decision. Source reviews contribute to the Quality rating."],
    ["Issue / observation", "An issue describes a potential technical defect. An observation records other evidence or context. Red highlights mark issues."],
    ["Severity / confidence", "Severity measures the impact of a defect. Confidence describes how strongly the evidence supports a judgment."],
    ["Difficulty", "Bars show actual model solve rates; longer bars mean more tasks solved. Current comparisons use Small Qwen3-Coder-30B-A3B-Instruct, Large Qwen3.5-122B-A10B, and Hosted GLM-5.3 on the same tasks. Every model gets 65,536 total context tokens, a 49,152-token maximum input, and 16,384 output tokens including reasoning. Large thinking is enabled; Hosted uses Low reasoning. Older models and 8k budgets are Historical, and their bars use actual model names. Filters and sorting use only current Qwen3.5-122B-A10B results. Click for attempts, native verifier outputs, applied settings, and 95% sampling uncertainty. Failed executions and unusable verifier results are counted separately as unverified."],
    ["Reviewer solve baseline", "The solver's outcomes on the small sample used for its quality review. Three judges evaluate each saved attempt; they do not count as three additional solver attempts. This starting sample can differ from the later paired difficulty sample."],
    ["Historical difficulty evidence", "Earlier Atlas runs and Marin issue #8942 measurements. Model identities, data releases, task populations, and budgets can differ from the current protocol. Historical GLM AWQ was once labeled Large; it is not the current Qwen3.5-122B-A10B model. Historical Qwen3.5-9B is not the current Small model. Their original evidence remains available."],
    ["Reasoning effort / output budget", "Effort controls the checkpoint's thinking setting, such as Low or Max. The output budget limits all generated tokens, including thinking. Thinking can use up this budget before a final answer appears. The saved trace records when that happened."],
    ["Context budget / token accounting", "The context budget limits input and output tokens. Token accounting checks the request with the model's tokenizer and records the provider's actual usage. Reasoning tokens count toward the output limit."],
    ["Serving precision / attention cache", "Weight precision describes how model parameters are stored. The attention cache stores information from earlier tokens; lower cache precision can affect answers. Saved serving settings identify weight and cache precision separately when available."],
    ["Attempt complete / stop reason", "An attempt can finish with an answer or reach the agent's time or turn limit. It counts toward difficulty only when the native verifier actually scored it. Infrastructure errors remain unverified. The stop reason records how the attempt ended."],
    ["Generation setting follow-up", "An additional run on the same sampled tasks with a documented setting change. The original measurements and traces remain available alongside the new result."],
    ["Hosted model follow-up", "An additional run with a provider-hosted model using the original paired comparison's tasks and verifier code. Its model identity, available version, and any LLM verifier judge changes are recorded separately. Score differences do not isolate the effect of the reasoning effort setting."],
    ["Verifier judge configuration", "Some datasets' own verifiers call an LLM to score an answer. Changing that judge can affect the score even when the verifier code stays the same. Deterministic verifiers do not use these judges."],
    ["Evidence / traces", "Saved model requests, responses, environment events, verifier logs, and source code supporting a review."],
    ["Revision / SHA", "A commit or content hash identifying the exact dataset, model checkpoint, or verifier code. MSkyRL means MarinSkyRL."],
    ["Dataset content equivalence", "A saved proof that the reviewed data file and component selection are unchanged at a later repository revision. Historical judgments can still apply when only the README changed; their actual review date and executed revision remain unchanged. This proof does not establish verifier code applicability."],
    ["Execution implementation applicability", "A saved check of the task loader, worker launcher, native identity check, and their code dependencies against the original difficulty run. Whole-file hashes remain recorded even when unrelated quality-review code changes. This check covers execution code, not model judgments or changed model settings."],
    ["Dataset / family", "Dataset identifies the pinned conversion input. Component rows describe selected subsets. Family describes the task domain."],
    ["Input rows / interaction", "Input rows count the selected dataset population before conversion or curation. Unknown counts stay blank; generators have no fixed count. Interaction describes support for one response or multiple conversation turns."],
    ["RLVR / Alignment / Agentic", "RLVR uses verifiable rewards; Alignment uses preferences or behavior objectives; Agentic tasks involve actions or tools."],
    ["Tags / applicability", "Source tags describe task types, interaction and search aliases; review tags label findings. Applicability records which dataset revision an imported review is known to describe."],
    ["Imported review / runtime review", "Task Trove is a catalog of tasks executed in Harbor. Imported reviews preserve its earlier audits. Runtime reviews include new task attempts and actual verifier execution."],
  ];
  const guide = document.createElement("aside");
  guide.className = "field-guide";
  guide.setAttribute("aria-label", "Field guide");
  const intro = document.createElement("p");
  intro.textContent = "Quality describes data reliability; red highlights identify technical issues. Current difficulty models: Small = Qwen3-Coder-30B-A3B-Instruct, Large = Qwen3.5-122B-A10B, Hosted = GLM-5.3. Shared limits: 65,536 context / 49,152 input / 16,384 output tokens. Earlier measurements are Historical.";
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
