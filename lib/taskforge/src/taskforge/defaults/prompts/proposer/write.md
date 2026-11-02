Develop ONLY slot {slot} of the portfolio below into a detailed construction blueprint,
written as one JSON object containing a typed header and a Markdown body in this format:

{document_template}
Format rules:
- Reply with the JSON object only, without a surrounding code fence.
- Copy header.id, header.source and header.grounding exactly as shown. header.null_reason is
  null for a proposed task. Include every header field, including empty lists.
- Encode newlines, quotes and backslashes with standard JSON escapes.
- {solver_machine_rule}
- research_need decides whether building this task requires looking anything up. It is
{research_criterion}
  Decide from what this slot needs, not from habit.
- research_reason: one line naming what drives the decision.
- research: when research_need is needed, one entry per lookup a builder must do, kind web or
  github only; when it is not_needed, `research: []`.
- Every lookup the body mentions (a web or repository search, a contamination or name-collision
  search, a license or version check, reading external documentation) is one of the research
  entries, and the body cites it as `research[i]` (0-based, in research array order) where it is
  mentioned. A lookup not worth an entry is not worth mentioning. If research_need is not_needed,
  the body mentions no lookup, search, browsing, or external-source check anywhere.
- build: one entry per build artifact, with a short name and a one-line description.
  The Build plan section explains each one.
- resources: relative paths (globs allowed) of the files the build will produce.
- The body has these level-two headings, once each, in this order:
  {headings}. Use `###` or lists inside a section.

What each section must contain:
- Task: the realistic user request as the solver sees it; the concrete visible inputs and
  deliverables; constraints; what makes it difficult; and how it exercises the capability. The
  solver-visible text states the deliverable and its format only; grading belongs in 'Grader design
  and controls', never in Task (no 'graded', 'checked', 'parsed', 'ignored', credit or tolerances in
  solver-visible text).
- Realism and workflow: who does this work and why, the workflow it reproduces, why this
  environment, solver machine and verification fit (justify any change from the planned slot),
  and what a real user can see and do versus what stays evaluator-only.
- Research plan: the research decision and its reason, and what is already known. If needed, for
  each research[i]: the query or URL, its purpose, a license check, and a fallback; mark every
  source unverified; repository searches name desired properties and a fallback; never invent an
  existing repository. If not_needed, say so in one line and describe no lookup.
- Build plan: the construction steps as a dependency graph of build sessions with handoff
  artifacts and acceptance checks; for shellsim, the simulator semantics and executable
  compatibility tests; for container, dependencies, reset and seed strategy, isolation, and
  resource estimates (to be measured).
- Grader design and controls: the observable success condition; the grader design (for judge or
  composite: anchored criteria, disqualifiers, weights, critical gates, and the exact aggregation
  formula); positive and negative controls with the reward band each must receive (the build
  computes exact values); anti-shortcut and
  prompt-injection controls; and how hidden evidence stays private (no answer leakage).
- Risks and null conditions: each risk with a mitigation and an abandon-if condition; data
  provenance, license, and split group (source/derivation groups stay in one train/eval split).
  Control contamination by construction (freshly generated values, no copied public items); a
  contamination search is a lookup like any other and is written only as a research entry.

Keep the slot's intended capability.
The slot's research_need and research_reason come from the portfolio plan. Keep the decision
unless developing the design shows otherwise; if you change it, say why in the Research plan.

A builder must be able to discover or download realistic artifacts, implement the environment, and
know whether it worked from this document alone. Give
concrete inputs, outputs, constraints, edge cases, hidden evidence, and negative controls. Do not
claim to have browsed or run code. Existing benchmarks are inspiration, never silently copied
evaluation tasks. Solver and grader execute separately: code verification does not require a
container for the solver.

If the slot cannot be made credible, write a substantive, double-quoted reason as the value of
null_reason instead of `null`, keep the other header fields (research_need may be not_needed
with `research: []`, and build and resources may be empty lists), and write a short body explaining
why. A null proposal is better than a contrived or ungradable one.

CAPABILITY RECORD:
{capability_record}

PORTFOLIO:
{portfolio}

SLOT TO DEVELOP:
{planned_slot}
