# [taskforge] Triage proposals with structural checks and a GLM rubric

Stacks on PR 9623 (`rollout-engine`) through `taskforge/03-proposal`, whose `TaskProposal` it
evaluates.

`taskforge.triage` runs deterministic checks first; a fatal failure or a null proposal rejects with
no model call. `GlmRubric` then scores seven integer axes in independent samples, and code decides:
ACCEPT or REJECT needs a strict majority of the per-sample rule, anything else is REPAIR.
`GlmRubric.repair` rewrites a proposal in one call and returns it for the caller to re-evaluate.
The axes are top-level tool fields because GLM sent a nested `scores` object as a JSON string.

Unit tests on this layer: 169 passed, 17 skipped. Live, every proposal from the proposal runs (103)
was scored with 3 samples at temperature 0.7: 311 calls, 0 rubric output errors, 31 ACCEPT and 72
REPAIR. Majority decisions agreed with an earlier run on 57 of 63 shared proposals. Two of three
repairs reached ACCEPT. The raw calls are in the gitignored `.evidence/triage/`.

Out of scope: the rubric has never recommended REJECT live, so that branch is unit-tested only. The
advisory `resources_in_build_plan` check has known false positives. Triage imports the proposal
seam type from `taskforge.proposal`; `DESIGN.md` says stage packages do not import each other, and
either the type moves down a layer or the rule changes.
