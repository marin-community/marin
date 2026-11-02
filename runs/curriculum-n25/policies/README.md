# Verification-rule texts for the curriculum-n25 run

`propose_curriculum.py --verification-rule-file PATH` replaces the stack's `VERIFICATION_RULE`
(`experiments/post_training/capability_driven_envs/proposal_source.py`) with the file's text, less its final
newline, for one run. `launch.sh` passes it when `POLICY_FILE` is set. Without the flag, the stack's rule is used. Each
`result.json` records the sha256 of the rule text actually used, of the system prompt built around it, and the
policy name (the file stem, or `stack_verification_rule`). A shard refuses to start on a root whose results
record a different provenance.

| Text (without the file's final newline) | sha256 (UTF-8 bytes) |
|---|---|
| Stack `VERIFICATION_RULE` at `a83a1fa3c6` (code-first, 2,880 chars) | `57e87aca6db07364db2878eb65076699bc3d1871ec304f35842ff44e157e7b9e` |
| `determinate_first_preferred.txt` (DFP, 4,494 chars) | `84259144193ba1104e4370c541c6be3821006f6d84d4179cf0b0ae8c6fab5b75` |
| Stack `SYSTEM_PROMPT` with the stack rule | `f971cc503a2624d4e4f75202a00f5dca8bba0918d77df75686858feebb208ade` |
| Stack `system_prompt(DFP)` | `1dfec5ddc36cc8bcb63767d00f19741dc7fc87c3d7ebc67982c1bed23ee76b6d` |
| phase11 `PROPOSAL_INSTRUCTION + "\n\n" + DFP` (the frozen Oct 30 prompt) | `ace5025d739813aafe1f2ee025217f4c54348c4afaa49bffcde017e6668f428f` |

`determinate_first_preferred.txt` is `NUDGE_POLICIES["determinate_first_preferred"]` from
`branches/phase11-proposals-all-20261102/experiments/post_training/capability_env_gen/phase11/proposal_nudge.py`,
written by importing it there. Joined to phase11's `PROPOSAL_INSTRUCTION` it reproduces the sha that
`proposal_production.py` freezes (`FROZEN_SYSTEM_PROMPT_SHA256`), so the file is byte-identical to the measured
policy text.

## How each harness composes its prompt

**phase11** (`proposals.py::generate_proposal`), one proposal per request, no portfolio:

- system: `PROPOSAL_INSTRUCTION + "\n\n" + policy_text`. `PROPOSAL_INSTRUCTION` is a short paragraph: design one
  realistic task for the request, choose environment and verification "according to the supplied policy", return
  the JSON proposal (header plus the six ordered body sections), use `null_reason` when no valid task exists, do
  not fabricate research, reserve composite.
- user: the capability request text.
- decoding: `document_request_fields(proposal_id, source)`, the same JSON schema the stack uses. Schema repairs are
  not capped.

**Stack** (`proposal_source.py`), one plan request then one request per slot:

- system (both phases): `system_prompt(rule)`, which is the role and realism paragraph, the
  solver-workspace-versus-private-evaluator paragraph, the environment definitions, `SOLVER_MACHINE_RULE`, then
  the rule.
- plan user turn: `plan_prompt` (portfolio diversity, host environments, pairing cap, research criterion, the
  capability record), then the `SlotPlan` schema instruction. It decodes under the plan's JSON schema.
- slot user turn: `proposal_prompt` (the JSON document template, format rules, `SECTION_GUIDE`, `SLOT_RULE`, the
  capability record, the whole portfolio and the slot to develop). It decodes under `document_request_fields`.
  The driver caps document repairs at 4 and plan repairs at 4.

## Is substituting DFP faithful?

Only the rule paragraph changes. The rest of the stack prompt still leans code-first and still asks for things
DFP does not mention, so a DFP run in this harness is a new, unmeasured combination. It is not a replay of the
Oct 30 measurement. Specifically:

- `plan_prompt` still says "choose each slot's verification by the verification rule: code unless the answer
  cannot be stated as fields".
- The plan validator exempts only `code` from the per-pairing cap (`UNCAPPED_VERIFICATIONS`), so a portfolio can
  hold any number of code slots but at most ceil(n/3) of any other kind per environment.
- The stack rule's last paragraph tells the proposal what to write for decision_model and llm_as_judge: the
  rubric items with pass criteria, equivalences, negation handling and weights; never naming a grader model;
  controls with reward bands; no hand-derived control rewards. DFP has none of this. `SECTION_GUIDE`'s "Grader
  design and controls" item still asks for the rubric and controls, and refers to "why the rule above selects
  it".
- The stack prompt plans a 25-slot portfolio with diversity constraints and quotes it into every slot. phase11
  writes each proposal independently.
- phase11's `PROPOSAL_INSTRUCTION` itself is not carried over; the stack's own system and user text takes its
  place.
