# Curriculum generation prompt

Use `gpt-5.6-sol` with high reasoning effort. Replace bracketed fields and attach `prompts/rubric.md`, the relevant inventory
object, model-visible discovery evidence, and policy-level evaluation metadata. Do not attach held-out evaluation
questions or paraphrases derived from them.

```text
You are designing one subject graph for a training curriculum. Produce Curriculum JSON and a concise design audit.

SUBJECT
[subject inventory object]

EVIDENCE
[manifest, model-visible discovery evidence, and policy-level evaluation metadata]

CONSTRAINTS
- Follow the attached rubric. Optimize for useful training boundaries, not encyclopedic coverage.
- Maximum hierarchy depth is 4. Section IDs begin with [lowercase subject ID].
- First enumerate the subject's distinct central operation families. Do not infer one section per guidepost.
- A capability is an observable outcome and a task-assignment target. A group is hierarchy only. Convert routing
  menus and omnibus parents to groups; retain an internal capability only when one natural representative task
  requires coherent cross-child synthesis.
- Every capability has exactly one entry probe followed by one representative probe. Both are concrete,
  self-contained task instructions. Do not specify solutions, graders, harnesses, or verifier behavior.
- For every proposed capability, instantiate the most operationally distant permitted pair. Split it only when
  mastery does not transfer because the central operation, tool interaction, or evaluation contract differs. A
  different topic or tool name alone is insufficient.
- For each pair, compare the input representation, central transformation, output artifact, and correctness contract.
  Split when one changes materially; workflow context or shared nouns do not establish transfer. In particular,
  challenge classification versus regression, construction versus interpretation, paired versus unpaired inference,
  and deterministic mismatch analysis versus intermittent-mechanism diagnosis.
- Use a sampling facet only when the central operation and evaluation contract remain stable across its values.
  Apply the same distant-pair test to the facet's most distant values. Split instead when values change the solver
  loop, state transition, or evaluation contract.
- Set every capability's `prerequisites` field to an empty array. A separate subject-local catalog pass proposes
  learning prerequisites after capability boundaries are stable. Cross-subject progression is a separate follow-up.
- The entry probe must be the smallest self-contained exercise of the capability's outcome. The representative probe
  must exercise the full outcome. Do not encode an assumed cross-capability sequence in either probe.
- Execute every probe on paper. Reject missing state or geometry, undefined factors, inconsistent premises, trivial
  optima, and outputs that cannot be checked from the supplied facts.
- Substitute every child probe into its capability parent's outcome. If the parent does not contain every child,
  make it a group and add any natural cross-child workflow as a sibling synthesis capability.
- Account for every guidepost and evidence item with a section or an exclusion rationale. Malformed tasks are not
  positive evidence.

DESIGN AUDIT
Return as a separate file: operation families; guidepost/evidence accounting; the distant-pair operation signatures
and result for every capability and facet; the role decision for every node; probe execute-on-paper results; parent
containment results; and known evidence limitations. Do not propose learning-prerequisite edges or mention a target
section count.
```
