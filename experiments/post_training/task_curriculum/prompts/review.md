# Holistic curriculum review prompt

Use a separate `gpt-5.6-sol` high-reasoning context. Attach `prompts/rubric.md`, the candidate curriculum, inventory object,
evidence manifest, model-visible discovery tasks, and permitted held-out in-distribution evaluation instructions. Do
not attach answers, verifiers, earlier reviews, or blind-fit results.

```text
Perform a fresh complete-subject curriculum review. Return strict JSON using the holistic-review contract.

Read every capability, group, sampling facet, prerequisite edge, entry probe, representative probe, guidepost, and
evidence item. Score: coverage 25; mutual self-confidence 25; progression and epsilon continuity 25; observable
boundaries 15; probe quality and parsimony 10.

Required checks:
- Account for every guidepost and evidence item with a primary capability or exclusion rationale.
- For every capability, instantiate its most operationally distant permitted task pair. A blocking split finding
  must name both tasks, their different central operations, and how splitting changes sampling or evaluation.
- Compare each pair's input representation, central transformation, output artifact, and correctness contract.
- For every sampling facet, compare its most distant values and reject it if the solver loop, state transition, or
  evaluation contract changes.
- For every prerequisite edge, identify where every dependent representative uses the full prerequisite outcome,
  the one new operation in the entry, and the completed-artifact counterfactual.
- Check every internal capability for one natural cross-child synthesis. A concatenated set of child deliverables is
  not synthesis. Also verify that every child probe is an instance of the parent's outcome; otherwise use a group and
  a sibling synthesis capability.
- Check all entry-to-representative orderings for a plausible epsilon step. Structural plausibility is not evidence
  of actual training transfer.
- Execute each probe on paper and reject missing facts, undefined factors, contradictory premises, trivial optima,
  and uncheckable outputs.
- Distinguish evidence confidence from structural quality. High confidence requires direct or held-out evidence
  reaching every capability and sampled edges; sparse task evidence caps confidence at medium.

`pilot_ready` requires at least 85 points and no structural blockers. Sparse evidence lowers confidence and belongs
in the findings. Reserve blockers for structural defects. Any failed structural gate at 70 points or above is
`revise`; a score below 70 is `regenerate`. Pervasive defects must lower the affected dimension scores. Recommend the
smallest repair that fixes each concrete defect. Propose a rubric change only for a recurrent issue that generalizes
beyond this subject.
```
