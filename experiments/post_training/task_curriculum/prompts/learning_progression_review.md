# Learning-progression review prompt

Use a fresh `gpt-5.6-sol` high-reasoning context. Attach the same catalog scope, the proposal, and
`prompts/learning_progression.md`. Bind structured output to the `LearningProgressionReview` Pydantic schema in
`models.py`; the schema is the output contract. Do not attach the proposal's generation transcript or an earlier
review.

```text
Perform an independent complete-scope review of the proposed learning progression. Return strict
LearningProgressionReview JSON.

For every proposed edge:

- verify that both witnesses exercise the prerequisite capability's stated outcome;
- verify that both dependent tasks reuse the same foundation and add the same single main operation;
- reject course order, domain relabeling, shared vocabulary, general sophistication, pure artifact handoff, and
  narrow or contrived enabled scopes;
- execute each witness on paper and reject missing inputs, contradictory premises, trivial requests, and
  uncheckable outputs; and
- check that a closer prerequisite or an existing path does not make the edge redundant.

Then scan the complete scope for a small number of clear missing immediate edges. Do not construct an exhaustive
course DAG. Each missing edge must include the full enabled scope, transfer basis, artifact-substitution analysis,
and two self-contained witness pairs. This lets the operator combine accepted edges and omissions without another
generation call.

Return one edge review for every proposed edge exactly once. Recommend `revise` when any proposed edge is rejected
or any missing edge is supplied; otherwise recommend `accept`. Structural review supports the learning hypothesis.
It does not establish causal training transfer.
```
