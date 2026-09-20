# Learning-progression prompt

Use `gpt-5.6-sol` with high reasoning effort after the capability catalog is stable. Attach the complete catalog or a
declared subject subset. Bind structured output to the `LearningProgression` Pydantic schema in `models.py`; the
schema, rather than field names inferred from this prose, is the output contract. The output is a separate working
artifact; do not rewrite capability sections during this pass.

```text
You are proposing learning-prerequisite edges between existing curriculum capabilities. Return strict
LearningProgression JSON.

An edge A -> B means that mastery of A materially raises the probability of solving at least one recurring family of
entry-level B tasks. This is the epsilon condition: a learner who has mastered A should have a non-trivial chance of
some success in B. A does not need to be necessary for every B task.

Apply every rule below:

1. Name the exact reusable foundation from A. It must be an operation, representation, invariant, or concept exercised
   in A's representative tasks and actively reused in B.
2. Each B witness reuses that foundation and adds one main operation or concept. It may add supplied domain facts,
   notation, or data, but it cannot require another unmodeled capability.
3. Give exactly two witness pairs from semantically distinct task families, not paraphrases or parameter variants.
   Each pair contains a representative A task and a self-contained entry B task. Both B tasks exhibit the same
   epsilon step. Exact duplicate pairs are rejected mechanically; the independent reviewer enforces semantic family
   diversity.
4. State the enabled scope within B. The scope must be a natural sampling stratum broad enough to train and evaluate
   independently. Reject a dependency that applies only to one formula, tool, or contrived example.
5. Distinguish conceptual transfer from artifact handoff. Supply every artifact that A could have produced. Keep the
   edge only when A mastery still helps choose a method, construct a model, maintain an invariant, detect an invalid
   result, or adapt the supplied artifact. Reject pure workflow order.
6. Reject course order, general sophistication, shared terminology, common parentage, and difficulty correlation.
   Reject A -> B when B uses an isolated fragment of A and would not benefit from mastery of A's stated outcome.
7. Prefer the closest useful foundation. Omit a transitive edge when the same transfer is already represented by a
   shorter accepted path. Multiple prerequisites are allowed only when each contributes a distinct foundation used
   in both B witnesses.
8. Do not infer an edge from the existing entry and representative labels. Rewrite witness tasks when catalog probes
   hide a genuine progression by supplying formulas or intermediate state.
9. Cross-subject edges are allowed. The dependent capability's subject must be in scope; its prerequisite may be in
   another catalog subject.
10. Keep the capability taxonomy unchanged. Ignore embedded prerequisites from earlier catalog versions. An honest
    empty graph is better than a dense syllabus sequence.

For every edge return prerequisite and dependent capability IDs, enabled scope, transfer basis,
artifact-substitution analysis, and exactly two witnesses. Do not target an edge count. The result is a structural
learning hypothesis, not evidence that training on A caused improvement on B.
```
