# NeMo Gym two-row sweep

The current NeMo Gym collection contains 29 task datasets and three training
blends. Two deterministic rows were sampled from each task dataset at its Hub
revision. Seventeen rows are accepted, six rejected, and thirty-five deferred.
The accepted set has two competitive-coding rows, two instruction-following
rows, two stateful Workplace rows, and eleven answer-only rows covered by shared
MCQA, math, or reference-answer verifiers.

The eleven static rows retain pinned source rows and provenance only in verifier
resources. Their task instructions remove source delivery and grading language;
plain text, boxed-LaTex, JSON, and XML are separate lowerings. Reference-answer
rows carry an explicit private judge policy, and a missing or malformed judge
verdict is recorded as infrastructure failure rather than a zero score.

The action-prediction findings in `reviews/agent-corpora.json` were superseded by
`reviews/action-wire-format.json`. The four rows depend on hidden tool results or
authentication state, so removing those records would change the task. They are
deferred instead of being normalized into bare next-action prediction.

The Science row at offset 69602 remains deferred because its source contract
requires stateful Python execution. Structured-output, citation, and other
format-sensitive rows also remain deferred until their source schema or format
semantics are recovered.
