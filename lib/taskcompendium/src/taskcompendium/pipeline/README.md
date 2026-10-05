# Task curation library

This package converts source records into TaskSpecs, gathers quality and grading
evidence, and produces final filtering decisions. The
[experiment](../../../../../experiments/post_training/task_curation/README.md)
chooses pinned sources and builds the artifact graph.

A `TaskPipeline` combines normalization, a rubric for the GLM review model and optional checks. A
`DatasetRecipe` binds that policy to source inputs and intended use. Start in
[datasets/](datasets/) when changing how a family is interpreted: keep its
normalization, rubric and task-specific grading construction together. Reuse a
family across sources with the same contract; add a family when the contract
differs.

Normalization preserves the boundary between the public problem and private
answers or fixtures. Checks and GLM review supply separate evidence. Filtering
commits to keep or reject, writing a new artifact with a complete audit view and
an accepted-task view. The audit retains reasons, confidence and cleanup history.
Optional rewriting creates a candidate that is checked and reviewed
again, with its original evidence retained. Quality acceptance and grader
readiness are separate; executable output requires a runnable grader.

Use [stages.py](stages.py) to follow execution through audit, filter, rewrite and
merge. Zephyr owns sharding and retry behavior. GLM completions use an exact-query
cache keyed by the query and model revision, so unrelated catalog changes need
not repeat inference. Keep artifact construction and inference-client binding in
the experiment.

For grading changes, follow [grader.py](../grader.py): a task carries a standard
VerifyIT spec or an ordinary script with private resources. Put generic reusable
verification components in [VerifyIT](../../../../../lib/verifyit); keep
dataset-specific behavior in the emitted grader. Runtime requirements must remain
explicit in the TaskSpec.

See the [task curation reference](../../../../../docs/references/task-curation.md)
for the full stage contracts, review policy, cache identities and output schema.
