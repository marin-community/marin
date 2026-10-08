# Dataset-family components

These modules normalize source rows, preserve grading contracts and construct
`TaskPolicy` bundles containing conversion, checks and review rubrics. They are
used by the [experiment declarations](../../../../../experiments/post_training/task_curation/datasets/README.md),
which own source pins and expose `pipeline()` factories.

Families cover answer tasks, preferences, code, repository work, instructions and
source-native actions. [nemotron/](nemotron/) handles structured-output
rows and question-placeholder restoration; [nemotron_ultra/](nemotron_ultra/)
handles Ultra rows and family policies; [reasoning_gym/](reasoning_gym/) handles
pinned generated tasks and their source scorer contract.

Normalization returns a TaskSpec or an explicit import rejection. It preserves
source scripts and test semantics; a broken test suite is rejected instead of
being repaired to pass a reference. An original evaluator that cannot run here
becomes a `NoGrader` that records the reason and the source contract. Quality
review and executable readiness are separate decisions in the [pipeline](../pipeline/README.md).

These components construct no ArtifactSteps and import no experiment modules.
Custom scorers belong to the task or its original installed source package;
common graders belong to VerifyIT.
