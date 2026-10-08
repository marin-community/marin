# TaskCompendium package

Start with the [task and grading contract](../../README.md). This package defines
tasks with their answer formats and graders, submission extraction and grading,
plus reusable ingestion procedures. Source declarations and ArtifactSteps live in
[the curation experiment](../../../../experiments/post_training/task_curation/README.md).

| Module or directory | Responsibility |
| --- | --- |
| [models.py](models.py) | TaskSpec, public context, resources, environment requirements, answer formats and grader kinds |
| [submission.py](submission.py) | Answer-format instructions, compatibility checks and model-visible requests |
| [grading.py](grading.py) | In-process verifyit grading and verdict normalization |
| [grader.py](grader.py) | Grader packages: a grader with its verifier resources |
| [convert/](convert/) | Shared conversion techniques: answer tasks, TaskTrove archives, image-installed source scorers, executable tasks |
| [importers/](importers/README.md) | Input format decoding and provenance |
| [pipeline/](pipeline/README.md) | Sampling, review, filtering, source gates and sidecars |
| [runtime/](runtime/README.md) | Acquired evidence, workspace capture and grading in a fresh Shellbox machine |

Common scoring algorithms live in VerifyIT. Custom source scorers remain in
upstream packages or task resources; importing a converter is not a runtime
requirement of its grader. The actor runtime presents the task's answer format
and keeps grader configuration and verifier resources out of the model request.
