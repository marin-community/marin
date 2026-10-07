# TaskCompendium package

Start with the [task and grading contract](../../README.md). This package defines
private tasks, submission extraction and grading, plus reusable ingestion
procedures. Source declarations and ArtifactSteps live in
[the curation experiment](../../../../experiments/post_training/task_curation/README.md).

| Module or directory | Responsibility |
| --- | --- |
| [models.py](models.py) | TaskSpec, public context, private resources and environment requirements |
| [submission.py](submission.py) | Submission conventions and model-visible requests |
| [grading.py](grading.py), [grading_contract.py](grading_contract.py) | Candidate scoring and verifier validation |
| [grader.py](grader.py), [native_grader.py](native_grader.py) | Serialized grader packages and native command declarations |
| [datasets/](datasets/README.md) | Dataset-family conversion policies and review rubrics |
| [importers/](importers/README.md) | Input format decoding and provenance |
| [pipeline/](pipeline/README.md) | Sampling, review, filtering, source gates and sidecars |
| [runtime/](runtime/README.md) | Acquired evidence, workspace capture and private Shellbox grading |

Common scoring algorithms live in VerifyIT. Custom source scorers remain in
upstream packages or task resources; importing a converter is not a runtime
requirement of its grader. The actor runtime chooses a submission convention and
keeps verifier configuration and files private.
