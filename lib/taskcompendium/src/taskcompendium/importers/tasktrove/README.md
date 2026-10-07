# TaskTrove archive import

[convert.py](convert.py) reads task archives as regular-file bytes without
extracting them. `archive_files` limits compressed and expanded size to 32 MiB
and counts at most 1,024 members. `read_archive` also checks the declared subset
and archive path against `task.toml`; the caller supplies release provenance.

[models.py](models.py) stores the decoded archive. [mcqa.py](mcqa.py) imports MCQ
tasks with the common VerifyIT scorer. Executable source conversion lives in the
[TaskTrove experiment](../../../../../../experiments/post_training/task_curation/datasets/README.md)
and its conversion components. Source-provided grader files retain their content
and semantics. This reader does not replace broken graders or manufacture a
golden solution. See the [importer boundary](../README.md).
