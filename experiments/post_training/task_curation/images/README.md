# Source evaluator images

Each directory contains a grader image recipe for its source family. Builds acquire
unchanged evaluators from pinned upstream commits and verify their hashes. The
runtime manifest selects the resulting image by digest. Grading runs with network
access disabled through Shellbox.

The recipes are [APPS](apps/README.md), [IFEval](ifeval/README.md),
[Nemotron Ultra](nemotron_ultra/README.md),
[Reasoning Gym](reasoning_gym/README.md), and
[SkyRL code/SQL](skyrl_code_sql/README.md).

The APPS image uses Python 3.10, `pyext==0.7` and `numpy==1.23.5` because its
original evaluator does not import under Python 3.11. The converter stages only
the task's tests and a Python runner that calls the installed evaluator and writes
its reward. Scoring and candidate execution remain in the upstream evaluator.
The IFEval image acquires the unchanged scorer at its pinned revision and checks
the source hash. Its task runner calls the scorer with normalized constraints.
Nemotron and SkyRL code/SQL images install their unchanged source modules.
Reasoning Gym retains separate package versions for TaskTrove, Ultra and generated
records; regeneration uses the original Python 3.11 environment.

Build an image from this checkout, then publish it and use its digest in the
campaign runtime manifest:

```bash
docker build --platform linux/amd64 -t REGISTRY/task-curation-apps:VERSION \
  experiments/post_training/task_curation/images/apps
docker push REGISTRY/task-curation-apps:VERSION
```

These grader images are separate from the shared Zephyr worker image. An
`iris-gvisor` runtime entry needs the grader image; a QEMU entry also needs a
worker image containing a matching Shellbox guest bundle.
