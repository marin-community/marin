# Grader images

Each directory holds the recipe for a grader image that runs a source's original
scorer. Builds acquire the scorers from pinned upstream commits and check their
hashes. [`__init__.py`](__init__.py) records every image the declarations use as an
`Image` constant: its digest, the Shellbox backends that can run it, and the QEMU
guest bundle path on the campaign worker image when QEMU can run it. Grading runs
with network access denied.

The recipes are [APPS](apps/README.md), [IFEval](ifeval/README.md),
[Nemotron Ultra](nemotron_ultra/README.md),
[Reasoning Gym](reasoning_gym/README.md), and
[SkyRL code/SQL](skyrl_code_sql/README.md).

The APPS image uses Python 3.10, `pyext==0.7` and `numpy==1.23.5` because its
original evaluator does not import under Python 3.11. The task supplies its test
cases and a small runner that calls the installed evaluator and writes its reward;
scoring and candidate execution stay in the upstream evaluator. The IFEval image
installs the unchanged scorer at its pinned revision and checks its hash.
Nemotron and SkyRL code/SQL images install their unchanged source modules.
Reasoning Gym keeps separate package versions for TaskTrove, Ultra and generated
records.

To change an image, build and publish it, then update its constant in
[`__init__.py`](__init__.py) with the new digest. The artifact of every source whose
converter references the constant changes with it.

```bash
docker build --platform linux/amd64 -t REGISTRY/task-curation-apps:VERSION \
  experiments/post_training/task_curation/images/apps
docker push REGISTRY/task-curation-apps:VERSION
```

Grader images are separate from the Zephyr worker image. QEMU verification boots
the guest bundle that the worker image carries for each grader image; gVisor and
Iris verification run the grader image directly.
