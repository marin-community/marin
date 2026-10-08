# Task curation tests

Each family test module (`test_skyrl.py`, `test_nemotron_ultra.py`, `test_arc.py`, `test_reasoning_gym.py`, `test_tasktrove_code.py`, `test_tasktrove_text.py`) defines `ROWS`, one representative source row per declaration, and converts every row through its declaration with [`conversion.py`](conversion.py). [`test_catalog.py`](test_catalog.py) checks that those rows cover the catalog, that declarations join the Atlas listings, and that declared grader images are buildable recipes. [`test_scorers.py`](test_scorers.py) imports each vendored scorer from the files a task ships and checks the patched APPS evaluator against the original.

[`test_pipeline.py`](test_pipeline.py) covers artifact identity and downloads; [`test_images.py`](test_images.py) covers image identity and the build step with the `docker` stand-in from [`image_builds.py`](image_builds.py); [`test_driver.py`](test_driver.py) and [`test_campaign.py`](test_campaign.py) cover campaign planning, full-mode admission and the shared pool.

[`fixtures/`](fixtures/README.md) holds representative archived tasks and source rows.

## Grading in the grader image

[`test_skyrl_grading.py`](test_skyrl_grading.py), [`test_nemotron_ultra_grading.py`](test_nemotron_ultra_grading.py), [`test_arc_grading.py`](test_arc_grading.py), [`test_reasoning_gym_grading.py`](test_reasoning_gym_grading.py) and [`test_tasktrove_grading.py`](test_tasktrove_grading.py) grade converted fixture rows, and run their declared controls, in the locally built grader image through [`local_grader.py`](local_grader.py). They check that each grade script scores a wrong answer 0 and the source's reference answer 1 where it has one, and that a scorer module or dependency that cannot import is an infrastructure error, never a zero.

These modules carry the `docker` marker, which the default test run excludes. Build the image and select them with `-m docker`:

```bash
docker build --platform linux/amd64 --build-context verifyit=lib/verifyit/src/verifyit \
  -t local/task-curation-grader:test experiments/post_training/task_curation/images/grader
uv run --with-editable './lib/taskcompendium[pipeline]' pytest experiments/post_training/task_curation/tests -m docker
```

They skip with a reason when Docker or the image is missing. Other tests do not show that a grader image runs a source's scorer; a source's `verify/report.json` records that evidence. See the [experiment overview](../README.md).
