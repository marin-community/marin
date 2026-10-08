# Task curation tests

Each family test module (`test_skyrl.py`, `test_nemotron_ultra.py`, `test_arc.py`, `test_reasoning_gym.py`, `test_tasktrove_code.py`, `test_tasktrove_text.py`) defines `ROWS`, one representative source row per declaration, and converts every row through its declaration with [`conversion.py`](conversion.py). [`test_catalog.py`](test_catalog.py) checks that those rows cover the catalog, that declarations join the Atlas listings, and that declared grader images are buildable recipes. Grader-script tests run the dataset scripts locally on passing and failing answers where the script needs no grader image.

[`test_pipeline.py`](test_pipeline.py) covers artifact identity and downloads; [`test_images.py`](test_images.py) covers image identity and the build step with the `docker` stand-in from [`image_builds.py`](image_builds.py); [`test_driver.py`](test_driver.py) and [`test_campaign.py`](test_campaign.py) cover campaign planning, full-mode admission and the shared pool.

[`fixtures/`](fixtures/README.md) holds representative archived tasks and source rows. These tests do not show that a grader image runs a source's scorer; a source's `verify/report.json` records that evidence. See the [experiment overview](../README.md).
