# TaskTrove Clean

Convert `open-thoughts/TaskTrove` (revision 0292300, 1,739,326 tasks, 93 sources) into
"TaskTrove Clean": every kept task graded by `tasktrove-verify` in one declared mode, with its own
edited Dockerfile, deduplicated, statically and in-process verified, and tagged for selection.
Plan artifact: https://claude.ai/code/artifact/adcc779a-78d3-4b19-8daf-11e1ec1e8fbc

Branch `tasktrove-conversion-pipeline`, one PR (#9061 is absorbed into it).

## Pipeline (`experiments/post_training/tasktrove/pipeline.py`)

raw → summaries → templates → converted → deduped → verified → clean

- Agents write `source_verdicts.json` (done) and `converters/<name>.py` (one per converter key).
- Everything else is mechanical and reruns from the checked-in files.

## Tasks

1. [x] Spec in `tests/verifier.toml`, flat modes, `--verify-tool-ref` (experiments module)
2. [x] `lib/tasktrove-verify`: 13 modes, CLI, library entry point, fixture tests
3. [x] Converter registry keyed by (family, tests/ code files), coverage.json, tags columns
4. [x] 19 converters (one per key, each with a fixture test and a local Docker sampling report under `converters/reports/`); every kept source is covered, every other source is dropped with a reason
5. [x] Dedup and cap step
6. [x] Verified step: spec, dockerfile, gold leak, empty, expected, perturbed, shape
7. [x] Clean step: tasks/ per source, ledger/, manifest.json, report.md, export CLI, README
8. [ ] Full run on Iris `cw-us-east-02a` with the pushed SHA as tool ref; measured counts in the PR body; 100-task random sample inspected

## Not in this PR

Container-based verification of every task, judge sampling against a model, PyPI release,
Hugging Face upload.
