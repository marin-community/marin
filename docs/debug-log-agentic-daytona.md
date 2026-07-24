# Debugging log for agentic Daytona validation

Validate a capped Terminal-Bench 2 run through the unified evaluation launcher, a minted Iris
capability URL, Daytona, and the v2 evaldash record/sample path.

## Initial status

The `qwen3-0.6b` serve started successfully on `v5litepod-4`, and the launcher minted a capability
URL for `/serve/inference-eadde315`. Harbor failed before creating Daytona sandboxes:

`ValueError: Tag 'latest' not found for dataset 'DCAgent2/terminal_bench_2'`.

## Hypothesis 1

The DCAgent benchmark identifiers imported from #7246 name Hugging Face repositories containing raw
Harbor task directories. The unified suite registered them as Harbor package-registry names, so
Harbor 0.20 interpreted the slash as a package reference and looked for a package tag.

The existing Marin Harbor evaluator already uses an `hf://` dataset convention and materializes those
repositories with `huggingface_hub.snapshot_download` before constructing a path-backed Harbor
dataset.

## Changes to make

- Reuse the existing `hf://` convention in the unified runner.
- Register the #7246 agentic presets as `hf://DCAgent...` sources with an explicit Hugging Face
  revision.
- Pass a materialized local task path across the isolated Harbor-driver boundary.
- Set both Harbor's legacy `version` and package-registry `ref` fields for registry-backed datasets.
- Add a regression test around Hugging Face dataset materialization and the suite's source metadata.

## Results

The materialization regression test and the 66 tests under `tests/evaluation` and `tests/evals`
pass. All eight imported DCAgent repositories resolve at the configured `main` revision.

The live retry is pending.

## Future work

- [x] Confirm all imported DCAgent preset repositories still exist at their configured revisions.
