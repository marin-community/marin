# Debugging log for agentic Daytona validation

Validate a capped Terminal-Bench 2 run through the unified evaluation launcher, a minted Iris
capability URL, Daytona, and the v2 evaldash record/sample path.

## Initial status

The `qwen3-0.6b` serve started successfully on `v5litepod-4`, and the launcher minted a capability
URL for `/serve/inference-eadde315`. Harbor failed before creating Daytona sandboxes:

`ValueError: Tag 'latest' not found for dataset 'DCAgent2/terminal_bench_2'`.

## Hypothesis 1

The DCAgent benchmark identifiers imported from
[#7246](https://github.com/marin-community/marin/pull/7246) name Hugging Face repositories containing raw
Harbor task directories. The unified suite registered them as Harbor package-registry names, so
Harbor 0.20 interpreted the slash as a package reference and looked for a package tag.

The existing Marin Harbor evaluator already uses an `hf://` dataset convention and materializes those
repositories with `huggingface_hub.snapshot_download` before constructing a path-backed Harbor
dataset.

## Changes to make

- Reuse the existing `hf://` convention in the unified runner.
- Register the [#7246](https://github.com/marin-community/marin/pull/7246) agentic presets as
  `hf://DCAgent...` sources with an explicit Hugging Face
  revision.
- Pass a materialized local task path across the isolated Harbor-driver boundary.
- Set both Harbor's legacy `version` and package-registry `ref` fields for registry-backed datasets.
- Add a regression test around Hugging Face dataset materialization and the suite's source metadata.

## Results

The materialization regression test and the 66 tests under `tests/evaluation` and `tests/evals`
pass. All eight imported DCAgent repositories resolve at the configured `main` revision.

## Hypothesis 2

The first retry materialized all 861 Terminal-Bench files, started two Daytona trials, and sent model
requests through the minted URL. Iris rejected those requests with
`endpoint-scoped token cannot access this endpoint`.

The direct inference lifecycle registered the endpoint with private access. Iris accepts a scoped
capability token only for an endpoint registered with link access. Harbor serve sessions now request
link access explicitly while in-cluster eval sessions retain the private default. The same retry also
exposed a result-path bug: the raw `hf://` dataset identifier was interpolated into the sample parquet
filename. Agentic runs now use the run-local `samples_harbor.parquet` name.

## Final validation

Run `20260724-004403-qwen3-0.6b-tb2-lite-8f6f` completed successfully on commit
`954f7cd8798f4a2be75b872828300bc0b186bdfc`. The group launcher served `qwen3-0.6b`
on a `v5litepod-4`, registered `/serve/inference-6f9ecea3` with link access, and minted a scoped
capability URL. Both Daytona trials made repeated model requests through that URL with HTTP 200
responses.

Harbor ran two Terminal-Bench 2 tasks, `adaptive-rejection-sampler` and `bn-fit-modify`. The
0.6-billion-parameter smoke model solved neither task, so the recorded accuracy and mean reward are
both 0.0. This validation measures plumbing rather than model quality: both trial directories,
verifier outputs, and normalized trajectories were uploaded, and the launcher wrote a successful
`EvalRunRecord`.

The run's `samples_harbor.parquet` contains two schema-v2 rows. Both have `kind=agentic`,
`Grading(method="harbor:verifier")`, and trajectory URIs for the corresponding task. Evaldash indexes
the run at
`https://evaldash.oa.dev/runs/20260724-004403-qwen3-0.6b-tb2-lite-8f6f`.

- [x] Confirm all imported DCAgent preset repositories still exist at their configured revisions.
- [x] Materialize a DCAgent task repository and create Daytona sandboxes.
- [x] Exercise the minted capability URL from both terminal-agent trials.
- [x] Upload Harbor results, trajectories, and schema-v2 agentic samples to evaldash storage.
