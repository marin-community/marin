# Harbor dataset revision: configured tag is absent

Verify and correct the Grug OpenCode dataset selector after the capped PR #7606 validation reached
the ready vLLM service but failed during Harbor task resolution.

## Initial status

The one-task run `/loom/eval-20260724-210018-grug-agentic-s3-step1903-6046` staged the object-store
model and brought the H100x8 vLLM service to readiness. Harbor then rejected
`DCAgent/dev_set_v2` with `ValueError: Tag '1.0' not found for dataset 'DCAgent/dev_set_v2'` before
starting a Daytona sandbox.

## Hypothesis 1

The inherited `1.0` value is a package version rather than a registry tag, so serializing it as
Harbor's `ref` field selects the wrong registry namespace.

## Changes to make

Inspect the installed Harbor schema and live package metadata, then bind the dataset using the field
that resolves the intended published revision. Add a regression assertion against the native Harbor
schema and repeat the same one-task validation.

## Results

Current Harbor interprets any slash-qualified dataset name as a package-registry identifier and
interprets `ref: "1.0"` as a package tag. The live package registry has no
`DCAgent/dev_set_v2` package. The intended dataset is instead the public Hugging Face repository
`DCAgent/dev_set_v2`, whose `main` branch resolved to
`377118ff3031c934f5a647ae2c425eb74eef3b21` during the investigation. The evaluation now pins that
commit and uses the launcher's existing `hf://` materialization path.

## Future work

- [x] Record the capped rerun result.

## Hypothesis 2

The corrected rerun completed all three Harbor attempts, then failed while persisting the trial tree.
The Harbor runner passed an S3 URL to helpers that always constructed a GCS filesystem, so `gcsfs`
interpreted `s3:` as a bucket name.

## Changes to make

Resolve the filesystem from each artifact URL and cover a complete trial-tree upload and restore
through a non-GCS remote protocol.

## Results

The upload and restore helpers now resolve their filesystem from the artifact URL. A regression test
round-trips `result.json` and the agent trajectory through an in-memory remote filesystem; the 128
evaluation, eval, and inference tests pass.

The first final-code rerun spent the full 40-minute CoreWeave startup grace period in `building`
without a worker attempt, log, failure, or preemption. Iris recorded `infra_failed` and cleaned up
the child. The unchanged retry
`/loom/eval-20260724-221717-grug-agentic-s3-step1903-a34d` then succeeded. It staged 45 model files,
started the H100x8 vLLM service, completed all three attempts for one OpenCode task, uploaded the
three trial trees, wrote the normalized samples and aggregate result, and recorded `status=succeeded`
at
`s3://marin-us-east-02a/marin/eval-metadata/runs/20260724-221717-grug-agentic-s3-step1903-grug-opencode-id-83c1/record.json`.
The task scored 0/3; the capped run validated the launch and persistence path, not model quality.
