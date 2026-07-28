# CoreWeave TensorStore GCS authentication

## Summary

The first d768 nested-MoE burn-in attempt failed before model initialization
because the canonical Datakit cache is in private GCS. The CoreWeave task
environment supplied fsspec with renewable user ADC, but TensorStore's native
GCS driver did not consume the fsspec configuration and attempted anonymous
reads.

No training steps or experimental measurements were produced by the failed
attempt.

## Impact

Both matched arms failed while opening the same Zarr metadata object:

- `nest-burn-001-e256-d768-s8192-e256-c4p14e18-r1`
- `nest-burn-001-fixed25-d768-s8192-e256-c4p14e18-r1`

The coordinator job trees were stopped after the identical deterministic
failure was confirmed, preventing further retries.

## Evidence

An authenticated fsspec probe on `cw-us-east-08a` read the canonical ledger.
Both training logs subsequently reported HTTP 401 from TensorStore while
opening an `input_ids/offsets/zarr.json` object. The error identified the
caller as anonymous.

The JAX distributed runtime and NCCL initialization completed before this
failure. The incident is therefore isolated to TensorStore credential
discovery, not training, routing, or accelerator communication.

## Remediation

At process startup, the Grug MoE training entrypoint now materializes the
structured token from `FSSPEC_GS` into a mode-0600 task-local ADC file and sets
`GOOGLE_APPLICATION_CREDENTIALS`. Existing explicit ADC paths take precedence.
The credential contents are never logged.

Validate the remediation with a CPU task on the target cluster that opens the
same Zarr metadata through TensorStore before launching replacement GPU jobs.

## Cleanup

Remove the experiment-only `FSSPEC_GS` key from the `iris-task-env` Kubernetes
secret after the burn-in and its evaluations finish.
