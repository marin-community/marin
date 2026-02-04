# Debugging log for training-v2-iris

Validate fray v2 migration for Levanter training and confirm Iris submission paths (smoke -> CPU tutorial -> TPU tutorial) work with correct TPU resource requests.

## Initial status

User requested fray v2 migration for training and Iris test runs using `uv run iris run`, with correct TPU type for TPU jobs.

## Hypothesis 1

If Iris jobs fail, the most likely causes are missing extras (CPU vs TPU deps) or incorrect TPU type passed to `iris run`.

## Changes to make

- Update Levanter tutorial scripts to import `ResourceConfig` from `fray.v2`.
- Update training design doc with spiral plan and Iris run steps.
- Use `uv run iris --config ... run` for smoke/CPU/TPU jobs with explicit `--tpu` for TPU.

## Future Work

- [ ] Confirm `default_train` works on Iris with fray v2 end-to-end.
- [ ] Align remaining tutorial scripts with `fray.v2` if needed.
- [ ] Add a minimal Iris job wrapper for Levanter tests if repeated manual runs are common.

## Results

Smoke job (local Iris cluster) succeeded using:

- `uv run iris --config lib/iris/examples/local.yaml run --extra marin:cpu -- python -c "print('smoke')"`

Observed autoscaler scale-up, worker registration, and job completion with log output `smoke`.

Attempted Levanter CPU tutorial (local Iris) without MARIN_PREFIX and saw:

- `ValueError: Must specify a prefix or set the MARIN_PREFIX environment variable`

Retried with MARIN_PREFIX, but tokenization step failed due to request timeouts
from Zephyr actor RPCs (Iris local cluster). The job ended in failed state after
timeouts. A follow-up run with `ZEPHYR_NUM_WORKERS=4` reduced worker fan-out but
still stalled with repeated retries/timeouts; I terminated the run to avoid a
runaway local controller.

Next: rerun with an even lower Zephyr worker count (e.g. `ZEPHYR_NUM_WORKERS=1`)
and consider increasing Iris actor call timeouts for Zephyr-heavy workloads.
Then attempt TPU tutorial on a TPU-capable Iris config with `--tpu v4-8` and
`--extra marin:tpu`.
