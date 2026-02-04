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

Pending: begin Iris runs and record outcomes.
