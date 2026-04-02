# Debugging log for PR 4303 activation grad RMS logging

Investigate why `grad/activation_rms` is computed on the PR branch but does not appear in W&B for the Grug GPU smoke run.

## Initial status

PR comment context established that `experiments/grug/base/tiny_gpu.py` trains successfully on GPU at PR head `6cde4d03a`, but the resulting W&B run did not contain `grad/activation_rms` even though `GrugTrainerConfig.log_activation_grad_rms=True`.

## Hypothesis 1

The backward metric is computed inside the jitted train step, but the outer training loop never logs the returned `metrics` mapping, so only callback-generated metrics make it to the tracker.

## Changes to make

- Inspect `experiments/grug/base/train.py` around `_make_train_step` and the main optimization loop.
- Compare the Grug loop against the standard Levanter trainer logging path.
- Add a regression test that exercises the JSON tracker end-to-end with `log_activation_grad_rms=True`.

## Future Work

- [ ] Check whether other Grug loops returning extra metrics have the same silent-drop behavior.
- [ ] Consider a helper that centralizes "log train-step metrics + callback metrics" to avoid this class of bug.

## Results

Confirmed the hypothesis. `_make_train_step` returns a metrics dict containing `grad/activation_rms`, but `_run_grug_local` only passed `metrics["train/loss"]` into callbacks and separately logged throughput/watch metrics. Adding `levanter.tracker.log(metrics, step=step)` in the callback section restores end-to-end metric logging.
