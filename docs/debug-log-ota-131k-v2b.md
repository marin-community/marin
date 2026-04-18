# Debugging Log For OT-Agent 131K v2b

Find a `v5p-32` Levanter configuration that can train 131K SFT stably without the checkpoint-time OOMs seen in the prior long-context attempt.

## Initial status

Prior work concluded that `v5p-256` was required for 131K SFT. The available evidence was mixed:
- `exp3897_sft_ota_131k_qwen3_8b.py` (`v2`) on `v5p-32` failed with exit 137 after running for many hours.
- `exp3897v3_sft_ota_131k_qwen3_8b_no_offload.py` existed, but the session summary treated it as inconclusive.

## Hypothesis 1

The `v5p-32` failure was caused by the offload strategy plus frequent temporary checkpoints, not by the 131K training shape itself.

## Changes to make

- Add a new self-contained experiment `experiments/exp3897v2b_sft_ota_131k_qwen3_8b.py`.
- Base it on the no-offload `v3` recipe.
- Disable time-based temporary checkpoints by overriding the nested Levanter checkpointer config to `save_interval=None`.
- Disable interim HF exports and only save one trainer checkpoint at the end of the half-run stability test.

## Results

- Controller bug reports confirmed the offload `v2` job repeatedly died during `jax.experimental.array_serialization.serialization` checkpoint commit activity.
- The old no-offload `v3` job was not OOM-killed. It was manually terminated after 1h29m and had already completed several checkpoint commits.
- This narrows the problem from "131K requires `v5p-256`" to "offload plus frequent checkpoint serialization is unstable on `v5p-32`."

## Hypothesis 2

If we keep the no-offload path and eliminate the 10-minute temporary checkpoints, `v5p-32` should remain stable long enough to reach the halfway point.

## Results

- `exp3897v2b` launched successfully on a fresh `v5p-32` slice after an initial scheduling delay.
- The `train_lm` child job started at `2026-03-30 08:44:10 UTC` on slice `marin-tpu-v5p-32-us-central1-a-20260330-0840-1dcab560`.
- All four TPU workers initialized `jax.distributed` cleanly and began loading the `Qwen/Qwen3-8B` HF checkpoint at `08:45:39 UTC`.
- The model shards loaded fully, and the trainer reached `Tracing train_step for jaxpr...` at `08:47:17 UTC`.
- All workers completed JAX tracing in about `104.7s` and lowered `train_step` to HLO in about `1.5s` by `08:49:07 UTC`.
- No OOM, exit 137, or XLA crash has appeared during startup.
- One new warning is unrelated to model memory: Levanter's config artifact dump cannot encode `save_interval=None`, so W&B skips logging the YAML config artifact. Training continues despite that warning.

## Hypothesis 3

`save_interval=None` is the right runtime setting for `v2b`, but it introduces a benign config-serialization warning in Levanter's tracker path.

## Changes to make

- Do not change the runtime behavior yet; the warning is non-fatal and the priority is to observe train-step stability.
- If `v2b` proves stable through compilation and early steps, consider a follow-up fix so Levanter can serialize an absent checkpoint interval cleanly.

## Future Work

- [ ] If `v2b` still OOMs, capture whether it happens during training, permanent checkpoint save, or some unrelated controller/scheduler event.
- [ ] If `v2b` is stable but still slow, inspect whether MFU improves after removing time-based checkpoint overhead.
- [ ] If needed, test a second variant with a different permanent checkpoint cadence or tensor-parallel layout.
