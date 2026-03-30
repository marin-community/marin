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

## Future Work

- [ ] If `v2b` still OOMs, capture whether it happens during training, permanent checkpoint save, or some unrelated controller/scheduler event.
- [ ] If `v2b` is stable but still slow, inspect whether MFU improves after removing time-based checkpoint overhead.
- [ ] If needed, test a second variant with a different permanent checkpoint cadence or tensor-parallel layout.
