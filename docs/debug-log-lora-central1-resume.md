# Debugging log for lora central1 resume

Resume the `endlboq3` LoRA-DPO run on `us-central1-a` / `v5p-8` without breaking the LoRA resume path.

## Initial status

The relaunch job `/ahmed/lora-bsv2-mi-b0p1-s2-lr7p5e6-b64-v5p8-central1-resume-20260330-2217` failed with:

`FileNotFoundError: Missing 12 arrays in OCDBT checkpoint`

The missing arrays were base-model weights like:

- `model/transformer/layers/stacked/self_attn/q_proj/weight`
- `model/transformer/layers/stacked/mlp/down_proj/weight`
- `model/embeddings/token_embeddings/weight`
- `model/lm_head/weight`

The checkpoint contents shown in the error were mostly LoRA tensors and optimizer state.

## Hypothesis 1

The relaunch used the wrong combination of resume flags.

Specifically, it passed both:

- `--trainer.initialize_from <step-835>`
- `--initialize_from_checkpoint_path <step-835>`

For LoRA DPO, the trainer checkpoint stores trainable adapter weights and optimizer state, not a full frozen base model. Passing that same path to `initialize_from_checkpoint_path` causes the LoRA resume reconstruction path to treat the trainer checkpoint like a full model checkpoint and then fail when base weights are absent.

## Changes to make

No code change is required for this root cause.

Relaunch with:

- the central1 `b64` config
- `RUN_ID=endlboq3`
- `--trainer.initialize_from <east5 step-835>`

Do **not** pass:

- `--initialize_from_checkpoint_path`

Leave `initialize_from_hf` pointing at the central1 HF base model so the existing LoRA resume fix can reconstruct:

- base model from HF
- resumed adapter weights from trainer checkpoint

## Results

The failing job logs show:

- `levanter.trainer Initializing from gs://.../step-835`
- then an OCDBT error listing only LoRA arrays and optimizer state in the checkpoint
- missing full-model arrays for the base model

This matches the launch-misconfiguration hypothesis exactly.

Root cause: the launch command was wrong, not the checkpoint contents and not worker corruption.

## Future Work

- [ ] Relaunch with `trainer.initialize_from` only
- [ ] Confirm logs show `Resuming from step 836` and no OCDBT missing-array error
