# Debugging log for LoRA checkpoint resume

Investigate whether LoRA checkpoint save/load is broken for DPO resume, with emphasis on the failing `endlboq3` run described in `.agents/logbooks/lora_ckpt.md`.

## Initial status

The failing run resumes from a LoRA-only checkpoint at step 799 and then reports eval loss around `0.686`, close to the untrained DPO baseline. The existing investigation already established:

- the checkpoint contains non-zero LoRA arrays
- the expected LoRA tensor keys are present
- the missing "LoRA" leaves reported by debug logging are non-array leaves (`dropout.inference`, `bias=None`) that are not supposed to be serialized

Two high-value questions remain:

1. Is the generic save/load path for LoRA-only trainer checkpoints structurally correct?
2. If the checkpoint is structurally fine, is resumed DPO eval accidentally reading the wrong model view?

## Hypothesis 1

The LoRA save/load mask is structurally wrong and resumed trainer state silently drops or mis-merges LoRA adapter weights.

## Changes to make

- Inspect `lora_trainable_params_filter`, `saveable_training_mask`, and `load_checkpoint_or_initialize`.
- Run a direct local save -> load -> output-parity check through the real trainer checkpoint path.
- Check whether `TrainerState.eval_model` preserves loaded LoRA weights after resume.

## Future Work

- [ ] Add a regression test for LoRA checkpoint round-trip output parity if the current code path proves correct.
- [ ] Remove temporary debug logging from `lib/levanter/src/levanter/checkpoint.py` after the investigation is done.
- [ ] If save/load is correct, instrument the resumed DPO eval path to compare policy vs reference norms directly.

## Results

- `lora_trainable_params_filter` stops recursion at `LowRankLinear`, not `LoraLinear`. That means the filter targets the adapter subtree (`lora_A`, `lora_B`) but not the wrapped base linear weights.
- This matches the existing checkpoint-debug evidence from the failing run: `42` LoRA arrays loaded for `21` adapted modules, which is consistent with `lora_A.weight` + `lora_B.weight` only.
- A direct local round-trip through `TrainerState.saveable_state` and `load_checkpoint_or_initialize` showed that the loaded LoRA tensors are preserved, but the resumed model output matches a fresh base-model initialization rather than the original pre-save model output when the base model differs.

## Hypothesis 2

LoRA checkpoint serialization is fine; the unified DPO resume path is reconstructing the wrong base model and then overlaying the checkpointed LoRA tensors onto it.

## Changes to make

- Compare the old LoRA-DPO / LoRA-LM initialization path to the unified `train_dpo.py` path.
- Patch `train_dpo.py` so adapter resumes rebuild the policy from the configured source before reusing checkpointed adapter weights.
- Add a regression test that simulates the bad state (`wrong base + right LoRA`) and verifies the repaired resume path recovers the original model output.

## Results

- The old LoRA-DPO path loaded the pretrained base model before calling `trainer.initial_state`. The unified `train_dpo.py` path does not: it builds a fresh policy model, loads only the checkpointed trainable leaves into that tree, and only consults `initialize_from_hf` / `initialize_from_checkpoint_path` on fresh starts (`state.step == 0`).
- For LoRA checkpoints this is wrong by construction because the checkpoint only stores the adapter weights. On resume, the base model must be reconstructed from the configured source before the loaded adapter weights are reused.
- I reproduced the failure locally with a tiny model:
  - save a LoRA-only trainer checkpoint
  - reload it into a different base-model initialization
  - observe that the resumed output matches `wrong base + loaded LoRA`, not the original model
- I patched `lib/levanter/src/levanter/main/train_dpo.py` so adapter resumes rebuild the policy model from source and then call `_restore_policy_model_from_partial_checkpoint(...)` to overlay the loaded adapter weights onto the correct base.
- I removed the temporary debug logging from `lib/levanter/src/levanter/checkpoint.py`.
- Verification:
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_dpo.py lib/levanter/tests/test_lora_dpo.py lib/levanter/tests/test_checkpoint.py`
  - result: `48 passed, 1 skipped`
