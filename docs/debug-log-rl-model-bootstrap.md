# Debugging log for RL model bootstrap

Remove launcher-time Hugging Face reads from RL experiments by making the RL step depend on the executor-managed regional model artifact.

## Initial status

The root Iris launcher for `/ahmed/iris-rl-oom-b120-uc1-r2` failed before submitting nested RL jobs because `make_rl_step()` called `AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")`, which hit Hugging Face and got HTTP 429.

The repo already has an executor-managed model download DAG in `experiments/models.py`. For Llama 3.1 8B Instruct, that step resolves to a stable region-local artifact such as `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`.

## Hypothesis 1

RL step construction is doing model config resolution too early. If model bootstrap moves into the executor step runtime and the step config references the downloaded model artifact, the executor will add the proper dependency edge and cached per-region model downloads will be reused.

## Changes to make

- Refactor `lib/marin/src/marin/rl/rl_experiment_utils.py` so `make_rl_step()` builds a runtime config with a resolved model artifact input rather than loading HF config in the launcher.
- Update RL experiment `ModelConfig` to point at executor-managed model artifacts.
- Update RL experiments to use `experiments.models.llama_3_1_8b_instruct`.

## Hypothesis 2

Even after launcher-time config loading is removed, RL worker runtime still assumes tokenizer/checkpoint strings are HF repo IDs. For `gs://.../hf` artifact paths, tokenizer and checkpoint loading need to use the Levanter helpers that support fsspec-backed paths, and HF-checkpoint detection must recognize local/remote HF exports.

## Changes to make

- Replace RL tokenizer loading with `levanter.compat.hf_checkpoints.load_tokenizer`.
- Teach `marin.rl.model_utils.is_hf_checkpoint` to recognize HF exports on `gs://` or local paths by checking for HF files.
- Update vLLM inference context tokenizer loading to the same helper.

## Future Work

- [ ] If vLLM itself rejects `gs://` model paths, add a local-first normalization layer at the vLLM boundary.
- [ ] Add a dedicated regression test for RL runs depending on cached model download steps across regions.

## Results

Pending.
