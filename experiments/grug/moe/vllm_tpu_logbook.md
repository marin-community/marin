# GrugMoE Native JAX TPU Logbook

Issue: https://github.com/marin-community/marin/issues/6106

## Branches

- Marin: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/marin/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-support`
  - Current local base before this native-JAX evidence update: [`4a909352`](https://github.com/marin-community/marin/commit/4a909352b27a1eedebc3556bf35bc9712394486c)
  - Native-JAX parity harness/docs commit used for TPU validation: [`9095972d`](https://github.com/marin-community/marin/commit/9095972d843652a22f925057a93498f04a6f3b1a)
  - Native-JAX strengthened parity harness commit used for the 2026-06-03 TPU validation: [`1394236f`](https://github.com/marin-community/marin/commit/1394236f0d1bfa71d931397872e522557b237c78)
  - Canonical inference artifact roundtrip harness commit used for the 2026-06-03 TPU validation: [`c3442afe`](https://github.com/marin-community/marin/commit/c3442afe3ccfb6703623d6a679d7398f5789cd73)
  - Normal Levanter export path roundtrip commit used for the 2026-06-03 TPU validation: [`14577de9`](https://github.com/marin-community/marin/commit/14577de9f460cb727f8b3a6ed2232051aed1359a)
  - Sharded Levanter/HF export smoke commit used for the 2026-06-03 TPU validation: [`f895c61f`](https://github.com/marin-community/marin/commit/f895c61f8546b9847fdb31316e04d2c6d351aa83)
  - Realistic full-canary training-state roundtrip commit used for the 2026-06-03 TPU validation: [`e2a4a4bf`](https://github.com/marin-community/marin/commit/e2a4a4bfb71ebeb56a09cc24c01cca18aa117678)
  - Deterministic full-forward generation parity commit used for the 2026-06-03 TPU validation: [`923ff3f2`](https://github.com/marin-community/marin/commit/923ff3f2b96b1938444de4ab05f9b4c984f0dc43)
- vLLM: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/vllm/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm`
  - Native-JAX replacement commit: [`d025e46d`](https://github.com/marin-community/vllm/commit/d025e46d3dfe0afc0f5bd1518c19ce337205db2d)
- tpu-inference: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/tpu-inference/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference`
  - Native-JAX implementation commit: [`4f83d210`](https://github.com/marin-community/tpu-inference/commit/4f83d2109ae650de7d7e4f521154fb0d5c1a23a1)
  - Canonical inference artifact loader commit: [`e42d7339`](https://github.com/marin-community/tpu-inference/commit/e42d7339e2c84b29b4c302b22483678b3862ab4e)
  - Standard sharded HF safetensors loader commit: [`c0d472c`](https://github.com/marin-community/tpu-inference/commit/c0d472c6c1ab085156767375d534c3272fbfd120)

## Milestones

- 2026-06-02: Refreshed `/home/romain/dev/marin` with `git -C /home/romain/dev/marin pull origin main`.
- 2026-06-02: Created the Marin worktree and cloned vLLM/tpu-inference on matching branches.
- 2026-06-02: Opened Marin experiment issue #6106 with `experiment` and `agent-generated` labels.
- 2026-06-02: Built and validated a PyTorch-first vLLM prototype, then superseded it after deciding the model owner should be native tpu-inference JAX.
- 2026-06-02: Removed the vLLM `GrugMoeForCausalLM` PyTorch model, vLLM GrugMoE test, and vLLM text-generation registry entry.
- 2026-06-02: Added native `tpu_inference.models.jax.grugmoe.GrugMoeForCausalLM`.
- 2026-06-02: Updated tpu-inference `MODEL_IMPL_TYPE=auto` resolution so `GrugMoeForCausalLM` goes through `flax_nnx`, not vLLM.
- 2026-06-02: Replaced the Marin parity harness so it imports native tpu-inference JAX GrugMoE, uses Levanter `moe_mlp` for component parity, and compares composed hidden states against a dense Levanter-parameter reference.
- 2026-06-02: Preserved the correctness-first scope: tiny seeded configs, component parity, composed parity, and targeted `v6e-4` TPU validation.
- 2026-06-02: Ran `/romain/grugmoe-native-jax-tpu-parity` on `v6e-4`; component and composed parity both passed and the job reached `succeeded`.
- 2026-06-03: Strengthened the checkpointless native JAX harness to use a 4-layer tiny config with sliding-window branches `[2, 2, 2, 4]`, still manually copy Levanter parameters into the native tpu-inference model, and compare final hidden states, `compute_logits` logits, and routed expert IDs.
- 2026-06-03: Ran `/romain/grugmoe-native-jax-strengthened-parity-sha` on `v6e-4`; component parity and strengthened full parity both passed and the job reached `succeeded`.
- 2026-06-03: Added a small canonical GrugMoE inference artifact boundary: `config.json` plus `model.safetensors` with stable HF/vLLM-style tensor names and checkpoint-oriented linear tensor layout.
- 2026-06-03: Added native tpu-inference JAX loading for that artifact without treating Levanter's training checkpoint tree as the serving contract.
- 2026-06-03: Extended the Marin parity harness with a seeded export/load roundtrip: initialize tiny Levanter GrugMoE, export the canonical artifact, load it into a second native JAX model, compare against the manual-copy native model for hidden states, `compute_logits` logits, and routed expert IDs, and assert that all exported tensors were consumed with no missing or unexpected tensors.
- 2026-06-03: Ran `/romain/grugmoe-inference-artifact-roundtrip` on `v6e-4`; component parity, strengthened full parity, and canonical artifact roundtrip all passed and the job reached `succeeded`.
- 2026-06-03: Moved canonical GrugMoE artifact export into the normal Levanter/HF path: `GrugModelConfig.to_hf_config`, `Transformer.to_state_dict`, `HFCheckpointConverter.save_pretrained`, `export_lm_to_hf`, and Marin `convert_checkpoint_to_hf_step`.
- 2026-06-03: Added `checkpoint_subpath` to the Levanter/Marin conversion path so standard LM checkpoints still default to `model` while Grug training-state checkpoints can export from `params`.
- 2026-06-03: Replaced the harness-owned safetensors writer with a saved-checkpoint roundtrip: save tiny GrugMoE `params`, export via `export_lm_to_hf`, load in native tpu-inference JAX, and compare fixed-prompt hidden states, `compute_logits` logits, and routed expert IDs against the manual-copy model.
- 2026-06-03: Ran `/romain/grugmoe-export-path-roundtrip-v2` on `v6e-4`; component parity, strengthened full parity, and saved-checkpoint Levanter export roundtrip all passed and the job reached `succeeded`.
- 2026-06-03: Added standard HF sharded safetensors support to the native tpu-inference GrugMoE loader: `model.safetensors.index.json` plus `model-*-of-*.safetensors`, while preserving the existing single-file `model.safetensors` path and canonical tensor names.
- 2026-06-03: Added a Levanter/HF `max_shard_size` export knob through `export_lm_to_hf` and Marin `convert_checkpoint_to_hf_step`, defaulting to the previous Levanter/HF shard size.
- 2026-06-03: Extended the parity harness to force a tiny 1024-byte sharded HF export and verify hidden states, `compute_logits` logits, and routed expert IDs against the manual-copy Levanter reference.
- 2026-06-03: Added an opt-in capped-size large smoke that creates a zero-weight GrugMoE Levanter checkpoint, exports standard sharded safetensors, and verifies the native tpu-inference loader consumes the artifact.
- 2026-06-03: Ran `/romain/grugmoe-sharded-export-large-smoke` on `v6e-4`; single-file export parity, forced tiny sharded export parity, and the 1.25GB sharded loader smoke all passed and the job reached `succeeded`.
- 2026-06-03: Added a realistic roundtrip mode that builds a seeded non-zero `GrugTrainState` with `initial_state`, saves the full training-state checkpoint, exports through `export_lm_to_hf` with `checkpoint_subpath="params"`, requires standard sharded HF safetensors, and compares loaded native tpu-inference hidden states, logits, and routed expert IDs against the Levanter/manual-copy reference.
- 2026-06-03: Ran a scaled local realistic sanity check preserving the production structure: 64 experts, top-4 routing, shared expert, GQA, and short/long sliding-window layer pattern.
- 2026-06-03: Ran `/romain/grugmoe-canary-training-state-roundtrip` on `v6e-4` with the full `GRUG_MOE_TRIAL_MODEL` canary config; the real training-state checkpoint export produced 26 standard HF safetensors shards, native tpu-inference consumed all 217 canonical tensors with no missing/unexpected tensors, and hidden states, logits, and routed expert IDs matched the Levanter/manual-copy reference.
- 2026-06-03: Added deterministic greedy generation parity to the realistic roundtrip: no sampling, three full-forward generated tokens, Levanter/manual-copy reference comparison at each step, and loaded native tpu-inference artifact comparison at each step.
- 2026-06-03: Re-ran the scaled local realistic sanity check with generation; the sharded loaded artifact generated `[858, 3205, 1165]` and matched Levanter logits across all three steps.
- 2026-06-03: Ran `/romain/grugmoe-canary-generation-parity` on `v6e-4`; the full canary loaded sharded artifact generated `[57524, 45040, 67859]`, matched Levanter generation logits across all three full-forward steps, preserved hidden-state/logit/expert-routing parity, and consumed all 217 canonical tensors with no missing/unexpected tensors.

## Local Verification

Commands used for local verification. The vLLM removal check first passed on
2026-06-02; the focused pytest and Marin parity harness were re-run after the
canonical artifact roundtrip update on 2026-06-03.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
rg -n "GrugMoe|GrugMoE|grugmoe|grug_moe" .
```

Result: no matches; `rg` exited `1`, which is the expected no-match status.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
uv run --no-project python -m py_compile \
  vllm/model_executor/models/registry.py \
  tests/models/registry.py
```

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference
python -m py_compile \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/jax/test_grugmoe.py
uv run --no-project --with ruff ruff check \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/jax/test_grugmoe.py
JAX_PLATFORMS=cpu \
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference:/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run --no-project \
  --with-requirements requirements.txt \
  --with-requirements /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m pytest \
    tests/models/jax/test_grugmoe.py \
    tests/models/common/test_model_loader.py::TestGetModel::test_get_model_auto_resolves_to_flax_nnx_for_grug_moe \
    -q
```

Result on 2026-06-03: focused pytest `2 passed, 25 warnings in 10.87s`.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
python -m py_compile \
  experiments/grug/moe/model.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  lib/levanter/src/levanter/compat/hf_checkpoints.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
uv run --with ruff ruff check \
  experiments/grug/moe/model.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  lib/levanter/src/levanter/compat/hf_checkpoints.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
./infra/pre-commit.py \
  experiments/grug/moe/model.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  lib/levanter/src/levanter/compat/hf_checkpoints.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
uv run --with pytest --with pytest-xdist pytest lib/levanter/tests/test_export_to_hf.py -q
JAX_PLATFORMS=cpu \
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference:/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run \
  --with-requirements ../grugmoe-vllm-tpu-inference/requirements.txt \
  --with-requirements ../grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m experiments.grug.moe.vllm_tpu_parity \
  --tpu-inference-root ../grugmoe-vllm-tpu-inference
```

Final parity output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
```

Additional 2026-06-03 focused results:

- `./infra/pre-commit.py ...`: `OK`
- `uv run --with ruff ruff check ...`: `All checks passed!`
- `uv run --with pytest --with pytest-xdist pytest lib/levanter/tests/test_export_to_hf.py -q`: `1 passed, 14 warnings in 18.19s`

Additional 2026-06-03 sharded-export results:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference
uv run python -m py_compile \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/jax/test_grugmoe.py
uv run --with ruff ruff check \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/jax/test_grugmoe.py
JAX_PLATFORMS=cpu \
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference:/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run --no-project \
  --with-requirements requirements.txt \
  --with-requirements /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m pytest tests/models/jax/test_grugmoe.py -q
JAX_PLATFORMS=cpu \
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference:/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run --no-project \
  --with-requirements requirements.txt \
  --with-requirements /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m pytest \
    tests/models/common/test_model_loader.py::TestGetModel::test_get_model_auto_resolves_to_flax_nnx_for_grug_moe \
    -q
```

Results:

- `ruff`: `All checks passed!`
- `tests/models/jax/test_grugmoe.py`: `3 passed, 11 warnings in 3.89s`
- `test_get_model_auto_resolves_to_flax_nnx_for_grug_moe`: `1 passed, 20 warnings in 21.65s`

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run python -m py_compile \
  experiments/grug/moe/vllm_tpu_parity.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
uv run --with ruff ruff check \
  experiments/grug/moe/vllm_tpu_parity.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
./infra/pre-commit.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
uv run --with pytest --with pytest-xdist pytest lib/levanter/tests/test_export_to_hf.py -q
JAX_PLATFORMS=cpu \
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference:/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run \
  --with-requirements ../grugmoe-vllm-tpu-inference/requirements.txt \
  --with-requirements ../grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m experiments.grug.moe.vllm_tpu_parity \
  --tpu-inference-root ../grugmoe-vllm-tpu-inference
```

Results:

- `ruff`: `All checks passed!`
- `./infra/pre-commit.py ...`: `OK`
- `lib/levanter/tests/test_export_to_hf.py`: `1 passed, 14 warnings in 22.02s`
- Local parity harness final output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact-single-file: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
artifact-sharded: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
```

The forced tiny local sharded export used `max_shard_size=1024`, produced 50 shard files, totaled about 109.63KB, and intentionally did not emit a top-level `model.safetensors`.

Additional 2026-06-03 realistic training-state roundtrip local sanity check:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run python -m py_compile experiments/grug/moe/vllm_tpu_parity.py
uv run --with ruff ruff check experiments/grug/moe/vllm_tpu_parity.py
./infra/pre-commit.py experiments/grug/moe/vllm_tpu_parity.py
JAX_PLATFORMS=cpu \
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference:/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run \
  --with-requirements ../grugmoe-vllm-tpu-inference/requirements.txt \
  --with-requirements ../grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m experiments.grug.moe.vllm_tpu_parity \
  --tpu-inference-root ../grugmoe-vllm-tpu-inference \
  --component-only \
  --realistic-roundtrip \
  --realistic-config scaled \
  --realistic-max-shard-size 16777216 \
  --realistic-generation-tokens 3
```

Results:

- `py_compile`: passed
- `ruff`: `All checks passed!`
- `./infra/pre-commit.py experiments/grug/moe/vllm_tpu_parity.py`: `OK`
- Scaled config: `vocab_size=4096`, `hidden_dim=256`, `intermediate_dim=128`, `shared_expert_intermediate_dim=256`, `num_experts=64`, `num_experts_per_token=4`, `num_layers=4`, `num_heads=4`, `num_kv_heads=1`, `head_dim=64`, `max_seq_len=256`, `sliding_window=128`, `qk_mult=1.3`.
- Dtype policy: `params=float32,compute=bfloat16,output=bfloat16`; native parity forward used `float32`.
- Output root: `/tmp/grugmoe-realistic-roundtrip-anye_xyy`
- Training-state checkpoint: `/tmp/grugmoe-realistic-roundtrip-anye_xyy/checkpoints`, `117,807,669` bytes.
- Sharded HF artifact: `/tmp/grugmoe-realistic-roundtrip-anye_xyy/grugmoe-inference`, `117,747,718` bytes, 9 shards, `max_shard_size=16,777,216`, no top-level `model.safetensors`.
- Tensor accounting: `expected_tensors=84`, `consumed_tensors=84`, `missing=[]`, `unexpected=[]`.
- Greedy full-forward generation: `3` new tokens, no sampling, no KV cache; generated IDs `[858, 3205, 1165]`.
- Final output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts
realistic-generation: manual-copy native reference full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-generation: prompt_ids=[1, 42, 128, 2048, 17, 3072, 5, 63] generated_ids=[858, 3205, 1165] final_token_ids=[1, 42, 128, 2048, 17, 3072, 5, 63, 858, 3205, 1165]
realistic-generation: loaded native artifact full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs
realistic-roundtrip: checkpoint_dir=/tmp/grugmoe-realistic-roundtrip-anye_xyy/checkpoints checkpoint_bytes=117807669 artifact_dir=/tmp/grugmoe-realistic-roundtrip-anye_xyy/grugmoe-inference artifact_bytes=117747718 shard_count=9 max_shard_size=16777216 expected_tensors=84 consumed_tensors=84 missing=[] unexpected=[] generation_tokens=3 generated_ids=[858, 3205, 1165]
```

## TPU Verification

Status: passed for the normal Levanter export path roundtrip branch heads listed below.

Exact validation SHAs:

- Marin: `14577de9f460cb727f8b3a6ed2232051aed1359a`
- tpu-inference: `e42d7339e2c84b29b4c302b22483678b3862ab4e`
- vLLM: `d025e46d3dfe0afc0f5bd1518c19ce337205db2d`

Repro command:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --extra marin-core:tpu \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 1800 \
  --cpu 2 \
  --memory 16GB \
  --disk 50GB \
  --job-name grugmoe-export-path-roundtrip-v2 \
  -- bash -lc 'set -euxo pipefail; echo marin_sha=14577de9f460cb727f8b3a6ed2232051aed1359a; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-vllm-tpu-inference; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm; echo tpu_inference_sha=$(git -C /tmp/grugmoe-vllm-tpu-inference rev-parse HEAD); echo vllm_sha=$(git -C /tmp/grugmoe-vllm-tpu-vllm rev-parse HEAD); export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; PYTHONPATH=/tmp/grugmoe-vllm-tpu-inference:/tmp/grugmoe-vllm-tpu-vllm uv run --with-requirements /tmp/grugmoe-vllm-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-vllm-tpu-vllm/requirements/common.txt --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-vllm-tpu-inference'
```

Result:

- Job: `/romain/grugmoe-export-path-roundtrip-v2`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `1 minute and 10.79 seconds`

Remote SHA log lines:

```text
marin_sha=14577de9f460cb727f8b3a6ed2232051aed1359a
tpu_inference_sha=e42d7339e2c84b29b4c302b22483678b3862ab4e
vllm_sha=d025e46d3dfe0afc0f5bd1518c19ce337205db2d
```

Final parity log lines:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
```

Note: `/romain/grugmoe-export-path-roundtrip` failed before validation because the packaged Iris workspace is not a Git checkout, so the initial `git rev-parse HEAD` assertion returned an empty string. The replacement `/romain/grugmoe-export-path-roundtrip-v2` job above removed only that pre-test assertion and is the authoritative TPU validation.

## Sharded Export TPU Verification

Status: passed for the sharded HF safetensors loader and Levanter export branch heads listed below.

Exact validation SHAs:

- Marin: `f895c61f8546b9847fdb31316e04d2c6d351aa83`
- tpu-inference: `c0d472c6c1ab085156767375d534c3272fbfd120`
- vLLM: `d025e46d3dfe0afc0f5bd1518c19ce337205db2d`

Repro command:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --extra marin-core:tpu \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 3600 \
  --cpu 8 \
  --memory 64GB \
  --disk 100GB \
  --job-name grugmoe-sharded-export-large-smoke \
  -- bash -lc 'set -euxo pipefail; echo marin_sha=f895c61f8546b9847fdb31316e04d2c6d351aa83; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-vllm-tpu-inference; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm; echo tpu_inference_sha=$(git -C /tmp/grugmoe-vllm-tpu-inference rev-parse HEAD); echo vllm_sha=$(git -C /tmp/grugmoe-vllm-tpu-vllm rev-parse HEAD); export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; PYTHONPATH=/tmp/grugmoe-vllm-tpu-inference:/tmp/grugmoe-vllm-tpu-vllm uv run --with-requirements /tmp/grugmoe-vllm-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-vllm-tpu-vllm/requirements/common.txt --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-vllm-tpu-inference --large-smoke'
```

Result:

- Job: `/romain/grugmoe-sharded-export-large-smoke`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `2 minutes and 29.62 seconds`

Remote SHA log lines:

```text
marin_sha=f895c61f8546b9847fdb31316e04d2c6d351aa83
tpu_inference_sha=c0d472c6c1ab085156767375d534c3272fbfd120
vllm_sha=d025e46d3dfe0afc0f5bd1518c19ce337205db2d
```

Final validation log lines:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact-single-file: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
Will save 50 shards with max size 4.1 KB
artifact-sharded: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
Checkpoint size -/1250800896 ...
Will save 27 shards with max size 83.89 MB
Saved a sharded checkpoint with 27 shards, max size 83.89 MB
Progress on:Checkpoint size 1.25GB/1.25GB ...
artifact-large-sharded: zero-weight Levanter export loaded in native tpu-inference (1250833717 bytes, 27 shards)
```

Artifact evidence:

- Tiny single-file and forced-sharded artifacts were written to TPU-local temporary directories under `/tmp/tmph9zr4mb4` and `/tmp/tmp8dwngmm8`.
- The capped-size large smoke artifact was written to `/tmp/tmpa8u4tbxf/grugmoe-large-inference`.
- The large smoke artifact was `1,250,833,717` bytes across 27 standard HF safetensors shards.
- These paths were TPU-local temporary directories and are not persistent artifact locations.
- No checkpoint or artifact was copied to local disk or the Hugging Face Hub.

## Realistic Full-Canary Training-State TPU Verification

Status: passed for the full `GRUG_MOE_TRIAL_MODEL` canary config using a seeded non-zero `GrugTrainState` checkpoint.

Exact validation SHAs:

- Marin: `e2a4a4bfb71ebeb56a09cc24c01cca18aa117678`
- tpu-inference: `c0d472c6c1ab085156767375d534c3272fbfd120`
- vLLM: `d025e46d3dfe0afc0f5bd1518c19ce337205db2d`

Submitted command:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --extra marin-core:tpu \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 7200 \
  --cpu 16 \
  --memory 128GB \
  --disk 200GB \
  --job-name grugmoe-canary-training-state-roundtrip \
  -- bash -lc 'set -euxo pipefail; echo marin_sha=$(git rev-parse HEAD); git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-vllm-tpu-inference; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm; echo tpu_inference_sha=$(git -C /tmp/grugmoe-vllm-tpu-inference rev-parse HEAD); echo vllm_sha=$(git -C /tmp/grugmoe-vllm-tpu-vllm rev-parse HEAD); export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; PYTHONPATH=/tmp/grugmoe-vllm-tpu-inference:/tmp/grugmoe-vllm-tpu-vllm uv run --with-requirements /tmp/grugmoe-vllm-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-vllm-tpu-vllm/requirements/common.txt --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-vllm-tpu-inference --component-only --realistic-roundtrip --realistic-config canary --realistic-output-dir /tmp/grugmoe-canary-roundtrip --realistic-max-shard-size 268435456'
```

Note: the submitted command tried to print `marin_sha=$(git rev-parse HEAD)` inside the packaged Iris workspace, but that workspace is not a Git checkout, so the remote log line printed `marin_sha=`. The authoritative Marin SHA is the pushed local branch head listed above; the remote harness output confirms the full-canary code path from that branch was executed.

Result:

- Job: `/romain/grugmoe-canary-training-state-roundtrip`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `9 minutes and 6.46 seconds`
- Resources: `cpu=16`, `memory=128GB`, `disk=200GB`

Config and dtype evidence:

- Config source: `GRUG_MOE_TRIAL_MODEL`
- Config: `vocab_size=128256`, `hidden_dim=1024`, `intermediate_dim=512`, `shared_expert_intermediate_dim=1024`, `num_experts=64`, `num_experts_per_token=4`, `num_layers=11`, `num_heads=8`, `num_kv_heads=2`, `head_dim=128` inferred, `max_seq_len=4096`, `sliding_window=4096`, `initializer_std=0.015625`, `qk_mult=1.3`.
- Layer window pattern: short window `2048` on layers not congruent to `3 mod 4`; long window `4096` on every fourth layer.
- Dtype policy: `params=float32,compute=bfloat16,output=bfloat16`.
- Native parity forward dtype: `float32`.
- Fixed prompt IDs: `[1, 42, 128, 2048, 17, 3072, 5, 63]`.

Artifact and tensor evidence:

- Checkpoint layout: real `GrugTrainState` from `initial_state`, exported with `checkpoint_subpath="params"`.
- Training-state checkpoint: `/tmp/grugmoe-canary-roundtrip/checkpoints`, `5,762,369,168` bytes.
- HF artifact: `/tmp/grugmoe-canary-roundtrip/grugmoe-inference`, `5,762,168,484` bytes.
- HF artifact layout: `model.safetensors.index.json` plus `model-00001-of-00026.safetensors` through `model-00026-of-00026.safetensors`; single-file `model.safetensors` was rejected by the harness for this milestone.
- Requested `max_shard_size`: `268,435,456` bytes.
- Actual max shard logged by Levanter: `525.34 MB`, because the embedding/lm-head tensors are individually larger than the requested shard size and HF sharding keeps individual tensors intact.
- Shard count: `26`.
- Tensor accounting: `expected_tensors=217`, `consumed_tensors=217`, `missing=[]`, `unexpected=[]`.

Final validation log lines:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts
Will save 26 shards with max size 525.34 MB
Saved a sharded checkpoint with 26 shards, max size 525.34 MB
Progress on:Checkpoint size 5.76GB/5.76GB ...
realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs
realistic-roundtrip: checkpoint_dir=/tmp/grugmoe-canary-roundtrip/checkpoints checkpoint_bytes=5762369168 artifact_dir=/tmp/grugmoe-canary-roundtrip/grugmoe-inference artifact_bytes=5762168484 shard_count=26 max_shard_size=268435456 expected_tensors=217 consumed_tensors=217 missing=[] unexpected=[]
```

Artifact persistence: the checkpoint and HF artifact were left in TPU-local `/tmp` for the duration of the job. They were not copied to local disk, GCS, or the Hugging Face Hub.

Additional 2026-06-03 full-canary deterministic generation parity:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --extra marin-core:tpu \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 7200 \
  --cpu 16 \
  --memory 128GB \
  --disk 200GB \
  --job-name grugmoe-canary-generation-parity \
  -- bash -lc 'set -euxo pipefail; echo marin_sha=923ff3f2b96b1938444de4ab05f9b4c984f0dc43; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-vllm-tpu-inference; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm; echo tpu_inference_sha=$(git -C /tmp/grugmoe-vllm-tpu-inference rev-parse HEAD); echo vllm_sha=$(git -C /tmp/grugmoe-vllm-tpu-vllm rev-parse HEAD); export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; PYTHONPATH=/tmp/grugmoe-vllm-tpu-inference:/tmp/grugmoe-vllm-tpu-vllm uv run --with-requirements /tmp/grugmoe-vllm-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-vllm-tpu-vllm/requirements/common.txt --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-vllm-tpu-inference --component-only --realistic-roundtrip --realistic-config canary --realistic-output-dir /tmp/grugmoe-canary-generation-parity --realistic-max-shard-size 268435456 --realistic-generation-tokens 3'
```

Result:

- Job: `/romain/grugmoe-canary-generation-parity`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `11 minutes and 8.96 seconds`
- Remote SHAs:
  - Marin branch head: `923ff3f2b96b1938444de4ab05f9b4c984f0dc43`
  - `tpu_inference_sha=c0d472c6c1ab085156767375d534c3272fbfd120`
  - `vllm_sha=d025e46d3dfe0afc0f5bd1518c19ce337205db2d`
- Config source: `GRUG_MOE_TRIAL_MODEL`.
- Dtype policy: `params=float32,compute=bfloat16,output=bfloat16`; native parity forward used `float32`.
- Fixed prompt IDs: `[1, 42, 128, 2048, 17, 3072, 5, 63]`.
- Greedy full-forward generation: `3` new tokens, no sampling, no KV cache; generated IDs `[57524, 45040, 67859]`.
- Training-state checkpoint: `/tmp/grugmoe-canary-generation-parity/checkpoints`, `5,762,306,365` bytes.
- HF artifact: `/tmp/grugmoe-canary-generation-parity/grugmoe-inference`, `5,762,168,484` bytes.
- HF artifact layout: `model.safetensors.index.json` plus 26 shard files; single-file `model.safetensors` remains rejected by the harness for the realistic milestone.
- Requested `max_shard_size`: `268,435,456` bytes.
- Actual max shard logged by Levanter: `525.34 MB`, because the embedding/lm-head tensors are individually larger than the requested shard size and HF sharding keeps individual tensors intact.
- Tensor accounting: `expected_tensors=217`, `consumed_tensors=217`, `missing=[]`, `unexpected=[]`.

Final validation log lines:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts
realistic-generation: manual-copy native reference full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-generation: prompt_ids=[1, 42, 128, 2048, 17, 3072, 5, 63] generated_ids=[57524, 45040, 67859] final_token_ids=[1, 42, 128, 2048, 17, 3072, 5, 63, 57524, 45040, 67859]
Will save 26 shards with max size 525.34 MB
Saved a sharded checkpoint with 26 shards, max size 525.34 MB
Progress on:Checkpoint size 5.76GB/5.76GB ...
realistic-generation: loaded native artifact full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs
realistic-roundtrip: checkpoint_dir=/tmp/grugmoe-canary-generation-parity/checkpoints checkpoint_bytes=5762306365 artifact_dir=/tmp/grugmoe-canary-generation-parity/grugmoe-inference artifact_bytes=5762168484 shard_count=26 max_shard_size=268435456 expected_tensors=217 consumed_tensors=217 missing=[] unexpected=[] generation_tokens=3 generated_ids=[57524, 45040, 67859]
```

## Fork Stack Smoke 2026-06-07

Purpose: validate the GrugMoE native TPU path on the Marin-owned fork stack while
Marin moves from the opaque `vllm-tpu` package path to explicit
`marin-community/vllm` plus `marin-community/tpu-inference` git pins.

Validation branch heads:

- Marin smoke branch: `codex/grugmoe-fork-stack-smoke-20260607`
  - TPU validation bundle commit, before this logbook update: `55c2d0d7ace48be373995e973d6d9b0b1dc59e71`
  - Fork-stack base: `origin/romain/marin-vllm-tpu-deponly-20260605`
  - Root `pyproject.toml` pins:
    - `vllm=54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
    - `tpu-inference=22a5fcccf542cf1b77d71e1db495be4ddae01bac`
- vLLM smoke branch: `codex/grugmoe-fork-stack-smoke-20260607`
  - `54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
  - This branch is a reproducibility pointer to `origin/romain/tpu-vllm-lkg-v0.20.0-candidate`; no extra Grug code was needed in vLLM.
- tpu-inference smoke branch: `codex/grugmoe-fork-stack-smoke-20260607`
  - `13474e22a97739b44d795d43253e8663885aa8f7`
  - Includes the GrugMoE native JAX commits plus a small `JaxLmHead` bridge needed by the fork-stack base.

Local dependency and unit gates:

- `VLLM_TARGET_DEVICE=tpu uv lock --check`: passed after repinning and lock regeneration.
- `VLLM_TARGET_DEVICE=tpu uv export --locked --package marin-core --extra vllm --extra eval --no-dev --no-emit-project --no-emit-workspace`: passed, `3409` requirement lines.
- `VLLM_TARGET_DEVICE=tpu uv export --locked --package marin-core --extra vllm --extra evalchemy --no-dev --no-emit-project --no-emit-workspace`: passed, `3600` requirement lines.
- `VLLM_TARGET_DEVICE=tpu uv export --locked --package marin-core --extra gpu --no-dev --no-emit-project --no-emit-workspace`: passed, `2498` requirement lines.
- `rg -n "^(vllm|vllm-tpu|tpu-inference)(==| @|$)" /tmp/marin-gpu-requirements.txt`: no matches, expected.
- `VLLM_TARGET_DEVICE=tpu uv export --locked --package marin-core --extra vllm --extra tpu --no-dev --no-emit-project --no-emit-workspace`: failed as expected because `marin-core[vllm]` and `marin-core[tpu]` intentionally conflict.
- Focused Marin dependency tests:
  `tests/rl/test_orchestration.py`,
  `tests/evals/test_inference_proxy.py::test_brokered_vllm_workers_use_self_contained_vllm_extra`,
  and `tests/evals/test_tpu_vllm_dependency_groups.py`: `13 passed in 6.07s`.
- tpu-inference Grug focused pytest with the fork-stack source checkouts:
  `tests/models/jax/test_grugmoe.py` plus
  `tests/models/common/test_model_loader.py::TestGetModel::test_get_model_auto_resolves_to_flax_nnx_for_grug_moe`:
  `4 passed, 23 warnings in 6.87s`.
- Marin Grug py_compile and ruff checks passed for
  `experiments/grug/moe/model.py`,
  `experiments/grug/moe/vllm_tpu_parity.py`,
  `lib/levanter/src/levanter/compat/hf_checkpoints.py`,
  `lib/levanter/src/levanter/main/export_lm_to_hf.py`, and
  `lib/marin/src/marin/export/levanter_checkpoint.py`.
- tpu-inference py_compile and ruff checks passed for
  `tpu_inference/layers/jax/linear.py`,
  `tpu_inference/models/jax/grugmoe.py`,
  `tests/models/jax/test_grugmoe.py`, and
  `tests/models/common/test_model_loader.py`.

Local scaled GrugMoE roundtrip:

```bash
cd /home/romain/dev/marin-wt/grugmoe-fork-stack-smoke
JAX_PLATFORMS=cpu \
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-fork-stack-smoke-tpu-inference:/home/romain/dev/marin-wt/grugmoe-fork-stack-smoke-vllm \
VLLM_TARGET_DEVICE=tpu \
uv run \
  --with-requirements /home/romain/dev/marin-wt/grugmoe-fork-stack-smoke-tpu-inference/requirements.txt \
  --with-requirements /home/romain/dev/marin-wt/grugmoe-fork-stack-smoke-vllm/requirements/common.txt \
  --with 'torch==2.11.0+cpu' \
  --with 'torchvision==0.26.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m experiments.grug.moe.vllm_tpu_parity \
    --tpu-inference-root /home/romain/dev/marin-wt/grugmoe-fork-stack-smoke-tpu-inference \
    --component-only \
    --realistic-roundtrip \
    --realistic-config scaled \
    --realistic-max-shard-size 16777216 \
    --realistic-generation-tokens 3
```

Result:

- Component parity passed.
- Config source: scaled production-structured fallback.
- Checkpoint: `/tmp/grugmoe-realistic-roundtrip-gb0sxfx3/checkpoints`, `117,800,671` bytes.
- HF artifact: `/tmp/grugmoe-realistic-roundtrip-gb0sxfx3/grugmoe-inference`, `117,748,760` bytes.
- Shards: `9`, requested `max_shard_size=16,777,216`.
- Tensor accounting: `expected_tensors=84`, `consumed_tensors=84`, `missing=[]`, `unexpected=[]`.
- Greedy full-forward generation: `3` new tokens, generated IDs `[858, 3205, 1165]`.

Final local output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts
realistic-generation: manual-copy native reference full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-generation: prompt_ids=[1, 42, 128, 2048, 17, 3072, 5, 63] generated_ids=[858, 3205, 1165] final_token_ids=[1, 42, 128, 2048, 17, 3072, 5, 63, 858, 3205, 1165]
realistic-generation: loaded native artifact full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs
realistic-roundtrip: checkpoint_dir=/tmp/grugmoe-realistic-roundtrip-gb0sxfx3/checkpoints checkpoint_bytes=117800671 artifact_dir=/tmp/grugmoe-realistic-roundtrip-gb0sxfx3/grugmoe-inference artifact_bytes=117748760 shard_count=9 max_shard_size=16777216 expected_tensors=84 consumed_tensors=84 missing=[] unexpected=[] generation_tokens=3 generated_ids=[858, 3205, 1165]
```

Full GrugMoE canary TPU validation:

```bash
cd /home/romain/dev/marin-wt/grugmoe-fork-stack-smoke
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --extra marin-core:tpu \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 7200 \
  --cpu 16 \
  --memory 128GB \
  --disk 200GB \
  --job-name grugmoe-fork-stack-canary-20260607 \
  -e VLLM_TARGET_DEVICE tpu \
  -- bash -lc 'set -euxo pipefail; echo marin_sha=55c2d0d7ace48be373995e973d6d9b0b1dc59e71; git clone --depth 1 --branch codex/grugmoe-fork-stack-smoke-20260607 https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-fork-stack-tpu-inference; git clone --depth 1 --branch codex/grugmoe-fork-stack-smoke-20260607 https://github.com/marin-community/vllm.git /tmp/grugmoe-fork-stack-vllm; echo tpu_inference_sha=$(git -C /tmp/grugmoe-fork-stack-tpu-inference rev-parse HEAD); echo vllm_sha=$(git -C /tmp/grugmoe-fork-stack-vllm rev-parse HEAD); export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; export VLLM_TARGET_DEVICE=tpu; PYTHONPATH=/tmp/grugmoe-fork-stack-tpu-inference:/tmp/grugmoe-fork-stack-vllm uv run --with-requirements /tmp/grugmoe-fork-stack-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-fork-stack-vllm/requirements/common.txt --with "torch==2.11.0+cpu" --with "torchvision==0.26.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-fork-stack-tpu-inference --component-only --realistic-roundtrip --realistic-config canary --realistic-output-dir /tmp/grugmoe-fork-stack-canary-20260607 --realistic-max-shard-size 268435456 --realistic-generation-tokens 3'
```

Result:

- Job: `/romain/grugmoe-fork-stack-canary-20260607`
- Dashboard: `https://iris.oa.dev/#/job/%2Fromain%2Fgrugmoe-fork-stack-canary-20260607`
- TPU: `v6e-4`
- Region: `europe-west4`
- Worker: `marin-tpu-v6e-preemptible-4-europe-west4-20260606-1908-edbc06d1-worker-0`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `701,917ms`
- Remote SHAs:
  - `marin_sha=55c2d0d7ace48be373995e973d6d9b0b1dc59e71`
  - `tpu_inference_sha=13474e22a97739b44d795d43253e8663885aa8f7`
  - `vllm_sha=54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
- Config source: `GRUG_MOE_TRIAL_MODEL`.
- Dtype policy: `params=float32,compute=bfloat16,output=bfloat16`; native parity forward used `float32`.
- Fixed prompt IDs: `[1, 42, 128, 2048, 17, 3072, 5, 63]`.
- Greedy full-forward generation: `3` new tokens, no sampling, no KV cache; generated IDs `[57524, 45040, 67859]`.
- Training-state checkpoint: `/tmp/grugmoe-fork-stack-canary-20260607/checkpoints`, `5,762,423,241` bytes.
- HF artifact: `/tmp/grugmoe-fork-stack-canary-20260607/grugmoe-inference`, `5,762,169,526` bytes.
- HF artifact layout: `model.safetensors.index.json` plus `26` shard files.
- Requested `max_shard_size`: `268,435,456` bytes.
- Actual max shard logged by Levanter: `525.34 MB`, because the embedding/lm-head tensors are individually larger than the requested shard size and HF sharding keeps individual tensors intact.
- Tensor accounting: `expected_tensors=217`, `consumed_tensors=217`, `missing=[]`, `unexpected=[]`.

Final TPU validation log lines:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts
realistic-generation: manual-copy native reference full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-generation: prompt_ids=[1, 42, 128, 2048, 17, 3072, 5, 63] generated_ids=[57524, 45040, 67859] final_token_ids=[1, 42, 128, 2048, 17, 3072, 5, 63, 57524, 45040, 67859]
Will save 26 shards with max size 525.34 MB
Saved a sharded checkpoint with 26 shards, max size 525.34 MB
realistic-generation: loaded native artifact full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs
realistic-roundtrip: checkpoint_dir=/tmp/grugmoe-fork-stack-canary-20260607/checkpoints checkpoint_bytes=5762423241 artifact_dir=/tmp/grugmoe-fork-stack-canary-20260607/grugmoe-inference artifact_bytes=5762169526 shard_count=26 max_shard_size=268435456 expected_tensors=217 consumed_tensors=217 missing=[] unexpected=[] generation_tokens=3 generated_ids=[57524, 45040, 67859]
```

Companion direct vLLM TPU smoke:

```bash
cd /home/romain/dev/marin-wt/grugmoe-fork-stack-smoke
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --extra marin-core:vllm \
  --extra marin-core:eval \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 3600 \
  --cpu 16 \
  --memory 128GB \
  --disk 200GB \
  --job-name vllm-fork-stack-direct-smoke-20260607 \
  -e VLLM_TARGET_DEVICE tpu \
  -- python -c 'import os; os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=98304"; import importlib.metadata as md, json; from marin.evaluation.evaluators.evaluator import ModelConfig; from marin.inference.vllm_server import resolve_model_name_or_path; from vllm import LLM, SamplingParams; cfg = ModelConfig(name="test-llama-200m", path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m", engine_kwargs={"enforce_eager": True, "max_model_len": 1024, "max_num_batched_tokens": 1024}, generation_params={"max_tokens": 16}); model, cfg = resolve_model_name_or_path(cfg); print("vllm_version=" + md.version("vllm")); print("model=" + model); print("engine_kwargs=" + json.dumps(cfg.engine_kwargs, sort_keys=True)); outputs = LLM(model=model, **cfg.engine_kwargs).generate(["Write a short haiku about TPUs."], SamplingParams(max_tokens=16, temperature=0.0)); print("generated=" + repr([(o.outputs[0].token_ids, o.outputs[0].text) for o in outputs]))'
```

Result:

- Job: `/romain/vllm-fork-stack-direct-smoke-20260607`
- Dashboard: `https://iris.oa.dev/#/job/%2Fromain%2Fvllm-fork-stack-direct-smoke-20260607`
- TPU: `v6e-4`
- Region: `europe-west4`
- Worker: `marin-tpu-v6e-preemptible-4-europe-west4-20260606-2003-f15d49ff-worker-0`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `112,478ms`
- Installed vLLM version: `0.20.1rc1.dev148+g54a6eb69d.tpu`.
- Model: `gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m`.
- Engine kwargs:
  `{"enforce_eager": true, "load_format": "runai_streamer", "max_model_len": 1024, "max_num_batched_tokens": 1024}`.
- vLLM resolved `MODEL_IMPL_TYPE=auto` to `flax_nnx`.
- RunAI streamer loaded `147/147` safetensors and streamed `757.1 MiB`.
- KV cache evidence: maximum concurrency for `1,024` tokens per request was logged as `908.00x`.
- Generated token IDs:
  `[1102, 596, 1949, 323, 499, 649, 1005, 433, 311, 3350, 264, 33894, 922, 4205, 13, 1102]`.
- Generated text:
  `" It's free and you can use it to write a poem about anything. It"`

Sharp edges from this fork-stack run:

- Keep `VLLM_TARGET_DEVICE=tpu` set for `uv lock`, `uv export`, Iris `uv sync`, and any source-build metadata path that can see vLLM. Without it, git-source vLLM can take the CUDA metadata/build path.
- `marin-core[vllm]` and `marin-core[tpu]` intentionally conflict. For a vLLM TPU job, request TPU hardware with Iris `--tpu v6e-4` but install the Python stack with `--extra marin-core:vllm`; do not also install `marin-core:tpu`.
- The Grug native canary used `--extra marin-core:tpu` plus explicit `PYTHONPATH` source checkouts because the Grug model owner is native tpu-inference JAX, not vLLM's text-generation registry.
- The fork-stack tpu-inference base did not include `JaxLmHead`; the smoke branch adds that narrow bridge instead of pulling broad upstream changes.
- The old Grug support command's `torch==2.10.0+cpu` is stale for this fork stack. The focused Grug source-checkout commands need `torch==2.11.0+cpu` and `torchvision==0.26.0+cpu`.
- Do not use the fork-stack vLLM `requirements/cpu.txt` alongside tpu-inference `requirements.txt` for Grug tests; it conflicts on `numba==0.65.0` versus tpu-inference's `numba==0.62.1`. Use vLLM `requirements/common.txt` for this source-checkout harness.
- Importing vLLM from a source checkout via `PYTHONPATH` emits `No module named 'vllm._version'` and "vLLM package was not found" warnings. The installed `marin-core:vllm` direct smoke did not rely on that checkout path and reported the expected `.tpu` version.
- `uv run --no-project` inside the tpu-inference checkout can create a tiny local `uv.lock`; remove it before committing.
- Iris log fetching tails by default in practice. Use `iris job logs --no-tail` when the early setup/version lines matter.
- The direct vLLM smoke used a known `us-east5` GCS model path from a `europe-west4` TPU. It is useful evidence but still a cross-region read, so prefer an `europe-west4` model artifact for future repeated validation.
- Remote `uv` warned about hardlink fallback because cache and target directories are on different filesystems. This was harmless; set `UV_LINK_MODE=copy` only if the warning noise matters.
- The installed direct-vLLM smoke logs `No module named 'vllm._C'` warnings on TPU. They did not block model load or generation in this run.

## Installed Fork-Stack GrugMoE Smoke 2026-06-07

Purpose: close the remaining evidence gap from the fork-stack smoke by testing
GrugMoE through the installed/user-facing Marin dependency path rather than a
`PYTHONPATH` source-checkout harness.

Validation branch and pins:

- Marin branch: `codex/grugmoe-installed-path-smoke-20260607`
  - Worktree: `/home/romain/dev/marin-wt/grugmoe-installed-path-smoke`
  - Base: `origin/codex/grugmoe-fork-stack-smoke-20260607`
  - Repin commit: `31d4e0dd96a129b0d36b4c9f7a421d4ff5a053da`
- vLLM pin: `54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
- tpu-inference pin before this smoke: `22a5fcccf542cf1b77d71e1db495be4ddae01bac`
- tpu-inference pin tested here: `13474e22a97739b44d795d43253e8663885aa8f7`

Setup commands:

```bash
git -C /home/romain/dev/marin pull origin main
git -C /home/romain/dev/marin worktree add \
  /home/romain/dev/marin-wt/grugmoe-installed-path-smoke \
  -b codex/grugmoe-installed-path-smoke-20260607 \
  origin/codex/grugmoe-fork-stack-smoke-20260607
cd /home/romain/dev/marin-wt/grugmoe-installed-path-smoke
VLLM_TARGET_DEVICE=tpu uv lock
```

Local installed dependency checks:

```bash
cd /home/romain/dev/marin-wt/grugmoe-installed-path-smoke
VLLM_TARGET_DEVICE=tpu uv lock --check
VLLM_TARGET_DEVICE=tpu uv export \
  --locked \
  --package marin-core \
  --extra vllm \
  --extra eval \
  --no-dev \
  --no-emit-project \
  --no-emit-workspace \
  --output-file /tmp/marin-grug-installed-vllm-eval-requirements.txt
rg -n "^(tpu-inference|vllm|jax|jaxlib|libtpu)(==| @)" \
  /tmp/marin-grug-installed-vllm-eval-requirements.txt
VLLM_TARGET_DEVICE=tpu uv run \
  --locked \
  --package marin-core \
  --extra vllm \
  python -c 'import importlib.util; import importlib.metadata as md; print(md.version("vllm")); print(importlib.util.find_spec("tpu_inference.models.jax.grugmoe"))'
```

Results:

- Before the tpu-inference repin, installed `marin-core[vllm]` resolved
  `tpu-inference==0.0.0` but `grugmoe_spec=None` and `has_JaxLmHead=False`.
- After the repin, installed `marin-core[vllm]` resolved:
  - `tpu-inference @ git+https://github.com/marin-community/tpu-inference.git@13474e22a97739b44d795d43253e8663885aa8f7`
  - `vllm @ git+https://github.com/marin-community/vllm.git@54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
  - `jax==0.9.2`, `jaxlib==0.9.2`, and `libtpu==0.0.39`
- The post-repin installed package exposed
  `tpu_inference.models.jax.grugmoe` from `site-packages` and
  `JaxLmHead` from `tpu_inference.layers.jax.linear`.
- A local tiny Grug HF config artifact with
  `architectures=["GrugMoeForCausalLM"]` showed the same model-resolution
  boundary as TPU:
  - plain `AutoConfig.from_pretrained(...)` rejects `model_type="grug_moe"`;
  - after runtime registering a `PretrainedConfig` subclass for `grug_moe`,
    vLLM accepts the config object;
  - vLLM `ModelConfig` rejects the architecture because
    `GrugMoeForCausalLM` is not in the vLLM model registry.

Authoritative TPU command shape:

```bash
cd /home/romain/dev/marin-wt/grugmoe-installed-path-smoke
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 3600 \
  --cpu 16 \
  --memory 128GB \
  --disk 200GB \
  --job-name grugmoe-installed-vllm-path-uvrun5-20260607 \
  -e VLLM_TARGET_DEVICE tpu \
  -- bash -lc 'set -euo pipefail; export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; VLLM_TARGET_DEVICE=tpu uv run --locked --package marin-core --extra vllm --extra eval python - <<'"'"'PY'"'"'
# Python script:
# - print Marin, vLLM, and tpu-inference SHAs;
# - import installed JAX/vLLM/tpu-inference on the TPU worker;
# - print direct_url.json for tpu-inference and vLLM;
# - write a temporary HF-style Grug config artifact;
# - verify unregistered Transformers rejects grug_moe;
# - runtime-register GrugMoeHfConfig with Transformers and vLLM config utils;
# - call vllm.LLM(model=artifact_dir, runner="generate", skip_tokenizer_init=True, ...).
PY'
```

Authoritative TPU result:

- Job: `/romain/grugmoe-installed-vllm-path-uvrun5-20260607`
- Dashboard: `https://iris.oa.dev/#/job/%2Fromain%2Fgrugmoe-installed-vllm-path-uvrun5-20260607`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `41.63 seconds`
- Remote printed SHAs:
  - `marin_sha=31d4e0dd96a129b0d36b4c9f7a421d4ff5a053da`
  - `vllm_pin=54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
  - `tpu_inference_pin=13474e22a97739b44d795d43253e8663885aa8f7`
- Runtime package evidence:
  - `remote_cwd=/app`
  - `jax_version=0.9.2`
  - `jax_devices=[TpuDevice(...), TpuDevice(...), TpuDevice(...), TpuDevice(...)]`
  - `vllm_version=0.20.1rc1.dev148+g54a6eb69d`
  - `tpu_inference_version=0.0.0`
  - `grugmoe_spec=ModuleSpec(... origin='/app/.venv/lib/python3.11/site-packages/tpu_inference/models/jax/grugmoe.py')`
  - `has_JaxLmHead=True`
  - `tpu-inference_direct_url` commit
    `13474e22a97739b44d795d43253e8663885aa8f7`
  - `vllm_direct_url` commit
    `54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
- Config boundary:
  - `unregistered_autoconfig=ValueError:The checkpoint you are trying to load has model type 'grug_moe' but Transformers does not recognize this architecture.`
  - `registered_vllm_config=GrugMoeHfConfig:['GrugMoeForCausalLM']`
- Installed vLLM path result:
  - `vllm.LLM(...)` reached `ModelConfig` and failed with:
    `Value error, Model architectures ['GrugMoeForCausalLM'] are not supported for now.`
  - Final marker:
    `installed_path_result=blocked:vllm_registry_missing_GrugMoeForCausalLM`

Conclusion:

- Installed `marin-core[vllm]` can include the Grug-capable tpu-inference
  branch if Marin's tpu-inference pin is moved to
  `13474e22a97739b44d795d43253e8663885aa8f7`.
- GrugMoE is not currently usable through the installed fork-stack
  `vllm.LLM(...).generate(...)` path.
- The exact blocker is the vLLM model registry/model-resolution gap:
  `GrugMoeForCausalLM` is present in the artifact config and can be made known
  to Transformers/vLLM config loading, but the installed vLLM registry does not
  have a model implementation registered for that architecture.

Sharp edges from this installed-path smoke:

- Iris job-level `--extra marin-core:vllm` and `--extra vllm` syncs did not
  materialize `jax` for the direct user command in this workflow; those early
  jobs failed with `ModuleNotFoundError: No module named 'jax'`.
- The reliable installed-path command was an explicit in-job
  `VLLM_TARGET_DEVICE=tpu uv run --locked --package marin-core --extra vllm --extra eval ...`.
- Remote `uv run` executed under `/app`; do not assume the user command is in a
  Git checkout. Print known SHAs or package `direct_url.json` rather than
  running `git rev-parse HEAD` inside the job.
- This vLLM fork's `LLM` constructor uses `runner="generate"`; `task="generate"`
  fails before model resolution.
- `JaxLmHead` is exported from `tpu_inference.layers.jax.linear`, not from a
  `tpu_inference.layers.jax.lm_head` module.
- The TPU vLLM startup still logs `No module named 'vllm._C'`, Triton-driver,
  duplicate op-registration, and `os.fork()`/JAX warnings. They did not block
  reaching the model registry failure.

## Scope

In scope:

- Native tpu-inference JAX model wiring.
- QB routing: biased top-k selection and unbiased sigmoid combine weights.
- Tiny deterministic component parity against Levanter `moe_mlp`.
- Tiny deterministic composed parity using Levanter parameters/equations, including final hidden states, `compute_logits` logits, and routed expert IDs.
- Tiny deterministic canonical inference artifact export/load roundtrip.
- Tiny deterministic saved-checkpoint export through Levanter `export_lm_to_hf` and Marin-compatible `checkpoint_subpath="params"`.
- Standard HF sharded safetensors loading in native tpu-inference: `model.safetensors.index.json` plus `model-*-of-*.safetensors`.
- Tiny deterministic forced-shard Levanter/HF export roundtrip.
- Capped-size 1.25GB zero-weight sharded Levanter/HF export smoke.
- Full `GRUG_MOE_TRIAL_MODEL` local seeded non-zero training-state checkpoint roundtrip.
- Deterministic full-forward greedy generation parity for the loaded native tpu-inference artifact path.
- Strict artifact tensor accounting: all exported tensors consumed, no missing tensors, no unexpected tensors.
- Targeted Iris `v6e-4` validation.

Out of scope:

- Performance tuning.
- Fused TPU kernels.
- vLLM PyTorch GrugMoE ownership.
- Direct loading of Levanter's native training checkpoint tree as the serving contract.
- Production generation serving and KV-cache decode.
- External trained-checkpoint smoke testing remains out of scope for this validation; the completed milestone uses a seeded non-zero local `GrugTrainState` checkpoint with the full canary config.
