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

## Installed GrugMoE vLLM Registry Fix 2026-06-07

Purpose: fix the narrow installed-path registry gap from the previous smoke so
`vllm.LLM(...).generate(...)` can resolve `GrugMoeForCausalLM` to the native
tpu-inference JAX/Flax NNX implementation without restoring the removed
PyTorch-first vLLM Grug model.

Branches and commits:

- tpu-inference branch: `codex/grugmoe-vllm-registry-20260607`
  - Worktree:
    `/home/romain/dev/marin-wt/grugmoe-vllm-registry-tpu-inference`
  - Base: `13474e22a97739b44d795d43253e8663885aa8f7`
  - Registry commit: `71e5f24656c413c798579f064e31119a0c9ace1a`
  - Alias commit tested on TPU:
    `1730991a92bbd1169ff79e3cc67321b7db281adb`
- Marin branch: `codex/grugmoe-installed-vllm-registry-20260607`
  - Worktree:
    `/home/romain/dev/marin-wt/grugmoe-installed-vllm-registry`
  - Base:
    `origin/codex/grugmoe-installed-path-smoke-20260607` at
    `1af6bb934704562adc927083d9c5c6ce591d189e`
  - First pin commit:
    `e79784123b7d5f09972a53fc445f9ccb39d8e84e6`
  - Final tested pin commit:
    `2d3ec227b0059210546ccedc54193dfce9537b93`
- vLLM pin: `54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`

tpu-inference fix:

- Added `GrugMoeHfConfig`, a small `PretrainedConfig` adapter for
  `model_type="grug_moe"`.
- Added TPU-gated registration in `tpu_inference.layers.vllm.register_layers`:
  - register `grug_moe` with Transformers `AutoConfig`;
  - register `grug_moe` with vLLM's config registry;
  - register `GrugMoeForCausalLM` with vLLM's model registry, backed by the
    native tpu-inference implementation.
- Added HF/vLLM config aliases in `GrugMoeHfConfig` for fields such as
  `hidden_size`, `intermediate_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, and
  `max_position_embeddings`.

Setup commands:

```bash
git -C /home/romain/dev/marin pull origin main
git -C /home/romain/dev/marin worktree add \
  /home/romain/dev/marin-wt/grugmoe-installed-vllm-registry \
  -b codex/grugmoe-installed-vllm-registry-20260607 \
  origin/codex/grugmoe-installed-path-smoke-20260607
git clone https://github.com/marin-community/tpu-inference.git \
  /home/romain/dev/marin-wt/grugmoe-vllm-registry-tpu-inference
cd /home/romain/dev/marin-wt/grugmoe-vllm-registry-tpu-inference
git checkout -b codex/grugmoe-vllm-registry-20260607 \
  13474e22a97739b44d795d43253e8663885aa8f7
```

Local validation commands:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-registry-tpu-inference
pre-commit run ruff --files \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/common/test_model_loader.py
pre-commit run isort --files \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/common/test_model_loader.py
python -m py_compile \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/common/test_model_loader.py

cd /home/romain/dev/marin-wt/grugmoe-installed-vllm-registry
VLLM_TARGET_DEVICE=tpu uv lock
VLLM_TARGET_DEVICE=tpu uv lock --check
VLLM_TARGET_DEVICE=tpu uv run --locked --package marin-core --extra vllm python - <<'PY'
# Python script:
# - assert PYTHONPATH is absent;
# - print direct_url.json for installed vLLM and tpu-inference;
# - import tpu_inference.layers.vllm.register_layers from site-packages;
# - write a tiny Grug config;
# - verify AutoConfig returns GrugMoeHfConfig;
# - verify vLLM ModelConfig reports hidden/head size 8;
# - verify ModelRegistry resolves GrugMoeForCausalLM as text generation.
PY
./infra/pre-commit.py pyproject.toml uv.lock
git diff --check
```

Local result:

- Installed package path, no `PYTHONPATH` source checkout:
  `/home/romain/dev/marin-wt/grugmoe-installed-vllm-registry/.venv/lib/python3.11/site-packages/tpu_inference`.
- `tpu-inference_direct_url` commit:
  `1730991a92bbd1169ff79e3cc67321b7db281adb`.
- `vllm_direct_url` commit:
  `54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`.
- `autoconfig_class=GrugMoeHfConfig`.
- `model_config_hidden_size=8`.
- `model_config_head_size=8`.
- `registry_impl_name=GrugMoeForCausalLM`.
- `registry_text_generation=True`.

Authoritative TPU command shape:

```bash
cd /home/romain/dev/marin-wt/grugmoe-installed-vllm-registry
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
  --job-name grugmoe-installed-vllm-registry-real4-20260607 \
  -e VLLM_TARGET_DEVICE tpu \
  -- bash -lc 'set -euo pipefail; export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; export PYTHONUNBUFFERED=1; VLLM_TARGET_DEVICE=tpu uv run --locked --package marin-core --extra vllm --extra eval python - <<'"'"'PY'"'"'
# Python script:
# - print remote cwd and direct_url.json package SHAs;
# - import installed vLLM/tpu-inference/JAX/libtpu from site-packages;
# - call tpu_inference.layers.vllm.register_layers();
# - write a tiny HF-style Grug artifact with config.json and model.safetensors;
# - use vocab_size=32 and untied lm_head.weight;
# - verify AutoConfig, ModelConfig, and ModelRegistry resolution;
# - call vllm.LLM(..., runner="generate", skip_tokenizer_init=True, ...);
# - generate from prompt_token_ids=[1, 2, 3].
PY'
```

Authoritative TPU result:

- Job: `/romain/grugmoe-installed-vllm-registry-real4-20260607`
- Dashboard:
  `https://iris.oa.dev/#/job/%2Fromain%2Fgrugmoe-installed-vllm-registry-real4-20260607`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `59.5 seconds`
- Remote printed SHAs:
  - `marin_sha=2d3ec227b0059210546ccedc54193dfce9537b93`
  - `vllm_pin=54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`
  - `tpu_inference_pin=1730991a92bbd1169ff79e3cc67321b7db281adb`
- Runtime package evidence:
  - `remote_cwd=/app`
  - `vllm_version=0.20.1rc1.dev148+g54a6eb69d.tpu`
  - `tpu-inference_version=0.0.0`
  - `jax_version=0.9.2`
  - `libtpu_version=0.0.39`
  - `grugmoe_spec=ModuleSpec(... origin='/app/.venv/lib/python3.11/site-packages/tpu_inference/models/jax/grugmoe.py')`
  - `artifact_tensor_count=27`
  - `artifact_bytes=43821`
- Resolution evidence:
  - `registered_grug_config=<class 'tpu_inference.models.jax.grugmoe.GrugMoeHfConfig'>`
  - `autoconfig_class=GrugMoeHfConfig`
  - `autoconfig_architectures=['GrugMoeForCausalLM']`
  - `hidden_size_alias=8`
  - `model_config_hidden_size=8`
  - `model_config_head_size=8`
  - `registry_impl_name=GrugMoeForCausalLM`
  - `registry_text_generation=True`
  - vLLM logged `Resolved architecture: GrugMoeForCausalLM`
  - tpu-inference logged `Resolved MODEL_IMPL_TYPE 'auto' to 'flax_nnx'`
- Installed generation result:
  - `llm_initialized=True`
  - `generated=[([1, 2, 3], [0], '')]`
  - `installed_path_result=works:real_artifact_generate`

Conclusion:

- GrugMoE now works through the installed
  `marin-core[vllm]` / `vllm.LLM(...).generate(...)` path on TPU for a tiny
  HF-style Grug artifact.
- The fix stays in tpu-inference ownership and does not restore a vLLM
  PyTorch-first Grug model.
- The Marin dependency pin that was tested successfully is
  `tpu-inference@1730991a92bbd1169ff79e3cc67321b7db281adb` with
  `vllm@54a6eb69daafc23b72dd1bc3c78d097b7f4cd997`.
- No remaining registry/resolution blocker was observed in this validation.

Sharp edges from this registry-fix smoke:

- Do not call `jax.devices()` in the parent process before constructing
  `vllm.LLM`; the first attempt reached engine initialization and then hung
  around the JAX/fork boundary.
- Tiny local artifacts need `vocab_size >= 20`; vLLM/tpu-inference
  `gather_logprobs` precompile calls top-k with `k=20`.
- vLLM reads HF-style config aliases such as `hidden_size` and
  `num_attention_heads`; a Grug-only config with only `hidden_dim` and
  `num_heads` can reach model init but fail later with hidden size `0`.
- Keep `runner="generate"` and `skip_tokenizer_init=True` for this tokenizerless
  smoke.
- The expected TPU startup warnings remain: missing `vllm._C`, Triton disabled,
  duplicate op registration, and JAX buffer-donation warnings. They did not
  block loading or generation.

Intermediate TPU attempts:

- `/romain/grugmoe-installed-vllm-registry-real-20260607` was stopped after it
  reached registry/model resolution and then hung because the script initialized
  JAX in the parent before vLLM forked its engine.
- `/romain/grugmoe-installed-vllm-registry-real2-20260607` failed with
  `ValueError: k argument to top_k must be no larger than size along axis`
  because the tiny artifact had `vocab_size=16`.
- `/romain/grugmoe-installed-vllm-registry-real3-20260607` failed after native
  model initialization because HF aliases were missing and vLLM-derived
  architecture metadata reported hidden size `0`; fixed by
  `1730991a92bbd1169ff79e3cc67321b7db281adb`.

## Current PR-Stack Installed Full-Canary TPU Validation 2026-06-15

Objective:

- Validate the GrugMoE TPU integration on top of the current fork-stack PR heads:
  - `marin-community/vllm#6`
  - `marin-community/tpu-inference#5`
  - `marin-community/marin#6288`
- Keep vLLM unchanged unless testing proves otherwise.
- Run the installed/user-facing `marin-core[vllm]` path with no user
  `PYTHONPATH`, `VLLM_TARGET_DEVICE=tpu`, a regenerated full
  `GRUG_MOE_TRIAL_MODEL` HF artifact, and `vllm.LLM(...).generate(...)`.

Current PR heads at audit time:

- vLLM PR #6:
  `c0010021cfc7d4439eab0bf2a6564c38444fa2e3`
- tpu-inference PR #5:
  `22a5fcccf542cf1b77d71e1db495be4ddae01bac`
- Marin PR #6288:
  `b6f0d7a92ab1c42ad3654499bf4c6d8de337860b`

Branches and tested pins:

- tpu-inference worktree:
  `/home/romain/dev/marin-wt/grugmoe-pr5-registry-tpu-inference-20260615`
- tpu-inference branch:
  `codex/grugmoe-pr5-registry-20260615`
- tpu-inference base:
  PR #5 head `22a5fcccf542cf1b77d71e1db495be4ddae01bac`
- tpu-inference tested commit:
  `80bc72a68a216ddf86367db9d93263e0b2a288ce`
- Marin worktree:
  `/home/romain/dev/marin-wt/grugmoe-prstack-installed-vllm-20260615`
- Marin branch:
  `codex/grugmoe-prstack-installed-vllm-20260615`
- Marin base:
  PR #6288 head `b6f0d7a92ab1c42ad3654499bf4c6d8de337860b`
- Marin tested commit before this logbook entry:
  `db0f11fe5`
- vLLM worktree used only for inspection:
  `/home/romain/dev/marin-wt/grugmoe-pr6-vllm-plugins-20260615`
- vLLM branch:
  `codex/grugmoe-pr6-general-plugins-20260615`
- vLLM tested commit:
  PR #6 head `c0010021cfc7d4439eab0bf2a6564c38444fa2e3`
- No vLLM code changes were required. The real `LLM(...)` construction path
  goes through `EngineArgs`, which calls `load_general_plugins()` before model
  config resolution.

tpu-inference rebase/fix:

- Re-applied the GrugMoE TPU registry/config support onto PR #5.
- Added the missing PR #5-era native JAX `grugmoe.py` implementation.
- Adapted the Grug head from the older `JaxLmHead` helper to PR #5's current
  `JaxEinsum` layer.
- Tried and reverted an import-time auto-registration hook because it made a
  vLLM circular import observable during import-only probes. The installed
  `LLM(...)` path does not need that hook.

Marin integration changes:

- `pyproject.toml` and `uv.lock` pin:
  - `tpu-inference`:
    `https://github.com/marin-community/tpu-inference.git@80bc72a68a216ddf86367db9d93263e0b2a288ce`
  - `vllm`:
    `https://github.com/marin-community/vllm.git@c0010021cfc7d4439eab0bf2a6564c38444fa2e3`
- Added `experiments/grug/moe/installed_vllm_full_canary_smoke.py`.
- The final harness keeps the parent process JAX-free, runs full artifact
  generation in one subprocess, waits for that subprocess to exit, and then
  launches the installed vLLM serving smoke in a second subprocess.

Local validation:

```bash
cd /home/romain/dev/marin-wt/grugmoe-pr5-registry-tpu-inference-20260615
pre-commit run ruff --files \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/common/test_model_loader.py
pre-commit run isort --files \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/common/test_model_loader.py
python -m py_compile \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/common/test_model_loader.py

cd /home/romain/dev/marin-wt/grugmoe-prstack-installed-vllm-20260615
VLLM_TARGET_DEVICE=tpu uv lock
VLLM_TARGET_DEVICE=tpu uv lock --check
env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu \
  uv run --locked --package marin-core --extra vllm python - <<'PY'
# Construct EngineArgs and verify installed package direct_url SHAs,
# AutoConfig -> GrugMoeHfConfig, ModelConfig hidden size/head size,
# and vLLM ModelRegistry -> GrugMoeForCausalLM text generation.
PY
env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu \
  /home/romain/dev/marin-wt/grugmoe-prstack-installed-vllm-20260615/.venv/bin/python \
  -m pytest \
  /home/romain/dev/marin-wt/grugmoe-pr5-registry-tpu-inference-20260615/tests/models/common/test_model_loader.py \
  -k grugmoe -q
python -m py_compile \
  experiments/grug/moe/installed_vllm_full_canary_smoke.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  experiments/grug/moe/model.py \
  lib/levanter/src/levanter/compat/hf_checkpoints.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
./infra/pre-commit.py \
  pyproject.toml \
  uv.lock \
  experiments/grug/moe/installed_vllm_full_canary_smoke.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  experiments/grug/moe/vllm_tpu_logbook.md \
  experiments/grug/moe/vllm_tpu_reference.md \
  experiments/grug/moe/model.py \
  lib/levanter/src/levanter/compat/hf_checkpoints.py \
  lib/levanter/src/levanter/main/export_lm_to_hf.py \
  lib/marin/src/marin/export/levanter_checkpoint.py
git diff --check
```

Local result:

- `uv lock --check` passed.
- Installed registry probe passed with no user `PYTHONPATH`.
- Installed direct URL evidence showed:
  - `vllm@c0010021cfc7d4439eab0bf2a6564c38444fa2e3`
  - `tpu-inference@80bc72a68a216ddf86367db9d93263e0b2a288ce`
- `AutoConfig.from_pretrained(...)` returned `GrugMoeHfConfig`.
- vLLM `ModelConfig` reported hidden size `8`.
- `inspect_arch=GrugMoeForCausalLM`.
- `registry_text_generation=True`.
- Focused Grug loader pytest passed:
  `1 passed, 19 deselected`.
- Marin pre-commit and `git diff --check` passed before TPU rerun; this
  section was added after the successful TPU run.

Failed one-process TPU harness attempt:

- Job: `/romain/grugmoe-prstack-full-installed-vllm-20260615`
- Dashboard:
  `https://iris.oa.dev/#/job/%2Fromain%2Fgrugmoe-prstack-full-installed-vllm-20260615`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `failed`
- Exit: `1`
- Failures/preemptions: `1`/`0`
- Duration: `12 minutes and 0.59 seconds`
- Useful evidence before failure:
  - full canary config source was `GRUG_MOE_TRIAL_MODEL`;
  - native/manual-copy parity matched Levanter hidden states, logits, routed
    experts, and full-forward greedy generation;
  - generated reference IDs were `[57524, 45040, 67859]`;
  - HF artifact saved as 26 shards, `5.76GB`;
  - loaded native artifact consumed `217/217` tensors with
    `missing=[]` and `unexpected=[]`;
  - artifact bytes were `5762169526`.
- Blocker:
  - the parent process had initialized JAX/libtpu while generating the
    artifact, then the installed vLLM engine subprocess failed with
    `RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by process with pid 341`.
- Fix:
  - commit `db0f11fe5` changed the harness to run artifact generation and
    installed serving in separate child processes.

Successful two-process TPU command:

```bash
cd /home/romain/dev/marin-wt/grugmoe-prstack-installed-vllm-20260615
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 7200 \
  --cpu 16 \
  --memory 256GB \
  --disk 300GB \
  --job-name grugmoe-prstack-full-installed-vllm-2phase-20260615 \
  -e VLLM_TARGET_DEVICE tpu \
  -- bash -lc 'set -euo pipefail; unset PYTHONPATH; export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; export PYTHONUNBUFFERED=1; export VLLM_TARGET_DEVICE=tpu; VLLM_TARGET_DEVICE=tpu uv run --locked --package marin-core --extra vllm --extra eval python -m experiments.grug.moe.installed_vllm_full_canary_smoke --output-dir /tmp/grugmoe-prstack-full-installed-vllm-2phase-20260615 --max-shard-size 268435456 --generation-tokens 3'
```

Successful two-process TPU result:

- Job: `/romain/grugmoe-prstack-full-installed-vllm-2phase-20260615`
- Dashboard:
  `https://iris.oa.dev/#/job/%2Fromain%2Fgrugmoe-prstack-full-installed-vllm-2phase-20260615`
- TPU: `v6e-4`
- Region: `europe-west4`
- State: `succeeded`
- Exit: `0`
- Failures/preemptions: `0`/`0`
- Duration: `11 minutes and 31.95 seconds`
- Remote package evidence:
  - `remote_cwd=/app`
  - `marin_sha=unavailable` because the Iris workspace is not a git checkout;
    authoritative Marin commit for this run is the pushed branch commit
    `db0f11fe5`;
  - `vllm_direct_url` and `serve_vllm_direct_url`:
    `c0010021cfc7d4439eab0bf2a6564c38444fa2e3`;
  - `tpu-inference_direct_url` and `serve_tpu-inference_direct_url`:
    `80bc72a68a216ddf86367db9d93263e0b2a288ce`;
  - `vllm_version=0.20.1rc1.dev146+gc0010021c.tpu`;
  - `tpu-inference_version=0.0.0`;
  - `jax_version=0.9.2`;
  - `libtpu_version=0.0.39`;
  - `grugmoe_spec` resolved to
    `/app/.venv/lib/python3.11/site-packages/tpu_inference/models/jax/grugmoe.py`.
- Full artifact evidence:
  - `artifact_generation_process=started`;
  - `realistic-roundtrip: config_source=GRUG_MOE_TRIAL_MODEL`;
  - native/manual-copy parity matched Levanter hidden states, logits, routed
    experts, and full-forward greedy generation;
  - reference prompt IDs were `[1, 42, 128, 2048, 17, 3072, 5, 63]`;
  - reference generated IDs were `[57524, 45040, 67859]`;
  - HF export saved 26 shards, `5.76GB`;
  - loaded native artifact consumed all tensors:
    `expected_tensors=217 consumed_tensors=217 missing=[] unexpected=[]`;
  - `checkpoint_bytes=5762310165`;
  - `artifact_bytes=5762169526`;
  - `full_canary_artifact_bytes=5762169526`;
  - `full_canary_shard_count=26`;
  - `artifact_generation_process=completed`.
- Installed vLLM serving evidence:
  - vLLM logged `Registered JAX model GrugMoeForCausalLM with tpu_inference and vLLM registries`;
  - vLLM logged `Resolved architecture: GrugMoeForCausalLM`;
  - tpu-inference logged `Resolved MODEL_IMPL_TYPE 'auto' to 'flax_nnx'`;
  - `llm_initialized=True`;
  - `installed_full_canary_generated=[([1, 42, 128, 2048, 17, 3072, 5, 63], [91542, 58518, 8334], '')]`;
  - `installed_path_result=works:full_canary_generate`.

Conclusion:

- Full GrugMoE now loads and generates through the installed
  `marin-core[vllm]` / `vllm.LLM(...).generate(...)` path on TPU with the
  current PR-stack pins.
- The full canary artifact path also preserves native tpu-inference parity with
  Levanter and consumes all 217 exported tensors.
- The installed vLLM smoke validated serving-path load/generate success, not
  token parity against the non-KV-cache native full-forward reference. The
  installed vLLM generated IDs were `[91542, 58518, 8334]`, while the native
  full-forward reference IDs were `[57524, 45040, 67859]`. Treat vLLM decode
  token parity as a remaining correctness follow-up if exact serving semantics
  are required.

Sharp edges from this PR-stack validation:

- Current PR heads mattered. Earlier branches pinned older vLLM/tpu-inference
  commits and were not representative of the stack that is expected to land.
- PR #5's JAX layer surface no longer had the older `JaxLmHead`; Grug needed a
  current `JaxEinsum` head.
- Do not auto-register Grug from tpu-inference package import time. It can run
  before vLLM finishes initializing and cause a circular import through
  `vllm.platforms.current_platform`.
- Bare `AutoConfig` probes can still fail to know about `grug_moe` unless the
  vLLM general plugin path has loaded. The real `LLM(...)` path works because
  `EngineArgs` loads general plugins before `ModelConfig` is built.
- Do not generate the full artifact and run installed vLLM serving from the
  same process after JAX/libtpu has initialized. Use separate child processes
  or separate jobs; otherwise vLLM's engine subprocess can fail because TPU is
  already owned by the parent.
- Iris runs from `/app`, not a git checkout, so `git rev-parse HEAD` is not
  reliable inside the job. Use pushed branch commits plus direct URL package
  SHAs as the authoritative run identity.
- Expected non-blocking startup warnings include missing `vllm._C`, Triton
  disabled on TPU, duplicate op registration, quantization overwrites, and JAX
  buffer-donation warnings.

## 2026-06-15 - GrugMoE Installed vLLM Logprob/Routing Validation

Branches and fork-stack commits:

- vLLM branch `codex/grugmoe-expert-indices-vllm-20260615`, commit
  `c6f0608ddadb6bdd39a16b857b5affe660b1259e`.
- tpu-inference branch `codex/grugmoe-expert-indices-tpu-20260615`, commit
  `d8c4579d3409d50ccaa4a81df44b0d62c9e9b958`.
- Marin branch `codex/grugmoe-correctness-validation-20260615` pins those
  commits in `pyproject.toml` / `uv.lock`.

Code changes under validation:

- vLLM now has first-class `ModelRunnerOutput.expert_indices` support for
  model runners that return routed experts while
  `enable_return_routed_experts=True`.
- vLLM scheduler accumulates runner-provided routed experts per request and
  still uses the existing GPU `RoutedExpertsReader` path for non-TPU devices.
- tpu-inference no longer monkeypatches vLLM scheduler internals; its TPU
  runner continues to populate `model_runner_output.expert_indices`.
- Marin installed smoke now writes a Levanter reference JSON with fixed prompt,
  fixed continuation, selected-token logprobs, routed experts, and router
  margins. The installed validation exports the full canary artifact, serves it
  through Marin `VllmEnvironment`, scores fixed continuation logprobs through
  installed vLLM, and compares routed experts.

Local checks:

- vLLM: `python -m py_compile ...`; `pre-commit run ruff-check`; `pre-commit
  run ruff-format`; focused scheduler test
  `tests/v1/core/test_scheduler.py::test_model_runner_expert_indices_are_returned_when_request_finishes`.
- tpu-inference: `python -m py_compile`; `pre-commit run ruff`; `pre-commit
  run isort`; no remaining references to the old scheduler monkeypatch helper.
- Marin: `VLLM_TARGET_DEVICE=tpu uv lock --check`; `./infra/pre-commit.py
  experiments/grug/moe/vllm_tpu_parity.py
  experiments/grug/moe/installed_vllm_full_canary_smoke.py pyproject.toml
  uv.lock`; `python -m py_compile`; installed-environment CLI smoke via
  `env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu uv run --locked --package
  marin-core --extra vllm --extra eval python -m
  experiments.grug.moe.installed_vllm_full_canary_smoke --help`.

TPU jobs and sharp edges:

- `/romain/grugmoe-correctness-installed-vllm-20260615-190509`: failed in
  score because TPU vLLM returned `prompt_logprobs` length `1` for an 11-token
  scored prompt. Adjusted harness to score fixed continuation targets through
  one-token completion logprobs instead of prompt logprobs.
- `/romain/grugmoe-correctness-installed-vllm-20260615-192504`: failed because
  TPU sample logprobs ignored `logprob_token_ids`; the returned dict contained
  only the sampled token (`available=[91542]`) and not target `57524`.
- `/romain/grugmoe-correctness-installed-vllm-20260615-193932`: failed because
  `SamplingParams(logprobs=-1)` produced an empty completion logprob list on
  TPU. Adjusted to set `max_logprobs=vocab_size` and request
  `logprobs=vocab_size`.
- `/romain/grugmoe-correctness-installed-vllm-20260615-195315`: produced the
  desired logprob and routing summaries, then failed on suspicious routing
  mismatches before the serve phase. Reordered all-phase validation to serve
  before score so serving evidence is retained when routing fails.
- `/romain/grugmoe-correctness-installed-vllm-20260615-200749`: final
  all-phase run. It served/generated successfully, then failed as intended on
  routed-expert mismatches beyond the `0.01` router-margin tolerance.

Final TPU evidence from
`/romain/grugmoe-correctness-installed-vllm-20260615-200749`:

- Full canary config source: `GRUG_MOE_TRIAL_MODEL`.
- Levanter/native full-forward generated continuation:
  `[57524, 45040, 67859]`.
- Reference continuation logprobs:
  `[-10.659404754638672, -10.7232666015625, -10.627751350402832]`.
- Sharded export/load:
  `artifact_bytes=5762169526`, `full_canary_shard_count=26`,
  `expected_tensors=217 consumed_tensors=217 missing=[] unexpected=[]`.
- Installed Marin serve smoke:
  `vllm_server_initialized=True`, `vllm_generate_status_code=200`,
  `installed_full_canary_generated=[([1, 42, 128, 2048, 17, 3072, 5, 63],
  [91542, 58518, 8334], '')]`,
  `installed_path_result=works:full_canary_generate`.
- Installed vLLM selected-token logprobs for the fixed continuation were
  collected with full-vocab completion logprobs:
  `score_full_vocab_logprobs=128256`,
  `score_continuation_generated_token_ids=[91542, 93785, 83484]`,
  `max_abs_delta=0.21515655517578125`,
  `mean_abs_delta=0.1790914535522461`, `max_allowed_abs_delta=5.0`.
- Per-token logprob deltas:
  - token `57524` at position `8`: Levanter `-10.659404754638672`, vLLM
    `-10.849434852600098`, delta `0.19003009796142578`;
  - token `45040` at position `9`: Levanter `-10.7232666015625`, vLLM
    `-10.855354309082031`, delta `0.13208770751953125`;
  - token `67859` at position `10`: Levanter `-10.627751350402832`, vLLM
    `-10.842907905578613`, delta `0.21515655517578125`.
- Routed-expert diagnostic:
  `token_layer_count=121`, `top_k=4`, `ordered_topk_match_count=8`,
  `ordered_topk_match_rate=0.06611570247933884`,
  `unordered_full_match_count=20`,
  `unordered_full_match_rate=0.1652892561983471`,
  `mean_unordered_overlap=0.7086776859504132`,
  `top1_match_count=81`, `top1_match_rate=0.6694214876033058`,
  `low_margin_boundary_mismatch_count=30`,
  `suspicious_mismatch_count=71`.

Conclusion:

- The vLLM/tpu-inference scheduler monkeypatch can be removed: TPU routed
  experts are now carried through the first-class runner output path.
- Installed `marin-core[vllm]` can export, serve, and generate the full GrugMoE
  canary on v6e-4 with the fork-stack pins.
- Fixed-continuation selected-token logprobs are close under the intentionally
  loose correctness threshold.
- Routed experts diverge substantially, including many mismatches with router
  boundary margins greater than `0.01`; this is not only low-margin drift. The
  validation intentionally fails after recording the serve and logprob evidence.
- Routing replay is still out of scope; the next debugging step should inspect
  route alignment/precision in the installed vLLM TPU path before adding replay.

## 2026-06-15 - Small Diagnostic GrugMoE Installed vLLM TPU Validation

Follow-on branch:

- Marin branch `codex/grugmoe-small-diagnostic-20260615`, based on
  `codex/grugmoe-correctness-validation-20260615`.
- vLLM remains pinned to
  `c6f0608ddadb6bdd39a16b857b5affe660b1259e`.
- tpu-inference remains pinned to
  `d8c4579d3409d50ccaa4a81df44b0d62c9e9b958`.

Code changes:

- Added `small-diagnostic` to the realistic GrugMoE parity/export config set.
- Exposed `--model-size {full-canary,small-diagnostic}` in the installed
  validation harness. The default remains `full-canary`.
- Preserved the existing full-canary path and checks. The small diagnostic is an
  explicit opt-in mode.
- Added model-shape metadata to the installed reference JSON and score logs.
- Routing mismatch details now print all mismatches when the token/layer grid is
  small enough; the full canary still caps details at the existing limit.

Small diagnostic shape:

```text
vocab_size=4096
hidden_dim=512
intermediate_dim=1024
shared_expert_intermediate_dim=512
num_layers=2
num_heads=4
num_kv_heads=1
head_dim=128
num_experts=4
num_experts_per_token=2
max_seq_len=16
sliding_window=8
initializer_std=0.03125
qk_mult=1.3
moe_implementation=scatter
```

Local checks:

```bash
cd /home/romain/dev/marin-wt/grugmoe-small-diagnostic-20260615
python -m py_compile \
  experiments/grug/moe/vllm_tpu_parity.py \
  experiments/grug/moe/installed_vllm_full_canary_smoke.py
VLLM_TARGET_DEVICE=tpu uv lock --check
./infra/pre-commit.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  experiments/grug/moe/installed_vllm_full_canary_smoke.py
env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu \
  uv run --locked --package marin-core --extra vllm --extra eval \
  python -m experiments.grug.moe.installed_vllm_full_canary_smoke --help
env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu \
  uv run --locked --package marin-core --extra vllm --extra eval \
  python -m experiments.grug.moe.vllm_tpu_parity --help
```

Result: all checks passed.

TPU command:

```bash
cd /home/romain/dev/marin-wt/grugmoe-small-diagnostic-20260615
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 7200 \
  --cpu 16 \
  --memory 256GB \
  --disk 300GB \
  --job-name grugmoe-small-diagnostic-installed-vllm-$(date +%Y%m%d-%H%M%S) \
  -e VLLM_TARGET_DEVICE tpu \
  -- env -u PYTHONPATH \
    VLLM_TARGET_DEVICE=tpu \
    PYTHONUNBUFFERED=1 \
    LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304 \
    uv run --locked --package marin-core --extra vllm --extra eval \
    python -m experiments.grug.moe.installed_vllm_full_canary_smoke \
    --model-size small-diagnostic \
    --output-dir /tmp/grugmoe-small-diagnostic-installed-vllm \
    --max-shard-size 4194304 \
    --generation-tokens 3
```

Final TPU job:

- `/romain/grugmoe-small-diagnostic-installed-vllm-20260615-221844`.
- State: failed after score, task exit `1`, duration
  `3 minutes and 40.29 seconds`.
- Failure reason: the diagnostic routing guard found suspicious routed-expert
  mismatches after export/load, installed serve, and selected-token logprob
  comparison completed.

Installed dependency identity:

- `vllm_direct_url` commit
  `c6f0608ddadb6bdd39a16b857b5affe660b1259e`.
- `tpu-inference_direct_url` commit
  `d8c4579d3409d50ccaa4a81df44b0d62c9e9b958`.

Export/load evidence:

- Config source: `small diagnostic GrugMoE`.
- Levanter/native full-forward fixed continuation:
  `[3045, 2536, 3106]`.
- Reference continuation logprobs:
  `[-6.945110321044922, -6.994075775146484, -7.133152961730957]`.
- Sharded export/load:
  `checkpoint_bytes=82052339`, `artifact_bytes=81845277`,
  `shard_count=14`, `max_shard_size=4194304`,
  `expected_tensors=46 consumed_tensors=46 missing=[] unexpected=[]`.

Installed serve/generate evidence:

- `vllm_server_initialized=True`.
- `vllm_generate_status_code=200`.
- Installed vLLM serve generated:
  `installed_full_canary_generated=[([1, 42, 128, 2048, 17, 3072, 5, 63],
  [3045, 2536, 223], '')]`.
- `installed_path_result=works:full_canary_generate`.

Installed vLLM score evidence:

- `score_full_vocab_logprobs=4096`.
- `score_prompt_token_ids=[1, 42, 128, 2048, 17, 3072, 5, 63, 3045, 2536, 3106]`.
- `score_generated_token_ids=[2951]`.
- `score_continuation_generated_token_ids=[3045, 2536, 3106]`.
- Selected-token logprob summary:
  `count=3`, `max_abs_delta=0.0033097267150878906`,
  `mean_abs_delta=0.0019470850626627605`,
  `max_allowed_abs_delta=5.0`.
- Per-token selected logprob deltas:
  - token `3045` at position `8`: Levanter `-6.945110321044922`, vLLM
    `-6.941800594329834`, delta `0.0033097267150878906`;
  - token `2536` at position `9`: Levanter `-6.994075775146484`, vLLM
    `-6.996557712554932`, delta `0.0024819374084472656`;
  - token `3106` at position `10`: Levanter `-7.133152961730957`, vLLM
    `-7.133103370666504`, delta `4.9591064453125e-05`.

Routed-expert diagnostic:

- `token_layer_count=22`, `top_k=2`.
- `ordered_topk_match_count=17`, `ordered_topk_match_rate=0.7727272727272727`.
- `unordered_full_match_count=19`,
  `unordered_full_match_rate=0.8636363636363636`.
- `mean_unordered_overlap=0.9318181818181818`.
- `top1_match_count=19`, `top1_match_rate=0.8636363636363636`.
- `mismatch_count=5`.
- `low_margin_boundary_mismatch_count=0`.
- `suspicious_mismatch_count=3`.
- `routing_mismatch_print_limit=22`, so all five mismatch locations printed.

Full routing mismatch list:

```json
[
  {"layer": 0, "levanter": [1, 2], "router_margin": 0.02071744203567505, "token_id": 1, "token_position": 0, "top1_match": false, "unordered_overlap": 1, "vllm": [0, 1]},
  {"layer": 1, "levanter": [3, 0], "router_margin": 0.21385124325752258, "token_id": 1, "token_position": 0, "top1_match": true, "unordered_overlap": 1, "vllm": [3, 1]},
  {"layer": 0, "levanter": [1, 0], "router_margin": 0.11697107553482056, "token_id": 42, "token_position": 1, "top1_match": false, "unordered_overlap": 2, "vllm": [0, 1]},
  {"layer": 1, "levanter": [2, 3], "router_margin": 0.1284528374671936, "token_id": 42, "token_position": 1, "top1_match": true, "unordered_overlap": 1, "vllm": [2, 1]},
  {"layer": 1, "levanter": [3, 1], "router_margin": 0.7457730770111084, "token_id": 128, "token_position": 2, "top1_match": false, "unordered_overlap": 2, "vllm": [1, 3]}
]
```

Conclusion:

- The small model reproduces the routed-expert divergence while making it much
  easier to inspect: only five ordered mismatches and three suspicious
  unordered mismatches across 22 token/layer comparisons.
- Selected-token logprobs are much closer than in the full canary run, with max
  delta about `0.00331`, so the mismatch appears concentrated in routed expert
  ID ordering/selection rather than gross logit corruption.
- The next recommended debugging step is to instrument one failing token/layer
  in the installed vLLM TPU path, starting with token position `0`, layer `0`
  (`levanter=[1,2]`, `vllm=[0,1]`). Compare raw router logits, router bias,
  biased logits, and top-k tie/order behavior immediately before
  `jax.lax.top_k` in Levanter/manual reference versus tpu-inference/vLLM.

Sharp edges:

- The compact artifact would not necessarily exercise indexed safetensors with
  the full-canary `256MiB` shard size, so the TPU run used
  `--max-shard-size 4194304` and produced 14 shards.
- The installed serve path still labels legacy output markers as
  `full_canary_*` for compatibility, even when `--model-size small-diagnostic`
  is selected.
- Iris still runs from `/app`, so `git rev-parse HEAD` remains unavailable
  inside the job; use the pushed branch commit and installed package direct URLs
  for identity.

## 2026-06-15 - Small Diagnostic Routing Mismatch Localization

Goal: determine whether the first small-diagnostic mismatch at token position
`0`, layer `0` is introduced inside installed GrugMoE model math or later in
route capture, token/layer ordering, scheduler slicing, or output plumbing.

Branches and commits:

- Marin: `codex/grugmoe-routing-localize-20260615` at
  `45697caf1c6d5cedbf2caea7ec74735be61470f9`.
- vLLM: unchanged at `c6f0608ddadb6bdd39a16b857b5affe660b1259e`.
- tpu-inference diagnostic branch: `codex/grugmoe-routing-debug-tpu-20260615`
  at `0cf0ed8cb92f1731ec9fb7bc1c291c9ab00364af`.

Diagnostic code changes:

- Added opt-in installed harness flags:
  `--routing-debug`, `--routing-debug-token-position`,
  `--routing-debug-layer`, and `--routing-debug-vector-limit`.
- The Levanter reference JSON now optionally records, for one token/layer, the
  router input hidden state summary, raw router logits, router bias, biased
  logits, top-k expert IDs, boundary expert/logit, router margin, and combine
  weights.
- The installed score phase sets `GRUGMOE_ROUTING_DEBUG=1` for the selected
  token/layer and prints the final `CompletionOutput.routed_experts` slice.
- The tpu-inference diagnostic branch adds a gated `jax.debug.callback` in
  `GrugMoeMLP.route` and prints the same route-point record from the installed
  TPU model.

Local checks:

```bash
cd /home/romain/dev/marin-wt/grugmoe-routing-debug-tpu-20260615
python -m py_compile tpu_inference/models/jax/grugmoe.py
pre-commit run --files tpu_inference/models/jax/grugmoe.py
git diff --check

cd /home/romain/dev/marin-wt/grugmoe-routing-localize-20260615
python -m py_compile \
  experiments/grug/moe/vllm_tpu_parity.py \
  experiments/grug/moe/installed_vllm_full_canary_smoke.py
VLLM_TARGET_DEVICE=tpu uv lock --check
./infra/pre-commit.py \
  experiments/grug/moe/vllm_tpu_parity.py \
  experiments/grug/moe/installed_vllm_full_canary_smoke.py \
  pyproject.toml \
  uv.lock
env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu \
  uv run --locked --package marin-core --extra vllm --extra eval \
  python -m experiments.grug.moe.installed_vllm_full_canary_smoke --help
env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu \
  uv run --locked --package marin-core --extra vllm --extra eval \
  python -m experiments.grug.moe.vllm_tpu_parity --help
```

Result: all local checks passed.

TPU command:

```bash
cd /home/romain/dev/marin-wt/grugmoe-routing-localize-20260615
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 7200 \
  --cpu 16 \
  --memory 256GB \
  --disk 300GB \
  --job-name grugmoe-routing-localize-installed-vllm-$(date +%Y%m%d-%H%M%S) \
  -e VLLM_TARGET_DEVICE tpu \
  -- env -u PYTHONPATH \
    VLLM_TARGET_DEVICE=tpu \
    PYTHONUNBUFFERED=1 \
    LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304 \
    uv run --locked --package marin-core --extra vllm --extra eval \
    python -m experiments.grug.moe.installed_vllm_full_canary_smoke \
    --model-size small-diagnostic \
    --output-dir /tmp/grugmoe-routing-localize-installed-vllm \
    --max-shard-size 4194304 \
    --generation-tokens 3 \
    --routing-debug \
    --routing-debug-token-position 0 \
    --routing-debug-layer 0 \
    --routing-debug-vector-limit 16
```

Final TPU job:

- `/romain/grugmoe-routing-localize-installed-vllm-20260615-225420`.
- Region/TPU: `europe-west4`, `v6e-4`.
- Final state: failed in the expected score-phase diagnostic guard with
  `AssertionError: 3 routed-expert mismatches had boundary margin > 0.01`.
- Duration: `3 minutes and 39.9 seconds`.
- Installed dependency identity:
  - vLLM: `c6f0608ddadb6bdd39a16b857b5affe660b1259e`.
  - tpu-inference: `0cf0ed8cb92f1731ec9fb7bc1c291c9ab00364af`.

Export/load and serve evidence:

- Config source: `small diagnostic GrugMoE`.
- Artifact/load: `artifact_bytes=81845277`, `shard_count=14`,
  `expected_tensors=46`, `consumed_tensors=46`, `missing=[]`,
  `unexpected=[]`.
- Serve/generate: `vllm_generate_status_code=200`.
- Installed vLLM serve generated `[3045, 2536, 223]` for prompt
  `[1, 42, 128, 2048, 17, 3072, 5, 63]`.
- Score prompt token IDs:
  `[1, 42, 128, 2048, 17, 3072, 5, 63, 3045, 2536, 3106]`.
- Score generated token IDs: `[2951]`.
- Fixed-continuation generated token IDs: `[3045, 2536, 3106]`.
- Selected-token logprobs stayed close:
  `max_abs_delta=0.0033097267150878906`,
  `mean_abs_delta=0.0019470850626627605`.

First failing token/layer evidence:

- Target: token position `0`, token ID `1`, layer `0`, top-k `2`.
- Levanter/manual reference:
  - router input hidden state dtype `float32`, shape `[512]`, first values
    `[0.8003653287887573, 0.9361541867256165, 0.4786311388015747,
    0.3288799822330475, 0.019369488582015038, -0.6378109455108643,
    0.14827126264572144, -0.4013029634952545]`, `l2=11.270750380335834`.
  - raw router logits:
    `[0.11579327285289764, 0.18306951224803925,
    0.1365107148885727, -0.06385284662246704]`.
  - router bias: `[0.0, 0.0, 0.0, 0.0]`.
  - biased router logits match raw logits.
  - top-k with boundary: experts `[1, 2, 0]`, logits
    `[0.18306951224803925, 0.1365107148885727,
    0.11579327285289764]`.
  - selected top-k experts: `[1, 2]`.
  - combine weights: `[0.5456399917602539, 0.5340747833251953]`.
  - router margin: `0.02071744203567505`.
- Installed tpu-inference route point:
  - router input hidden state dtype `bfloat16`, shape `[512]`, first values
    `[0.8515625, 0.98828125, 0.455078125, 0.349609375,
    0.1943359375, -0.4921875, 0.1845703125, -0.578125]`,
    `l2=11.272372436733875`.
  - raw router logits:
    `[0.19308093190193176, 0.13528966903686523,
    0.0202304869890213, -0.030783653259277344]`.
  - router bias: `[0.0, 0.0, 0.0, 0.0]`.
  - biased router logits match raw logits.
  - top-k with boundary: experts `[0, 1, 2]`, logits
    `[0.19308093190193176, 0.13528966903686523,
    0.0202304869890213]`.
  - selected top-k experts: `[0, 1]`.
  - combine weights: `[0.546875, 0.53515625]`.
  - router margin: `0.11505918204784393`.
- Final vLLM output plumbing:
  - `CompletionOutput.routed_experts` shape `[11, 2, 2]`.
  - token position `0`, all layer routes: `[[0, 1], [3, 1]]`.
  - token position `0`, layer `0`, reported top-k: `[0, 1]`.

Routing summary remained the same as the previous small diagnostic:

- `token_layer_count=22`, `top_k=2`.
- `ordered_topk_match_count=17`, `ordered_topk_match_rate=0.7727272727272727`.
- `unordered_full_match_count=19`,
  `unordered_full_match_rate=0.8636363636363636`.
- `top1_match_count=19`, `top1_match_rate=0.8636363636363636`.
- `mismatch_count=5`, `suspicious_mismatch_count=3`.

Conclusion:

- The first known mismatch appears before route capture or scheduler plumbing.
  The installed tpu-inference model's own route calculation for token `0`,
  layer `0` selects `[0, 1]`, and final vLLM `routed_experts` reports the same
  `[0, 1]` with shape/order `[token, layer, top_k]`.
- The mismatch first appears in the model math before top-k selection: the
  router input hidden state already differs between the Levanter/manual
  reference and installed TPU path, and the raw router logits differ before any
  router bias is applied. Router bias is zero on both paths.
- This is not evidence of token ordering, layer ordering, scheduler slicing, or
  output-plumbing corruption.
- I do not see a clean correctness fix in scheduler/output code. The likely
  next fix is to align the validation reference with the precision actually
  used by installed vLLM TPU (`dtype="bfloat16"`) or, if exact Levanter fp32
  routed-expert IDs are the serving contract, run/keep more of the installed
  model path in float32. A useful next experiment is to add a temporary
  `--vllm-dtype float32` score mode or to emit a bfloat16 native-reference
  route record, then rerun the same small diagnostic.

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
