# GrugMoE Native JAX TPU Logbook

Issue: https://github.com/marin-community/marin/issues/6106

## Branches

- Marin: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/marin/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-support`
  - Current local base before this native-JAX evidence update: [`4a909352`](https://github.com/marin-community/marin/commit/4a909352b27a1eedebc3556bf35bc9712394486c)
  - Native-JAX parity harness/docs commit used for TPU validation: [`9095972d`](https://github.com/marin-community/marin/commit/9095972d843652a22f925057a93498f04a6f3b1a)
  - Native-JAX strengthened parity harness commit used for the 2026-06-03 TPU validation: [`1394236f`](https://github.com/marin-community/marin/commit/1394236f0d1bfa71d931397872e522557b237c78)
  - Canonical inference artifact roundtrip harness commit used for the 2026-06-03 TPU validation: [`c3442afe`](https://github.com/marin-community/marin/commit/c3442afe3ccfb6703623d6a679d7398f5789cd73)
  - Normal Levanter export path roundtrip commit used for the 2026-06-03 TPU validation: [`14577de9`](https://github.com/marin-community/marin/commit/14577de9f460cb727f8b3a6ed2232051aed1359a)
  - Final pushed branch head is recorded in issue #6106 after this evidence update is pushed.
- vLLM: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/vllm/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm`
  - Native-JAX replacement commit: [`d025e46d`](https://github.com/marin-community/vllm/commit/d025e46d3dfe0afc0f5bd1518c19ce337205db2d)
- tpu-inference: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/tpu-inference/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference`
  - Native-JAX implementation commit: [`4f83d210`](https://github.com/marin-community/tpu-inference/commit/4f83d2109ae650de7d7e4f521154fb0d5c1a23a1)
  - Canonical inference artifact loader commit: [`e42d7339`](https://github.com/marin-community/tpu-inference/commit/e42d7339e2c84b29b4c302b22483678b3862ab4e)

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

## Scope

In scope:

- Native tpu-inference JAX model wiring.
- QB routing: biased top-k selection and unbiased sigmoid combine weights.
- Tiny deterministic component parity against Levanter `moe_mlp`.
- Tiny deterministic composed parity using Levanter parameters/equations, including final hidden states, `compute_logits` logits, and routed expert IDs.
- Tiny deterministic canonical inference artifact export/load roundtrip.
- Tiny deterministic saved-checkpoint export through Levanter `export_lm_to_hf` and Marin-compatible `checkpoint_subpath="params"`.
- Strict artifact tensor accounting: all exported tensors consumed, no missing tensors, no unexpected tensors.
- Targeted Iris `v6e-4` validation.

Out of scope:

- Performance tuning.
- Fused TPU kernels.
- vLLM PyTorch GrugMoE ownership.
- Direct loading of Levanter's native training checkpoint tree as the serving contract.
- Generation.
- Real-model smoke tests.
- Production KV-cache decode.
