# GrugMoE Native JAX TPU Logbook

Issue: https://github.com/marin-community/marin/issues/6106

## Branches

- Marin: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/marin/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-support`
  - Current local base before this native-JAX evidence update: [`4a909352`](https://github.com/marin-community/marin/commit/4a909352b27a1eedebc3556bf35bc9712394486c)
  - Final pushed branch head is recorded in issue #6106 after the evidence-doc commit is pushed.
- vLLM: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/vllm/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm`
  - Native-JAX replacement commit: [`d025e46d`](https://github.com/marin-community/vllm/commit/d025e46d3dfe0afc0f5bd1518c19ce337205db2d)
- tpu-inference: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/tpu-inference/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference`
  - Native-JAX implementation commit: [`4f83d210`](https://github.com/marin-community/tpu-inference/commit/4f83d2109ae650de7d7e4f521154fb0d5c1a23a1)

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

## Local Verification

Commands that passed on 2026-06-02:

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
  tests/models/jax/test_grugmoe.py \
  tests/models/common/test_model_loader.py
uv run --no-project --with ruff ruff check \
  tpu_inference/models/jax/grugmoe.py \
  tests/models/jax/test_grugmoe.py \
  tests/models/common/test_model_loader.py
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

Result: focused pytest `2 passed, 25 warnings in 12.36s`.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
python -m py_compile experiments/grug/moe/vllm_tpu_parity.py
uv run --with ruff ruff check experiments/grug/moe/vllm_tpu_parity.py
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
full: native GrugMoeModel hidden states match Levanter Transformer reference
```

## TPU Verification

Status: pending for the native-JAX replacement branch heads listed above.

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
  --job-name grugmoe-native-jax-tpu-parity \
  -- bash -lc 'git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-vllm-tpu-inference && git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm && PYTHONPATH=/tmp/grugmoe-vllm-tpu-inference:/tmp/grugmoe-vllm-tpu-vllm uv run --with-requirements /tmp/grugmoe-vllm-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-vllm-tpu-vllm/requirements/common.txt --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-vllm-tpu-inference'
```

## Scope

In scope:

- Native tpu-inference JAX model wiring.
- QB routing: biased top-k selection and unbiased sigmoid combine weights.
- Tiny deterministic component parity against Levanter `moe_mlp`.
- Tiny deterministic composed parity using Levanter parameters/equations.
- Targeted Iris `v6e-4` validation.

Out of scope:

- Performance tuning.
- Fused TPU kernels.
- vLLM PyTorch GrugMoE ownership.
- Checkpoint export/loading.
- Real-model smoke tests.
- Production KV-cache decode.
