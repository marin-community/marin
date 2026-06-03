# GrugMoE Native JAX TPU Reference

Issue: https://github.com/marin-community/marin/issues/6106

## Source Map

- GrugMoE reference: [`experiments/grug/moe/model.py`](./model.py)
  - Architecture summary: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
  - Routing path: router logits are fp32, top-k is selected from `router_logits + stop_gradient(router_bias)`, and combine weights are sigmoid of the unbiased selected logits.
  - Transformer path: all blocks are MoE blocks; every fourth layer uses the long sliding-window mask.
- Levanter MoE oracle: [`lib/levanter/src/levanter/grug/grug_moe.py`](../../../lib/levanter/src/levanter/grug/grug_moe.py)
  - `moe_mlp` accepts precomputed `selected_experts` and `combine_weights`, which makes it the component-parity target for native tpu-inference routing.
- Native tpu-inference implementation: `tpu_inference/models/jax/grugmoe.py` in the tpu-inference branch.
  - `GrugMoeForCausalLM` follows the JAX/NNX model shape used by the existing tpu-inference model registry.
  - `GrugMoeForCausalLM.load_weights` loads the canonical inference artifact, not Levanter's training checkpoint tree.
  - `tpu_inference/models/common/model_loader.py` resolves `GrugMoeForCausalLM` through the native `flax_nnx` path.
  - Exact validation SHA: `e42d7339e2c84b29b4c302b22483678b3862ab4e`.
- vLLM branch:
  - The PyTorch-first `vllm/model_executor/models/grugmoe.py` prototype and its test are removed.
  - `GrugMoeForCausalLM` is no longer registered in vLLM's text-generation model registry.
  - Exact validation SHA: `d025e46d3dfe0afc0f5bd1518c19ce337205db2d`.
- Marin parity harness:
  - `experiments/grug/moe/vllm_tpu_parity.py` validates both the manual-copy correctness path and the canonical inference artifact roundtrip.
  - Exact validation SHA: `c3442afe3ccfb6703623d6a679d7398f5789cd73`.

## Implementation Choice

The current implementation is correctness-first and native JAX inside tpu-inference:

- Use NNX parameters and tpu-inference `JaxModule`/pipeline patterns.
- Keep dense, direct attention and MoE equations for tiny seeded parity.
- Return the vLLM-facing `ForCausalLM` tuple shape so model-loader integration is wired.
- Add a narrow canonical inference artifact boundary: `config.json` plus `model.safetensors`.
- Use stable HF/vLLM-style tensor names with checkpoint-oriented linear tensor layout; native JAX parameter orientation remains a loader detail.
- Keep vLLM as a dependency/source checkout for shared config/import surfaces, not as the GrugMoE model owner.

This intentionally does not include:

- vLLM PyTorch model code or fused vLLM MoE kernels.
- Direct loading of Levanter's native training checkpoint tree.
- Generation.
- Production KV-cache decode behavior.
- Performance tuning or fused TPU kernels.
- Real-model smoke tests.

## Canonical Artifact

The GrugMoE inference artifact is a directory with two files:

- `config.json`: model hyperparameters plus common HF aliases such as `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and `num_key_value_heads`.
- `model.safetensors`: tensors named with a stable dotted schema.

Representative tensor names:

- `model.embed_tokens.weight`
- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.mlp.router.weight`
- `model.layers.0.mlp.experts.gate_proj.weight`
- `model.layers.0.mlp.experts.up_proj.weight`
- `model.layers.0.mlp.experts.down_proj.weight`
- `model.layers.0.shared_expert.gate_proj.weight`
- `model.norm.weight`
- `lm_head.weight`

Linear tensors in the artifact use checkpoint orientation: output features before input features. Expert tensors keep the semantic expert axis first and store gate/up/down separately rather than baking in the native JAX fused `w_gate_up` parameter. The native loader validates the artifact config, checks the full tensor-name set, raises on missing or unexpected tensors, and reports the consumed tensor names.

## Tiny Config

The smallest useful composed test shape keeps every Grug feature active:

- `hidden_dim=8`
- `intermediate_dim=12`
- `shared_expert_intermediate_dim=10`
- `num_experts=4`
- `num_experts_per_token=2`
- `num_layers=4`
- `num_heads=2`
- `num_kv_heads=1`
- `head_dim=4`
- `max_seq_len=8`
- `sliding_window=4`

The 4-layer shape exercises both sliding-window branches in the Grug block loop:
layers `0`, `1`, and `2` use the short window `2`, while layer `3` uses the
long window `4`. The full harness uses a 6-token sequence so the different
window lengths can affect attention.

The focused tpu-inference unit test uses an even smaller MoE-only shape to isolate the router-bias semantic.

## Oracle Coverage

The Marin parity harness keeps Levanter as the correctness oracle:

- Component parity instantiates native `GrugMoeMLP`, sets deterministic router/expert weights, computes selected experts and unbiased sigmoid combine weights, and compares the output against Levanter `moe_mlp`.
- Composed parity initializes a tiny seeded Levanter `Transformer`, copies its parameters into native tpu-inference `GrugMoeForCausalLM`, and compares final hidden states against a dense JAX reference using the same Levanter parameters/equations.
- The composed check also calls native `GrugMoeForCausalLM.compute_logits` and compares those logits against the Levanter output projection applied to the reference hidden states.
- The composed check records selected expert IDs from each native layer and compares them against the reference QB routing path.
- The artifact roundtrip initializes a tiny seeded Levanter `Transformer`, exports the canonical artifact, loads it into a second native tpu-inference JAX model, and compares hidden states, `compute_logits` logits, and routed expert IDs against the manual-copy native model.
- The artifact roundtrip asserts the loader consumed exactly the exported tensor set and reports no missing or unexpected tensors.
- The harness patches Levanter sharding helpers to a one-device runtime for local and single-slice TPU validation. A direct `Transformer.__call__` is not used as the composed oracle because the training sharding contract is stricter than this one-device inference harness.
- The harness sets `jax_default_matmul_precision=float32` so local CPU and TPU validation use the same strict numeric path.

## Local Commands

vLLM removal check:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
rg -n "GrugMoe|GrugMoE|grugmoe|grug_moe" .
uv run --no-project python -m py_compile \
  vllm/model_executor/models/registry.py \
  tests/models/registry.py
```

tpu-inference native JAX checks:

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

Marin parity harness:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
python -m py_compile experiments/grug/moe/vllm_tpu_parity.py
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
  --tpu-inference-root ../grugmoe-vllm-tpu-inference
```

Expected parity output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact: canonical safetensors load matches manual-copy hidden states, logits, and routed expert IDs
```

## TPU Command

Run after pushing the Marin, vLLM, and tpu-inference `grugmoe-vllm-tpu-support` branches:

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
  --job-name grugmoe-inference-artifact-roundtrip \
  -- bash -lc 'set -euxo pipefail; echo marin_sha=c3442afe3ccfb6703623d6a679d7398f5789cd73; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-vllm-tpu-inference; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm; echo tpu_inference_sha=$(git -C /tmp/grugmoe-vllm-tpu-inference rev-parse HEAD); echo vllm_sha=$(git -C /tmp/grugmoe-vllm-tpu-vllm rev-parse HEAD); PYTHONPATH=/tmp/grugmoe-vllm-tpu-inference:/tmp/grugmoe-vllm-tpu-vllm uv run --with-requirements /tmp/grugmoe-vllm-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-vllm-tpu-vllm/requirements/common.txt --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-vllm-tpu-inference'
```

Canonical inference artifact roundtrip validation result on 2026-06-03:

- Job: `/romain/grugmoe-inference-artifact-roundtrip`
- State: `succeeded`
- Exit: `0`
- TPU: `v6e-4`
- Region: `europe-west4`
- Duration: `1 minute and 6.63 seconds`
- Remote SHAs:
  - `marin_sha=c3442afe3ccfb6703623d6a679d7398f5789cd73`
  - `tpu_inference_sha=e42d7339e2c84b29b4c302b22483678b3862ab4e`
  - `vllm_sha=d025e46d3dfe0afc0f5bd1518c19ce337205db2d`
- Final output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact: canonical safetensors load matches manual-copy hidden states, logits, and routed expert IDs
```
