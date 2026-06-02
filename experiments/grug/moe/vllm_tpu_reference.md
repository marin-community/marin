# GrugMoE vLLM TPU Reference

Issue: https://github.com/marin-community/marin/issues/6106

## Source Map

- GrugMoE reference: [`experiments/grug/moe/model.py`](./model.py)
  - Architecture summary: "QB-routed MoE with GatedNorm, XSA, sigmoid combine weights."
  - Routing path: router logits are fp32, top-k is selected from `router_logits + stop_gradient(router_bias)`, and combine weights are sigmoid of the unbiased selected logits.
  - Transformer path: all blocks are MoE blocks; every fourth layer uses the long sliding-window mask.
- Levanter MoE kernel: [`lib/levanter/src/levanter/grug/grug_moe.py`](../../../lib/levanter/src/levanter/grug/grug_moe.py)
  - `moe_mlp` accepts precomputed `selected_experts` and `combine_weights`, so it is the right component-parity target for vLLM's local MoE.
- vLLM model registry: `vllm/model_executor/models/registry.py` in the vLLM branch
  - Text generation architectures map names to `(module, class)` tuples, so `GrugMoeForCausalLM` maps to `("grugmoe", "GrugMoeForCausalLM")`.
- tpu-inference implementation resolver: `tpu_inference/models/common/model_loader.py` in the tpu-inference branch
  - `MODEL_IMPL_TYPE=auto` resolves to vLLM for architectures in `_VLLM_PREFERRED_ARCHITECTURES`.

## Implementation Choice

The first vLLM implementation is intentionally unfused and model-local:

- Use ordinary `torch.nn.Parameter` tensors and explicit PyTorch ops.
- Do not use vLLM `FusedMoE` for Grug routing yet.
- Do not use vLLM attention/KV-cache kernels yet.

Reason: Grug's routing differs from common fused MoE routers. Selection is by biased logits, while output weights are sigmoid of unbiased logits. tpu-inference's vLLM MoE plugin receives raw router logits for its backend, so relying on a generic fused path would be easy to make semantically wrong.

## Tiny Config

The smallest useful test shape keeps every Grug feature active:

- `hidden_dim=8`
- `intermediate_dim=12`
- `shared_expert_intermediate_dim=10`
- `num_experts=4`
- `num_experts_per_token=2`
- `num_layers=2`
- `num_heads=2`
- `num_kv_heads=1`
- `head_dim=4`
- `max_seq_len=8`
- `sliding_window=4`

The focused vLLM unit test uses an even smaller MoE-only shape to isolate the router-bias semantic.

## Harness Notes

The parity harness sets `jax_default_matmul_precision=float32`. Without this,
the local CPU run passes strict tolerances, but JAX on TPU uses lower default
dot precision and the component check drifts from the PyTorch CPU model before
testing any Grug-specific semantic.

## Commands

Local component/full parity harness:

```bash
uv run --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m experiments.grug.moe.vllm_tpu_parity \
  --vllm-root ../grugmoe-vllm-tpu-vllm
```

vLLM focused test:

```bash
cd ../grugmoe-vllm-tpu-vllm
uv run --no-project \
  --with-requirements requirements/common.txt \
  --with pytest \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m pytest --confcutdir=tests/models tests/models/test_grugmoe.py -q
```

tpu-inference resolver test:

```bash
cd ../grugmoe-vllm-tpu-inference
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run --no-project \
  --with-requirements requirements.txt \
  --with-requirements /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m pytest tests/models/common/test_model_loader.py::TestGetModel::test_get_model_auto_resolves_to_vllm_for_grug_moe -q
```

Iris `v6e-4` TPU validation:

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
  --job-name grugmoe-vllm-tpu-parity-fp32 \
  -- bash -lc 'git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm && uv run --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --vllm-root /tmp/grugmoe-vllm-tpu-vllm'
```
