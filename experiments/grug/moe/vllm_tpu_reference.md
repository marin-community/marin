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
  - The loader accepts either a single `model.safetensors` file or standard HF sharded safetensors via `model.safetensors.index.json` plus `model-*-of-*.safetensors`.
  - `tpu_inference/models/common/model_loader.py` resolves `GrugMoeForCausalLM` through the native `flax_nnx` path.
  - Exact sharded-export validation SHA: `c0d472c6c1ab085156767375d534c3272fbfd120`.
- vLLM branch:
  - The PyTorch-first `vllm/model_executor/models/grugmoe.py` prototype and its test are removed.
  - `GrugMoeForCausalLM` is no longer registered in vLLM's text-generation model registry.
  - Exact validation SHA: `d025e46d3dfe0afc0f5bd1518c19ce337205db2d`.
- Marin parity harness:
  - `experiments/grug/moe/model.py` exposes `GrugModelConfig.to_hf_config` and `Transformer.to_state_dict` so `HFCheckpointConverter.save_pretrained` emits the canonical inference artifact.
  - `experiments/grug/moe/vllm_tpu_parity.py` validates the manual-copy correctness path, the saved-checkpoint single-file Levanter export roundtrip, the forced tiny sharded Levanter export roundtrip, an opt-in capped-size sharded loader smoke, a realistic seeded non-zero `GrugTrainState` checkpoint roundtrip, and deterministic full-forward greedy generation from the loaded native artifact.
  - Exact deterministic generation validation SHA: `923ff3f2b96b1938444de4ab05f9b4c984f0dc43`.
- Marin export step:
  - `lib/levanter/src/levanter/main/export_lm_to_hf.py` supports tokenizer-less save-only exports when the model config carries `vocab_size`.
  - `lib/levanter/src/levanter/main/export_lm_to_hf.py` and `lib/marin/src/marin/export/levanter_checkpoint.py` support `checkpoint_subpath`; Grug training-state checkpoints export from `params`, while existing LM exports still default to `model`.
  - `lib/levanter/src/levanter/main/export_lm_to_hf.py` and `lib/marin/src/marin/export/levanter_checkpoint.py` support `max_shard_size`, preserving the previous default while allowing forced sharded export tests.

## Implementation Choice

The current implementation is correctness-first and native JAX inside tpu-inference:

- Use NNX parameters and tpu-inference `JaxModule`/pipeline patterns.
- Keep dense, direct attention and MoE equations for tiny seeded parity.
- Return the vLLM-facing `ForCausalLM` tuple shape so model-loader integration is wired.
- Add a narrow canonical inference artifact boundary: `config.json` plus either `model.safetensors` or standard HF sharded safetensors, produced by the normal Levanter/HF export path.
- Use stable HF/vLLM-style tensor names with checkpoint-oriented linear tensor layout; native JAX parameter orientation remains a loader detail.
- Keep vLLM as a dependency/source checkout for shared config/import surfaces, not as the GrugMoE model owner.

This intentionally does not include:

- vLLM PyTorch model code or fused vLLM MoE kernels.
- Direct loading of Levanter's native training checkpoint tree.
- Production KV-cache decode behavior or generation serving integration.
- Performance tuning or fused TPU kernels.
- External trained-checkpoint smoke tests. The realistic validation milestone uses a seeded non-zero local `GrugTrainState` checkpoint with the full canary config.

## Canonical Artifact

The GrugMoE inference artifact is a directory emitted by `HFCheckpointConverter.save_pretrained`:

- `config.json`: model hyperparameters plus common HF aliases such as `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and `num_key_value_heads`.
- Either `model.safetensors`, or standard HF sharded safetensors: `model.safetensors.index.json` plus `model-*-of-*.safetensors`.

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

Linear tensors in the artifact use checkpoint orientation: output features before input features. Expert tensors keep the semantic expert axis first and store gate/up/down separately rather than baking in the native JAX fused `w_gate_up` parameter. The native loader validates the artifact config, checks the full tensor-name set, raises on missing or unexpected tensors, and reports the consumed tensor names. Sharding changes only file placement through the HF weight map; the canonical tensor schema is unchanged. The serving contract remains this canonical artifact, not the Levanter training checkpoint tree.

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

## Full Canary Config

The realistic training-state roundtrip uses the full `GRUG_MOE_TRIAL_MODEL` config from the MoE launch path:

- `vocab_size=128256`
- `hidden_dim=1024`
- `intermediate_dim=512`
- `shared_expert_intermediate_dim=1024`
- `num_experts=64`
- `num_experts_per_token=4`
- `num_layers=11`
- `num_heads=8`
- `num_kv_heads=2`
- inferred `head_dim=128`
- `max_seq_len=4096`
- `sliding_window=4096`
- `initializer_std=0.015625`
- `qk_mult=1.3`

This preserves the production-relevant structure required for validation: 64 experts, top-4 routing, a shared expert, GQA, and the short/long sliding-window layer pattern. The validation prompt is fixed to `[1, 42, 128, 2048, 17, 3072, 5, 63]`. The deterministic generation check greedily emits three full-forward tokens from that prompt with no sampling and no KV cache; the full-canary generated IDs are `[57524, 45040, 67859]`.

## Oracle Coverage

The Marin parity harness keeps Levanter as the correctness oracle:

- Component parity instantiates native `GrugMoeMLP`, sets deterministic router/expert weights, computes selected experts and unbiased sigmoid combine weights, and compares the output against Levanter `moe_mlp`.
- Composed parity initializes a tiny seeded Levanter `Transformer`, copies its parameters into native tpu-inference `GrugMoeForCausalLM`, and compares final hidden states against a dense JAX reference using the same Levanter parameters/equations.
- The composed check also calls native `GrugMoeForCausalLM.compute_logits` and compares those logits against the Levanter output projection applied to the reference hidden states.
- The composed check records selected expert IDs from each native layer and compares them against the reference QB routing path.
- The artifact roundtrip initializes a tiny seeded Levanter `Transformer`, saves a checkpoint under `params`, exports it through `export_lm_to_hf` with `checkpoint_subpath="params"`, loads the resulting canonical artifact into a second native tpu-inference JAX model, and compares hidden states, `compute_logits` logits, and routed expert IDs against the manual-copy native model.
- The artifact roundtrip runs both the default single-file export and a forced tiny sharded export using `max_shard_size=1024`.
- The artifact roundtrip asserts the exported safetensors key set matches the canonical schema, and the loader consumed exactly that tensor set with no missing or unexpected tensors.
- The opt-in large smoke creates a zero-weight GrugMoE Levanter checkpoint, exports standard sharded safetensors, and verifies native tpu-inference can consume the 1.25GB artifact without changing tensor names.
- The realistic roundtrip builds a seeded non-zero `GrugTrainState` via `initial_state`, saves the full training-state checkpoint, exports through `export_lm_to_hf` with `checkpoint_subpath="params"`, rejects single-file output, and requires `model.safetensors.index.json` plus shard files.
- The realistic roundtrip compares the loaded native model against the Levanter/manual-copy reference for final hidden states, `compute_logits` logits, and routed expert IDs. It also reports artifact size, shard count, dtype policy, and strict missing/unexpected tensor accounting.
- The deterministic generation check reuses the realistic roundtrip artifact, repeatedly recomputes the full sequence, greedily selects `argmax` from the final-position logits, and compares generated token IDs exactly plus final-position logits within the existing parity tolerance at each step. It runs for the manual-copy native reference and the loaded native tpu-inference artifact path.
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

Expected parity output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact-single-file: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
artifact-sharded: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
```

Scaled realistic local sanity check:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
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

Expected scaled realistic output includes:

```text
realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts
realistic-generation: manual-copy native reference full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-generation: loaded native artifact full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs
realistic-roundtrip: ... shard_count=9 ... expected_tensors=84 consumed_tensors=84 missing=[] unexpected=[] generation_tokens=3 generated_ids=[858, 3205, 1165]
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
  --job-name grugmoe-export-path-roundtrip-v2 \
  -- bash -lc 'set -euxo pipefail; echo marin_sha=14577de9f460cb727f8b3a6ed2232051aed1359a; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/tpu-inference.git /tmp/grugmoe-vllm-tpu-inference; git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm; echo tpu_inference_sha=$(git -C /tmp/grugmoe-vllm-tpu-inference rev-parse HEAD); echo vllm_sha=$(git -C /tmp/grugmoe-vllm-tpu-vllm rev-parse HEAD); export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; PYTHONPATH=/tmp/grugmoe-vllm-tpu-inference:/tmp/grugmoe-vllm-tpu-vllm uv run --with-requirements /tmp/grugmoe-vllm-tpu-inference/requirements.txt --with-requirements /tmp/grugmoe-vllm-tpu-vllm/requirements/common.txt --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --tpu-inference-root /tmp/grugmoe-vllm-tpu-inference'
```

Normal Levanter export path roundtrip validation result on 2026-06-03:

- Job: `/romain/grugmoe-export-path-roundtrip-v2`
- State: `succeeded`
- Exit: `0`
- TPU: `v6e-4`
- Region: `europe-west4`
- Duration: `1 minute and 10.79 seconds`
- Remote SHAs:
  - `marin_sha=14577de9f460cb727f8b3a6ed2232051aed1359a`
  - `tpu_inference_sha=e42d7339e2c84b29b4c302b22483678b3862ab4e`
  - `vllm_sha=d025e46d3dfe0afc0f5bd1518c19ce337205db2d`
- Final output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
```

Sharded HF safetensors and capped-size large smoke validation used the same `v6e-4` region with the latest branch heads:

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

Sharded validation result on 2026-06-03:

- Job: `/romain/grugmoe-sharded-export-large-smoke`
- State: `succeeded`
- Exit: `0`
- TPU: `v6e-4`
- Region: `europe-west4`
- Duration: `2 minutes and 29.62 seconds`
- Remote SHAs:
  - `marin_sha=f895c61f8546b9847fdb31316e04d2c6d351aa83`
  - `tpu_inference_sha=c0d472c6c1ab085156767375d534c3272fbfd120`
  - `vllm_sha=d025e46d3dfe0afc0f5bd1518c19ce337205db2d`
- Final output:

```text
component: native GrugMoeMLP matches Levanter moe_mlp
full: native GrugMoeModel hidden states, logits, and routed expert IDs match Levanter reference
artifact-single-file: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
artifact-sharded: saved-checkpoint Levanter export matches manual-copy hidden states, logits, and routed expert IDs
artifact-large-sharded: zero-weight Levanter export loaded in native tpu-inference (1250833717 bytes, 27 shards)
```

Full canary realistic training-state roundtrip:

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

Full canary validation result on 2026-06-03:

- Job: `/romain/grugmoe-canary-generation-parity`
- State: `succeeded`
- Exit: `0`
- TPU: `v6e-4`
- Region: `europe-west4`
- Duration: `11 minutes and 8.96 seconds`
- Remote SHAs:
  - Marin branch head: `923ff3f2b96b1938444de4ab05f9b4c984f0dc43`
  - `tpu_inference_sha=c0d472c6c1ab085156767375d534c3272fbfd120`
  - `vllm_sha=d025e46d3dfe0afc0f5bd1518c19ce337205db2d`
- Training-state checkpoint: `/tmp/grugmoe-canary-generation-parity/checkpoints`, `5,762,306,365` bytes.
- Sharded HF artifact: `/tmp/grugmoe-canary-generation-parity/grugmoe-inference`, `5,762,168,484` bytes, 26 shards.
- Greedy full-forward generation: `3` new tokens, no sampling, no KV cache; generated IDs `[57524, 45040, 67859]`.
- Tensor accounting: `expected_tensors=217`, `consumed_tensors=217`, `missing=[]`, `unexpected=[]`.
- Final output:

```text
realistic-roundtrip: manual-copy native reference matches Levanter hidden states, logits, and routed experts
realistic-generation: manual-copy native reference full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-generation: prompt_ids=[1, 42, 128, 2048, 17, 3072, 5, 63] generated_ids=[57524, 45040, 67859] final_token_ids=[1, 42, 128, 2048, 17, 3072, 5, 63, 57524, 45040, 67859]
realistic-generation: loaded native artifact full-forward greedy generation matched Levanter reference for token IDs and logits across 3 steps
realistic-roundtrip: sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs
realistic-roundtrip: checkpoint_dir=/tmp/grugmoe-canary-generation-parity/checkpoints checkpoint_bytes=5762306365 artifact_dir=/tmp/grugmoe-canary-generation-parity/grugmoe-inference artifact_bytes=5762168484 shard_count=26 max_shard_size=268435456 expected_tensors=217 consumed_tensors=217 missing=[] unexpected=[] generation_tokens=3 generated_ids=[57524, 45040, 67859]
```
