# Levanter TPU Inference Parity Spec

This spec pins the proposed implementation surface. It is intentionally narrower than the design: the goal is to make the kernel boundary and benchmark contract unambiguous without prescribing every internal optimization.

## Files

| Path | Purpose |
| --- | --- |
| `lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py` | Phase 1 apples-to-apples Levanter vs `vllm-tpu` benchmark harness for Qwen3 8B decode-heavy RL rollouts. |
| `lib/levanter/src/levanter/inference/tpu_kernels.py` | Backend enum, config, backend dispatch, and shared exceptions for paged TPU inference kernels. |
| `lib/levanter/src/levanter/inference/tpu_inference_adapter.py` | Adapter around required-on-TPU `tpu-inference`; imports the dependency only inside backend availability checks and calls so CPU/GPU installs still work. |
| `lib/levanter/tests/inference/test_tpu_paged_attention_backends.py` | Backend contract tests against existing paged-attention reference cases. |
| `lib/levanter/tests/inference/test_paged_attention.py` | Existing reference tests remain; add parametrization only if it keeps the file readable. |

Phase 1 lands only the benchmark harness. Phase 2 lands the backend files and attention integration after the baseline gap is measured.

## Public Python Surface

```python
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import haliax as hax
import jax.numpy as jnp
from haliax import NamedArray

from levanter.inference.page_table import PageBatchInfo
from levanter.layers.kv_cache import KvPageCache


class TpuPagedAttentionBackend(StrEnum):
    AUTO = "auto"
    TPU_INFERENCE = "tpu_inference"
    JAX_RPA = "jax_rpa"
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class TpuPagedAttentionConfig:
    backend: TpuPagedAttentionBackend | Sequence[TpuPagedAttentionBackend] = TpuPagedAttentionBackend.AUTO
    allow_autotune: bool = False
    fail_on_reference_fallback: bool = True
```

Contract: `backend` selects the implementation order. On TPU, `AUTO` expands to `TPU_INFERENCE` only; `tpu-inference` is required for TPU serving. On CPU/GPU, `AUTO` expands to `REFERENCE`. An explicit non-sequence backend must either run or raise `UnsupportedTpuPagedAttentionBackend`. A backend sequence tries entries in order, emits `TpuPagedAttentionFallbackWarning` for each failed earlier backend, and raises if no backend can run. `JAX_RPA` and `REFERENCE` are diagnostic/test backends on TPU, not production fallbacks. `allow_autotune` permits bounded on-device backend tuning after the static table misses; it must not run during ordinary pytest unless explicitly enabled.

Fallback truth table:

| Platform | Config | Missing dependency or unsupported shape | Result |
| --- | --- | --- | --- |
| TPU | `AUTO` | `TPU_INFERENCE` missing or unsupported | Raise `UnsupportedTpuPagedAttentionBackend`. |
| TPU | explicit `TPU_INFERENCE` or `JAX_RPA` | selected backend fails | Raise `UnsupportedTpuPagedAttentionBackend`. |
| TPU | explicit `REFERENCE` | `fail_on_reference_fallback=True` | Raise `ValueError` before lowering. |
| TPU | explicit `REFERENCE` | `fail_on_reference_fallback=False` | Run reference path for diagnostics/tests. |
| TPU | sequence including `REFERENCE` | TPU backends fail and flag is false | Warn and run reference path. |
| CPU/GPU | `AUTO` | n/a | Run reference path. |
| CPU/GPU | explicit TPU backend | n/a | Raise `UnsupportedTpuPagedAttentionBackend`. |

```python
class UnsupportedTpuPagedAttentionBackend(RuntimeError):
    """Raised when an explicitly selected paged-attention backend cannot run for the current platform or shape."""


class TpuPagedAttentionFallbackWarning(UserWarning):
    """Warning emitted when backend dispatch falls back from one configured backend to another."""
```

```python
def paged_attention_with_kv_update(
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jnp.ndarray,
    soft_cap: float | None,
    config: TpuPagedAttentionConfig,
) -> tuple[NamedArray, KvPageCache]:
    """Compute paged attention for the packed decode/prefill tokens and return the updated KV cache.

    The input shapes follow Levanter's current paged decode contract:
    `q` is `[position, kv_head, q_heads_per_group, head_size]`, `new_k` and
    `new_v` are `[position, kv_head, head_size]`, `kv_cache.kv_pages` is
    `[page, slot, 2 * kv_head, head_size]`, and `batch_info` describes the
    active sequence/page mapping for the packed token axis.

    Backends that own a fused KV-update plus attention kernel may update the
    cache internally. Backends that only provide attention must call
    `kv_cache.update(batch_info, new_k, new_v)` before attention. The returned
    attention has the same axes as `q`.

    The returned cache is the only valid cache for subsequent calls. Backends
    may use JAX donation or aliasing internally, so callers must not reuse
    `kv_cache` after invoking this function. The returned cache must preserve
    the original axes, dtype, and sharding. `batch_info.new_token_dests` must
    contain no duplicate valid destinations in a single call; duplicates raise
    `ValueError` before backend lowering. Packed token ordering is the existing
    Levanter position-axis order after scheduler packing.
    """
```

```python
def available_tpu_paged_attention_backends() -> tuple[TpuPagedAttentionBackend, ...]:
    """Return backends importable and platform-supported in the current process, excluding `AUTO`."""


@dataclass(frozen=True, slots=True)
class TpuPagedAttentionShape:
    platform: str
    device_kind: str
    dtype: jnp.dtype
    page_size: int
    head_size: int
    num_q_heads: int
    num_kv_heads: int
    q_heads_per_group: int
    max_model_len: int
    tensor_parallel_size: int


def tpu_paged_attention_supports_shape(
    backend: TpuPagedAttentionBackend,
    shape: TpuPagedAttentionShape,
) -> tuple[bool, str | None]:
    """Return whether `backend` supports `shape`, plus a human-readable rejection reason when unsupported."""
```

Initial supported shape matrix:

| Field | Initial target |
| --- | --- |
| Model | Qwen3 8B |
| TPU | v5p first; v6e optional secondary validation |
| dtype | bf16 |
| page size | 128 |
| head size | 128 after padding |
| query heads / KV heads | 32 / 8 |
| tensor parallel size | 4 on the Levanter `model` axis |
| max model length | 4096 |

## Config Integration

Add one optional field to `InferenceEngineConfig`:

```python
tpu_paged_attention: TpuPagedAttentionConfig = TpuPagedAttentionConfig()
```

Contract: this config is passed through the inference engine into Qwen/Llama-style paged attention. Training/full-sequence attention is unchanged. Non-TPU platforms may construct the config, but `TPU_INFERENCE` and `JAX_RPA` raise if explicitly selected off TPU. TPU serving entrypoints must ensure the `tpu-inference` dependency is installed before constructing the engine with `backend=AUTO`.

The implementation path is:

- `InferenceEngineConfig.tpu_paged_attention` owns the user-facing config.
- `InferenceEngine.from_model_with_config` passes the config to model setup without mutating the model.
- `LmHeadModel.decode` implementations that use paged attention accept the config as a keyword-only argument, defaulting to `TpuPagedAttentionConfig()`.
- `Qwen3LMHeadModel.decode`/`LlamaLMHeadModel.decode`, `LlamaTransformer.decode`, and decoder-layer decode paths thread the config to `Attention.paged_decode`.
- Other models using `Attention.paged_decode` must either thread the same config or explicitly use `TpuPagedAttentionConfig(backend=REFERENCE)` in tests. Full-sequence training attention is unaffected.

## Attention Integration

`Attention.paged_decode` changes from:

```python
kv_cache = kv_cache.update(batch_info, k, v)
attn_tokens = ragged_paged_attention(...)
```

to:

```python
attn_tokens, kv_cache = paged_attention_with_kv_update(
    q,
    k,
    v,
    kv_cache,
    batch_info,
    sm_scale=sm_scale,
    soft_cap=self.config.logits_soft_cap,
    config=...,
)
```

Contract: `Attention.paged_decode` remains the model-facing API. Its return type and caller behavior do not change.

## Benchmark CLI

```text
uv run python lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py \
  --model Qwen/Qwen3-8B \
  --levanter-checkpoint <hf-or-levanter-checkpoint> \
  --backend both \
  --max-model-len 4096 \
  --tensor-parallel-size 4 \
  --matrix rl_decode_qwen3_8b_v1 \
  --dump-levanter-kernels \
  --output-dir <path>
```

Required output files:

- `summary.json`: machine-readable metrics for every case.
- `summary.md`: human-readable table with Levanter, vLLM, and ratio columns.
- `env.json`: JAX, jaxlib, vllm-tpu, tpu-inference, libtpu, TPU device kind, XLA flags, and command line.
- `levanter_hlo/`: generation and prefill HLO/jaxpr artifacts when requested.
- `vllm_profiles/`: vLLM TPU profiles when requested and available.

Benchmark contract:

- `--model` names the HF model/tokenizer identity used by both backends.
- `--levanter-checkpoint` may differ only in storage format; it must represent the same weights as `--model`, and the harness records the converter or mapping used.
- Tokenizer, chat template, BOS/EOS handling, stop token IDs, temperature, top-p, and max output tokens are shared.
- Default measurement uses deterministic greedy decoding. Optional sampling runs must record seeds and acknowledge that vLLM TPU may not support per-request seeds.
- Each case runs warmups before measurement and records warmup count, measured repetitions, and whether output tokens were generated normally or forced by a fixed decode trace.
- `rl_decode_qwen3_8b_v1` is the milestone gate. It contains decode-heavy rollout cases with many active sequences, `input_len=1`, `output_len={128,512}`, and cloned-prefix `n > 1` cases. Prefill-only and mixed-serving matrices may exist, but they are diagnostic unless a later design promotes them.
- The default hardware target is TPU v5p. The harness may run on v6e, but v6e results do not satisfy the first milestone without v5p results.

Required metric keys per case:

```json
{
  "case_name": "decode_b32_i1_o128",
  "backend": "levanter:auto",
  "compile_including_seconds": 0.0,
  "steady_state_seconds": 0.0,
  "ttft_ms_p50": 0.0,
  "request_latency_ms_p50": 0.0,
  "request_latency_ms_p90": 0.0,
  "decode_tokens_per_second": 0.0,
  "total_tokens_per_second": 0.0,
  "hbm_used_bytes": 0,
  "compiled_shape_count": 0
}
```

## Errors

- `UnsupportedTpuPagedAttentionBackend`: explicit backend unavailable, unsupported TPU generation, unsupported dtype, unsupported head size, unsupported page size, or missing `tpu-inference` on TPU.
- `ValueError`: invalid backend sequence, `REFERENCE` selected as the only backend while `fail_on_reference_fallback=True` on TPU, or benchmark cases with incompatible prompt/output lengths.
- `RuntimeError`: no configured backend can run and no fallback is allowed.

## Out Of Scope

- Replacing Levanter's whole inference scheduler with vLLM's scheduler.
- Changing training/full-sequence attention.
- Supporting quantized weights, speculative decoding, MoE models, or non-Qwen3 architectures in the first parity milestone.
- Promising backwards compatibility for internal Levanter inference config names before this design lands.
