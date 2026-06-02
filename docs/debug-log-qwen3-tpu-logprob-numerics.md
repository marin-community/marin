# Debugging log for Qwen3 TPU logprob numerics

Goal: understand and fix the high Levanter full-forward vs paged-decode logprob error for Qwen3 8B on TPU, checking
the `tpu-inference` ragged paged attention path first and then the rest of the stack. Target: max absolute logprob error
around 0.05 or lower, accepting some slowdown.

## Initial status

The v5p refdiag run `/dlwh/qwen3-parity-levanter-refdiag-v5p8-n4-topk4096-pages1024-20260601-0122` showed top-1
agreement but nontrivial top-token logprob drift:

- max logit abs error: 0.25
- top-1 logprob abs error: 0.0754366
- reference top-4096 mean logprob abs error: 0.0247864
- reference top-4096 max logprob abs error: 0.112063

This means the original max-logit-only failure is not just a tail-token relative-error artifact.

## Hypothesis 1

The error may come from the `tpu-inference` ragged paged attention kernel or adapter, rather than from Levanter's
decode stack as a whole.

## Changes to make

Extend `bench_qwen3_tpu_inference_parity.py` so one TPU run can compare full causal logits against multiple paged-decode
backend and cache-dtype combinations. The first diagnostic matrix should run:

- `tpu_inference` with the required bf16 cache
- Levanter `reference` paged attention with the default cache dtype
- Levanter `reference` paged attention with bf16 cache

This isolates kernel/adapter drift from cache-dtype and non-RPA decode effects.

## Results

Local harness validation passed:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 54 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/src/levanter/layers/attention.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py --fix
# OK
```

Submitted TPU diagnostic matrix:

- `/dlwh/qwen3-parity-levanter-refmatrix-v5p8-n4-topk4096-pages1024-20260601-0713`

The job acquired a v5p worker and reached TPU initialization. Results are pending.

That job was preempted before emitting artifacts and is pending retry. To reduce exposure to preemption, the harness now
has `--reference-logit-only`, which logs the matrix immediately after writing `levanter_reference_logits.{json,md}` and
exits before serving benchmarks. Local validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 41 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

Submitted replacement:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-20260601-0725`

That replacement acquired v5p but failed before writing reference-logit artifacts:

```text
NameError: name 'config' is not defined
```

The earlier full matrix retry also failed before artifacts after one preemption, but at a later diagnostic point:

```text
TypeError: while_loop body function carry input and carry output must have equal types
input carry: bfloat16[1,8,4,128]
output carry: float32[1,8,4,128]
```

These were harness/backend-dispatch bugs rather than Qwen3 numerical results. Local fixes:

- Thread `TpuPagedAttentionConfig` into `_run_tpu_inference_backend` so the `out_dtype` diagnostic knob is defined.
- Initialize the slow reference paged-attention output accumulator as f32 so the reference loop carry dtype is stable.

Validation after both fixes:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 54 passed
./infra/pre-commit.py --files lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/src/levanter/inference/tpu_inference_adapter.py lib/levanter/src/levanter/layers/attention.py lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py docs/debug-log-qwen3-tpu-logprob-numerics.md --fix
# OK
```

Submitted fixed replacement:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-fixed-20260601-0733`
- Initial state: `JOB_STATE_PENDING`
- Pending reason: v5p scale-up, not a code failure.

Also submitted a secondary 4-chip fixed matrix to reduce dependence on v5p capacity:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v6e-v5lite4-fixed-20260601-0736`
- TPU alternatives: `v5litepod-4,v6e-4`
- Same backend/cache matrix and `--reference-logit-only` contract as the v5p fixed run.

The secondary fixed run reached the reference-logit check on `v6e-4` but failed because the diagnostic threshold was too
low before the artifact logger ran:

```text
AssertionError: Levanter reference-logit check failed; see
/dev/shm/qwen3-parity-refonly-matrix-v6e-v5lite4-fixed/levanter_reference_logits.json for details
```

Submitted a capture rerun with the same matrix and `--reference-logit-atol 999` so the artifact is logged:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v6e-v5lite4-capture-20260601-0743`

The v5p fixed job later started with the same low diagnostic threshold, so submitted a v5p capture twin:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-capture-20260601-0745`

The secondary capture reached a `v6e-4` worker but failed during f32 model loading with HBM allocation failure before
the reference-logit check:

```text
RESOURCE_EXHAUSTED: RuntimeBufferAllocationFailure: Attempting to allocate 288.00M.
There are 71.37M free.; (1x0x0_HBM0)
```

This is not a matrix result. The v5p capture remains the primary path for the f32-reference diagnostic.

The restarted v5p fixed run then completed the reference-logit check but failed while logging artifacts because
`log_output_artifacts` assumed `summary.md` exists. Reference-only runs do not write that file:

```text
FileNotFoundError: /dev/shm/qwen3-parity-refonly-matrix-fixed/summary.md
```

Fixed the artifact logger to log whichever artifacts exist. The pending v5p capture was submitted before this local
fix, so submitted a logfix v5p capture:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-logfix-20260601-0757`

That logfix job was also preempted shortly after TPU init. Submitted the same logfix matrix without `--preemptible`:

- `/dlwh/qwen3-parity-levanter-refonly-matrix-v5p8-logfix-nonpreempt-20260601-0807`

## Hypothesis 2

If the matrix shows that the high logprob error is specific to `tpu_inference`, the likely first fix is RPA precision.
The installed `tpu-inference==0.19.0` RPA API exposes `out_dtype`, described in its source as "the dtype of the output
and the accumulator for matmul" with lower dtype improving performance and higher dtype improving accuracy. Current
Levanter leaves this unset, so bf16 queries produce bf16 RPA output/accumulation. Levanter also casts attention output
back to the residual dtype before `o_proj`.

## Changes to make

Add diagnostic/fix controls:

- `TpuPagedAttentionConfig.tpu_inference_out_dtype` to pass `out_dtype` through to `tpu-inference` RPA.
- `TpuPagedAttentionConfig.preserve_attention_output_dtype` to optionally keep f32 attention output through the output
  projection instead of immediately casting back to bf16.
- Benchmark CLI flags `--levanter-tpu-inference-out-dtype` and `--levanter-preserve-attention-output-dtype`.

## Results

Local validation for these controls passed in the same focused test/pre-commit commands above. A follow-up TPU run should
use these controls if Hypothesis 1 identifies RPA precision as the source.

## Hypothesis 3

The refdiag prompt has `positions=1`, so the broad logit drift can occur before any multi-token attention math: for a
single-token sequence, paged decode writes V into the KV cache and reads it back, while full forward consumes the
projected V directly. That makes KV-cache dtype a first-order suspect. `tpu-inference` RPA supports f32 KV cache in its
tests/source, but Levanter had no benchmark knob to exercise f32 serving/cache dtype and the support predicate only
documented the initial bf16 target.

## Changes to make

- Add `--levanter-compute-dtype` to the parity harness so the Levanter inference engine can use f32 KV cache.
- Add `--levanter-trainer-mp` so smaller-TPU runs can try bf16 model-policy diagnostics.
- Add temporary diagnostics for f32 KV cache, then keep `tpu_inference` shape support bf16-only after TPU evidence showed
  that f32 output/cache paths fail for this Qwen3 decode shape.

## Results

Local validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 57 passed
./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/src/levanter/inference/tpu_kernels.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py docs/debug-log-qwen3-tpu-logprob-numerics.md .agents/projects/levanter_tpu_inference_parity/research.md scratch/20260601-0816_f32kv_v5p_monitoring_state.json scratch/20260601-0816_bf16mp_f32kv_v6e_v5lite_monitoring_state.json --fix
# OK
```

Submitted the authoritative v5p f32-KV reference-only diagnostic:

- `/dlwh/qwen3-parity-levanter-refonly-f32kv-v5p8-20260601-0816`
- It acquired `v5p-8`, loaded the f32 model, and failed in `tpu-inference` when the matrix reached f32 KV cache:

```text
TypeError: cannot reshape array of shape (2, 4, 128) (size 1024)
into shape (4, 128) (size 512)
```

This came from `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py` while storing decode output. So f32 KV cache
is not a drop-in fix for the current `tpu-inference==0.19.0` decode path. Submitted two narrower v5p diagnostics:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32out-v5p8-20260601-0823`: `tpu_inference`, bf16 cache, f32
  RPA accumulator/output, preserve attention output dtype.
- `/dlwh/qwen3-parity-levanter-refonly-reference-cachematrix-v5p8-20260601-0823`: slow Levanter `reference` backend,
  bf16-vs-f32 cache matrix.

The narrower `tpu_inference` f32-output run retried once after preemption and then failed with the same reshape error,
even with bf16 KV cache. That points at f32 `out_dtype` itself, not only f32 KV packing:

```text
TypeError: cannot reshape array of shape (2, 4, 128) (size 1024)
into shape (4, 128) (size 512)
```

Tried to submit the slow-reference cache matrix with `--no-preemptible`, but Iris has no non-preemptible `v5p-8`
group. The preemptible slow-reference job is retrying after one preemption.

Submitted a smaller-TPU same-dtype attempt:

- `/dlwh/qwen3-parity-levanter-refonly-bf16mp-f32kv-v6e-v5lite4-20260601-0816`
- It acquired `v6e-4` and loaded checkpoint shards, but still failed during model load before reference-logit artifacts:

```text
RESOURCE_EXHAUSTED: RuntimeBufferAllocationFailure:
Attempting to allocate 288.00M. There are 71.37M free.; (0x0x0_HBM0)
```

This means `--levanter-trainer-mp bf16` alone is not enough to make Qwen3 8B reference diagnostics fit on `v6e-4`;
the current useful path is waiting for v5p.

## Hypothesis 4

The slow-reference/f32-cache v5p run succeeded but did not clear the mismatch:

- `/dlwh/qwen3-parity-levanter-refonly-reference-f32cache-v5p8-20260601-0839`
- backend: `reference`
- cache dtype: `float32`
- positions: 1
- max logit abs error: 0.25
- top-4096 max logprob abs error: 0.159118
- top-4096 mean logprob abs error: 0.0391007

This rules out bf16 KV cache and page traversal as the sole cause. The remaining likely source is a decode-vs-full
forward dtype transition outside RPA, especially hidden/logit rounding before or at the LM head.

## Changes to make

Add dtype attribution to `levanter_reference_logits.{json,md}`:

- reference logit dtype
- decode logit dtype
- max abs error explained by rounding reference logits to the decode dtype
- residual max abs error after that rounding

This tells whether the observed 0.125/0.25-sized errors are just decode-logit quantization or a real arithmetic/cache
mismatch.

## Results

Focused validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 56 passed
```

Submitted a narrower v5p follow-up:

- `/dlwh/qwen3-parity-levanter-refonly-reference-f32cache-preserve-v5p8-20260601-0848`
- slow reference backend, float32 cache, `--levanter-compute-dtype float32`, and
  `--levanter-preserve-attention-output-dtype`

This run later acquired v5p and started loading Qwen3 shards.

## Hypothesis 5

The `tpu-inference` f32-output failure is likely an adapter/kernel dtype-packing mismatch, not proof that f32
accumulation is impossible. In `tpu-inference==0.19.0`, the RPA v3 wrapper allocates the output double buffer and output
shape from prepared Q dtype/packing:

```text
bo_double_buf = bq_double_buf
out_shape = q.shape, q.dtype
```

Therefore, passing `out_dtype=float32` with bf16 Q asks the kernel to store f32 output through a bf16-shaped output
buffer, matching the observed reshape error.

## Changes to make

Cast Q to the requested `out_dtype` before calling `tpu-inference` RPA. K/V and KV cache remain bf16 for the initial
Qwen3 target.

## Results

Focused validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 57 passed
```

Submitted the fixed `tpu_inference` f32-output diagnostic:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32qout-v5p8-20260601-0854`
- bf16 KV cache, f32 Q/output, preserved attention output dtype

## Hypothesis 6

The slow-reference preserve run is not a valid parity fix:

- `/dlwh/qwen3-parity-levanter-refonly-reference-f32cache-preserve-v5p8-20260601-0848`
- backend: `reference`
- cache dtype: `float32`
- `--levanter-compute-dtype float32`
- `--levanter-preserve-attention-output-dtype`

It completed but worsened the mismatch:

| positions | max logit abs | mean logit abs | p99 logit abs | top-1 logprob abs | top-4096 max logprob abs |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.15741 | 0.391914 | 0.748282 | 0.458656 | 0.458656 |

That indicates the full-forward baseline includes the usual cast back to residual/model dtype before output projection;
preserving f32 attention output changes the decode path rather than matching the baseline.

## Hypothesis 7

The fixed f32-output `tpu_inference` diagnostic failed before artifacts, but past the earlier reshape point:

```text
ValueError: Expected kv_cache.dtype=dtype(bfloat16) to be equal to k.dtype=dtype('float32')
and v.dtype=dtype('float32').
```

This is an adapter-boundary bug. `KvPageCache.update` casts new K/V into the cache dtype, but the fused
`tpu-inference` path bypassed that helper and passed raw f32 K/V directly to the external kernel.

## Changes to make

Cast new K/V to `kv_cache.kv_pages.dtype` before padding/packing for `tpu-inference`. Keep the earlier Q cast to
`out_dtype`, because the v3 wrapper uses Q dtype to size its output buffers.

## Results

Focused validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/inference/test_tpu_paged_attention_backends.py -q
# 16 passed
```

The replacement TPU run succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32qout-v5p8-20260601-0902`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: true
- environment: TPU v5, `tpu_inference==0.19.0`, `vllm_tpu==0.19.0`, `jax==0.9.2`

It cleared the runtime failures but reproduced the bad preserve-reference numerics:

| positions | ref dtype | decode dtype | max logit abs | mean logit abs | p99 logit abs | top-1 logprob abs | top-4096 max logprob abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `bfloat16` | `float32` | 1.15741 | 0.391914 | 0.748282 | 0.458656 | 0.458656 |

Most logits moved into the `0.3..1` absolute-error bucket (`113035 / 151936`). This confirms that preserving f32
attention output through the residual/output-projection path is not a fix. The next diagnostic should use f32
`tpu-inference` accumulator/output internally but leave `preserve_attention_output_dtype=false`, so Levanter casts the
attention output back to the residual/model dtype before `o_proj`.

## Hypothesis 8

If the cast-back run still misses the logprob target, we need to know whether the error is already present in the
transformer hidden state or is introduced only by `lm_head_logits`.

## Changes to make

Extend `levanter_reference_logits.{json,md}` with hidden-state attribution from the same full-forward/decode comparison:

- reference/decode hidden dtype
- hidden max absolute error
- hidden mean absolute error
- hidden RMS absolute error

This keeps the next diagnostic focused: hidden error means continue inside paged decode/RPA/residual numerics; low hidden
error with high logit error means inspect LM-head projection dtype/sharding/logprob handling.

## Results

Focused validation:

```text
uv run --with pytest --with pytest-timeout pytest lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 42 passed

./infra/pre-commit.py --files lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py --fix
# OK
```

The f32-output/cast-back diagnostic succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-bf16cache-f32out-castback-v5p8-20260601-0910`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: false

It improved top-1 but did not clear the whole requested logprob target:

| positions | ref dtype | decode dtype | max logit abs | mean logit abs | p99 logit abs | top-1 logprob abs | top-4096 max logprob abs | top-4096 mean logprob abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `bfloat16` | `bfloat16` | 0.25 | 0.0444493 | 0.15625 | 0.0341177 | 0.159118 | 0.0391007 |

This matches the slow-reference/f32-cache result, so the remaining error is not specific to `tpu-inference` RPA.
Next step is to rerun the cast-back case with hidden-state attribution enabled, then use that result to decide whether
to continue in transformer/paged decode or in LM-head/logprob projection.

## Hypothesis 9

The residual top-k logprob error after the `tpu-inference` adapter fixes may already be present in transformer hidden
states, not introduced by the LM head or logprob calculation.

## Results

The hidden-attribution rerun succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-castback-hidden-v5p8-20260601-0922`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: false
- positions: 1
- environment: TPU v5, `tpu_inference==0.19.0`, `vllm_tpu==0.19.0`, `jax==0.9.2`

Reference-logit diagnostics:

| ref hidden dtype | decode hidden dtype | hidden max abs | hidden mean abs | hidden RMS abs | ref logits dtype | decode logits dtype | max logit abs | top-1 logprob abs | top-4096 max logprob abs | top-4096 mean logprob abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bfloat16` | `bfloat16` | 1.3125 | 0.0160123 | 0.0325309 | `bfloat16` | `bfloat16` | 0.25 | 0.0341177 | 0.159118 | 0.0391007 |

This rules out LM-head-only and logprob-only explanations for the remaining gap. The transformer hidden state already
has a large pointwise outlier for a one-token prompt, while the aggregate hidden error is modest. Because previous
slow-reference/f32-cache and `tpu_inference` cast-back runs match the same top-k logprob result, the remaining shared
issue is in the common full-forward-vs-paged-decode transformer path rather than the external RPA kernel alone.

## Changes to make

Add a one-token direct-attention diagnostic to the reference-logit artifact. For one-token prompts, causal attention is
mathematically the value projection path (`softmax([qk]) == 1`), so the harness can run a direct transformer pass that
replaces each self-attention kernel with `V -> o_proj` and compare it to both full forward and paged decode:

- direct hidden dtype
- direct-vs-full hidden max absolute error
- direct-vs-decode hidden max absolute error
- direct-vs-full logits max absolute error
- direct-vs-decode logits max absolute error

Interpretation for the next TPU run:

- direct matches decode but not full forward: full-forward attention lowering is the reference-side source.
- direct matches full forward but not decode: paged decode remains the source.
- direct differs from both: the mismatch is likely a shared dtype/rounding boundary outside attention itself.

## Hypothesis 10

The remaining drift is not the singleton attention kernel itself but the layer container used by the full-forward
reference. Paged decode manually unrolls transformer layers, while `model.activations()` uses `Stacked.fold`. In bf16,
those two execution orders can differ enough to dominate the one-token reference-logit diagnostic.

## Changes to make

Keep the singleton-attention fast path for the safe `Q=K=1` case, but change the reference-logit checker to build its
full-sequence reference by explicitly unrolling `model.transformer.layers.unstacked()`. This keeps the reference causal
full-forward computation, but uses the same layer evaluation order as paged decode. It avoids changing Levanter's
training loss path, which still uses `model.activations()` and `Stacked.fold`.

## Results

Local bf16 reproduction after the harness change:

```text
unrolled-decode hidden max 0.0
```

Focused validation:

```text
uv run --with pytest --with pytest-timeout pytest \
  lib/levanter/tests/inference/test_tpu_paged_attention_backends.py \
  lib/levanter/tests/test_qwen3.py \
  lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py -q
# 61 passed, 1 skipped

./infra/pre-commit.py --files \
  lib/levanter/src/levanter/layers/attention.py \
  lib/levanter/scripts/bench/bench_qwen3_tpu_inference_parity.py \
  lib/levanter/tests/test_qwen3.py \
  lib/levanter/tests/test_qwen3_tpu_inference_parity_bench.py \
  docs/debug-log-qwen3-tpu-logprob-numerics.md \
  .agents/projects/levanter_tpu_inference_parity/research.md \
  scratch/20260601-0941_tpu_directv_v5p_monitoring_state.json --fix
# OK, including Pyrefly
```

Next TPU proof: rerun the one-token Qwen3 8B reference-logit check with the unrolled reference. If this is the true
source, the top-k logprob gap should drop from `0.159118` to near zero for the one-token rollout prompt.

The TPU proof run succeeded:

- `/dlwh/qwen3-parity-levanter-refonly-unrolledref-v5p8-20260601-0952`
- backend: `tpu_inference`
- cache dtype: `bfloat16`
- RPA output dtype: `float32`
- preserve attention output dtype: false
- `--reference-logit-atol 0.05`
- environment: TPU v5, `tpu_inference==0.19.0`, `vllm_tpu==0.19.0`, `jax==0.9.2`

Reference-logit diagnostics with the unrolled full-reference checker:

| hidden max abs | hidden mean abs | hidden RMS abs | max logit abs | top-1 logprob abs | top-4096 max logprob abs | top-4096 overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 | 0 | 1 |

The original high top-k logprob error was therefore a reference-side artifact from comparing paged decode against
`Stacked.fold` full-forward activations. It was not caused by `tpu-inference` RPA after the adapter fixes. For the
decode-heavy one-token rollout regime, Levanter paged decode with `tpu-inference` bf16 cache, f32 RPA output, and
cast-back before `o_proj` is numerically identical to the unrolled Levanter full-reference path.

The earlier direct-attention attribution run also succeeded and corroborates the same conclusion:

- `/dlwh/qwen3-parity-levanter-refonly-tpu-directv-v5p8-20260601-0941`

Key attribution row from that run, before switching the checker reference to unrolled full forward:

| hidden max abs | direct-ref hidden max abs | direct-decode hidden max abs | max logit abs | direct-ref logits max abs | direct-decode logits max abs | top-4096 max logprob abs |
| --- | --- | --- | --- | --- | --- | --- |
| 1.3125 | 1.3125 | 0 | 0.25 | 0.25 | 0 | 0.159118 |

So the exact one-token attention path matched paged decode and not the scanned full-forward reference. That independently
pins the original error on the scanned reference path rather than on RPA/cache update/LM-head/logprob code.
