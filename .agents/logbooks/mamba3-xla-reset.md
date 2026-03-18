# Mamba-3 XLA Reset: Research Logbook

## Scope
- Goal: rebuild the Mamba-3 TPU path around a clean SSD-style XLA decomposition, then benchmark it on v5p-8 before reconsidering any Pallas kernel.
- Primary metric(s): forward/backward steady-state time, compile-including time, tokens/s, and estimated contraction throughput.
- Constraints: SISO real-valued only, pure JAX/XLA is the default path, and no Pallas path ships unless it is both correct and faster.

## Design Note
- Direct recurrence:
  - `h_t = alpha_t * h_{t-1} + beta_t * v_{t-1} + gamma_t * v_t`
  - `v_t = B_t x_t^T`
  - `alpha_t = exp(dt_t * A_t)`
  - `beta_t = (1 - lambda_t) * dt_t * alpha_t`
  - `gamma_t = lambda_t * dt_t`
- Rewrite:
  - `q_t = (1 - lambda_t) * dt_t`
  - `g_t = h_t + q_{t+1} * v_t`
  - then `g_t = alpha_t * g_{t-1} + (lambda_t * dt_t + q_{t+1}) * v_t`
  - and `y_t = C_t^T g_t - q_{t+1} * (C_t^T B_t) * x_t`
- Chunked decomposition:
  - local intra-chunk quadratic SSD block,
  - chunk-end state summaries,
  - inter-chunk scan,
  - prefix-state emission,
  - diagonal Mamba-3 correction outside the hot block.
- Important detail: `q_{t+1}` must shift across chunk boundaries, not reset at the end of every chunk. The new chunked-scale helper handles the shift over the flattened `(chunks, chunk)` sequence.
- XLA vs Pallas:
  - XLA owns the shipping path.
  - Pallas is currently absent on purpose. It can come back only for the intra-chunk local block after profiling shows the XLA formulation is correct and that the local block dominates.

## Experiment Log
### 2026-03-18 - Reset in clean worktree
- Hypothesis: the right restart point is the SSD/XLA rewrite plus a direct paper oracle, not the earlier bespoke TPU kernel.
- Planned code refs:
  - `lib/levanter/src/levanter/kernels/pallas/ssd/`
  - `lib/levanter/src/levanter/kernels/pallas/mamba3/`
  - `lib/levanter/tests/kernels/test_pallas_mamba3.py`
  - `lib/levanter/scripts/bench/bench_mamba3_xla.py`
- Status: implementation in progress.

### 2026-03-18 - Algebraic rewrite and local validation
- Hypothesis: the `g_t = h_t + q_{t+1} v_t` rewrite is valid if and only if `q_{t+1}` is shifted across chunk boundaries rather than reset inside each chunk.
- Result:
  - Added a direct recurrent oracle on native `(dt, lam, A, B, C, x)` inputs.
  - Added a transformed sequential oracle and chunked SSD/XLA path.
  - Added explicit tests for chunk-boundary shifting and for direct-oracle parity.
  - Local validation: `uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_mamba3.py -q` passed with `11 passed`.
- Interpretation:
  - The rewrite is numerically correct in this tree.
  - The old “shift only along the last axis” helper would have been wrong for multi-chunk inputs; the new chunked-scale helper fixes that.
- Next action:
  - Benchmark direct reference vs XLA on v5p-8 and decide whether any Pallas follow-up is justified.

### 2026-03-18 - v5p-8 throughput on the clean XLA path
- Command:
  - `ssh dev-tpu-dlwh-mamba3-xla-03f9 '... .venv/bin/python lib/levanter/scripts/bench/bench_mamba3_xla.py --seq-lens 2048,8192 --batch-head-groups 16 --chunk-size 128 --state-dim 128 --value-dim 512 --steps 3 --warmup 1 --json'`
  - `ssh dev-tpu-dlwh-mamba3-xla-03f9 '... .venv/bin/python lib/levanter/scripts/bench/bench_mamba3_xla.py --seq-lens 16384 --batch-head-groups 16 --chunk-size 128 --state-dim 128 --value-dim 512 --steps 3 --warmup 1 --json'`
- Config:
  - Device: `v5p-8`
  - Dtype: `bfloat16`
  - Shape family: `batch_head_groups=16, chunk_size=128, state_dim=128, value_dim=512`
- Result:
  - `seq_len=2048`
    - reference forward: `0.02263 s`, `1.45M tok/s`, `0.62` estimated TFLOP/s
    - reference backward: `0.05855 s`, `0.56M tok/s`, `0.24` estimated TFLOP/s
    - XLA forward: `0.000531 s`, `61.74M tok/s`, `26.30` estimated TFLOP/s
    - XLA backward: `0.000854 s`, `38.37M tok/s`, `16.35` estimated TFLOP/s
  - `seq_len=8192`
    - reference forward: `0.09004 s`, `1.46M tok/s`, `0.62` estimated TFLOP/s
    - reference backward: `0.23393 s`, `0.56M tok/s`, `0.24` estimated TFLOP/s
    - XLA forward: `0.001676 s`, `78.23M tok/s`, `33.32` estimated TFLOP/s
    - XLA backward: `0.002911 s`, `45.03M tok/s`, `19.18` estimated TFLOP/s
  - `seq_len=16384`
    - reference forward: `0.17998 s`, `1.46M tok/s`, `0.62` estimated TFLOP/s
    - reference backward: `0.46756 s`, `0.56M tok/s`, `0.24` estimated TFLOP/s
    - XLA forward: `0.003179 s`, `82.46M tok/s`, `35.13` estimated TFLOP/s
    - XLA backward: `0.005687 s`, `46.10M tok/s`, `19.64` estimated TFLOP/s
- Interpretation:
  - The clean XLA decomposition is decisively faster than the direct paper oracle, with forward speedups of about `42.6x`, `53.7x`, and `56.6x` at `seq_len` `2048`, `8192`, and `16384`.
  - The main bottleneck is no longer algebraic correctness; the structured XLA local block plus scan is already giving respectable TPU throughput.
  - Reintroducing Pallas is not justified yet. The prior branch’s TPU Pallas experiments were both slower and incorrect, while this tree now has a correct XLA default with strong long-context performance.
- Decision:
  - Keep XLA as the only shipping/default implementation in this reset.
  - Defer any new Pallas work until profiling isolates a dominant intra-chunk local block that is still worth attacking after this rewrite.
