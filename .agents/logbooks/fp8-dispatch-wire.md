---
topic: fp8-dispatch-wire
issue: https://github.com/marin-community/marin/issues/7665
description: MXFP8 forward-dispatch wire for expert-parallel MoE — quantize once before the dispatch collective and feed the bytes straight to the MXFP8 grouped GEMM.
author: Claude agent (supervised by @mcwitt)
---

# FP8 forward-dispatch wire: Task Logbook

## Scope

- **Goal:** carry the EP forward dispatch payload as MXFP8 with no added quantization
  compute, by relocating the expert MLP's existing post-dispatch quantize to before the
  collective.
- **Primary metrics:** p50 MFU at a matched drop regime; exposed collective time; added
  quantize compute (ms/step). Fidelity gates: gradient parity vs the bf16-wire arm,
  drop-fraction parity, backward cotangent underflow rate.
- **Constraints:** forward dispatch leg only. Reduction legs stay bf16 (the Hopper
  permutation-legs-only rule, #6911). Collective count must not increase — the EP64 profile
  attributes cost to many small per-layer legs and leg-batching is one of the few measured
  wins.
- **Coordinating issue:** https://github.com/marin-community/marin/issues/7665
- **ID prefix:** `FP8W`. **Tags:** `FP8W`, `7665`, `7279`, `7282`.

## Baseline

- **Date:** 2026-07-26
- **Branch:** `research/mcwitt/7279-fp8-dispatch-wire`, based on
  `research/mcwitt/7282-mxfp8-blackwell` @ `0a3785463` — the only tree carrying both
  `MxFp8MoeMlpOp` and `levanter.grug._moe.fp8_wire`.
- **Code refs:**
  - `lib/levanter/src/levanter/grug/_moe/fp8_wire.py` — the round-trip wire under test as
    the control. `_fp8_all_gather_impl` dequantizes to bf16 before returning.
  - `lib/levanter/src/levanter/grug/_moe/ep_ring.py` — ring EP backend; `fp8_all_gather`
    at the dispatch, `expert_mlp_op` consumes bf16 `x_dispatch`.
  - `experiments/grug/moe/mxfp8.py` — `_forward_pipeline` calls `_dual_quantize` to build
    both orientations from bf16; column orientation is a VJP residual for the wgrads.
  - `experiments/grug/moe/standalone/mxfp8_grouped/quantize.py` — `quantize_mxfp8`
    (feature axis), `quantize_mxfp8_tokens` (token axis), `build_sfa_fast` (swizzle).
- **Baseline numbers carried in from prior threads (not re-measured here):**
  - #7279 round-trip FP8 permutation wire, bf16 GEMMs: **-2.02pp**, with 936 ms/step of
    exposure recovered and 2182 ms/step of quantize compute added (break-even 920).
  - #7282 MXFP8 GEMMs with wire disabled: 1.308x clean at 64 GPUs (MXFP8-010); 0.749x at
    EP1 D6144 in the #7201 stack; +7.2% end-to-end at the #7271 66B-token quality gate.
  - #6911 Hopper: fused e4m3 dispatch (H3/H4) 1.467x vs round-trip wire 1.375x at EP32.

## Hypothesis Queue

Living queue; updated as hypotheses are proposed, blocked, falsified, or promoted.

| id | hypothesis | state |
|---|---|---|
| H1 | Quantizing before the dispatch is bit-identical to quantizing after, so the forward operand is free. | **confirmed** (FP8W-001, CPU) |
| H2 | The wgrad column operand can be rebuilt from the arrived payload without material loss. | **confirmed** (FP8W-001, CPU) |
| H3 | Packing scales inside the payload buffer keeps the collective count unchanged at 33/64 of bf16 bytes. | proposed; **revised** by FP8W-003 — the packed uint8 buffer must stay inside the wire's `custom_vjp` |
| H8 | The quantized payload can cross the op boundary as a differentiated argument, so the quantize need not be fused into the op's VJP. | **FALSIFIED** (FP8W-005). FP8W-003's confirmation was an artifact of a hand-written backward; both carrier dtypes corrupt a real cotangent. Fusing is required. |
| H9 | Packing payload and scales into one uint8 buffer keeps the byte ratio at 33/64 and the collective count at one. | **confirmed** (FP8W-005, unit test) |
| H4 | Relocating the quantize reduces producer volume by ~topk x cf (8.5x at 8-of-256, cf1.0625). | proposed |
| H5 | The net effect is positive at layer level inside the real step with remat on. | proposed, blocked on implementation |
| H6 | Most of #7279's -2.02pp was the round trip, not the per-token scaling granularity. | proposed (control arm) |
| H7 | `quantize_mxfp8` NaNs on all-zero blocks on GPU as it does on XLA CPU. | proposed, needs sm100 |

## Entry Log

### 2026-07-26 14:20 - FP8W-001: CPU numerics study, both fidelity objections retired

- **Hypothesis:** H1 and H2. MXFP8 blocks lie along the feature axis while the dispatch
  permutes the token axis, so quantization is a per-row attribute that travels with its
  row (H1). The wgrad's token-axis orientation cannot cross the permutation — its scale is
  a property of a set of 32 rows and routing dissolves that set — so it must be rebuilt
  from the arrived payload; because e8m0 scales are exact powers of two and e4m3 is a
  floating-point grid, that rebuild should be a mantissa-preserving exponent shift rather
  than a second rounding, diverging only into subnormals (H2).
- **Commit hash:** see the commit adding
  `experiments/grug/moe/standalone/study_fp8_wire_numerics.py` on this branch.
- **Command:**
  `uv run python experiments/grug/moe/standalone/study_fp8_wire_numerics.py`, importing
  `mxfp8_grouped/quantize.py` unmodified. CPU jax 0.10.1. Synthetic dispatch buffers
  `[2048, 1024]` with per-token log-normal magnitude spread and a minority of high-gain
  feature channels, swept over four regimes from benign (sigma=0.3, no outlier channels)
  to adversarial (sigma=3.0, 5% of channels at 1000x). Both the activation and the
  cotangent are drawn from the regime under test.
- **Config:** Path A (today) = `quantize_mxfp8_tokens(x)` from the bf16 original. Path B
  (proposed) = `quantize_mxfp8(x)` -> `dequantize_mxfp8` (exactly what arrives off the
  wire) -> `quantize_mxfp8_tokens`. The H1 test additionally applies a routing-shaped
  gather: arbitrary order, tokens replicated to multiple slots, tokens absent, plus a
  validity mask, comparing quantize-then-gather against gather-then-quantize.
- **Result:**
  - H1: **bit-identical**. All 1,726 valid rows of 2,048 matched on e4m3 bytes, e8m0
    scales, and dequantized values.
  - H2: added wgrad error `||B-A|| / ||A-ref||` = 2.7e-6 (benign), 6.0e-6 (typical),
    3.1e-5 (harsh), 1.1e-3 (adversarial), against an accepted noise floor `relerr(A)` of
    3.0e-2 to 4.2e-2 in the same arms. Value-level agreement 100.00% / 99.99% / 99.88% /
    96.88%. Divergence confined to underflow as predicted: extra flush-to-zero +0.0002pp,
    +0.0008pp, +0.0081pp, +0.2000pp.
  - Incidental: `quantize_mxfp8` returns **NaN on an all-zero 32-element block** on XLA
    CPU. `amax = 0` -> e8m0 byte 0 -> scale `2^-127`, a subnormal f32; XLA CPU flushes
    denormal divisors, so the division is 0/0. Confirmed directly:
    `jit(0.0 / 2**-127)` is NaN, `jit(1.0 / 2**-127)` is inf.
- **Interpretation:** the fidelity objection to the design is retired. The forward operand
  is free, not merely acceptable. The wgrad rebuild sits four to five orders of magnitude
  below the quantization noise floor the recipe already accepts, so it is not a plausible
  source of quality regression. Limitations: synthetic activations rather than captured
  dispatch buffers, and this exercises the XLA producer's math — the CuTe producer is
  separate code, though the exponent-shift argument does not depend on the producer.
  The NaN finding is on the production XLA producer path (`dual_quantize_activation` calls
  `quantize_mxfp8` and `quantize_mxfp8_tokens` on the activation) and zero rows are routine
  between dropped slots and the 256-row pad; #7271 trained 66B tokens on this path, which
  is strong evidence it does not bite on GPU. It should be checked, not assumed. Note the
  wire is better behaved here: masking payload and scale bytes to zero yields exact zeros.
- **Next action:** implement the wire (H3, H4) on the ring EP backend, then FP8W-002 layer
  A/B. Fold the GPU NaN probe (H7) into the first sm100 job rather than paying for a
  dedicated one.

### 2026-07-26 14:45 - FP8W-002: scaffolding and compute survey

- **Result:** experiment issue #7665 created; this logbook started; research branch and an
  isolated worktree created at `/home/marin/projects/marin-fp8wire` so the primary
  checkout's untracked work is untouched. Compute survey: `cw-us-east-08a` (GB200) and
  `cw-us-east-02a` controllers are healthy but scaled to zero workers with no scale groups
  configured; `marin` reports 455/455 healthy workers.
- **Interpretation:** implementation is the blocking path and needs no accelerator. Defer
  any sm100 request until there is something to measure, and bundle the H7 probe into it.
- **Next action:** implement `quantize_source` on the op protocol and the packed payload
  buffer.

### 2026-07-26 15:10 - FP8W-003: the payload carrier dtype decides whether the gradient survives

- **Hypothesis:** H8. The design has the backend quantize before the collective, permute
  the payload, and hand it to the op, so a cotangent must reach the bf16 source through a
  quantized intermediate. If that is impossible, the modular boundary collapses and the
  quantize has to be fused into the op's `custom_vjp` — the shape Hopper's H3/H4 used
  ("byte all_gather -> byte take -> w13 wgmma under one custom_vjp", #6911).
- **Commit hash:** see the commit adding
  `experiments/grug/moe/standalone/study_fp8_wire_ad.py` on this branch.
- **Command:** `uv run python experiments/grug/moe/standalone/study_fp8_wire_ad.py`.
- **Result:** with the op modelled as a `custom_vjp` handing back a straight-through bf16
  cotangent, exactly as `fp8_all_gather` does today:

  | carrier | gradient | verdict |
  |---|---|---|
  | bfloat16 (control) | `[1 1 1 1]` | propagates |
  | float8_e4m3fn | `[1 1 1 1]` | propagates |
  | uint8 (bitcast) | `[0 0 0 0]` | silently zero, no error raised |

- **Interpretation:** H8 holds, so no fusing is required and the modular boundary in the
  design survives — but under a constraint the obvious implementation violates. A uint8
  payload has a float0 tangent type, so JAX drops the cotangent without raising. The
  existing `fp8_wire` module bitcasts its payload to uint8 by deliberate design
  ("permutation collectives move bytes, and this keeps the wire format independent of
  backend FP8 dtype support"), which is safe there only because `_fp8_all_gather_impl`
  dequantizes to bf16 inside its own `custom_vjp`, so uint8 never crosses an autodiff
  boundary. Reusing that idiom across the new boundary would train on a silently zeroed
  gradient. This is the same failure class as #7279's fp8 underflow that `crash_on_nan`
  could not see: a wrong-but-finite value that every guard passes.

  Consequence for H3: the packed `[T, 33H/32]` uint8 buffer stays *inside* the wire's
  `custom_vjp` — bitcast to uint8 for the collective, split, and bitcast the payload back
  to `float8_e4m3fn` before returning. The e8m0 scales are returned as a separate uint8
  array and are never differentiated, so their dtype is unconstrained. Whether NCCL through
  XLA accepts a float8-typed collective directly is now moot, since the collective still
  sees uint8.
- **Next action:** implement against that constraint, and add a gradient-nonzero assertion
  to the parity tests so a future refactor back to a uint8 carrier fails loudly.

### 2026-07-26 15:25 - FP8W-004: collective and gather dtype support for the payload

- **Hypothesis:** the payload must be float8-typed at the op boundary (FP8W-003), so the
  question is whether the backend's collective and gather accept that dtype, or whether a
  uint8 bitcast pair is needed around the collective.
- **Command:** ad-hoc CPU check, `jax.lax.all_gather(..., tiled=True)` under `shard_map`
  on a one-device mesh with `check_rep=False`, plus `jnp.take`.
- **Result:** `jnp.take` on `float8_e4m3fn` returns `float8_e4m3fn`. `all_gather(tiled=True)`
  lowers for `bfloat16`, `uint8`, and `float8_e4m3fn` alike.
- **Interpretation:** on CPU the backend can carry a float8-typed packed buffer end to end,
  which is the simplest shape: the op returns `[T, H + H/32]` float8 (e4m3 payload followed
  by e8m0 scale bytes reinterpreted as e4m3, recovered by bitcast on arrival), and the
  backend applies its existing collective and gather to it unchanged. This is untested on
  GPU, where NCCL datatype support is the open question and is the stated reason the
  existing `fp8_wire` bitcasts to uint8. Fallback if NCCL rejects fp8: wrap
  `bitcast -> collective -> bitcast` in a small dtype-generic `custom_vjp` on the levanter
  side, so the uint8 window does not straddle an autodiff boundary. Keeping the collective
  itself on uint8 would otherwise reintroduce the FP8W-003 silent-zero failure in the
  middle of the backend.
- **Next action:** implement, then verify the collective dtype on the first sm100 job
  alongside the H7 zero-block probe.

### 2026-07-26 16:05 - FP8W-005: wire primitive landed; H8 falsified, fusing is required

- **Hypothesis:** H3, H9 (packing and byte ratio), and a re-test of H8 under a real
  downstream consumer rather than the hand-written backward FP8W-003 used.
- **Commit hash:** see the commit adding `lib/levanter/src/levanter/grug/_moe/mxfp8_wire.py`.
- **Command:**
  `XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=$(ls -d lib/*/src | tr '\n' ':')
  .venv/bin/python -m pytest experiments/grug/moe/test_mxfp8_wire_parity.py
  lib/levanter/tests/grug/test_mxfp8_wire.py -q -n 0` — 17 passed. The device-count flag is
  required: the sharded tests skip under two devices, so a default CPU run silently covers
  only the unsharded half.
- **Result:**
  - The quantizer is bit-exact against the vendored `mxfp8_grouped.quantize` reference the
    expert kernels use, over three seeds of heavy-tailed activations.
  - Quantize-then-gather equals gather-then-quantize through a routing-shaped gather, now
    as a unit test rather than a study script. Masking a row after quantization equals
    quantizing a masked row.
  - Packing gives exactly `33/64` of bf16 bytes (asserted on `nbytes`), one collective.
  - **H8 is falsified.** FP8W-003 concluded a float8 carrier propagates the cotangent
    correctly. It propagates, but JAX matches the cotangent to the primal's tangent type,
    so a bf16 `dx` handed back through the payload is downcast to *unscaled* e4m3. A
    downstream `custom_vjp` returning a 1e-6 bf16 gradient — ordinary for a cotangent —
    gets flushed to exactly zero, since 1e-6 is below e4m3's smallest subnormal. Pinned as
    `test_a_cotangent_crossing_the_payload_is_silently_corrupted`.
- **Interpretation:** FP8W-003's probe was wrong because I wrote the backward by hand and
  returned bf16 directly, so nothing downstream ever produced a cotangent *of the payload*;
  the cast never happened. Under a real consumer it does. Both carrier dtypes therefore
  fail, differently and silently: uint8 zeroes the gradient outright, float8 quantizes it
  to a format with no range. The design's modular boundary does not survive, and the wire
  must be fused — quantize through expert MLP under one `custom_vjp`, bf16 on both sides.
  That is the shape Hopper's H3/H4 used, and this is presumably why.

  The fused wrapper cannot call the op's `custom_vjp` either, for the same reason: its
  activation input would be the quantized payload. It has to drive `_forward_pipeline` and
  `_backward_pipeline` in `experiments/grug/moe/mxfp8.py` directly, which those are already
  factored as module-level functions for. That is a larger protocol change than
  `quantize_source`, so it is not written yet; the module ships the verified primitive with
  `mxfp8_all_gather` documented as non-differentiable.

  Two process notes. A default CPU pytest run skips every sharded test, so the corruption
  would not have been caught without forcing the device count. And the worktree has no
  `.venv`, so `pre-commit.py` reports ~1,465 repo-wide missing-import errors; ruff was run
  directly instead and pyrefly still needs a real run before any PR.
- **Next action:** write the fused `custom_vjp` against the pipeline seams, then the layer
  A/B. GB200 capacity is available on `cw-us-east-08a` (the earlier 0/0 worker reading was
  wrong — the config has on-demand `gb200-4x` scale groups and jobs were completing during
  this session).

### 2026-07-26 22:34 - FP8W-006: fused path runs on GB200; end-to-end parity and first timing

- **Hypothesis:** H4, H5, and the fused-VJP shape forced by FP8W-005.
- **Commit hash:** `ac4874e1f` (fused path at `3999982e5`).
- **Command:** `iris --cluster cw-us-east-08a job run --user mwittmann --gpu GB200x4
  --enable-extra-resources --cpu 16 --memory 64g --extra gpu --job-name
  fp8w-006-dispatch-parity-r2 -- python
  experiments/grug/moe/standalone/test_mxfp8_dispatch_gpu.py --tokens 512 --hidden 512`
- **Config:** 1x GB200x4 (sm100, cc 10.0), jax 0.10.1, ring EP over 4 devices, XLA
  producer, T=512 gathered tokens, D=512, I=256, E=16, top-4, capacity factor 1.25.
  Deliberately small: this is a correctness and smoke run, not the operating point.
- **Result:** control (bf16 dispatch, op quantizes on arrival) vs treatment (quantize
  before the collective):

  | metric | value |
  |---|---|
  | dx nonzero | true |
  | dw13 relfrob | 0.0 |
  | dw2 relfrob | 0.0 |
  | dx relfrob | 9.88e-4 |
  | forward relfrob | 7.67e-4 |
  | fwd speedup | 1.035x |
  | fwd+bwd speedup | 1.017x |

- **Interpretation:** the fused custom VJP works end to end on real kernels and the
  dispatch gradient survives, which is the FP8W-005 failure mode cleared. Both weight
  gradients are exactly equal. The speedups are at a toy shape where the collective is a
  negligible share of the step, so they carry no information about the operating point;
  they are reported only to show the direction is not negative. The forward relfrob of
  7.67e-4 contradicted the bit-identical prediction and is chased in FP8W-007.
- **Next action:** isolate the forward difference at operand level.

### 2026-07-26 22:38 - FP8W-007: the wire is exact; the e2e delta is downstream of the op

- **Hypothesis:** the row-orientation operand is bit-identical (H1 on real kernels), so the
  FP8W-006 forward difference arises somewhere other than the wire.
- **Commit hash:** `ac4874e1f`.
- **Command:** as above with `--check-operands --hidden 512`, job
  `fp8w-007-operands-r2`. Feeds one dispatch buffer (512 rows, a 96-row all-zero tail,
  uneven group sizes) through `_forward_pipeline` and `_forward_pipeline_quantized` and
  diffs the operands the grouped kernels receive.
- **Result:**

  | check | value |
  |---|---|
  | row operand bit-identical | true |
  | row scales bit-identical | true |
  | column operand bit-identical | false |
  | y relfrob | 0.0 |
  | control x_q has NaN | false |
  | treatment x_q has NaN | false |

- **Interpretation:** H1 confirmed on sm100, not just on CPU: quantizing before the
  dispatch produces byte-identical forward operands and an exactly identical forward
  output. The column operand differs, which is the intended rebuild measured in FP8W-001.
  The FP8W-006 end-to-end difference is therefore downstream of the op. The most likely
  source is the bf16 scatter-add in the ring combine, whose accumulation order over
  duplicate token indices XLA is free to schedule differently between the two graphs;
  7.67e-4 is the right magnitude for bf16 accumulation-order noise. Not yet proven, and it
  is the one loose end in the parity story.

  **H7 is answered as a side effect: no NaN on sm100.** The control path quantizes an
  all-zero-tail buffer through the vendored reference and produces no NaN, so the 0/0 that
  XLA CPU produces is a denormal-flush artifact of that backend and does not affect the
  #7271 training path. The wire's masking remains the more robust behaviour but is not
  fixing a live GPU bug.
- **Next action:** confirm the combine-ordering hypothesis, then run the layer A/B at the
  #7201 operating point (d5120 or d6144, 48 layers, 4-of-128), where the collective is a
  real share of the step and the timing means something.
