# Per-long-layer KV heads (grug MoE)

## Goal
Long layers (`i % 4 == 3 or is_last`, 12 of 48 at L48) use **4** KV heads; short layers keep the
baseline **10** (40 query heads ÷ gqa_ratio 4). Everything else baseline. Motivation: cut the
attention/KV-cache cost of the wider-window "long" layers, which are the expensive ones.

## Why it isn't a config knob
- The model has a **single global** `num_kv_heads` (`GrugModelConfig.num_kv_heads`). Long vs short
  layers differ only in mask (window), PKO, RoPE — never KV heads.
- With `SCALE_SCAN_LAYERS=1` all 48 layers are one `ArrayStacked[Block]` run through **one
  `lax.scan`** over a **uniform** stacked weight tensor. Different KV heads → different k/v-proj
  shapes → not representable in one uniform stack.
- FA4 (`_fa4_cute_backend`) sets `qhead_per_kvhead = q.shape[2] // k.shape[2]` at trace time and
  **compiles one kernel per ratio**. Long (40/4=10) and short (40/10=4) need **two** attention
  kernels. A single scan cannot switch kernels per step.

## Options

### A. Unroll + per-layer KV heads (`SCALE_SCAN_LAYERS=0`)
Build each layer as its own `Block`; long-layer blocks get `num_kv_heads=4`, short get 10. Each
layer's attention compiles for its own ratio (fine, they're separate subgraphs).
- **Pros:** localized change (layer-construction loop + a per-layer `num_kv_heads`); exact.
- **Cons:** 48 separate layer subgraphs → **slow compile** and higher per-layer overhead; changes
  the baseline's scan perf profile, so a throughput A/B is no longer apples-to-apples (would need
  the baseline unrolled too). Fine for a **quality** experiment; muddy for **throughput**.

### B. Block-of-4 scan (recommended for throughput)
The long-layer pattern is exactly `[S, S, S, L] × 12` (long at 3,7,…,47). Restructure into a
**12-block scan**, each block applying 3 short layers (10-KV weights) then 1 long layer (4-KV
weights). Two weight stacks: `short[12,3,…]` (10 KV), `long[12,1,…]` (4 KV); two compiled attention
kernels (one per ratio). Preserves layer order and keeps scanning.
- **Pros:** exact; stays scanned (compile-efficient); honest throughput number.
- **Cons:** **large** change — model construction, the scan body, weight init, sharding
  (`ArrayStacked` layout), and the long/short attention wiring all move to the block structure.
  Touches `model.py` heavily and interacts with FSDP/expert sharding.

### C. Masked/replicated KV in the uniform scan — rejected
Keep 10-KV weights, slice/replicate to 4 distinct KV heads on long layers. Blocked: FA4's fixed GQA
ratio means the attention op still runs at ratio 4 (10 KV), so no compute saving, and replicating 4
distinct KV into 10 slots is a hack that doesn't reduce KV cache. Doesn't achieve the goal.

## Recommendation
- If the question is **"does per-long-layer 4-KV hurt/help quality"** → **A (unroll)**, quickest to a
  correct experiment; ignore the throughput delta.
- If the question is **"throughput/MFU of the 4-KV long-layer variant"** → **B (block scan)**, the
  only apples-to-apples option, but a multi-file change worth its own PR.

## Open decision
Which experiment (quality via A, or throughput via B)? Drives the implementation path.
