# Wall Attention (tilde-research) — TPU/JAX port

Replace BOTH the sliding-window (short) and full-causal (long) attention in the Grug MoE
baseline with **wall attention**: softmax attention + per-channel multiplicative decay.

## Math (from wall_attn/reference.py)
Per head, per-channel log-decay gate `g_t ≤ 0` (data-dependent). Cumulative `P_i = Σ_{t≤i} g_t`.

    score_ij = scale · Σ_k q_ik · k_jk · exp(P_ik − P_jk)     (causal j ≤ i)
    logit_ij = score_ij + (C_i − C_j)                         (+ optional FoX scalar gate, C = cumsum(g_scalar))
    softmax_j(logit_ij), with an attention-sink term exp(sink − m) added to the denominator
    o_i = Σ_j softmax_ij · v_j
    window: keep only j with (i − j) < window_size

`g=0` ⇒ vanilla softmax. Short layers = wall + window (sliding_window//2); long layers = wall + no window.

## Numerical stability (chosen approach: chunked flash-style JAX, no Pallas)
Naive `q·exp(P)`, `k·exp(−P)` rescale overflows over 4k tokens. Fix: **per-query-block reference**.
For a query block ending at row `b` (block size C_blk), reference `P_b` (per channel):
- `q'_ik = q_ik · exp(P_ik − P_bk)`  → exponent ∈ [0, block_decay], bounded (i ≤ b).
- `k'_jk = k_jk · exp(P_bk − P_jk)`  → exponent ≤ 0 for j ≤ b; far-back keys underflow to 0 (correct).
- `score_block = scale · (q'_block @ k'^T)` is **exact** (`exp(P_i−P_j)` for any i in block) and stable.
Then masked (causal + window + segment) softmax with sink, `@ v`. `lax.scan` over query blocks;
memory O(C_blk·S) per block, total compute O(S²·d) (same as attention — no asymptotic penalty).

Segment handling: cross-doc pairs are masked in the softmax (−inf). P need NOT reset at boundaries —
`P_i − P_j` for same-doc contiguous i,j already excludes other docs' decay (cancels in the difference).

## Gate / norms
- Per-channel `g`: low-rank projection (GLA-style) + logsigmoid / normalizer, per KV head (GQA-repeated).
- Scalar `g_scalar`: per-head linear + logsigmoid (optional, default off initially).
- `sink_bias`: learnable per-head scalar (default on).
- QK RMSNorm on q,k (matches baseline CausalSelfAttention).
- GQA: num_heads / num_kv_heads / head_dim mirror the baseline (params match).

## Files
- `experiments/grug/moe/wall_attention.py` — `wall_attention_chunk(...)` + `WallAttention(eqx.Module)`.
- `model.py` — `use_wall_attention` config; route ALL attention layers to WallAttention (window only on short).
- `wall_combined_d512.py` — launcher (env knobs), same MuonH recipe as the SWA baseline.

## Status
- [x] Plan
- [ ] Core forward + correctness test vs brute-force reference (small S)
- [ ] Module + GQA + gate + sink + QK norm
- [ ] Model wiring (replace short+long attn) + launcher + launch d512 comparison vs SWA (c4 3.5931)
