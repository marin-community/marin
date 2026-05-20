# Context Norm + Head-wise Gate: Research Logbook

## Scope
- Goal: evaluate whether inserting a per-head RMSNorm on the attention context
  vector (post-XSA, pre head-wise sigmoid gate) improves the grug-moe recipe.
- Primary metric: effective speedup vs. v16 compute-optimal baselines at d512
  (2.19e17) and d768 (1.70e18) per `experiments/grug/moe/agent.md` (gate 1).
- Constraints: identical architecture to baseline except for the inserted norm;
  same heuristic-derived (num_layers, batch_size, num_steps, optimizer); v5p-8
  preemptible per house style.

## Baseline (from `experiments/grug/moe/README.md`)
- d512  / 2.19e17 → macro 3.8104, v5p-8 avg tok/s 405,630
- d768  / 1.70e18 → macro 3.4339, v5p-8 avg tok/s 273,532

Scaling law (L∞=1.6 pinned): `loss(C) = 1.6 + 95.18 · C^(-0.0941)`.

## Design
Insert a per-head RMSNorm between XSA and the head-wise sigmoid gate in
`CausalSelfAttention.__call__`:

```
attn_out = attention(q, k, v, mask)         # [B, S, N, D_head]
# XSA: subtract projection on v
attn_out = attn_out - (dot/||v||^2) * aligned_v
# NEW: per-head RMSNorm with learned scale [N, D_head]
attn_out = context_rms_norm(attn_out)
# Existing head-wise sigmoid gate
attn_out = gate * attn_out
out = w_o(rearrange(attn_out, '... n d -> ... (n d)'))
```

Implementation: `GrugModelConfig.use_context_norm: bool = False`. When True,
`CausalSelfAttention.context_norm` is a `[num_heads, head_dim]` fp32 weight
initialized to ones; default behavior is unchanged.

Code refs:
- `experiments/grug/moe/model.py` (CausalSelfAttention)
- `experiments/grug/moe/context_norm_gate1.py` (launch)

## Experiment Log
### 2026-05-19 - Kickoff
- Hypothesis: a learned per-head RMSNorm on the context vector pre-gate may
  stabilize per-head activation scales after XSA's projection-subtraction,
  helping the gate operate on a well-conditioned input.
- Branch: `moe_context_norm_gate1` (off main).
- Command: see `context_norm_gate1.py` docstring.
- Config: heuristic-derived d512 (steps=6387, batch=32) and d768
  (steps=10343, batch=64). `use_context_norm=True`. All other recipe knobs
  match baseline.
- Result: stopped before completion. d512 finished with one preemption.
  (W&B: `context-norm-gate1-v1-d512-2.19e17`.)
- Next action: switch to a different variant (below).

### 2026-05-19 - Variant change: replace XSA with norm+gate
- Hypothesis: XSA may be redundant once a learnable per-head RMSNorm is
  inserted before the head-wise gate. Try dropping XSA entirely.
- Code change: new `GrugModelConfig.use_xsa: bool = True` toggle; the no-XSA
  variant sets `use_xsa=False, use_context_norm=True`.
- Branch (same): `moe_context_norm_gate1`.
- Launch: `experiments/grug/moe/context_norm_no_xsa_gate1.py`.
- Iris job: `/kaiyue/iris-run-job-20260520-023742` (preemptible v5p-8).
- W&B group: `context-norm-no-xsa-gate1` (`-v1-d{512,768}-...`).
- Result: **gate-1 PASS at both scales** (L∞=1.6, α=0.0941):
  - d512: variant_loss=3.8293, variant_tps=449,488 → speedup **1.0124** (baseline 3.8104 / 405,630)
  - d768: variant_loss=3.4478, variant_tps=312,981 → speedup **1.0558** (baseline 3.4339 / 273,532)
- Interpretation: variant has slightly worse macro_loss but enough faster
  throughput (1.108× at d512, 1.144× at d768) to win wall-clock. Removing
  XSA's per-head projection + subtraction is the throughput source.
- Next action: launch gate-2 at d1024 (9.00e18) and d1280 (2.83e19); fit
  scaling law on four points; compare projection at 1e21 / 1e23 to baseline.

### 2026-05-20 - Gate-2 kickoff
- Launch: `experiments/grug/moe/context_norm_no_xsa_gate2.py`.
- Configs: d1024 (L=11, batch=128, steps=12649), d1280 (L=13, batch=256,
  steps=11807). Same `use_context_norm=True, use_xsa=False`.
- W&B group: `context-norm-no-xsa-gate2`.
- Result: pending (d1024 ~10.5h baseline; d1280 ~26.8h baseline).
