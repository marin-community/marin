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
- Result: pending.
- Next action: submit d512 and d768 to Iris (preemptible v5p-8), babysit.
