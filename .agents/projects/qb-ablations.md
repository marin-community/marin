# QB / QK-norm ablations — d2048 @ 60×, 64× H100

Branch: `qb_ablations` (off `aug_hero_run_ablations`). Three isolated runs, each = baseline + one
change, on **64 H100**, model **d2048** (from the sweep), token budget **60×**, LR 1.0×.

Techniques are from the **Kimi K3 tech report** (arxiv "Kimi K3: Open Frontier Intelligence").

## Runs

| run | change |
|-----|--------|
| `qb-abl-d2048-60x-baseline` | none (the sweep d2048 config) |
| `qb-abl-d2048-60x-qknorm`   | + learnable per-head QK-norm scale → Adam |
| `qb-abl-d2048-60x-qbhist`   | + histogram/quantile QB stats (K3 "Quantile Balancing") |

## Technique 1 — learnable per-head QK-norm scale (→ Adam)

Today the hero QK path is **non-parametric**: `q = rms_norm(q)`, `k = rms_norm(k)`, then a *fixed*
scalar `q = q * qk_mult` (=1.3) (`model.py:465-466, :502`). Make that scale a **learnable per-head
parameter**.

- Add `qk_norm_scale: jax.Array` of shape `[num_heads]` to `CausalSelfAttention`, init to `qk_mult`
  (1.3). Replace `q = q * self.cfg.qk_mult` with `q = q * qk_norm_scale[None, None, :, None]`.
- Gate on a config flag (e.g. `learnable_qk_scale: bool = False`).
- **Optimizer group:** a 1-D `[num_heads]` array is `ndim=1`, so `grugmuon_hero.py:76` (`ndim not in
  {2,3,4}`) routes it to **Adam** automatically — matching the spec. Verify it lands in the adam group
  (log its update norm).
- K3 basis: K3 uses a per-head learnable scale `A_h` (bounded-log-decay for KDA) under Muon; here we
  adapt the *idea* (per-head learnable attention scale) to the QK-norm, under Adam, per the request.

Open: q-only vs q **and** k; init 1.3 vs 1.0; per-head only vs per-head-per-layer (the layers are
stacked, so a `[num_layers, num_heads]` scale is the per-layer variant).

## Technique 2 — histogram QB stat (K3 "Quantile Balancing")

**We already do QB.** `router_bias` (`[num_experts]`, zeros-init) biases *selection*
(`biased_logits = router_logits + stop_gradient(router_bias)`, `model.py:734`), and `qb_alpha` is the
`(K+1)`-th top biased logit per token (`:738`). The QB stat today (`model.py:754-773`) is:

```python
s_minus_alpha = router_logits - qb_alpha            # per-token margins [tokens, experts]
def _local_qb_beta(s_ma):                            # runs per device (shard_map over batch)
    topk_vals, _ = top_k(s_ma.T, qb_count)           # qb_count = local_tokens * k / n
    beta = topk_vals[:, -1]                           # LOCAL (1-k/n)-quantile per expert
    return pmean(beta, axis=_BATCH_AXES)              # then AVERAGE the local quantiles
```

i.e. each device computes a **local** per-expert quantile of the margins and we **average** them —
a biased estimate of the true whole-batch quantile (an average of quantiles ≠ quantile of the pool).

**The change:** replace the local-quantile-then-`pmean` with a **pooled per-expert histogram** →
true global quantile (this is all the ablation is):
- Margin definition is unchanged: `m_{t,j} = router_logits_{t,j} − qb_alpha_t` (`s_minus_alpha`).
- Bin each expert's margin column into `H ∈ ℕ[num_experts × B]` on each device, then **one integer
  `psum(H, axis=_BATCH_AXES)`** → pooled global counts (counts are additive → exact whole-batch
  histogram regardless of sharding).
- Read `beta_j = quantile_{1−k/n}` from the pooled counts (cumulative-count crossing at rank
  `global_tokens · k/n`, interpolated within the bin), then the existing bias update uses it. Keep the
  zero-mean normalization K3 applies (`b ← b − mean(b)`).
- Sigmoid-combine still on **unbiased** logits (`:741-742`) — only selection uses the bias. Unchanged.
- Gate on a flag `qb_histogram: bool = False`. The bias stays a non-gradient routing param.

### Histogram specs — be smart (this is the point)

Confirmed from the K3 report: `B=1000` bins, one integer all-reduce, quantile read from pooled counts,
zero-mean bias. The paper does **not** hand us the numeric bin range, and that's the sensitive knob —
the target is the **`1−k/n = 1−4/128 ≈ 0.969` quantile** (the near-zero margin boundary between
selected/not), so resolution must be good *there*:

- **Range:** to keep it to a **single** all-reduce (no separate min/max reduce), use a **fixed** margin
  range with **clip-to-edge** (out-of-range margins land in the end bins). Safe because the 0.969
  quantile is interior, not the extreme tail. Center on 0; margins are logit-differences, so a range
  like `[−R, R]` with `R` a small multiple of the router-logit scale (start ~[−8, 8]; tune from the
  observed margin distribution logged in the first steps).
- **Bins:** `B=1000` uniform to start (bin width `2R/B ≈ 0.016` logits at R=8) — fine near the
  boundary. Could non-uniformly concentrate bins near 0 if uniform proves coarse, but start uniform.
- **Validate:** log the pooled-histogram quantile vs the old `pmean`-of-local-quantiles for the first
  N steps — they should agree in the balanced limit and diverge under imbalance (the whole point).

Open: exact `R` (tune from logged margins); whether to EMA the bias for stability; update cadence
(every step, matching today).

## H100 launch — kernels auto-select (not a blocker)

The MoE/attention backends **auto-select a working kernel for the H100 arch** (per user); no manual
backend swap needed. Just target the H100 resources and carry over the sweep's validated flags:
- Resources: `ResourceConfig` H100, 8 GPU/node → **8 nodes = 64 GPU**.
- `offload_opt_state=False`, PGLE off (`-e JAX_ENABLE_PGLE 0`), eval-bf16 cast — all carry over.
- Cluster + region-local `MARIN_PREFIX` holding the datakit store: confirm which H100 fleet.

Still run **baseline first** as a quick sanity that d2048 trains on 64 H100 before the two variants.

## Implementation order

1. Resolve the H100 kernel/backend + cluster choice (below) and get the baseline running on 64 H100.
2. Add `learnable_qk_scale` (small, contained in `CausalSelfAttention`). Test locally (config builds,
   param shape, adam group).
3. Add `qb_histogram` (larger; new update rule + histogram all-reduce). Test the quantile/bias math
   against a small reference.
4. Launch the three runs; compare loss + eval bpb + router balance (expert-usage entropy) curves.

## Decisions needed from user

1. **QK scale:** q only or q+k? init 1.3 or 1.0? per-head or per-head-per-layer?
2. **QB histogram:** match K3 exactly (B=1000, the dual-LP `α`)? update cadence?
3. **H100 kernels + cluster:** which MoE/attention backend on H100, and which H100 cluster?
4. **Isolation:** each variant = baseline + one change (assumed), not cumulative?
