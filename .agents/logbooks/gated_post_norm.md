# Gated Post-Norm: Research Logbook

Series ID: **MOE-GPN**
Issue: https://github.com/marin-community/marin/issues/5938
Branch: `research/moe-gated-post-norm` (off `main`)
Worktree: `/tiger/u/kaiyue/workspace/TPU/Marins/depth_mup/marin-moe-gated-post-norm`

## Scope

- Goal: test whether adding a second RMSNorm+GatedNorm on the residual stream *after* each sublayer (sandwich pre+post) improves the MoE scaling curve.
- Primary metric: `eval/paloma/macro_loss` at four compute-optimal scales (d512, d768, d1024, d1280) on Nemotron mix.
- Secondary metric: `throughput/tokens_per_second` (avg last 100 steps) — the post-norm adds parameters and one extra silu+sigmoid per sublayer, so step-time will rise.
- Constraints: A/B vs. canonical `experiments/grug/moe/README.md` baseline. **No other architectural deltas.** Same heuristic-derived (model, optimizer, batch, num_steps).

## Variant

```
r_{L+1} = GatedNorm(RMSNorm(r_L + f(GatedNorm(RMSNorm(r_L)))))
```

applied at both attention and MLP sub-blocks of every `Block`. Independent RMSNorm + GatedNorm parameters at the post position (no weight sharing with the pre-sublayer norm/gate). `embed_*` and `final_*` norms unchanged.

Implementation: `experiments/grug/moe/model.py` — `GrugModelConfig.gated_post_norm: bool = False` flag; `Block` gains four optional `*_post` modules constructed only when the flag is on. Forward pass: `attn_gated_norm_post(rms_attn_post(x))` after attention residual; `mlp_gated_norm_post(rms_mlp_post(x))` after MLP residual.

## Baseline (from README.md compute-opt table)

| Series  | Dim  | Layers | Budget  | Paloma macro | Tokens  | v5p-8 tok/s | Runtime |
|---------|------|--------|---------|--------------|---------|-------------|---------|
|         | d512 | 6      | 2.19e17 | **3.8104**   | 8.37e8  | 405,630     | 0.6h    |
|         | d768 | 8      | 1.70e18 | **3.4339**   | 2.71e9  | 273,532     | 2.8h    |
|         | d1024| 11     | 9.00e18 | **3.1605**   | 6.63e9  | 175,165     | 10.5h   |
|         | d1280| 13     | 2.83e19 | **3.0065**   | 1.24e10 | 128,277     | 26.8h   |

Scaling law: `loss(C) = 1.6 + 95.18 · C^(-0.0941)`.

## Stop criteria

- Gate 1 passes only if the variant shows **effective speedup > 1** at both d512 *and* d768 against the table above (formula in `experiments/grug/moe/agent.md`).
- Then gate 2 (d1024, d1280) **runs only after gate 1 results are in** (per current user policy — no parallel two-tier execution).
- Promote if gate 2 passes both per-scale speedup *and* refit projection at 1e21 / 1e23 is lower than baseline (2.606 / 2.252).

## Experiment Log

### 2026-05-25 04:50 — kickoff
- Branch off `origin/main`. Confirmed canonical model.py (no PKO / prope / last_layer_pko / expert scale).
- Implemented `gated_post_norm` flag; added four optional `*_post` modules to `Block`.
- Created `experiments/grug/moe/gated_post_norm_gate1.py` for d512 + d768.
- Wandb destination: `marin-community/marin_moe`, group `gated-post-norm`.
- Iris submission command (per agent.md + preemptible memory):
  ```
  .venv/bin/iris --config lib/iris/config/marin.yaml job run \
    --no-wait --preemptible --reserve v5p-8 \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -- python -m experiments.grug.moe.gated_post_norm_gate1
  ```
- Iris job: **TBD** (filled in after submission).
- Hypothesis: a learned per-token gate on the residual output should compensate for activation-norm drift in deeper layers (the existing pre-gate only sees the input). At small scale we expect a small loss improvement and a 5-10% tok/s hit from the extra gate+norm.
- Confidence at kickoff: `exploratory`.

## Prior context (do NOT repeat the mistake)

First attempt branched off `moe_may_arch` and stacked PKO + partial RoPE + last_layer_pko + k4e256 + gated_post_norm. That conflates four variables and breaks the apples-to-apples comparison the agent.md promotion gates require. Discarded.
