# Gated Post-Norm: Research Logbook

Series ID: **MOE-GPN**
Issue: https://github.com/marin-community/marin/issues/5938
Branch: `research/moe-gated-post-norm-muonh` (off `origin/moe_muonh_may_arch_gn_muonh_1pct_noclip`, issue #5763)
Worktree: `/tiger/u/kaiyue/workspace/TPU/Marins/depth_mup/marin-moe-gated-post-norm`

## Scope

- Goal: test whether adding a second RMSNorm+GatedNorm on the residual stream *after* each sublayer (sandwich pre+post) improves on the **muonh_may_arch_gn_muonh_1pct_noclip** recipe (issue #5763).
- Primary metric: `eval/paloma/macro_loss` at gate-1 scales (d512, d768), then gate 2 (d1024, optionally d1280) sequentially.
- Secondary: `throughput/tokens_per_second` (avg last 100 steps).

## Recipe context (issue #5763)

Baseline for this experiment:
- may_arch model: 256 experts, PKO every_4th, partial rope, last_layer_pko, no router/logit z-loss, embed → AdamH.
- Optimizer: `GrugMoeMuonHMayArchGNMuonHConfig` — MuonH on matrices **and** all four GatedNorms; 1% warmup; no gradient clipping; LR 1.0× heuristic.
- MuonH parameter mask matches `"gated_norm" in path_lower` → new `*_post.w_*` weights automatically inherit MuonH routing.

Comparison targets from issue #5763:
- gn-0p7 (GN at 0.7× adam_lr, 2% warmup, with clip): d512=3.6518, d768=3.3038
- default (GN at 1.0× adam_lr, 2% warmup, with clip): d512=3.6598
- recipe runs (1% warmup, no clip): see wandb group `muonh-may-arch-gn-muonh-1pct-noclip-sweep`.

## Variant

```
r_{L+1} = GatedNorm(RMSNorm(r_L + f(GatedNorm(RMSNorm(r_L)))))
```

`gated_post_norm: bool = False` in `GrugModelConfig`. When True, `Block` instantiates `rms_attn_post`, `attn_gated_norm_post`, `rms_mlp_post`, `mlp_gated_norm_post` (independent of the pre-sublayer modules) and the forward applies them after each residual add. No other architectural deltas vs. the muonh recipe.

## Stop criteria

- Gate 1: per `experiments/grug/moe/agent.md` effective-speedup formula vs. recipe baseline final macro_loss + tok/s. Need speedup > 1 at d512 *and* d768 to proceed.
- Gate 2 (d1024 sequential): need speedup > 1 at all three points + projected curve below baseline at 1e21, 1e23.

## Experiment Log

### 2026-05-25 06:00 — kickoff on muonh recipe
- Reset off issue #5763's branch `moe_muonh_may_arch_gn_muonh_1pct_noclip` (commit `f184e31d9`). Earlier attempts (off `main`, then off `moe_may_arch`) had wrong baseline. Now matches the user-requested recipe.
- `model.py`: added `gated_post_norm` flag + four optional `*_post` modules in `Block`. Flag off = byte-identical to recipe.
- Launcher: `experiments/grug/moe/gated_post_norm_muonh_gate1.py` (d512 + d768 only).
- Wandb group: `muonh-may-arch-gn-muonh-1pct-noclip-gated-post-norm` in `marin-community/marin_moe`.
- Submission (per issue #5763 — us-central1-a, lib/iris/examples/marin.yaml config):
  ```
  .venv/bin/iris --config lib/iris/examples/marin.yaml job run \
    --no-wait --preemptible --zone us-central1-a \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -- python -m experiments.grug.moe.gated_post_norm_muonh_gate1
  ```
- Confidence: `exploratory`.

### 2026-05-25 07:05 — d512 finished

State: `JOB_STATE_SUCCEEDED`, step 6386/6386. Confidence: `exploratory`.

| Metric              | Baseline (muonh recipe) | gated_post_norm | Δ        |
|---------------------|-------------------------|-----------------|----------|
| `eval/paloma/macro_loss` | 3.6427                  | **3.6539**      | +0.011 (worse) |
| `throughput/tokens_per_second` | 332,184          | **319,008**     | -3.96% |
| `throughput/total_tokens` | 837,156,864        | 837,156,864     | —      |

**Effective speedup at d512 = 0.906** (per `agent.md` formula, L∞=1.6, α=0.0941). Speedup < 1 means the variant takes longer wall-clock to reach baseline's final loss than the baseline itself. **Fails gate 1 at d512.**

Per agent.md gate 1 promotion criterion ("effective speedup > 1 at *both* scales"), this rules out promotion regardless of the d768 outcome. d768 is still running and will be completed for the full negative-result picture.

Interpretation: doubling the gated norm cost (extra silu+sigmoid per sublayer) costs throughput but does not deliver enough loss improvement to compensate. At this scale, the sandwich post-norm appears redundant with the existing pre-norm.

Run: https://wandb.ai/marin-community/marin_moe/runs/muonh-may-arch-gn-muonh-1pct-noclip-gpn-v1-d512-2.19e17

Confidence: `exploratory` (single seed, single scale; d768 still pending).

### 2026-05-25 07:15 — d768 stopped early; close out

User direction: close the issue since gate 1 has already failed at d512 (`effective_speedup=0.906`, both worse loss and lower tok/s) and gate 1 requires speedup > 1 at *both* scales. Continuing d768 for ~2.5 more hours of TPU would not change the verdict.

Iris job `/kaiyue/iris-run-job-20260525-060206` stopped at d768 step 2673/10343 (26%). Partial d768 snapshot when killed:
- step 2673, paloma_macro = 3.939 (mid-training, not comparable to baseline final 3.304), tok/s = 220,425
- baseline tok/s at d768 was 233,410 → variant ~5.6% slower (consistent with d512's -4% throughput hit from the doubled gated norm cost)

**Conclusion (`stable` w.r.t. d512, `exploratory` w.r.t. d768 trend):** the sandwich post-norm pattern `r_{L+1} = GatedNorm(RMSNorm(r_L + f(GatedNorm(RMSNorm(r_L)))))` on the muonh_may_arch_gn_muonh_1pct_noclip recipe does **not** outperform the recipe baseline at gate-1 scales. The variant pays ~4-6% throughput for marginally worse loss. The existing pre-sublayer RMSNorm+GatedNorm appears sufficient at these scales; a second copy is redundant.

### Negative-results index
- d512 (`MOE-GPN-001`): loss +0.011, tok/s -4.0%, speedup 0.91 → fail.
- d768 (`MOE-GPN-002`): stopped at 26% (not enough to compute final speedup), but throughput already -5.6% suggesting same direction.

### Next steps (small, ordered)
1. **Don't pursue this direction further** without a separate hypothesis for why the post-norm would only kick in at larger scale.
2. If revisited: try sharing parameters between pre and post (cuts param overhead in half), or only post (no sandwich), to isolate the throughput-vs-loss trade.

## Prior dead branches (do NOT repeat)

- `moe_may_arch_gated_post_norm`: branched off may_arch, ran gpn on top of `may_arch_sweep`. Wrong wandb project, wrong (combined) baseline. Discarded.
- `research/moe-gated-post-norm`: branched off `main`. Apples-to-apples, but the user's intended recipe is the muonh recipe in issue #5763, not the canonical README baseline. Discarded.
