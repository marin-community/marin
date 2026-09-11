# Gated latent routing

Ablation of the EP hero model, suggested by Zihan Qiu.
Tracking: [#9110](https://github.com/marin-community/marin/issues/9110).

The control routes the full-width pre-MLP feature `h` and dispatches
`RMSNorm(h @ W_latent_down)`. The treatment dispatches
`z = GatedNorm(h @ W_latent_down)` and routes `z`. GatedNorm is the existing
rank-128 multiplicative gate; there is no RMSNorm after the latent projection
in the treatment. Both arms keep the pre-MLP RMSNorm/GatedNorm, shared experts,
QB bias, sigmoid combine weights, and shared latent up-projection.

The model and trainer are copied from `moe_hero_ep` at
`d891fba48a7d3729fdac2595df3c6ca292a53b74`. The hero optimizer is reused:
latent gate matrices use MuonH; the router uses Adam. Common parameter RNG keys
are preserved. The combined intervention also changes activation scale and
router parameter count; this experiment cannot isolate the two components.

## Comparison

Follow the gate progression in [Agent MoE](../moe/agent.md), reusing its existing
May Recipe baseline runs. **Do not launch new baseline jobs.** The full-width
control remains in code for numerical parity checks and budget calculation.
The treatment uses v5p-8, EP1, the ring
backend's dropless local path, TPU Splash attention, 4096-token sequences,
384 experts, top-8, latent/expert width d/2, and two shared experts.
The Nemotron/StarCoder/ProofPile mixture and llama3 tokenization come from the
Agent MoE catalog. The optimizer follows the EP hero's Aug MuonH heuristic.
This is a small-scale architecture gate, not a reproduction of the hero's
Harrier data or GB200 EP64 transport/drop dynamics.

| Gate | Hidden | Layers | Batch | Control FLOP budget |
| --- | --- | --- | --- | --- |
| 1 | 512 | 6 | 32 | 3.82e17 |
| 1 | 768 | 8 | 64 | 2.81e18 |
| 2 | 1024 | 12 | 128 | 1.16e19 |
| 2 | 1280 | 14 | 256 | 3.46e19 |

The scaled hero control's analytic FLOPs fix the treatment's steps, tokens,
and optimizer schedule at each budget. This does not require training a control.
Analytic FLOPs follow the existing matmul convention: the treatment adds
`4 * latent_dim * 128` gate FLOPs and changes router FLOPs by
`2 * num_experts * (latent_dim - hidden_dim)` per token per layer.
Elementwise normalization/gating overhead is represented by measured throughput.

Run the treatment at one width at a time with an exact run ID:

```bash
uv run python -m experiments.grug.moe_latent_gated_router.launch \
  --dim 512 --arm gated_latent --run-id moe-lgr-9110-d512-gated \
  --version dev
uv run python -m experiments.grug.moe_latent_gated_router.launch \
  --dim 768 --arm gated_latent --run-id moe-lgr-9110-d768-gated \
  --version dev
```

These commands print the plan. Add `--run` inside an Iris CPU parent to submit
its TPU child. `--stop-after-steps` supports a separately named bounded smoke
while retaining the full schedule. Checkpoints are written every 15 minutes
and at completion under the artifact's output; restarts use the same run ID.

Final comparison requires W&B state `finished`, Paloma macro loss, final total
tokens, last-100-step mean token throughput, router entropy/counts and drop rate,
and a complete final checkpoint. The existing references are
`marin-community/marin_moe/moe_may_compute_opt_d{dim}_ep1`. Their architecture,
optimizer recipe, token count, and v4-32 hardware differ from this scaled hero.
Report compute-equivalent gain separately from measured wall-clock comparison,
using each run's own token count and throughput. This comparison does not isolate
the gated-router intervention. A Gate 1 pass requires effective speedup >1 at both
widths; only then run the two larger treatment cells and fit the scaling projection.
