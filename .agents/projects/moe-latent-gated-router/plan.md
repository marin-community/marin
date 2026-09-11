# Gated latent routing ablation

Issue: https://github.com/marin-community/marin/issues/9110
Idea: Zihan Qiu.
Base: main at d891fba48a7d3729fdac2595df3c6ca292a53b74.

1. Copy the EP hero model and training loop into `experiments/grug/moe_latent_gated_router`.
   Keep the existing full-width router / latent RMSNorm as an explicit control arm.
   The treatment computes `z = GatedNorm(h @ W_latent_down)` before routing;
   both `z @ W_router` and the selected experts read this same `z`. The gate
   replaces RMSNorm and has rank 128. Shared experts keep their original input.
2. Preserve common parameter initialization. The new gate receives a folded-in key;
   creating it must not perturb expert, attention, shared, or projection initialization.
   Reuse the hero optimizer: latent gate matrices use MuonH and router uses Adam.
3. Verify control parity against the source hero, router invariance to the down-projection
   nullspace, gradients from router statistics into the latent path, and routed output
   against a dense all-expert reference.
4. Reuse the Agent MoE Nemotron/StarCoder/ProofPile dataset catalog and v5p-8 / EP1
   hardware. Scale the hero architecture to d512/d768 (6/8 layers), 384 experts,
   top-8, latent/expert width d/2, and two shared experts. Reuse the existing Agent MoE
   May Recipe baselines as requested by the user; do not launch new control jobs.
   Those results differ in architecture, data defaults, optimizer and hardware.
5. Use the Agent MoE baseline budgets (3.82e17/2.81e18 non-embedding training FLOPs)
   to derive the treatment schedule using the untrained scaled hero control's analytic
   FLOPs and batches (32/64). Count the gate and reduced router FLOPs separately.
6. Run a bounded accelerator startup check before full Gate 1. Each arm gets a unique
   run ID/output root. Confirm final evaluation/checkpoint writing and restart handling.
7. Submit only the d512/d768 gated_latent cells, verify children and W&B identity,
   and monitor until terminal. Reuse `moe_may_compute_opt_d{dim}_ep1` in
   `marin-community/marin_moe`. Recenter the baseline loss at alpha=0.0941 and
   L_inf=1.6. Report compute-equivalent gain and wall-clock comparison separately,
   accounting for each run's token count and throughput. Document hardware and
   recipe confounds rather than attributing the whole difference to gated routing.
8. Advance only if both widths have effective speedup >1. Run d1024/d1280 at
   1.16e19/3.46e19 baseline FLOPs, fit the treatment's four-point scaling curve with
   L_inf=1.6, and compare projections to the existing guide baseline at 1e21/1e23.
   No new baseline jobs are authorized at Gate 2 either.

No production hero code or running hero job is changed.
