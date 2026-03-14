# JAX vs Megatron MoE GPU Gap: Research Logbook

## Scope
- Goal: identify the concrete root causes behind the discrepancy between the negative JAX custom-call result in `#3665` and the more positive torch/Megatron results in `#3641` and `#3666`.
- Primary metric(s):
  - matched H100x8 forward/backward wall time
  - phase-level timing deltas where possible
  - attributable delta by hypothesis category
- Constraints:
  - Use CoreWeave Iris H100x8 via `~/llms/cw_ops_guide.md`.
  - Keep the experiment falsifiable: matched comparisons first, wider tuning second.
  - Include a direct JAX/Marin vs Megatron-LM head-to-head.
- Experiment issue: https://github.com/marin-community/marin/issues/3677

## Baseline
- Date: 2026-03-14
- Code refs:
  - `#3641` seal: `moe-deepep-hybrid-ep-seal-20260314`
  - `#3665` seal: `moe-deepep-jax-layout-ffi-h100-matrix-20260314`
  - `#3666` seal: `moe-megatron-qwen-scale-h100-matrix-20260314`
- Baseline numbers:
  - `#3665` fixed-shape H100x8:
    - `current` beat `deepep_layout_ragged_a2a` by about `1.48x` to `1.81x` on distributed cells
    - `deepep_layout_ragged_a2a` stayed roughly tied with `ragged_a2a`
  - `#3641` fixed-shape torch-side H100x8:
    - patched `hybrid_ep` / `hybrid_ep_permute` beat `deep_ep` on the replicated `runs` slice
  - `#3666` Megatron full-layer H100x8:
    - `deepep` won most `128`-expert Qwen-like points
    - `hybridep` won the authoritative `32` and `64` expert reruns

## Initial Hypotheses
- The dominant root cause is kernel coverage: `#3665` only replaced `get_dispatch_layout`, while `#3641` / `#3666` benefited from real dispatch/combine transport kernels.
- The next root cause is benchmark scope: `#3665` timed the sealed `#3633` fixed-shape path, while `#3666` timed a full `MoELayer` with grouped GEMM and different compute/communication balance.
- JAX-specific overheads may matter, but only after the first two causes are controlled for; otherwise they are easy to overstate.
- A direct same-shape torch dispatch/combine benchmark with optional JAX->Torch bridging should help isolate whether the missing gains are mostly in transport kernels or mostly in the broader Megatron methodology.

## Stop Criteria
- Produce a direct JAX/Marin vs Megatron-LM head-to-head on CoreWeave H100x8.
- Produce a root-cause table that identifies which differences are necessary and/or sufficient to explain the gap.
- Update the issue body with a concise conclusion that is more specific than "JAX is slower" or "Megatron is better."

## Experiment Log
### 2026-03-14 21:05 - Kick off the new root-cause thread
- Commands:
  ```bash
  git worktree add -b research/moe-jax-megatron-root-cause \
    /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    moe-megatron-qwen-scale-h100-matrix-20260314

  sed -n '1,220p' /Users/romain/dev/marin/.agents/skills/agent-research/SKILL.md
  sed -n '1,260p' /Users/romain/llms/cw_ops_guide.md
  gh issue view 3641 --repo marin-community/marin --json title,body,state,comments --comments
  gh issue view 3665 --repo marin-community/marin --json title,body,state,comments --comments
  gh issue view 3666 --repo marin-community/marin --json title,body,state,comments --comments
  ```
- Config:
  - branch: `research/moe-jax-megatron-root-cause`
  - worktree: `/Users/romain/marin-wt/moe-jax-megatron-root-cause`
  - CoreWeave kubeconfig: `~/.kube/coreweave-iris`
- Result:
  - Reloaded the `agent-research` workflow and the CoreWeave ops guide.
  - Re-established the three relevant sealed baselines:
    - torch-side DeepEP / Hybrid-EP wins in `#3641`
    - JAX layout-only negative result in `#3665`
    - Megatron full-layer ranking shift in `#3666`
  - Decided to structure the new thread around root-cause separation rather than another one-off benchmark.
- Initial root-cause matrix:
  - `methodology`: dispatch-only vs full `MoELayer`
  - `kernel coverage`: layout-only vs dispatch/combine transport
  - `backend`: JAX `ragged_all_to_all` / current ring EP vs DeepEP / Hybrid-EP
  - `interop`: optional JAX -> Torch bridge cost
  - `timing hygiene`: dummy GEMM, warmup length, fixed inputs, router dtype
- Next action:
  - Create the experiment issue and link it back here.
  - Audit the relevant JAX / torch / Megatron code paths before launching the first H100x8 comparison.
