# MoE DeepEP / Hybrid-EP: Research Logbook

## Scope
- Goal: benchmark DeepEP and Hybrid-EP torch-targeted MoE dispatch/combine kernels on GPU from Marin research code, starting from the sealed `#3633` snapshot.
- Primary metric(s): steady-state dispatch/combine wall time and derived `tokens/s` on H100x8.
- Constraints:
  - Do not restart or reconfigure the CoreWeave Iris cluster without explicit approval.
  - Keep the first experiment intranode and torch-side; do not imply a production-ready JAX bridge before one exists.
  - Reuse the sealed `moe-gpu-ragged-all-all-h100-seal-20260313` snapshot as the benchmark baseline.
- Experiment issue: https://github.com/marin-community/marin/issues/3641

## Baseline
- Date: 2026-03-13
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `lib/levanter/scripts/bench/bench_deepep_torch.py`
  - DeepEP mainline `README.md`
  - DeepEP hybrid branch `docs/README_Hybrid-EP.md`
- Prior GPU baseline from `#3633`:
  - fixed shape: `tokens=32768`, `hidden=2048`, `mlp_dim=768`, `experts=128`, `shared_expert_dim=2048`
  - `current` beat `ragged_a2a` on every measured `EP > 1` point on H100x8

## Initial Hypotheses
- DeepEP's torch kernels may beat the JAX `ragged_a2a` path because they use a more specialized GPU dispatch/combine implementation.
- Hybrid-EP may improve intranode efficiency further by reducing SM pressure and integrating permute/unpermute more tightly.
- The first meaningful result is likely op-level dispatch/combine performance, not a full end-to-end MoE MLP replacement.

## Stop Criteria
- A repo-local benchmark script can successfully call DeepEP / Hybrid-EP kernels on H100x8.
- We collect at least one steady-state intranode benchmark table for DeepEP and Hybrid-EP on a controlled routing distribution.
- If build or runtime blockers appear, leave a reproducible setup command and a documented blocker state.

## Experiment Log
### 2026-03-13 17:05 - Follow-up kickoff from sealed GPU ragged snapshot
- Hypothesis: the right starting point is to branch the new DeepEP work from the sealed `#3633` snapshot so the follow-up has an explicit baseline and reproducible parent.
- Command:
  ```bash
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all add \
    .agents/logbooks/moe-gpu-ragged-all-all.md \
    .agents/projects/moe-gpu-ragged-all-all-issue.md
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all commit -m "Seal GPU ragged all-to-all experiment artifacts"
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all tag -a \
    moe-gpu-ragged-all-all-h100-seal-20260313 \
    -m "Sealed GPU ragged all-to-all experiment snapshot"
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all push -u origin research/moe-gpu-ragged-all-all
  git -C /Users/romain/marin-wt/moe-gpu-ragged-all-all push origin moe-gpu-ragged-all-all-h100-seal-20260313
  git worktree add -b research/moe-deepep-hybrid-ep \
    /Users/romain/marin-wt/moe-deepep-hybrid-ep \
    moe-gpu-ragged-all-all-h100-seal-20260313
  ```
- Config:
  - seal tag: `moe-gpu-ragged-all-all-h100-seal-20260313`
  - follow-up branch: `research/moe-deepep-hybrid-ep`
- Result:
  - The prior experiment is sealed and the follow-up worktree now starts from that immutable snapshot.
- Interpretation:
  - The DeepEP thread can now reference a stable parent baseline in both local artifacts and GitHub.
- Next action:
  - Create the follow-up experiment issue.
  - Add a repo-local torch benchmark for DeepEP / Hybrid-EP.

### 2026-03-13 17:12 - DeepEP source inspection and scope cut
- Hypothesis: a practical first step is a torch-side benchmark harness, because DeepEP is exposed as a PyTorch extension and the hybrid branch adds a separate `HybridEPBuffer` wrapper with a richer dispatch/permute path.
- Command:
  ```bash
  git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP-codex
  git clone --depth 1 --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP-hybrid-codex
  sed -n '1,260p' /tmp/DeepEP-codex/README.md
  sed -n '1,260p' /tmp/DeepEP-hybrid-codex/docs/README_Hybrid-EP.md
  sed -n '1,260p' /tmp/DeepEP-hybrid-codex/deep_ep/hybrid_ep_buffer.py
  sed -n '1,320p' /tmp/DeepEP-hybrid-codex/tests/test_hybrid_ep.py
  ```
- Config:
  - DeepEP mainline API: `deep_ep.Buffer`
  - Hybrid-EP API: `deep_ep.HybridEPBuffer`
- Result:
  - DeepEP mainline is a torch extension around `Buffer.dispatch` / `Buffer.combine`.
  - The hybrid branch adds `HybridEPBuffer.dispatch`, `combine`, `dispatch_with_permute`, and `combine_with_unpermute`.
  - Intranode bring-up on a single H100x8 host is supported without the full multi-node RDMA path.
- Interpretation:
  - A true JAX-to-Torch or JAX custom-call bridge is not the first step.
  - The highest-leverage first experiment is a repo-local torch benchmark script that drives DeepEP / Hybrid-EP directly on the same H100x8 worker used in `#3633`.
- Next action:
  - Implement the torch benchmark harness in this worktree.
  - Run an intranode H100x8 smoke/benchmark via Iris.
