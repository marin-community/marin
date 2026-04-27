# DPO + SFT Mixture Training

> **Logbook:** `.agents/logbooks/dpo_sft.md` — all tactical decisions, design questions, dataset analyses, and experiment results live there. This file stays high-level.

## Goals

1. **DPO for instruction following.** Train models to follow instructions via Direct Preference Optimization on chat-style preference pairs (chosen / rejected). Build on the existing DPO stack on the `dpo-lora-clean` branch.
2. **DPO + SFT mixture training.** Train on a single dataset that interleaves DPO preference pairs with plain SFT (instruction → response) examples. Each batch element computes either the DPO loss or the supervised cross-entropy loss; the trainer averages a single scalar per step.

This branch (`dpo_sft`, forked from `origin/dpo-lora-clean`) is the working branch for both goals.

## Why mixture training

- Preference data alone can drift from the SFT distribution. Mixing in SFT examples acts as an anchor toward fluent instruction following and reduces reward hacking on `delta_pi`.
- A unified mixture dataset avoids juggling two trainers / two checkpoints and lets us tune the SFT:DPO ratio as a hyperparameter.
- Many open preference datasets (UltraFeedback, IFBench-style verifier rollouts) ship alongside SFT data — we want one config that consumes both.

## Scope

**In:**
- A `MixtureDpoSftConfig` and entrypoint that loads two caches (preference + SFT chat) and trains end-to-end.
- DPO-only and SFT-only ablations runnable from the same entrypoint.
- An IF-style data pipeline (rollout pool → programmatic verifier → pair + SFT-on-passers) so we can produce both halves of the mixture from one source.
- Tests covering DPO-only, SFT-only, and mixture loss paths.

**Out (deferred):**
- Online / iterative DPO (re-rollout from current policy).
- SimPO, IPO, and other DPO variants.
- Multi-source SFT mixture weighting.
- RL with verifiable rewards (GRPO etc.).

## Project-level risks

- **Frozen-reference model interactions** — see `.agents/projects/dpo_levanter.md` for the Haliax/Levanter changes that make the frozen reference model work. Mixture training must not regress these.
- **Cost** — Marin compute is preemptible and large rollout runs are real money. Every real-money pipeline gets a smoke test first and explicit sign-off on the projected dollar figure before scaling.
- **Bug-1 pathology** — DPO debug-accum probes reproduce across datasets (see memory). Mixture experiments are not the place to chase Bug-1.

## Pointers

- Logbook: `.agents/logbooks/dpo_sft.md`
- Related design notes: `.agents/projects/dpo_levanter.md` (Haliax/Levanter DPO infrastructure analysis)
