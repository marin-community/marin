---
topic: MRCR context perplexity for Grug MoE
issue: https://github.com/marin-community/marin/issues/7181
description: Measure final-turn perplexity reduction from retained MRCR context.
author: Helw150
---

# MRCR Context Perplexity: Task Logbook

## Current TL;DR

The `MOE-MRCR` series is preparing its first d512 run. No experiment result is available yet.

## Scope

- Goal: measure final-turn PPL with and without retained MRCR context on the smallest standard Grug scale.
- Primary metrics: `eval/mrcr/context_ppl`, `eval/mrcr/no_context_ppl`, `eval/mrcr/ppl_reduction`, `eval/mrcr/ppl_ratio`, and the corresponding context-bin metrics.
- Constraints: v5p-8 in `us-east5-a`; model sequence length 8192; left truncation; final assistant turn is the only scored target.
- Coordinating issue: https://github.com/marin-community/marin/issues/7181
- Stop criterion: one finished d512 run with finite paired metrics across MRCR context bins, or a documented unrecoverable infrastructure or evaluation failure.

## Baseline

- Date: 2026-07-14
- Code refs: `experiments/grug/moe/README.md`, `experiments/grug/moe/launch.py`, and issue #7181.
- Baseline numbers: no prior MRCR paired-PPL result exists. The historical d512 Grug reference reports Paloma macro loss 3.5422 at 433,986 tokens/s on v4-32; this run is an evaluation integration study rather than an architecture speedup comparison.

## Source Ledger

- OpenAI MRCR dataset card and viewer: https://huggingface.co/datasets/openai/mrcr
- Local Grug baseline and evaluation wiring: `experiments/grug/moe/launch.py` and `experiments/grug/moe/train.py`.
- Local Grug experiment operating guide: `experiments/grug/moe/agent.md`.

## Hypothesis Queue

### Active

- `MOE-MRCR-001`: a d512 Grug model will produce finite paired final-turn PPL metrics, and retained context will reduce aggregate PPL relative to the final-user-only condition. Next test: run the d512 recipe in `us-east5-a`.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Entry Log

### 2026-07-14 17:00 - MOE-MRCR-001 proposed

- Hypothesis: retained MRCR context reduces final-turn perplexity on a compute-optimal d512 Grug model.
- Commit Hash: baseline `cc6b2e1c9152e20920888b3415bbe314f81f5252`; experiment snapshot pending.
- Command: planned Iris submission from `experiments/grug/moe/agent.md` using `python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: budget 3.82e17 FLOPs, hidden dimension 512, batch 32, 3494 derived steps, sequence length 8192, eval every 1000 steps, max 8 batches per eval set, v5p-8 in `us-east5-a`.
- Result: implementation and launcher prepared; no remote run yet.
- Interpretation: d512 is the smallest standard Grug gate scale and is large enough to test learned use of context, unlike a one-step initialization smoke.
- Next action: run focused tests, snapshot the research branch, and submit the Iris job.
