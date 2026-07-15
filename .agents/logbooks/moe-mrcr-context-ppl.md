---
topic: MRCR context perplexity for Grug MoE
issue: https://github.com/marin-community/marin/issues/7181
description: Measure final-turn perplexity reduction from retained MRCR context.
author: Helw150
---

# MRCR Context Perplexity: Task Logbook

## Current TL;DR

The `MOE-MRCR-001` d512 recovery run is submitted as Iris job `/held/moe-mrcr-001-d512-r1` in `us-east5-a`. No experiment result is available yet.

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

- `MOE-MRCR-001`: a d512 Grug model will produce finite paired final-turn PPL metrics, and retained context will reduce aggregate PPL relative to the final-user-only condition. Evidence: Iris accepted recovery job `/held/moe-mrcr-001-d512-r1`. Next test: monitor training and collect final MRCR metrics.

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

### 2026-07-14 17:15 - MOE-MRCR-001 submission preflight blocked

- Hypothesis: the prepared d512 snapshot can be submitted with the standard Grug Iris command.
- Commit Hash: `4534ffb60170cc3bdbb348dce081c525d60042cb`.
- Command: `test -n "${WANDB_API_KEY:-}"` and `wandb status` before Iris submission.
- Config: branch `research/helw150/7181-mrcr-context-ppl`; v5p-8 in `us-east5-a`; W&B project `marin-community/marin_moe`.
- Result: the environment variable is absent and W&B reports `api_key: null`; no Iris job was submitted.
- Interpretation: submitting without the required key would spend capacity on a run that cannot emit the experiment's required metrics.
- Next action: resume the standard Iris submission as soon as W&B authentication is available in the submitter environment.

### 2026-07-14 17:20 - MOE-MRCR-001 submitted

- Hypothesis: the d512 run will exercise the paired MRCR evaluation end to end and quantify context-conditioned PPL reduction.
- Commit Hash: `4534ffb60170cc3bdbb348dce081c525d60042cb`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512 --zone us-east5-a -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: budget 3.82e17 FLOPs, hidden dimension 512, batch 32, 3494 steps, sequence length 8192, eval every 1000 steps, v5p-8 training resource pinned to `us-east5-a`.
- Result: Iris accepted job `/held/moe-mrcr-001-d512`; dashboard https://iris.oa.dev/#/job/%2Fheld%2Fmoe-mrcr-001-d512.
- Interpretation: the required W&B authentication is now available. The first submission with the additional `--reserve v5p-8` constraint was rejected because no non-preemptible CPU coordinator group matched that live preemptible availability constraint in `us-east5-a`; the retry retained the explicit zone pin, while the launcher independently pins its v5p-8 child resource to the same zone.
- Next action: monitor Iris logs and W&B until terminal, recovering the self-submitted job if necessary.

### 2026-07-14 17:33 - MOE-MRCR-001 recovered missing runtime dependency

- Hypothesis: declaring the transform's existing `tiktoken` lockfile dependency at the root will make it available in the Iris coordinator environment.
- Commit Hash: `32003287679aa042cc1ccb68ee0f728b84e58ecc`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512-r1 --zone us-east5-a -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_RUN_ID MOE-MRCR-001-d512-r1 -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: unchanged d512 experiment; recovery run ID `MOE-MRCR-001-d512-r1`.
- Result: original job `/held/moe-mrcr-001-d512` failed during coordinator import with `ModuleNotFoundError: No module named 'tiktoken'`, before TPU dispatch. The root dependency was declared, checks passed, and Iris accepted `/held/moe-mrcr-001-d512-r1`.
- Interpretation: this was packaging-only and does not alter the dataset transform or experiment configuration.
- Next action: verify that the recovery coordinator reaches MRCR preprocessing, TPU dispatch, and W&B registration.
