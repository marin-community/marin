---
topic: MRCR context perplexity for Grug MoE
issue: https://github.com/marin-community/marin/issues/7181
description: Measure final-turn perplexity reduction from retained MRCR context.
author: Helw150
---

# MRCR Context Perplexity: Task Logbook

## Current TL;DR

`MOE-MRCR-001-d512-r6` finished 3,494 steps on a `v5p-8` in `us-east5-a`. Conditioning on the left-truncated 8,192-token context reduced aggregate final-turn PPL from 21.0855 to 10.4012, a 2.0272x ratio and 0.7067 nat/token NLL reduction. The 2-, 4-, and 8-needle subsets each showed approximately 2x lower PPL. This is an exploratory result from one d512 run.

## Scope

- Goal: measure final-turn PPL with and without retained MRCR context on the smallest standard Grug scale.
- Primary metrics: aggregate and per-needle final-user-only PPL, full-context PPL, context PPL reduction, context PPL ratio, and context NLL reduction.
- Constraints: v5p-8 in `us-east5-a`; model sequence length 8192; left truncation; final assistant turn is the only scored target.
- Coordinating issue: https://github.com/marin-community/marin/issues/7181
- Stop criterion: one finished d512 run with finite aggregate and per-needle paired metrics, or a documented unrecoverable infrastructure or evaluation failure.

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

- None.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- `MOE-MRCR-001`: retained MRCR context reduced aggregate final-turn PPL from 21.0855 to 10.4012 in the d512 run. Evidence: [W&B run](https://wandb.ai/marin-community/marin_moe/runs/MOE-MRCR-001-d512-r6). Decision: extract the paired evaluator into a production PR.

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

### 2026-07-14 17:38 - MOE-MRCR-001 scoped root package sync

- Hypothesis: explicitly syncing `marin-root` will install the root experiment dependency that Iris's default `--all-packages` workspace-member sync omitted.
- Commit Hash: `32003287679aa042cc1ccb68ee0f728b84e58ecc`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512-r2 --zone us-east5-a --sync-package marin-root -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_RUN_ID MOE-MRCR-001-d512-r2 -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: unchanged d512 experiment; recovery run ID `MOE-MRCR-001-d512-r2`; Iris sync target `marin-root`.
- Result: `/held/moe-mrcr-001-d512-r1` repeated the import failure because the default workspace-member sync did not install root-only dependencies. Iris accepted `/held/moe-mrcr-001-d512-r2` with the explicit root sync.
- Interpretation: the declared dependency was correct; the remaining failure was limited to coordinator environment selection.
- Next action: verify coordinator import and continue monitoring through TPU dispatch and W&B registration.

### 2026-07-14 17:58 - MOE-MRCR-001 removed pre-tokenization binning

- Hypothesis: MRCR PPL reduction needs model-tokenized full-context and final-user-only pairs, not a separate tokenizer used only for reporting bins.
- Commit Hash: `f010c2009cdd753150f43220d4551c8c4d67bf6f`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512-r3 --zone us-east5-a -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_RUN_ID MOE-MRCR-001-d512-r3 -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: unchanged d512 model and paired left-truncated evaluation; aggregate and per-needle metrics replace pre-tokenization context bins.
- Result: `/held/moe-mrcr-001-d512-r2` failed during its explicitly scoped environment build because Iris applies `--no-group dev` to `marin-root`, which has no such dependency group. The separate tokenizer and its dependency were removed; the focused transform test passed, repository checks passed, and Iris accepted `/held/moe-mrcr-001-d512-r3` with default syncing.
- Interpretation: the evaluation now has one tokenization authority: the model tokenizer used by the supervised cache. This is simpler and directly matches the requested aggregate context-conditioned PPL reduction.
- Next action: verify coordinator startup, MRCR preprocessing, TPU dispatch, and W&B registration.

### 2026-07-14 18:00 - MOE-MRCR-001 corrected the dataset artifact version

- Hypothesis: the tokenizer-free transform will proceed once its immutable artifact version uses Marin's accepted calendar format.
- Commit Hash: `257199771f642a7c279936a6f4c934b24bcabfd3`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512-r4 --zone us-east5-a -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_RUN_ID MOE-MRCR-001-d512-r4 -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: unchanged d512 model and paired evaluation; MRCR artifact version `2026.07.14.1`.
- Result: `/held/moe-mrcr-001-d512-r3` failed before preprocessing because version `2026.07.14-1` was not a valid Marin calendar version. The corrected run downloaded all six MRCR parquet shards to `gs://marin-us-east5/raw/openai/mrcr/2026.07.14.1`, then its 1 GB coordinator was OOM-killed during the inline processed transform before TPU dispatch.
- Interpretation: dataset discovery and regional download succeeded; the remaining failure was the coordinator memory limit, not the transform semantics or training resource.
- Next action: reuse the cached download and increase only the CPU coordinator memory.

### 2026-07-14 18:03 - MOE-MRCR-001 preprocessing recovered

- Hypothesis: a 3 GB CPU coordinator is sufficient for the inline MRCR transform and cache orchestration.
- Commit Hash: `257199771f642a7c279936a6f4c934b24bcabfd3`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512-r5 --zone us-east5-a --memory 3GB -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_RUN_ID MOE-MRCR-001-d512-r5 -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: unchanged d512 model and paired evaluation; CPU coordinator memory increased from 1 GB to 3 GB; v5p-8 child remains pinned to `us-east5-a`.
- Result: Iris accepted `/held/moe-mrcr-001-d512-r5`. The cached raw dataset was reused, the processed transform completed at approximately 850 MB RSS, and all six model-tokenized validation-cache pipelines started. All three 800-document final-user-only caches and their probes succeeded; the 2-, 4-, and 8-needle full-context caches remain active.
- Interpretation: removing pre-tokenization and increasing coordinator memory resolved the ingestion path without changing the requested left-truncated paired metric.
- Next action: monitor the full-context caches through TPU dispatch, W&B registration, training, and terminal metrics.

### 2026-07-14 18:29 - MOE-MRCR-001 reduced the d512 host reservation

- Hypothesis: the d512 run can safely use the former 128 GB Grug TPU-host default and fit an occupied regional host or the next available slice.
- Commit Hash: `d03b1cab6df40726a57d1b4bb8965e8a78ab1e9e`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512-r6 --zone us-east5-a --memory 3GB -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_RUN_ID MOE-MRCR-001-d512-r6 -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: unchanged d512 model and paired evaluation; TPU child host reservation reduced from the current generic v5p default of 224 GB to the former Grug default of 128 GB.
- Result: all six paired caches and probes completed under `/held/moe-mrcr-001-d512-r5`. Its child could not fit the five partially occupied hosts at 224 GB, and 19 attempted new regional slices failed during an availability backoff. The parent was stopped before TPU assignment. `/held/moe-mrcr-001-d512-r6` reused all cached preprocessing and reached the TPU queue; the memory constraint is resolved, and it is pending only because all five live `us-east5-a` v5p-8 slices are occupied.
- Interpretation: the remaining delay is regional capacity. Historical Grug runs used 128 GB, and the d512 model does not need the larger default introduced to protect large-model checkpoint saves.
- Next action: keep the interactive child queued until a regional slice is available, then verify W&B registration and paired metrics.

### 2026-07-15 09:30 - MOE-MRCR-001 finished with a 2.027x context PPL ratio

- Hypothesis: retained MRCR context reduces final-turn perplexity on a compute-optimal d512 Grug model.
- Commit Hash: `d03b1cab6df40726a57d1b4bb8965e8a78ab1e9e`.
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --job-name moe-mrcr-001-d512-r6 --zone us-east5-a --memory 3GB -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_RUN_ID MOE-MRCR-001-d512-r6 -- python -m experiments.grug.moe.launch_mrcr_d512`.
- Config: budget 3.82e17 FLOPs, hidden dimension 512, batch 32, 3,494 steps, sequence length 8,192, final-assistant-target-only loss, rightmost-window truncation, eval every 1,000 steps, max 8 batches per validation set, v5p-8 in `us-east5-a`.
- Result: Iris job `/held/moe-mrcr-001-d512-r6` finished. [W&B](https://wandb.ai/marin-community/marin_moe/runs/MOE-MRCR-001-d512-r6) recorded the following final paired metrics:

  | Subset | Final-user-only PPL | Full-context PPL | NLL reduction | PPL ratio | PPL reduction |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Aggregate | 21.0855 | 10.4012 | 0.7067 | 2.0272x | 10.6843 |
  | 2 needles | 20.8989 | 10.2703 | 0.7104 | 2.0349x | 10.6286 |
  | 4 needles | 21.6428 | 10.6670 | 0.7075 | 2.0289x | 10.9758 |
  | 8 needles | 20.7468 | 10.2837 | 0.7018 | 2.0174x | 10.4631 |

  Aggregate conditioning removed 50.67% of final-user-only PPL. Paloma macro loss was 3.6643 and micro loss was 3.4159 at the final evaluation.
- Interpretation: the paired metric is finite and consistent across all three needle counts. Retained context approximately halved final-turn PPL at the model's 8,192-token training length. This run does not compare model context lengths, so the expected reward for longer-context models remains untested.
- Next action: extract the reusable MRCR dataset and paired-loss reporting changes into a production PR linked to issue #7181.
