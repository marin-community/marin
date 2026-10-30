# [SFT] Cold Start Subsampling: H100 reproduction and a cheaper ablation baseline

## TL;DR

We successfully reproduced the SFT cold-start workflow on a new H100 cluster: import the base, stage the mixture, train, resume at a different GPU count, and export verified checkpoints. Agentic evals look good (provisionally, they're not quite finished yet). Roughly 10% of the full cold-start training budget may be enough to recover most of the useful performance, which would make ablations much cheaper.

We initially thought the reproduction had a large agentic capability gap. A major confound was in serving: exported checkpoints were missing the original model's second stop token, and thinking was not reliably enabled. We fixed the generation configs on Hugging Face and explicitly enabled thinking in the rerun configs. On the same SWE task, step 222 went from batches of up to 33 tool calls to at most two. The corrected partial evals are much stronger.

## Experiment

Start from [Snowball 67B-A2B base](https://huggingface.co/open-athena/snowball-67b-a2b-base-262k-qk175-skew8), with the source mixture used by the [10-05 Datakit SFT release](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05), and reproduce cold start on CoreWeave `cw-rno2a`.

| Setting | Reproduction |
|---|---|
| Model | 67.08B parameters, 26 layers, hidden size 2,560, 256 routed experts, top-4 routing, shared expert |
| Supervised mixture | The 216-source 09-29 catalog behind the 10-05 release; 88 packed sequences held out for validation |
| Replay | About 20.35% of scheduled positions, using existing S3 replay caches and long-document weighting; selection differs from the original |
| Context / global batch | 262,144 positions × 256 sequences = 67,108,864 scheduled positions per update |
| Full reference schedule | 4,750 updates, 318.77B scheduled positions: 253.89B SFT + 64.88B replay in the reproduction plan |
| Learning rate | 5e-5; no warmup; final 475-step cooldown to 0.1×, not reached by the evaluated reproduction checkpoints |
| Training | 128 H100s, then resume from step 471 on 256 H100s; global batch unchanged, accumulation reduced from 8 to 4 microsteps |
| Precision / initialization | BF16 parameters and compute, fresh optimizer from verified HF base import, frozen router biases |
| Exports | Steps 222, 471, 1024, 1536, 2048, 2338; BF16, original SFT tokenizer, 502 tensor mappings and 39 shards verified |
| Current eval cohort | Steps 222, 471, 1024, 2048 × SWE-bench Verified, OT-TBLite, Terminal-Bench 2.1 v6, DS-1000, BixBench, BFCL-Parity |

Training is stopped. The corrected 24-cell eval cohort is running under [eval policy #10197](https://github.com/marin-community/marin/issues/10197). No original-10-05 or base rerun is included in this cohort.

“Successful reproduction” here means a working cold-start training/export pipeline with promising downstream recovery. It is not a bitwise recipe match: the audit found different replay selection, fresh optimizer history, BF16 rather than FP32 stored training parameters, and different optimizer routing for stacked expert tensors. Those differences are documented for future ablations; the current scores do not establish their effects.

Training runs: [128 H100](https://wandb.ai/nyu-dice-lab/sft-pipe-repro/runs/snowball-sft-production-128h100-20261024b) · [256 H100 resume](https://wandb.ai/nyu-dice-lab/sft-pipe-repro/runs/snowball-sft-production-256h100-20261027c) · [original 10-05 run](https://wandb.ai/marin-community/marin_moe_sft/runs/grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped).

## Findings so far

### 1. Training: throughput, loss and replay

The new cluster ran the full 262k-context recipe with the same global batch at both GPU counts. Moving from 128 to 256 H100s reduced accumulation from eight microsteps to four and made each update about 1.8× faster.

| Training run | Mean seconds / update | Median seconds / update | Middle 80% of update times |
|---|---:|---:|---:|
| [128 H100](https://wandb.ai/nyu-dice-lab/sft-pipe-repro/runs/snowball-sft-production-128h100-20261024b) | 157.2 | 153.6 | 145.0–167.5 s |
| [256 H100, resumed from step 471](https://wandb.ai/nyu-dice-lab/sft-pipe-repro/runs/snowball-sft-production-256h100-20261027c) | 87.8 | 83.8 | 78.4–93.7 s |

At the observed 256-H100 rate, **5% of the 4,750-update schedule takes about 5 hours 48 minutes** (238 updates), and **10% takes about 11 hours 35 minutes** (475 updates). These are steady training-time projections, excluding setup, compilation and failed launch attempts. We actually trained the first 471 steps on 128 H100s; we did not measure a fresh 256-H100 run from step zero. The exported step-222 and step-471 checkpoints are the nearby 4.7% and 9.9% points.

[Timing summary](https://github.com/marin-community/marin/blob/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/training-timing/summary.json) and [logged timing/loss evidence (compressed JSON)](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/training-timing/evidence.json.gz).

Loss dropped sharply early and then improved more slowly:

| Held-out loss | Initialization | 450 updates (logged step 449) | 2300 updates (logged step 2299) |
|---|---:|---:|---:|
| Combined | 0.966 | 0.510 | 0.488 |
| SFT | 1.000 | 0.366 | 0.336 |
| Pretraining replay | 0.886 | 0.846 | 0.842 |

The resumed run's first and last 50 logged training losses averaged 0.594 and 0.547. That is consistent with continued learning, but held-out loss is the cleaner comparison because the training mixture changes over the schedule. These losses do not establish downstream quality parity with the original run. The [original 10-05 W&B run](https://wandb.ai/marin-community/marin_moe_sft/runs/grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped) is linked for comparison.

We slightly varied the replay recipe to reuse data already available in S3. The supervised side uses the same 216-source catalog, apart from 88 packed validation sequences. Replay still accounts for about **20.35% of scheduled positions**, with **8× long-document weighting**, but we reused Datakit cache `store_4d2e363d` with Harrier main-phase weights instead of reconstructing the original `replay_skew8.json` selection. That changes replay source selection and weights, so this is a close recipe reproduction rather than an identical data stream. The [training confound audit](https://gist.github.com/penfever/dfb8550a3643849dfea701df44050757) records the differences.

### 2. Learning curves, and the agentic outliers that exposed the serving issue

These figures preserve the **historical evals, before the serving correction**. They show why we investigated; they should not be read as corrected performance estimates. Each plot includes benchmarks whose endpoint change exceeds five percentage points, so the plots deliberately omit smaller movements.

![Base to step 222: broad early benchmark gains](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures/base_to_step222.png)

The first 222 updates produced broad gains: 17 of 26 benchmarks moved by more than five points, all upward. MATH-500 rose from 10.4% to 93.0%, AIME from 13.0% to 72.3%, and HumanEval+ from 17.7% to 74.4%. A small cold start changes what the base model can demonstrate, especially on the non-agentic benchmarks.

![Step 222 through step 2048: gains, plateaus and declines](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures/step222_to_step2048_all.png)

After that, the curves looked like a plausible mixture of gains, plateaus and tradeoffs. Eighteen of 26 benchmarks stayed within five points between steps 222 and 2048. Four improved by more than five points, including BFCL-Parity (39.8% → 66.7%) and SWE-bench (11.5% → 18.8%); four declined, including GSM8K (71.9% → 61.6%). This alone would not suggest a broken experiment.

![Reproduction checkpoints compared with the separate original 10-05 release](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures/step222_to_full_all.png)

The suspicious part was the persistent agentic gap. At step 2048, OT-TBLite was still 3.2% versus the original release's 25.6%, Terminal-Bench was 0.8% versus 12.4%, and DS-1000 was 7.0% versus 27.1%. The dashed segments end at the **separately trained original 10-05 release**, not a later checkpoint of our run. They made the gap conspicuous, but did not prove that more training caused it.

Those outliers prompted manual trace comparisons. We found oversized native tool-call batches and then audited the serving configs: the exported models lacked the original second EOS token, and thinking was not reliably enabled. Correcting those settings substantially changed the smoke trace and improved the partial rerun scores below. The curves helped us find the problem; the trace and config checks supplied the evidence.

<details>
<summary>Additional figure splits and source data</summary>

![Step 222 to 2048: gains](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures/step222_to_step2048_gains.png)

![Step 222 to 2048: declines](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures/step222_to_step2048_declines.png)

![Original-release comparison: largest changes](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures/step222_to_full_largest.png)

![Original-release comparison: other changes](https://raw.githubusercontent.com/marin-community/marin/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures/step222_to_full_other.png)

[All figures, CSVs and provenance](https://github.com/marin-community/marin/tree/codex/sft-cold-start-reference-20261030/experiments/sft_cold_start/figures). The historical “Terminal-Bench 2” plot label refers to the audited 2.1 v6 benchmark. Historical step-2048 SWE coverage was 453/500; these plots preserve that snapshot rather than silently replacing it with corrected scores.

</details>

### 3. Corrected agentic evals: provisional means

At 16:00 UTC on October 30, these were the raw partial means. The standalone tracker linked below carries denominators, source records, provisional snapshots and audited final columns.

| Checkpoint | Fraction of 4,750 updates | SWE-bench mean (scored / 500) | OT-TBLite mean (scored / 300) |
|---|---:|---:|---:|
| Step 222 | 4.7% | 24.7% (170) | 17.9% (259) |
| Step 471 | 9.9% | 32.3% (186) | 23.9% (238) |
| Step 1024 | 21.6% | 34.7% (196) | 20.9% (241) |
| Step 2048 | 43.1% | 31.4% (153) | 23.5% (213) |

The [historical 10-05 reference cells in the standalone tracker](https://gist.github.com/penfever/33f9c55379bd33eae6709687ec1c251f) are 39.4% SWE-bench and 25.6% OT-TBLite. These are not matched completion subsets or a fresh matched-runtime control, but step 471 is already in a useful range. Our working hypothesis is that ablations can use roughly 10% of cold-start exposure and retain most of the performance. Step 471 represents 31.61B scheduled positions, including replay; it does not prove that a carefully selected 10% of unique source documents would do the same thing.

The partial means exclude unfinished/unscored attempts and still need verifier review. Some setup failures were recorded as zeros, and slower tasks have not finished. OT-TBLite is mean fractional reward, not binary accuracy. We will keep the claim provisional and update it when the corrected evals land.

## The stop-token confound

The original release's `generation_config.json` specifies EOS `[128001, 128009]` and BOS `128000`. Our exported reproduction repositories lacked that file, so normal loading could stop only on `128001` and continue past the assistant end-of-turn token `128009`. The reproduction's default chat-template behavior also did not reliably request `/think`.

We copied the original generation config into all six reproduction repositories and verified that every weight shard and other file stayed unchanged. The rerun profiles pin the new revisions, explicitly override the same EOS/BOS settings, and force thinking in the server template. Live probes confirmed `/think` and immediate stopping on both EOS tokens.

The same-task step-222 SWE smoke changed from seven assistant responses with 166 tool calls (maximum batch 33) to 115 responses with 120 calls (maximum batch two). It now emits much more explicit reasoning. That is a concrete improvement in interaction structure, not a guarantee of task success: the smoke still failed its tests. Terminus command batches are a separate format and can still be large. The corrected cohort pins the Marin vLLM fork at `01911be34fac` and Harbor at `9d099696`; historical runtime/agent versions varied, as documented in the [confound audit](https://gist.github.com/penfever/dfb8550a3643849dfea701df44050757). Because EOS, thinking and runtime standardization changed together, we have not isolated an EOS-only effect.

## What to do next

1. Finish the corrected evals, audit verifier execution and coverage, and update the standalone tracker and the release-policy tracker cells. Keep the original confounded results as history.
2. If the full results support it, run data/recipe ablations from the same base for roughly 471 updates rather than the full 4,750. For follow-on experiments that do not change cold-start data, the exported step-471 checkpoint is also a reusable initializer. Check more than the two currently available benchmark means.
3. Clean the cold-start data with GLM: schema inconsistencies, invented arguments, repetitive repair churn, and other structural flaws. Repair a trace only when the correction can be grounded and verified; otherwise drop it. Keep the unchanged corpus as a control.
4. Improve instruction-following SFT for the [behaviors observed in our eval trace review](https://gist.github.com/penfever/dfb8550a3643849dfea701df44050757): claiming work that never happened, inventing conversation history or environment state, and substituting invented measurements for supplied data. Teach the model to use the supplied facts, ask for missing inputs, verify its actions, and distinguish attempted from completed work.
5. Carry over the lessons from the [earlier Snowball/R2E-Gym report](https://claude.ai/artifact/TefbJfphaTBsYFtFgrNgMY): do not transfer patch-file conventions into a harness that cannot consume them; edit the actual source, test it, stop, and recover from a busy terminal without repeating commands forever. That report concerns a separate Stage-3 checkpoint, so recheck its source attributions against this mixture. Excessive retained reasoning and unfinished demonstrations also deserve attention.
6. Try [laion/calibforge-relay-traces](https://huggingface.co/datasets/laion/calibforge-relay-traces) after decontamination against the eval suite and holdouts. Preserve its teacher-only `trained=true` supervision masks; do not feed it unchanged into an all-token-loss loader. Match trained-token budgets and test the addition separately from cleaning.

The data audit sampled 921 documents across all 216 supervised sources (12.6M tokens). It found specific schema/grounding problems, but did not find counterparts to the huge native tool-call batches. That distinction matters: fix the serving confound in serving, and clean data defects on their own evidence.

## Artifacts

- [SFT Cold Start Subsampling: all checkpoints, base and original reference](https://huggingface.co/collections/open-athena/sft-cold-start-subsampling-6ae4c434d2f87e2ae7ea0392)
- [Training-data audit overview, provenance and sampling manifest](https://gist.github.com/penfever/40c099e9d771a53268c25e79342d789c)
- [Protocol audit, scanner and candidates](https://gist.github.com/penfever/76dc092af41477a20ab8c5411c519036)
- [Grounding audit, scanner, calibration and review ledger](https://gist.github.com/penfever/7350eef547e078ddace6880ae6e05217)
- [Recovery/schema audit, scanner and review ledger](https://gist.github.com/penfever/b28df7784c3da6ffb95d6ca546be0325)
- [Consolidated reference branch](https://github.com/marin-community/marin/tree/codex/sft-cold-start-reference-20261030)

- [Standalone provisional and audited-final score tracker](https://gist.github.com/penfever/33f9c55379bd33eae6709687ec1c251f)
- [Trace diagnostics and training/serving confound audit](https://gist.github.com/penfever/dfb8550a3643849dfea701df44050757)

- [GLM cleaning and instruction-following follow-up #10788](https://github.com/marin-community/marin/issues/10788)

- [Reference-only draft PR #10789](https://github.com/marin-community/marin/pull/10789), closed as requested because the full archival diff exceeds 1,000 lines. The branch preserves all code/config snapshots with original hashes and a restore command; it is not intended to merge.

Compact extractions:

- [Corrected four-model eval profiles #10791](https://github.com/marin-community/marin/pull/10791)

- [Preserve BF16 checkpoint payloads #10792](https://github.com/marin-community/marin/pull/10792)
- [Preserve integral mixture allocations #10793](https://github.com/marin-community/marin/pull/10793)
- [Bound pending replay-cache batches #10794](https://github.com/marin-community/marin/pull/10794)
- [Handle singleton FA4 metadata sharding #10796](https://github.com/marin-community/marin/pull/10796)
- [Serialize tokenizer staging across processes #10797](https://github.com/marin-community/marin/pull/10797)
- [Preserve reference generation config during HF export #10798](https://github.com/marin-community/marin/pull/10798)
- [Existing multipart-copy fix #10639](https://github.com/marin-community/marin/pull/10639), reused rather than duplicated

The library fixes received independent subagent review before publication. Their PRs record focused regression checks, broader test results, and failures independently reproduced on unchanged main. Nothing has been merged by this experiment publication.
