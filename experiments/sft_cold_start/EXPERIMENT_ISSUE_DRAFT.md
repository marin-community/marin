# SFT Cold Start Subsampling: H100 reproduction and a cheaper ablation baseline

Agent-authored experiment draft. Results below are provisional as of October 30, 2026; corrected evals are still running.

## TL;DR

We successfully reproduced the SFT cold-start workflow on a new H100 cluster: import the base, stage the mixture, train, resume at a different GPU count, and export verified checkpoints. The first corrected agentic evals look good. Roughly 10% of the full cold-start training budget may be enough to recover most of the useful performance, which would make ablations much cheaper.

We initially thought the reproduction had a large agentic capability gap. A major confound was in serving: exported checkpoints were missing the original model's second stop token, and thinking was not reliably enabled. We fixed the generation configs on Hugging Face and explicitly enabled thinking in the rerun configs. On the same SWE task, step 222 went from batches of up to 33 tool calls to at most two. The corrected partial evals are much stronger. We do not currently have a reason to launch another training run; finish these evals first.

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

At 16:00 UTC, these were the raw partial means. The standalone tracker linked below carries denominators, source records, provisional snapshots and audited final columns.

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

Reference PR and compact extraction PR links are added below as publication completes.
