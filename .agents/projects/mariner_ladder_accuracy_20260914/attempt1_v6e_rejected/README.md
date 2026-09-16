# MARINER 1e21 accuracy evaluation — 14 September 2026

The user authorized resuming the previously deferred accuracy evaluation for completed 1e21 ladder checkpoints. This batch contains Proportional and UniMax-8 only: both have verified final step-22056 HF exports. The MARINER and matched-Olmix 1e21 checkpoints were incomplete at launch and are not substituted with intermediate checkpoints or historical policies. No training is submitted.

## Submission

Iris acknowledged `/calvinxu/mariner-ladder-accuracy-1e21-20260914` at 2026-09-14 05:52:18 UTC (13 September, 22:52 PDT). The parent uses interactive priority, one CPU, 4 GB RAM, nonpreemptible placement, a 48-hour timeout and no parent retries. Parent, child TPUs, checkpoints, dataset cache and outputs all use us-east5/us-east5-a. Two v6e-8 children are requested concurrently with 8 CPUs, 64 GB host RAM and 128 GB disk each. Initial parent state: building its runtime, zero failures. `submit.sh` is the exact command; `submission.log` contains acknowledgment.

## Evaluation and retention

The existing easy-overlap definitions give eleven families and 67 leaf tasks: all 57 MMLU subjects, ARC Easy, ARC Challenge, CommonsenseQA, HellaSwag, WinoGrande, SocialIQA, PIQA, SciQ, Lambada and MedMCQA. Each checkpoint is evaluated on every example in the evaluation split: 44,248 documents total. All tasks use five shots except Lambada (zero); seed 0, no chat template, context length 4,096, total batch size eight, BF16 weights and compute.

This yields choice accuracy (including normalized accuracy where returned) and Lambada greedy-word accuracy. It is the eleven-family accuracy overlap, not the complete 51-head OlmoBaseEval Easy likelihood objective: the math/code generation families are not included. No headline aggregation has been selected. Averaging all 67 leaves equally would heavily weight MMLU; retain subject and task-group results so the paper can choose a clearly described summary.

Every returned metric, standard error, task configuration, per-document sample/response, and Levanter per-request score is preserved. Each checkpoint writes `results.json.gz`, `summary_metrics.json`, `provenance.json`, and finally `SUCCESS.json`. The provenance contains the full plan, checkpoint object identities, runtime/source pins, task counts and live device topology. Readback hashes and complete task/sample coverage are mandatory. Resubmission reuses a completed row only after those checks pass.

The worker stages the complete east5 Hugging Face cache and loads all tasks with network dataset access disabled before initializing the model. The checkpoint's own tokenizer and archived Qwen3 architecture are retained through both converter constructions; no default Qwen tokenizer is fetched.

## Verification

Complete final HF exports and all three weight shards per checkpoint were verified by object size, generation and CRC32C; only metadata was read locally. The regional cache contains every task, including all 57 MMLU subjects. Its manifest uses layout version 2 despite the path suffix v1. Source review, empty-cache offline converter fixture, strict result-retention probes, repository lint, targeted pyrefly, local plan/input validation and exact-command regional validation passed. `accuracy_review.md` records the independent review and remaining live checks. The workspace bundle contains every frozen source and the exact plan; its size is 23.2 MiB.

## Collection

Read each `summary_metrics.json` first. Verify its `SUCCESS.json` identities and complete full-result coverage through `verified_result(plan, row)` before treating the result as complete. Keep these fresh accuracy results separate from older BPB outputs. Update the scaling table/figure and appendix protocol only after both matching task coverage and checkpoint provenance have been checked. The manuscript currently contains no fresh accuracy measurements.

## Provenance

- Plan SHA-256: `4ba9833e97257047722b01665a760d3fd19204b7d4bf3cdf2fb6786bc4736482`.
- Canonical outputs: `gs://marin-us-east5/experiments/mariner_ladder_accuracy_20260914/4ba9833e97257047722b01665a760d3fd19204b7d4bf3cdf2fb6786bc4736482/`.
- Dataset cache: `gs://marin-us-east5/raw/eval-datasets/olmo-base-easy-overlap-v1`.
- Fieldbook experiment: `exp_01kvvvv6zxrf0j7tkp4f7k6y66`; parent job: `job_01m2f7j2ccs856r8tk4scnvkwz`.
- Evaluation run `proportional_1e21-2f1a48_accuracy67`: `run_01m2f7g0wknzqq5prazf680a9r`; parent training run `run_01kvvvwj657tsjqbhqpcmkdn8f`.
- Evaluation run `unimax8_1e21-d685cd_accuracy67`: `run_01m2f7g16506yna154azs3m4wr`; parent training run `run_01kvvvwj8n4hza9yt79bg2g0jz`.
- Proportional final checkpoint: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/proportional_1e21-2f1a48/hf/step-22056`.
- UniMax-8 final checkpoint: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/unimax8_1e21-d685cd/hf/step-22056`.
