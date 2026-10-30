## Overview

This document is an attempt to formalize Marin’s policy towards what constitutes reportable, public-facing, canonical eval results for Marin models as well as the baselines we compare against.

Why do we need this document? With the first “snowball” release of a 67B A2B MoE, Marin is entering a regime where we regularly compare Marin models to existing open and closed-source models. Because Marin’s unique claim is that we are open development, we will need to hold ourselves to a high standard of transparency on how we report eval scores of Marin models and what claims we make about Marin model capabilities based on those scores.

What is in scope? This document is intended to act as a living snapshot, trustworthy for any given period in time, as to the what and the how of Marin’s canonical set of evals, a set of standards to which we will aspire to hold ourselves. By "the what", we mean “which evals did we run, with what consistent set of hyperparameters and experimental constraints”. By "the how", we mean “where is the self-contained codebase that is both necessary and sufficient to reproduce our eval results independently”.

## Previous version notes

* Released Sept 24, 2026: 18 non-agentic benchmarks, 8 Harbor agentic benchmarks, and 4 reserved out-of-distribution benchmarks.

## Complete eval list, with hyperparameters

### In Distribution

| Eval | Framework | Metric | Shots | Repetitions | Thinking mode | Allowed context length | Mean runtime | Runtime SD |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| MATH-500 | Evalchemy | accuracy | 0 | 1 run × 1 sample per item | on | 73,664 tokens | 24m 36s | 31m 21s |
| HumanEval+ | Evalchemy | accuracy | 0 | 1 run × 1 sample per item | off | 73,664 tokens | 3m 15s | 6m 01s |
| MBPP+ | Evalchemy | accuracy | 0 | 1 run × 1 sample per item | off | 73,664 tokens | 6m 46s | 10m 41s |
| OlympiadBench (30-question subset) | Evalchemy | accuracy | 0 | 1 run × 10 samples per item | on | 73,664 tokens | 2h 01m | 2h 26m |
| GSM8K | lm-eval-harness | accuracy | 0 | 1 run × 1 sample per item | off | 73,664 tokens | 12m 06s | 30m 27s |
| PIQA | lm-eval-harness | accuracy | 0 | 1 run × 1 choice decision per item | not applicable | 73,664 tokens | 2m 31s | 4m 33s |
| WinoGrande | lm-eval-harness | accuracy | 5 | 1 run × 1 choice decision per item | not applicable | 73,664 tokens | 5m 02s | 20m 51s |
| BoolQ | lm-eval-harness | accuracy | 0 | 1 run × 1 choice decision per item | not applicable | 73,664 tokens | 4m 35s | 8m 14s |
| TruthfulQA | lm-eval-harness | mc2 | 0 | 1 run × 1 scored answer set per item | not applicable | 73,664 tokens | 5m 59s | 18m 13s |
| TriviaQA | lm-eval-harness | accuracy | 5 | 1 run × 1 sample per item | off | 73,664 tokens | 1h 15m | 2h 38m |
| AIME24 | Evalchemy | accuracy | 0 | 1 run × 10 seeded samples per item | on | 73,664 tokens | 2h 05m | 1h 59m |
| MMLU-Pro | Evalchemy | accuracy | 0 | 1 run × 1 sample per item | on | 65,536 tokens | 9h 41m | 13h 03m |
| GPQA Diamond | Evalchemy | accuracy | 0 | 1 run × 3 samples per item | on | 73,664 tokens | 1h 49m | 2h 59m |
| CRUXEval (output) | Evalchemy | accuracy (output) | 0 | 1 run × 1 sample per item | off | 73,664 tokens | 44m 08s | 1h 19m |
| FinanceBench | Evalchemy | accuracy | 0 | 1 run × 1 sample per item | off | 73,664 tokens | 1m 08s | 2m 02s |
| IFBench | Evalchemy | strict prompt-level accuracy | 0 | 1 run × 1 sample per item | off | 73,664 tokens | 10m 40s | 10m 26s |
| MRCR | Evalchemy | mean sequence similarity | 0 | 1 run × 1 sample per selected item | off | model-specific; at most 73,728 tokens | 20m 42s | 13m 16s |
| NUPA stratified (Loose) | Evalchemy | accuracy | 0 | 1 run × 1 sample per 5,000 selected items | model default | 73,664 tokens | 8h 23m | 9h 28m |
| SWE-bench Verified | Harbor / mini-swe-agent 2.1.0 | accuracy | 0 | 1 agent trial per task | model default | 32,768 tokens | 7h 57m | 8h 41m |
| OpenThoughts-TBLite 2.0 | Harbor / Terminus-2 | accuracy | 0 | 3 agent trials per task | model default | 32,768 tokens | 5h 15m | 2h 42m |
| Terminal-Bench 2.1 | Harbor / Terminus-2 | accuracy | 0 | 3 agent trials per task | model default | 65,536 tokens; 16,384 per turn | 6h 18m | 2h 52m |
| DS-1000 mini-200 | Harbor / Terminus-2 | accuracy | 0 | 1 agent trial per task | model default | 32,768 tokens | 51m 42s | 42m 53s |
| BFCL-Parity @ Pi | Harbor / Pi | accuracy | 0 | 1 agent trial per task | model default | 32,768 tokens | 26m 06s | 1h 04m |
| BixBench CLI corrections @ Pi | Harbor / Pi | accuracy | 0 | 1 agent trial per task | model default | 65,536 tokens; 16,384 per turn | 3h 28m | 3h 20m |
| τ³-bench @ Pi | Harbor / Pi ACP | accuracy | 0 | 1 agent trial per task | model default | 131,072 tokens; 32,768 per turn | 6h 28m | 3h 45m |
| SOTOPIA-hard | Harbor / SOTOPIA agent | length-controlled normalized overall | 0 | 1 episode per scenario | model default | 32,768 tokens | 33m 30s | 22m 19s |

Runtime statistics use all 34 completed, tracker-linked runs per benchmark and report the sample
standard deviation across models. Evalchemy and lm-eval-harness runtimes use the evaluator's
recorded evaluation duration. Harbor runtimes span the first trial start through the last trial
finish; a policy-valid partial seal ends at its immutable freeze time. Scheduler wait and model
server startup are excluded.

### Eval health check

This snapshot covers the 884 canonical tracker cells available on October 16, 2026. Error rates
are computed per model and reported as the mean ± sample standard deviation across the 34-model
cohort. Grade-able errors retain a benchmark score; infrastructure errors do not. Pre-emptions are
retry-scheduled producer or evaluator-client attempts whose Iris backend state was
`WORKER_FAILED`, `KILLED`, or `PREEMPTED`. Job failures count exact backend `FAILED` outcomes per
100 observed terminal producer or client attempts; normal producer teardown is not a failure.

| Eval | Pre-emptions / run | Grade-able error rate | Infrastructure error rate | Failed jobs / 100 attempts |
| --- | ---: | ---: | ---: | ---: |
| MATH-500 | 0.09 | 2.99% ± 17.14% | 0.45% ± 1.84% | 0.00 |
| HumanEval+ | 0.09 | 0.13% ± 0.73% | 0.02% ± 0.10% | 0.00 |
| MBPP+ | 0.06 | 2.47% ± 14.43% | 0.01% ± 0.05% | 0.00 |
| OlympiadBench | 0.00 | 3.20% ± 17.14% | 6.48% ± 19.03% | 0.00 |
| GSM8K | 0.06 | 0.00% ± 0.01% | 0.16% ± 0.91% | 0.00 |
| PIQA | 0.06 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 |
| WinoGrande | 0.03 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 |
| BoolQ | 0.03 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 |
| TruthfulQA | 0.00 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 |
| TriviaQA | 0.18 | 1.94% ± 11.32% | 2.20% ± 10.04% | 0.00 |
| AIME24 | 0.00 | 3.18% ± 17.13% | 3.47% ± 9.84% | 0.00 |
| MMLU-Pro | 0.06 | 2.95% ± 17.15% | 0.78% ± 2.13% | 0.00 |
| GPQA Diamond | 0.03 | 3.08% ± 17.13% | 2.20% ± 9.03% | 0.00 |
| CRUXEval | 0.00 | 1.32% ± 7.68% | 0.06% ± 0.27% | 0.00 |
| FinanceBench | 0.03 | 2.29% ± 13.38% | 3.47% ± 13.18% | 0.00 |
| IFBench | 0.00 | 0.66% ± 3.83% | 0.14% ± 0.61% | 0.00 |
| MRCR | 0.00 | 3.70% ± 16.80% | 0.00% ± 0.01% | 0.00 |
| NUPA stratified (Loose) | 0.12 | 3.15% ± 17.13% | 1.57% ± 3.84% | 0.00 |
| SWE-bench Verified | 0.62 | 1.22% ± 3.22% | 0.93% ± 1.80% | 5.68 |
| OpenThoughts-TBLite 2.0 | 2.59 | 29.34% ± 26.05% | 3.72% ± 2.97% | 3.21 |
| Terminal-Bench 2.1 | 0.00 | 31.54% ± 25.39% | 3.66% ± 2.18% | 5.88 |
| DS-1000 mini-200 | 0.32 | 10.44% ± 21.79% | 0.38% ± 0.90% | 5.06 |
| BFCL-Parity @ Pi | 0.03 | 1.15% ± 1.08% | 0.29% ± 1.18% | 5.80 |
| BixBench CLI corrections @ Pi | 0.53 | 2.17% ± 3.42% | 0.95% ± 2.18% | 2.33 |
| τ³-bench @ Pi | 0.12 | 10.42% ± 16.73% | 2.14% ± 3.01% | 16.90 |
| SOTOPIA-hard | 0.03 | 0.03% ± 0.17% | 0.18% ± 0.46% | 1.45 |

The observed error counts below are campaign totals. Taxonomy is framework-specific:
`AgentTimeoutError` is grade-able in Harbor but infrastructure-classified under the pinned
Evalchemy implementation. The one grade-able `ArtifactUploadTimeoutError` retained a verifier
reward in an operator-reviewed Harbor seal; it does not change that error's default infrastructure
classification.

| Class | Recognized error type | Count |
| --- | --- | ---: |
| Grade-able | `malformed_model_response` | 33,729 |
| Grade-able | `AgentTimeoutError` | 8,082 |
| Grade-able | `NonZeroAgentExitCodeError` | 186 |
| Grade-able | `ContextLengthExceededError` | 33 |
| Grade-able exception | `ArtifactUploadTimeoutError` with retained reward | 1 |
| Infrastructure | `model_transport` | 12,931 |
| Infrastructure | `AgentTimeoutError` | 7,955 |
| Infrastructure | `no_result_written` | 302 |
| Infrastructure | `VerifierRuntimeError` | 245 |
| Infrastructure | `TmuxBatchProtocolError` | 187 |
| Infrastructure | `grader_infrastructure` | 145 |
| Infrastructure | `EnvironmentStartTimeoutError` | 121 |
| Infrastructure | `TmuxCommandError` | 69 |
| Infrastructure | `TmuxSessionEndedError` | 63 |
| Infrastructure | `VerifierTimeoutError` | 62 |
| Infrastructure | `Tau3RuntimeInfrastructureError` | 56 |
| Infrastructure | unclassified legacy errors | 49 |
| Infrastructure | `AgentSetupTimeoutError` | 45 |
| Infrastructure | `AgentKilledBySignalError` | 18 |
| Infrastructure | `AddTestsDirError` | 15 |
| Infrastructure | `PhaseCancellationError` | 6 |
| Infrastructure | `DaytonaProcessError` | 5 |
| Infrastructure | `SandboxBuildFailedError` | 5 |
| Infrastructure | `DaytonaSandboxStopError` | 2 |
| Infrastructure | `ArtifactUploadTimeoutError` | 2 |

The [machine-readable CSV](https://gist.github.com/penfever/26168603aaaebe890093826c9187413c#file-eval-health-check-20261016-csv) and
[audit JSON](https://gist.github.com/penfever/26168603aaaebe890093826c9187413c#file-eval-health-check-20261016-json) preserve per-benchmark type counts, rates,
run-level inputs, and Iris-attempt denominators.

### Out of Distribution

* AAII 4.3
* Terminal-Bench Science
* DeepSWE 1.1 (deep-swe-litecontainer@1.1)

### In Distribution but Not Currently Evaluated (by contamination)

* Harbor-Index

### Details on our Evals

- **MATH-500.** OpenAI's [May 2023 process-supervision release](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/) used a 500-problem subset of the original MATH competition-math benchmark. The campaign scores one zero-shot generated solution per problem by final-answer accuracy.
- **HumanEval+.** The [May 2023 EvalPlus paper](https://arxiv.org/abs/2305.01210) introduced expanded tests for HumanEval programs to catch incorrect code that passes sparse original tests. The campaign executes one zero-shot completion per problem against the stronger tests.
- **MBPP+.** The [November 24, 2023 EvalPlus release](https://github.com/evalplus/evalplus/releases/tag/v0.2.0) extends MBPP with additional functional tests. The campaign's pinned config is zero-shot and scores one generated program per problem.
- **OlympiadBench.** The [February 2024 paper](https://arxiv.org/abs/2402.14008) introduced bilingual, multimodal Olympiad-level math and science problems. This campaign uses Evalchemy's 30-question text subset, ten sampled solutions per question, and the configured answer judge.
- **GSM8K.** The [October 2021 paper](https://arxiv.org/abs/2110.14168) introduced grade-school math word problems requiring multi-step arithmetic. The campaign uses the zero-shot `gsm8k-0shot` completion route and scores the extracted final answer.
- **PIQA.** The [November 2019 paper](https://arxiv.org/abs/1911.11641) introduced two-choice questions about physical commonsense. The campaign selects the answer by length-normalized log-likelihood with no examples in the prompt.
- **WinoGrande.** The [July 2019 paper](https://arxiv.org/abs/1907.10641) introduced adversarial pronoun/coreference choices designed to reduce superficial cues. The campaign uses five-shot multiple-choice scoring.
- **BoolQ.** The [May 2019 paper](https://arxiv.org/abs/1905.10044) introduced naturally occurring yes/no questions paired with evidence passages. The campaign scores zero-shot choice accuracy.
- **TruthfulQA MC2.** The [September 2021 paper](https://arxiv.org/abs/2109.07958) introduced questions that elicit common misconceptions from language models. The campaign's MC2 metric sums probability mass on all true answers within each question's candidate set, rather than counting a single selected answer.
- **TriviaQA.** The [May 2017 paper](https://arxiv.org/abs/1705.03551) introduced trivia questions with independently collected evidence documents. The campaign runs five-shot answer generation and reports answer accuracy.
- **AIME24.** The [Mathematical Association of America's 2024 AIME](https://maa.org/maa-invitational-competitions/) supplies the contest problems. Evalchemy samples ten solutions per question in one run through the shared seeded-repeat implementation and scores the integer final answers.
- **MMLU-Pro.** The [June 2024 paper](https://arxiv.org/abs/2406.01574) introduced harder, more reasoning-focused multiple-choice questions across fourteen subjects. The campaign runs zero-shot generation with a 65,536-token cap and reports micro-averaged accuracy.
- **GPQA Diamond.** The [November 2023 paper](https://arxiv.org/abs/2311.12022) introduced expert-written graduate-level science questions, including the Diamond subset. The campaign averages accuracy over three generated answers per question.
- **CRUXEval (output).** The [January 2024 paper](https://arxiv.org/abs/2401.03065) introduced input- and output-prediction tasks for short Python programs. The tracker reports the output-prediction direction, scored on one generated answer per item.
- **FinanceBench.** The [November 2023 paper](https://arxiv.org/abs/2311.11944) introduced questions grounded in public financial filings. The campaign scores one generated answer per question with its pinned LLM judge.
- **IFBench.** The [July 2025 paper](https://arxiv.org/abs/2507.02833) introduced verifiable constraints that test generalization in precise instruction following. The campaign reports strict prompt-level accuracy, requiring every constraint to hold in the raw response.
- **MRCR.** [Gemini's September 2024 report](https://arxiv.org/abs/2409.12640) introduced multi-round coreference resolution; [OpenAI's public MRCR release (April 2025)](https://huggingface.co/datasets/openai/mrcr) supplies the corrected multi-needle data used here. The campaign scores nonce-gated sequence similarity and selects only context-length bins that fit the served model.
- **NUPA stratified (Loose).** The [November 2024 Number Cookbook paper](https://arxiv.org/abs/2411.03766) introduced tests of numerical understanding and processing across representations and tasks. The campaign uses Evalchemy's pinned `NUPA5K-Loose` benchmark: a deterministic 5,000-record stratified sample covering 44 tasks and 2,391 strata, scored with permissive answer extraction rather than the paper's direct-answer protocol.
- **SWE-bench Verified.** OpenAI's [August 13, 2024 release](https://openai.com/index/introducing-swe-bench-verified/) introduced the 500-task, human-validated repository-issue repair set. The campaign runs Harbor's canonical `swebench-verified@1.0` implementation with mini-swe-agent 2.1.0, one trial per issue, and executable verification.
- **OpenThoughts-TBLite 2.0.** The [February 18, 2026 release](https://www.openthoughts.ai/blog/openthoughts-tblite) introduced a 100-task, faster-running terminal-agent panel calibrated against Terminal-Bench 2. The campaign runs its pinned `openthoughts-tblite@2.0` tasks with Terminus-2, three trials per task, and reports mean verifier success across trials.
- **Terminal-Bench 2.1.** The [November 7, 2025 release](https://www.tbench.ai/news/announcement-2-0) introduced a harder, more thoroughly verified terminal-agent task set alongside Harbor. The campaign uses `terminal-bench/terminal-bench-2-1` at ref 6 through Harbor, with three Terminus-2 trials per task, a 65,536-token context window, and a 16,384-token per-turn output cap.
- **DS-1000 mini-200.** The [November 2022 paper](https://arxiv.org/abs/2211.11501) introduced data-science code-generation problems spanning seven Python libraries. The campaign evaluates a fixed seed-42 200-task Harbor subset with Terminus-2, building task images from the pinned Dockerfiles.
- **BFCL-Parity @ Pi.** Berkeley's [February 2024 BFCL release](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) introduced executable function-calling evaluation. This campaign uses the Harbor `bfcl_parity@1.0` adaptation with the Pi agent, so its score is the pinned adaptation's verified solve rate rather than an unqualified BFCL leaderboard score.
- **BixBench CLI corrections @ Pi.** BixBench is a computational biology benchmark. The [February 2025 BixBench paper](https://arxiv.org/abs/2503.00096) introduced multi-step computational-biology analysis tasks. This campaign uses Pi on 205 `bixbench-cli-marin-corrections@3` tasks and reports binary answer-equivalence reward, which is distinct from canonical BixBench CLI v1.5.
- **τ³-bench @ Pi.** Tau bench tests whether a model can reliably help someone complete a task from beginning to end. Sierra's [March 18, 2026 τ³-Bench launch](https://sierra.ai/blog/bench-advancing-agent-benchmarking-to-knowledge-and-voice) introduced knowledge and voice tracks alongside corrections to the existing airline and retail tasks. The campaign runs pinned `tau3-bench@1.0.1` in Harbor through the Pi ACP relay and scores task verification.
- **SOTOPIA-hard.** SOTOPIA is a role-playing benchmark where the agent must assume a fictional character and pursue a goal in a simulated scenario, pitted against other agents. The [October 2023 SOTOPIA paper](https://arxiv.org/abs/2310.11667) introduced interactive social-agent evaluation and identified a hard scenario subset. The campaign runs the 100 hard episodes against a fixed `openai/gpt-oss-120b` counterpart and judge. It linearly normalizes each of the seven official dimensions, averages them into an overall score, and applies a cohort-level response-length control. Normalized goal achievement remains available as a non-default diagnostic.

### How do we handle in-distribution and out-of-distribution evals?

#### IN-DISTRIBUTION

**We will** -- dedup against our SFT data, look at model scores while we develop the model, inspect the correlation of our ID evals with our OOD evals, remove or modify ID evals if we think the results are broken, unintuitive or wrong (reporting the eval as modified)

**We may** -- develop synthetic RL tasks modeled on specific tasks in ID evals

**We will not** -- train on teacher rollouts from eval questions, use our ID eval instructions directly as the basis of RL data, deliberately modify ID evals to favor our own models

#### OUT-OF-DISTRIBUTION

**We will** -- de-dup against our SFT raw data, look at meta-data about these evals (e.g., how big they are, general knowledge domain, etc), eval baselines on these evals

**We may** -- evaluate our final models on out-of-distribution benchmarks

**We will not** -- look at the individual questions in the evals, look at any of our own model scores from these evals until our model is frozen, model any synthetic tasks on tasks in the evals (edited), modify OOD evals

### How do we choose which model to release?

We track dozens of candidate checkpoints. Including every candidate in the baseline ranking would
make the critical-difference plot unreadable and would let the number of closely related Snowball
checkpoints bias the cohort. We therefore select one release checkpoint in an internal tournament,
then compare only that checkpoint with the baseline cohort.

The tournament includes every Snowball- or Grug-named candidate with a complete set of observed,
durably linked scores. It does not impute missing cells. FinanceBench and PIQA are excluded because
they did not discriminate reliably within this candidate cohort; TriviaQA is excluded because its
answer extractor proved overly sensitive to response format. The other 23 benchmarks receive equal
weight.

For each model pair and benchmark, let `d = score_a - score_b` and let `s_a` and `s_b` be the
task-level SEMs after multiplying each by `1 / sqrt(0.9)` for the infrastructure-failure tolerance.
The outcome is decisive only when `abs(d) > 1.96 * hypot(s_a, s_b)`. Its sign determines the winner;
otherwise the benchmark is a tie. A win is worth 1, a tie 0.5, and a loss 0. Each pairwise win rate
is the mean of those values across the 23 benchmarks. A candidate's tournament score is the
unweighted mean of its pairwise win rates against every other candidate. If candidates have exactly
the same tournament score, pooled benchmark win rate and then model ID break the tie.

The completed 21-model tournament selected
`open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10`, with a mean pairwise win rate of
0.627. This checkpoint is labeled **Snowball** in the baseline figures.

### Metric definitions

- **`accuracy`:** the fraction of scored items that pass the benchmark's configured correctness check. Almost all of our benchmarks use this.
- **`mc2`:** Probability mass on all true answers, normalized over the question's answer set and averaged over questions. It is not a binary selection accuracy, and chance performance is above zero.
- **`accuracy (output)`:** CRUXEval output-prediction accuracy. Given a function and input, the model predicts the output; this is distinct from CRUXEval input prediction, which reconstructs an input from the function and desired output.
- **`strict prompt-level accuracy`:** the fraction of IFBench prompts for which the raw response satisfies every required constraint.
- **`mean sequence similarity`:** MRCR's mean `SequenceMatcher` ratio between the response's answer span and the reference answer. A response that fails MRCR's nonce check receives zero credit. 32,768-token models use bins through 16,384 and the 65,536-token model uses bins through 32,768.
- **`length-controlled normalized overall score`:** SOTOPIA-hard's seven official dimensions are each linearly normalized to `[0, 1]`, then averaged per episode. Across the full comparison cohort, a fractional logistic GLM fits model, task, and `tanh`-transformed response-length effects. The reportable score predicts each episode at zero response-length difference from that task's cohort median, then averages those predictions. Length is the evaluated character's total emitted characters over the episode. The counterpart and evaluator are fixed to `openai/gpt-oss-120b` on Together for every model. Normalized goal achievement is retained as an explicit non-default scoring mode.

### Other Metric Notes

- **Scale:** every metric in the suite is reported on a 0-1 scale with 1 the best and 0 the worst; there are no error counts, ranks, or unbounded scores.
- **Allowed context length:** This is allowed prompt + completion context length. If a model's maximum context length is less than the allowed context length, we adopt the model's maximum context length. For single-turn benchmarks, we subdivide the space into max prompt length = longest prompt in the dataset plus a bit of cushion, max response length = the remaining context window. Unless a benchmark specifies another cap, multi-turn generation is limited to 8192 tokens per turn. Context-ablation labels are `context window / per-turn output limit`: both values are passed explicitly, with the first applied to vLLM `max_model_len` and Pi `model_info.max_input_tokens`, and the second applied to Pi `model_info.max_output_tokens`. For example, `65k-16k` means a 65,536-token serving and Pi context window with a 16,384-token per-turn output limit. Ling Lite's native 32,768-token window uses `mrcr-32k.yaml`; the historical 65,536-token Snowball candidate uses `mrcr-65k.yaml`; every release-figure model uses `mrcr.yaml` otherwise.
- **τ³ context selection:** [The Qwen3.6 τ³ ablation](https://github.com/marin-community/marin/issues/9743) selected 131,072 total context and 32,768 per-turn output (153/373 scoreable trials; estimated accuracy 0.410) as the optimal for this eval.
- **Pi runtime:** Agentic Pi configs must pin `@earendil-works/pi-coding-agent` explicitly. The corrected context control pins Pi `0.87.0`.

### LLM judges

Five evals involve an LLM judge.

- **FinanceBench:** every scored response is judged. The judge sees the question, the
  reference answers, and the candidate, and returns one label: `correct`, `incorrect`, or
  `not_attempted` (equivalent answers in substance count as correct — algebraic equivalence,
  unit conversions, rounding). Only `correct` scores.
- **BixBench CLI correction set:** every submitted answer is judged for semantic equivalence
  to the task's ideal answer and receives a binary reward. A missing answer receives zero
  without invoking the judge.
- **OlympiadBench:** Minerva/SymPy grades first, and the judge answers equivalence only for
  answers the deterministic check cannot establish. Empty extracted answers score zero before
  any judge request. The reported scores use the fixed-grader replay of sealed responses described
  below, rather than the initial judge labels in the generation jobs.
- **SOTOPIA-hard:** after each completed social interaction, the official episode evaluator
  scores believability, relationship, knowledge, secret keeping, social-rule compliance,
  financial and material benefit, and goal achievement. The verifier normalizes all seven
  dimensions and uses their mean as the default per-episode reward. Cohort analysis applies the
  response-length correction defined above; the judge does not perform that correction.
- **τ³-bench:** the verifier combines deterministic task checks with an LLM evaluator for
  natural-language assertions. The separately configured LLM that simulates the user is part of
  the task environment, not the judge.

Judge deployment is benchmark-specific:

- **FinanceBench** and **OlympiadBench** use locally hosted `MiniMaxAI/MiniMax-M3-MXFP8` at revision
  `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba` for their final offline regrades. The judge runs
  on eight H100s in RNO2A with tensor parallelism 8, a 262,144-token context window, and the
  `minimax_m3` reasoning parser. The request constrains the response to `correct`, `incorrect`, or
  `not_attempted` with vLLM `structured_outputs.choice`; Evalchemy retains each benchmark's native
  prompt, parsing, deterministic checks, and score aggregation. The replay reads the already sealed
  model responses and changes only the judge deployment.
- **BixBench** uses a co-hosted `MiniMaxAI/MiniMax-M3-MXFP8` judge at revision
  `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`. Marin serves it on eight H100s with tensor
  parallelism 8, a 32,768-token context window, the `minimax_m3` reasoning parser, prefix
  caching, and CUDA graphs enabled. The verifier receives the co-hosted endpoint and model
  identity directly from Marin; it does not use the Together judge environment.
- **SOTOPIA-hard** uses `openai/gpt-oss-120b` through Together for both the fixed counterpart
  and the official episode evaluator. Its LiteLLM identifier is
  `openai/openai/gpt-oss-120b`: the first segment selects the OpenAI-compatible provider and
  the remainder is Together's model ID.
- **τ³-bench** uses that same Together-hosted model and LiteLLM identifier for both the user
  simulator and the natural-language assertion evaluator. The two roles are independently
  configured even though this campaign pins them to the same checkpoint.

Judge hyperparameters, per path:

- **Answer-equivalence judge** (FinanceBench, OlympiadBench): a single user message with the
  three-way rubric, `temperature=0`, request timeout 300 s with 5 transport retries, and 16
  concurrent judgments per run. `max_tokens` escalates 128 → 512 → 2048 when a completion is
  empty. Both final replays deploy MiniMax with a 262,144-token judge context window and constrain
  the output to one of the three labels through the pinned vLLM request option. The raw completion
  and parsed label are retained for audit. The judge window deliberately exceeds the 73,664-token
  generation window because responses can expand when re-tokenized by MiniMax.
- **BixBench verifier:** the verifier calls the co-hosted MiniMax Chat Completions endpoint
  with `max_tokens=1024` and a structured response schema containing one boolean score.
  Sampling follows the endpoint's service defaults.
- **SOTOPIA evaluator:** the pinned SOTOPIA v0.1.5 evaluator uses the fixed judge model after
  each episode; the model under evaluation never acts as its own judge. The verifier preserves
  all seven scores, the normalized overall reward, and the evaluated agent's emitted character
  count. `goal` may be selected explicitly for diagnostic runs, but is not the release default.
- **τ³ natural-language assertions:** the pinned τ³ adapter passes only assertions that need
  semantic evaluation to its configured LLM; deterministic assertions remain programmatic.

When a judge call fails after retries, the sample is recorded as grader infrastructure
failure (counted against the run's error tolerance) rather than silently scored zero.

### Thinking mode

The main table records the policy for every benchmark. `on` and `off` set
`enable_thinking` through the model or benchmark config; a benchmark override wins over the model
default. `model default` means the benchmark adds no override, and `not applicable` means the task
does not use a generated chat response. If a model cannot turn reasoning off (e.g. GPT-OSS), an `off` benchmark
uses the model's lowest available reasoning setting instead.

### Errors and Exceptions

#### Infrastructure vs Model Error

We categorize errors during the evaluation process as either infrastructure errors or model errors.
Model errors are scoreable and do not count toward the infrastructure-error threshold. Unless this
policy explicitly identifies an error as a model error, it is an infrastructure error.

#### Infrastructure Error Tolerance

An evaluation may be sealed as soon as at least 90% of its declared trials have valid scores under
the campaign configuration. Up to 10% of declared trials may be absent because of infrastructure
failures; the result does not need to wait for those trials to be recovered. Reported uncertainty
must include this infrastructure-failure tolerance. Task-level outcomes, including agent timeouts
recorded as scores of zero, are scored trials rather than infrastructure failures.

#### Zero-score trace audit

A score of exactly zero shall be reported only after agent review of at least ten representative
traces, or every trace when the evaluation has fewer than ten. The review will check model output,
harness parsing, verifier output, and serving or sandbox errors and accept the zero when the traces support model or task failure and no untracked infrastructure failure explains the result. Otherwise, the result should be rejected and re-run under the policy.

#### Harbor

For Harbor evals, the following error types are currently treated as score-able model errors.

- `AgentTimeoutError`: the agent reached its configured execution timeout; Harbor preserves
  the work produced so far and runs the verifier.
- `ContextLengthExceededError`: the model or agent exhausted the available context; Harbor
  treats that as a classified end to the attempt and runs the verifier.

For these Harbor score-able model errors, we terminate the trial and then run the verifier on the output of the agent, if any output is present. Unlike the Evalchemy timeout rule above, we do *not* automatically score Harbor model errors as 0's. This is because sometimes, agents will solve the task correctly and then get stuck in a loop of self-doubt or fail to submit properly in their harness, and we regard that outcome as closer to a success than a failure.

##### Ambiguous Harbor Errors

- `NonZeroAgentExitCodeError`: the agent process exited non-zero after potentially producing scoreable work. We treat these errors as score-able but if a trial returns more than 30% NonZeroAgentExitCodeError the agent or human executing the policy should review the traces to look for infrastructure errors and invalidate the run if any are discovered.

#### Evalchemy

Under the current Evalchemy pin, `malformed_model_response` is the only named error category that
remains scoreable. It means the model endpoint returned a response that Evalchemy could not parse,
or returned no usable final-answer content. Evalchemy substitutes an empty generation, preserves
the diagnostic category, and lets the benchmark grade that generation normally.

`AgentTimeoutError`, `model_transport`, and `grader_infrastructure` are currently serialized with
the `EVALCHEMY_INFRASTRUCTURE_ERROR` marker and excluded from scored coverage. They therefore count
against the 10% infrastructure-failure allowance rather than as zero-valued model outcomes. Other
task-level failures, including `generation_policy`, `generation`, `grading`, `serialization`,
`incomplete_evaluation`, and `invalid_result`, do not produce a scoreable trial.

This exposes a mismatch with the intended Evalchemy timeout rule below: the release policy says an
`AgentTimeoutError` should score `0.0`, but the current pinned implementation treats it as
infrastructure. Results and coverage from this pin must follow the recorded implementation behavior;
making Evalchemy timeouts scoreable requires a new code pin and provenance.

#### Timeout Policy

The following limits apply to every policy benchmark; there are no benchmark-specific exceptions.

| Evaluator phase | Limit | Outcome when the limit expires |
| --- | ---: | --- |
| Evalchemy trial | 30 minutes | Record `0.0` as an `AgentTimeoutError`. This is a scoreable model outcome. |
| Harbor agent setup | 6 minutes | Record `AgentSetupTimeoutError`. This is a retryable infrastructure error. |
| Harbor agent execution | 30 minutes | Record `AgentTimeoutError`, preserve the agent's work, and run the verifier. This is a scoreable model outcome. |
| Harbor verifier | 5 minutes | Record `VerifierTimeoutError`. This is a retryable infrastructure error. |

The Evalchemy limit covers the full trial, including request retries and backoff, after the trial
acquires an evaluator slot.

Harbor launch configs override task-manifest values with
`agents[].override_setup_timeout_sec: 360`, `agents[].override_timeout_sec: 1800`, and
`verifier.override_timeout_sec: 300`. The repository-owned campaign validator checks these values
before submission, so the same limits apply even when a task manifest omits its own timeout.

## Baseline Models

### How we select baseline models

The process for selecting baseline models in our policy is fundamentally heuristic and community-driven. For each new model release, we will strive to select a representative cohort of open-weights and open-source models that are roughly comparable to Marin models in terms of parameter count and document them in the policy. We will look up reported results for those models in the research literature and, where available, use them as targets for internal reproduction, noting, however, that typically the variance in hyperparameters will make published results not directly comparable to our own. We will sometimes substitute quantized variants of the listed models for the canonical models, especially for larger models where our hardware makes it challenging to run inference on the non-quantized variants.

### Snowball 67B A2B Baseline Models

The current release analysis uses the following fixed 13-model baseline cohort:

* `Aleph-Alpha/Kolibri-1`
* `IFM/K2-Horizon-MoVA-36B-A4B`
* `Qwen/Qwen3-Coder-30B-A3B-Instruct`
* `Qwen/Qwen3-Next-80B-A3B-Instruct`
* `Qwen/Qwen3.6-35B-A3B`
* `arcee-ai/Trinity-Mini`
* `google/gemma-4-26B-A4B-it`
* `inclusionAI/Ling-lite-1.5`
* `moonshotai/Kimi-Linear-48B-A3B-Instruct`
* `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
* `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
* `openai/gpt-oss-20b`
* `zai-org/GLM-4.7-Flash`

## Codebase pins

The single pin authority is the checked-out Marin revision. Its generated
[`external_dependencies.py`](https://github.com/marin-community/marin/blob/75a5690b0ff2e1510d73e576649e3e059f06da9a/lib/marin/src/marin/external_dependencies.py#L65-L81)
contains the immutable Evalchemy and Harbor revisions; the campaign validator also requires the
Harbor lockfile to resolve to that same Harbor commit. Do not copy these pins into this document or
an artifact-side environment file. The release pins Marin commit
[`75a5690b0ff2e1510d73e576649e3e059f06da9a`](https://github.com/marin-community/marin/tree/75a5690b0ff2e1510d73e576649e3e059f06da9a).
Every launch snapshot records that exact Marin commit, which identifies all three codebase revisions.

### Configuration immutability

For convenience, Marin packages a launcher for policy evals. The launcher must not alter model-serving settings or benchmark semantics (except model-specific chat templates and model-specific Harbor context and compaction values). It may resolve credentials, cluster placement, output locations, concurrency, and other operational values.

Each durable run record preserves the effective model configuration, evaluator configuration, code revision, and Harbor policy digest. Any other semantic change requires a new checked-in config and code revision.

## Eval launch commands (Marin)

These commands require access to the `marin` Iris controller and the `cw-rno2a` federated cluster.
Set `MARIN_DIR` to a clean checkout of the pinned campaign revision and `MODEL` to a filename stem under
the repository's campaign `model-configs/` directory. The launcher validates every pin and canonical config, resolves the
required credentials, snapshots the effective command and configs, and submits one shared-serving
root. It deliberately supplies no eval-policy label.

```bash
REPRO_DIR=$(mktemp -d)
git clone https://github.com/marin-community/marin.git "$REPRO_DIR/marin"
git -C "$REPRO_DIR/marin" checkout 75a5690b0ff2e1510d73e576649e3e059f06da9a
export MARIN_DIR="$REPRO_DIR/marin"
CAMPAIGN="$MARIN_DIR/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled"
MODEL=REPLACE_WITH_MODEL_CONFIG_FILENAME_STEM
cd "$CAMPAIGN"
./validate-campaign.sh
./launch-campaign.sh --submit --model "$MODEL"
```

Use `--suite nonagentic`, `--suite agentic`, or `--suite terminal-bench` for a complete suite
subset. Use repeatable `--evalchemy NAME` or `--harbor NAME` selectors for a named recovery. AIME24,
OlympiadBench, and GPQA Diamond perform their repeated sampling inside one launch using the shared
seeded-repeat implementation; AIME24 is not launched ten times. Omitting `--submit` is a dry run.

### Three-benchmark reproduction check

For a release-level smoke reproduction, use model-config stem `Qwen-Qwen3.6-35B-A3B` and these
fixed benchmarks: GSM8K 0-shot (`0.908`), MATH500 (`0.984`), and DS-1000 (`0.270`). Launch each
benchmark separately with the Marin command above:

```bash
MODEL=Qwen-Qwen3.6-35B-A3B
./launch-campaign.sh --submit --wait --limit 50 --suite nonagentic --model "$MODEL" --evalchemy gsm8k-0shot
./launch-campaign.sh --submit --wait --limit 50 --suite nonagentic --model "$MODEL" --evalchemy math500
./launch-campaign.sh --submit --wait --limit 50 --suite agentic --model "$MODEL" --harbor ds-1000-local
```

Each launch prints a run ID such as
`20260929-150059-Qwen-Qwen3.6-35B-A3B-piqa-ef1c`. After the waiting command exits, inspect that
run's durable record with the following command. It prints the scoreable-trial count followed by
the primary score; repeat it with the run ID from each launch.

```bash
RUN_ID=REPLACE_WITH_PRINTED_RUN_ID
RECORDS_PREFIX=s3://marin-us-east-02a/marin/evals
bash -lc '
  source ./common.sh
  resolve_coreweave_credentials
  uv run --project "$MARIN_DIR" fsutil cat --raw \
    "'$RECORDS_PREFIX'/'$RUN_ID'/record.json"
' | python3 -c '
import json, sys
record = json.load(sys.stdin)
status = record["status"]
if status != "completed":
    error = record.get("error")
    raise SystemExit(f"run did not complete: {status}: {error}")
coverage = next(iter(record["coverage"].values()))
metrics = next(iter(record["canonical_metrics"].values()))
score = next(value for name, value in metrics.items() if not name.endswith("_stderr"))
print(coverage["n_scored"], score)
'
```

After at least 50 scoreable trials, compare their mean with the corresponding reported score. For this
provisional smoke check, parity means an absolute difference no greater than `0.146`. This is the
worst-case 95% normal half-width for 50 independent `[0, 1]` trial rewards,
`1.96 × 0.5 / sqrt(50)`, expanded by `1 / sqrt(0.9)` for the policy's infrastructure tolerance.
This gate is intentionally a smoke reproduction, not a replacement for the full-run confidence
intervals or release result.

## Eval launch commands (Standalone)

The standalone path uses the same checked-out Marin campaign directory as the commands above. It
requires an OpenAI-compatible endpoint reachable from both the local evaluator
and Daytona sandboxes and served with the checkpoint, vLLM flags, topology, and sampling settings
in the matching repository model YAML. `MODEL_BASE_URL` is the API root ending in `/v1`; `MODEL_ID`
must match the name returned by `/models`; `TOKENIZER_ID` is the Hugging Face repository and
`TOKENIZER_REVISION` its immutable revision. Evalchemy executes generated code for HumanEvalPlus
and MBPPPlus.

```bash
EVALCHEMY_COMMIT=$(PYTHONPATH="$MARIN_DIR/lib/marin/src" python3 -c \
  'from marin.external_dependencies import EVALCHEMY; print(EVALCHEMY.commit)')
git clone https://github.com/marin-community/evalchemy.git
cd evalchemy
git checkout "$EVALCHEMY_COMMIT"
uv sync --python 3.12 --extra benchmarks --extra serve-eval

export MODEL_BASE_URL=https://example-model-endpoint.invalid/v1
export MODEL_ID="REPLACE_WITH_SERVED_MODEL_NAME"
export MODEL_API_KEY="REPLACE_WITH_MODEL_ENDPOINT_API_KEY"
export TOKENIZER_ID="REPLACE_WITH_HUGGING_FACE_TOKENIZER_ID"
export TOKENIZER_REVISION="REPLACE_WITH_IMMUTABLE_TOKENIZER_REVISION"
export TOGETHER_API_KEY="REPLACE_WITH_TOGETHER_API_KEY"
export JUDGE_API_KEY="$TOGETHER_API_KEY"
export JUDGE_BASE_URL=https://api.together.xyz/v1
export JUDGE_MODEL=openai/gpt-oss-120b
export MODEL_CONTEXT=73664

mkdir -p results/evalchemy

run_chat_eval() {
  task_name=$1
  num_fewshot=$2
  max_tokens=$3
  max_length=$4
  seed=$5
  output_name=$6
  thinking=$7
  shift 7
  extra=(--batch_size 1 --log_samples --apply_chat_template)
  [ "$seed" = default ] || extra+=(--seed "$seed")
  [ "$max_tokens" = default ] || extra+=(--max_tokens "$max_tokens" --gen_kwargs "max_gen_toks=$max_tokens")
  model_args="model=$MODEL_ID,base_url=${MODEL_BASE_URL%/}/chat/completions,api_key=$MODEL_API_KEY,tokenizer_backend=none,tokenized_requests=False,num_concurrent=16,max_retries=8,timeout=1800,max_length=$max_length"
  if [ "$thinking" != default ]; then
    thinking_key=enable_thinking
    thinking_value=$thinking
    if [ "$thinking" = false ]; then
      case "$TOKENIZER_ID" in
        openai/gpt-oss-20b|IFM/K2-Horizon-MoVA-36B-A4B)
          thinking_key=reasoning_effort
          thinking_value='"low"'
          ;;
      esac
    fi
    model_args+=",chat_template_kwargs={\"$thinking_key\":$thinking_value}"
  fi
  uv run eval \
    --model local-chat-completions \
    --model_args "$model_args" \
    --tasks "$task_name" \
    --num_fewshot "$num_fewshot" \
    "${extra[@]}" \
    --output_path "results/evalchemy/$output_name" \
    "$@"
}

run_completion_eval() {
  task_name=$1
  num_fewshot=$2
  max_length=$3
  uv run eval \
    --model local-completions \
    --model_args "model=$MODEL_ID,base_url=${MODEL_BASE_URL%/}/completions,api_key=$MODEL_API_KEY,tokenizer=$TOKENIZER_ID,revision=$TOKENIZER_REVISION,tokenizer_backend=huggingface,trust_remote_code=True,tokenized_requests=False,num_concurrent=16,max_retries=8,timeout=1800,max_length=$max_length" \
    --tasks "$task_name" \
    --num_fewshot "$num_fewshot" \
    --batch_size 1 \
    --seed 42 \
    --log_samples \
    --output_path "results/evalchemy/$task_name"
}

run_chat_eval MATH500 0 default "$MODEL_CONTEXT" 42 math500 true
run_chat_eval AIME24 0 default "$MODEL_CONTEXT" 42 aime24 true
run_chat_eval HumanEvalPlus 0 default "$MODEL_CONTEXT" 42 humanevalplus false --confirm_run_unsafe_code
run_chat_eval MBPPPlus 0 default "$MODEL_CONTEXT" 42 mbppplus false --confirm_run_unsafe_code
run_chat_eval OlympiadBench 0 default "$MODEL_CONTEXT" 42 olympiadbench true
run_chat_eval gsm8k 0 default "$MODEL_CONTEXT" 42 gsm8k-0shot false
run_chat_eval triviaqa 5 default "$MODEL_CONTEXT" 42 triviaqa false
run_chat_eval MMLUPro 0 32768 65536 42 mmlu-pro true
run_chat_eval GPQADiamond 0 default "$MODEL_CONTEXT" 42 gpqa-diamond true
run_chat_eval CruxEval 0 default "$MODEL_CONTEXT" 42 cruxeval false
run_chat_eval FinanceBench 0 default "$MODEL_CONTEXT" 42 financebench false
run_chat_eval IFBench 0 default "$MODEL_CONTEXT" 42 ifbench false
run_chat_eval MRCR 0 4096 "$MODEL_CONTEXT" 42 mrcr false
run_chat_eval NUPA5K-Loose 0 default 73664 20222943 nupa5k default

run_completion_eval piqa 0 "$MODEL_CONTEXT"
run_completion_eval winogrande 5 "$MODEL_CONTEXT"
run_completion_eval boolq 0 "$MODEL_CONTEXT"
run_completion_eval truthfulqa_mc2 0 "$MODEL_CONTEXT"
```

Run the Harbor evaluations from a separate directory so Harbor's environment does not share
Evalchemy's project environment:

```bash
HARBOR_COMMIT=$(PYTHONPATH="$MARIN_DIR/lib/marin/src" python3 -c \
  'from marin.external_dependencies import HARBOR; print(HARBOR.commit)')
git clone https://github.com/marin-community/harbor.git
cd harbor
git checkout "$HARBOR_COMMIT"
uv sync --all-packages

export CAMPAIGN="$MARIN_DIR/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled"
export MODEL_STEM=REPLACE_WITH_MODEL_CONFIG_FILENAME_STEM
export MODEL_BASE_URL=https://example-model-endpoint.invalid/v1
export MODEL_ID="REPLACE_WITH_SERVED_MODEL_NAME"
export MODEL_API_KEY="REPLACE_WITH_MODEL_ENDPOINT_API_KEY"
export TOGETHER_API_KEY="REPLACE_WITH_TOGETHER_API_KEY"
export DAYTONA_API_KEY="REPLACE_WITH_DAYTONA_API_KEY"
export OPENAI_API_KEY="$TOGETHER_API_KEY"
export OPENAI_BASE_URL=https://api.together.xyz/v1
export TAU2_USER_MODEL=openai/openai/gpt-oss-120b
export TAU2_NL_ASSERTIONS_MODEL="$TAU2_USER_MODEL"
export BIXBENCH_JUDGE_BASE_URL=https://example-minimax-judge-endpoint.invalid/v1
export BIXBENCH_JUDGE_MODEL=minimax-m3-mxfp8-judge
export BIXBENCH_JUDGE_API_KEY="REPLACE_WITH_MINIMAX_ENDPOINT_API_KEY"

source "$CAMPAIGN/common.sh"
reset_staging_root
stage_model_config "$MODEL_STEM"
stage_harbor_configs "$MODEL_STEM"

for config in swebench-verified ot-tblite-recovery tb2-recovery ds-1000-local \
              bfclparity-pi; do
  uv run harbor run \
    --config "$STAGED_HARBOR_DIR/$config.yaml" \
    --model "hosted_vllm/$MODEL_ID" \
    --ak "api_base=${MODEL_BASE_URL%/}" \
    --ae "OPENAI_API_KEY=$MODEL_API_KEY" \
    --jobs-dir "results/harbor/$config" --yes
done

# τ³ uses its benchmark-specific served model YAML, not the base model YAML.
# Native-32k models have no τ³ override and use their base YAML instead.
TAU3_MODEL_CONFIG="$CAMPAIGN/model-configs/tau3/$MODEL_STEM.yaml"
if [ ! -f "$TAU3_MODEL_CONFIG" ]; then TAU3_MODEL_CONFIG="$CAMPAIGN/model-configs/$MODEL_STEM.yaml"; fi
stage_model_config "$MODEL_STEM" "$TAU3_MODEL_CONFIG"
# Serve STAGED_MODEL_CONFIG at a separate endpoint, then set its API root here.
export MODEL_BASE_URL=https://replace-with-tau3-endpoint.invalid/v1
uv run harbor run \
  --config "$STAGED_HARBOR_DIR/tau3-pi.yaml" \
  --model "hosted_vllm/$MODEL_ID" \
  --ak "api_base=${MODEL_BASE_URL%/}" \
  --ae "OPENAI_API_KEY=$MODEL_API_KEY" \
  --jobs-dir results/harbor/tau3-pi --yes

# Serve MiniMaxAI/MiniMax-M3-MXFP8 at revision
# c5454eb03678d8710e54a4e0fc681b9f3b4a3dba with the matching repository judge-model YAML.
uv run harbor run \
  --config "$STAGED_HARBOR_DIR/bixbench-pi.yaml" \
  --model "hosted_vllm/$MODEL_ID" \
  --ak "api_base=${MODEL_BASE_URL%/}" \
  --ae "OPENAI_API_KEY=$MODEL_API_KEY" \
  --ve "OPENAI_API_KEY=$BIXBENCH_JUDGE_API_KEY" \
  --ve "OPENAI_BASE_URL=${BIXBENCH_JUDGE_BASE_URL%/}" \
  --ve "MODEL_NAME=$BIXBENCH_JUDGE_MODEL" \
  --jobs-dir results/harbor/bixbench-pi --yes

uv run harbor run \
  --config "$STAGED_HARBOR_DIR/sotopia-hard.yaml" \
  --model "hosted_vllm/$MODEL_ID" \
  --ak "api_base=${MODEL_BASE_URL%/}" \
  --ae "OPENAI_API_KEY=$MODEL_API_KEY" \
  --jobs-dir results/harbor/sotopia-hard --yes
```

## Run resumption

Resume an interrupted Harbor run only when its model, checked-out repository revision, runtime
limit, and dataset state are unchanged. Read the original run ID and effective configuration from
its durable `record.json`. Use the model and Harbor config from the same Marin commit, then add the
original run ID:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model-config "$CAMPAIGN/model-configs/<model-config>.yaml" \
  --harbor-config "$STAGED_HARBOR_DIR/<harbor-config>.yaml" \
  --resume-run-id <original-run-id>
```

The durable `<records-prefix>/<original-run-id>/results/harbor_jobs/` tree must remain accessible.
Harbor preserves completed scored trials and reruns missing attempts and exceptions in its pinned
resume taxonomy. Use a clean replacement when model, policy, pins, limits, records prefix, or
dataset state changed, or when the unscored error is outside that taxonomy. Never delete a trial
directory to force recovery.

## Baseline-specific launch settings

The YAML files under
`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/` are authoritative for checkpoint and tokenizer
revisions, serving context, GPU count, tensor and data parallelism, reasoning parser, tool parser,
sampling overrides, and Pi `thinking_format`. Use the matching file unchanged. Pi uses
`qwen-chat-template` for Qwen3.6 and Qwen3-Next and `chat-template` for the other models.

## Updating codebase pins

Use the newest campaign pin for each codebase across the full release, assuming forward
compatibility with earlier score-producing runs. If a pin change causes an observed result
regression, stop launches, fix or revert the change, and issue a new policy version when the
evaluation behavior intentionally changes. Preserve the original run's exact pins in its immutable
config snapshot even after the release-wide pins advance.

## Standards for presenting Marin evals

* Display score cells in the standard tracker and any tournament-only tracker to three significant digits. Apply the same convention to a tracker formed from their union. The linked durable run records retain unrounded scores. Tournament and figure analyses recover full-precision task aggregates from that durable evidence; tracker values are presentation-only.
* Select the release Snowball checkpoint by the uncertainty-aware paired tournament defined in “How do we choose which model to release?” Release figures include only the selected checkpoint, not the other Snowball candidates. The Snowball candidate family includes Grug 67B A2B checkpoints as well as checkpoints named Snowball. For each benchmark, plot the selected Snowball checkpoint and every non-Snowball baseline with an available score and total-training-FLOP estimate. Performance is on the y-axis and total training FLOPs are on a logarithmic x-axis. A baseline with an incomplete suite may appear in the scatterplots for its completed benchmarks, but it is excluded from cross-benchmark critical-difference analysis unless it has the full common benchmark panel.
* Produce uncertainty-aware normalized critical-difference diagrams with and without a FLOP control. Normalize scores within each benchmark across the complete-panel cohort. Estimate each score's task-level sample SEM from the scored trial rewards, then multiply the SEM by `1 / sqrt(0.9)` to account conservatively for the policy's 10% infrastructure-failure tolerance. With a fixed random seed, draw 100,000 score panels from independent normal sampling distributions centered on the reported scores and clipped to the metric's `[0, 1]` range. Rank models within every benchmark for each draw, using average ranks for ties, and report mean ranks with simultaneous 95% bands based on the draw-wise maximum absolute rank deviation. Report the Nemenyi critical difference at `alpha = 0.05`; it depends on the number of models and benchmarks, while score uncertainty affects the simulated rank distribution and its simultaneous bands.
* For the FLOP-controlled diagram, regress the draw's benchmark-normalized scores on `log10(total training FLOPs)` with one common linear slope, rank the within-benchmark residuals, and otherwise use the same uncertainty procedure. Treat FLOP estimates as fixed point estimates and disclose that their estimation uncertainty is not propagated. The uncontrolled diagram ranks the sampled normalized scores directly.
* We use a 6ND proxy for pretraining FLOPS, and 6 * max_tokens_per_rollout / max_context_length * n_rollouts * active_params proxy for RL FLOPS; that said, we are the only lab, as of this writing, which releases RL FLOPS, so meaningful comparisons are impossible. From the few labs who release midtraining / SFT information, we derive an unweighted mean percentage of FLOPS devoted to midtraining. We then estimate midtraining FLOPS for other labs using that mean. We estimate total FLOPS as the sum of pretraining FLOPS + (all midtraining / non-RL) FLOPS + (all RL and OPD) FLOPS. For any model which doesn't give sufficient evidence for any of the three stages, use the mean FLOPS over all models in the cohort as the estimated  FLOPS for that stage.
* In a separate standalone table, include all raw scores in XX.y format, with 95% confidence intervals derived from the trial count and the infrastructure error tolerance -- do not leave gaps, use table captions for results that require further context, use the exact same model family in the table as was present in the x-y scatterplot(s)
* Eval results are reportable IFF the eval in question completed with no more than 10% of trials encountering infrastructure errors
* Eval results are reportable IFF they include provenance, in the form of model traces, and where available, verifier traces
* Any eval result that is publicly presented should clearly distinguish whether the benchmarks are in-distribution (part of the dev set), held out, or a mix of the two
* Any eval result that is publicly presented in a major release should report on *all* ID and OOD benchmarks, avoiding cherry picking, substitution, or new evals outside of a formal policy update
* For active development and internal communication, it is fine to define working subsets of the ID benchmarks and report on those in isolation

### Reproduce SOTOPIA-hard length control

SOTOPIA-Hard length control was introduced post-hoc, after the agent traces were generated. For this generation of eval policy, it needs to be run separately. The details on how to do so live here.

The Harbor verifier writes the unadjusted normalized overall reward and
`sotopia_response_characters` for every completed episode. The release score is a cohort
statistic, so it must be regenerated across all compared models rather than copied from one
Harbor run's aggregate. From the pinned Marin checkout, with CoreWeave S3 read credentials and
the campaign tracker available locally, run:

```bash
TRACKER=/absolute/path/to/TRACKER.md
OUTPUT_DIR=/absolute/path/to/artifacts/sotopia-length-controlled-overall
uv run python experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/sotopia_length_control/rank.py \
  --tracker "$TRACKER" --output-dir "$OUTPUT_DIR"
```

The tool reads the seven dimension scores and transcripts from each sealed SOTOPIA-hard archive,
measures only the evaluated agent's emitted characters, and groups cross-validation folds by task.
It writes both raw and length-controlled rankings, plus the tracker SHA-256, every source archive,
the selected regularization strength, and all cross-validation losses. Its counterfactual sets the
response-length difference to zero relative to each task's cohort median. Adding or removing a
model changes that median and requires recomputing the complete cohort. A policy-valid
infrastructure failure retains its configured zero; unscored infrastructure failures are excluded
under the campaign's sealing rule.

The tracker-linked October 15 cohort output is preserved at
`s3://marin-us-east-02a/marin/evals/ablations/sotopia-overall-length-controlled-20261015-c5ef5789/`.

### Reproduce the Snowball tournament and critical-difference figures

Use the campaign tracker as a point-in-time input: every numeric cell must contain its durable
S3 result link, either a native `/results` archive or a corrected per-trial replay JSON. The
tournament requires complete observed scores for every Snowball or Grug candidate on its
23-benchmark selection panel; it excludes FinanceBench, PIQA, and TriviaQA. It derives each cell's
full-precision task aggregate and task-level SEM from the durable trial evidence, then expands the
SEM by `1 / sqrt(0.9)`. The three-decimal tracker score is presentation-only. A benchmark is a
win only when `abs(score_a - score_b) > 1.96 * hypot(adjusted_sem_a, adjusted_sem_b)`; otherwise it
is a tie. A win is one vote, a tie half a vote, and each candidate's selection statistic is the
unweighted mean of its paired win rates against the other candidates. Preserve the generated
tracker and recovered-metrics SHA-256 values, and do not combine a tournament selection with
figures generated from a different tracker snapshot.

The analysis tools are in the Marin campaign worktree under
`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/`, in the
`snowball_tournament/` and `critical_difference/` subdirectories. From the pinned Marin checkout,
with the campaign tracker and output directory available locally, run:

```bash
TRACKER=/absolute/path/to/TRACKER.md
OUTPUT_DIR=/absolute/path/to/artifacts
uv run python experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/snowball_tournament/pairwise.py \
  --tracker "$TRACKER" --recovered-metrics "$OUTPUT_DIR/recovered-record-metrics.json" \
  --output-dir "$OUTPUT_DIR/snowball-tournament"
uv run python experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/critical_difference/plot.py \
  --tracker "$TRACKER" --selection "$OUTPUT_DIR/snowball-tournament/snowball_selection.json" \
  --recovered-metrics "$OUTPUT_DIR/recovered-record-metrics.json" \
  --output-dir "$OUTPUT_DIR/critical-difference"
```

Both commands need read access to the scored results in CoreWeave S3; configure the normal
Marin `fsutil` credentials first. It reads canonical `record.json` files for trial counts and
normalized sample rewards for non-binary metrics. If a NUPA record has not been written, the
tool requires its sealed native Evalchemy archive and verifies its coverage and canonical score.
For cells sealed from an interrupted executor,
`recovered-record-metrics.json` supplies the independently aggregated scored-trial count and
reward standard error; retain that audit file with the tracker. The complete-panel cohort is the selected
Snowball plus the non-Snowball models listed in `critical_difference/baseline_flops.csv`; a
benchmark enters the diagrams only if every cohort model has a numeric, linked score. The output
includes every benchmark-level pair outcome, the full paired-win matrix and selection summary,
per-cell sample statistics, FLOP estimates, rank tables, and uncontrolled and FLOP-controlled PNG,
PDF, and SVG figures. Regenerate
both commands after any tracker change. The uncertainty, normalization, FLOP-control, and
critical-difference calculations are defined above; the random seed and draw count are fixed in
the plotting code.

## Future considerations

* We would like to post-train multiple Snowball checkpoints, experimenting with different degrees of compute applied to post-training, to establish trade-off lines / domains
* We would like to think about a "marin model card" that would set minimum standards for true open development
* amd/Instella-MoE-16B-A3B-Think is not runnable currently from our pinned vLLM fork, but we would like to look at this model in the future
* We still weren't able to land a trusted K2 Horizon baseline set of scores yet
* evaluate our baselines on OOD as well as ID
* JudgeArena multilingual, maybe one of those cultural idiom / include benchmarks, restore simpleQA-mini, a long context supplement for MRCR
* Produce a summary of why we think the previously flagged TB2 / TB4 task description overlap are likely not problematic, if they are not
* Find a better instruction following benchmark than IFBench
* Consider a smaller subset of MMLU-Pro and NUPA so they run faster
* Fix the inconsistency in Evalchemy timeout error handling
* In the future, we may deliberately use subsets of very large benchmarks like BFCL and NUPA as training material; in this case, we must mark the full benchmarks as explicitly contaminated, adding a new section to our eval policy
* Properly unify on the co-hosted MiniMax judge
* Consider using a judge panel and requiring all judges to agree on equivalence
* Consider using cruxeval input as well as output
* Drop PIQA as non-discriminating, TriviaQA needs its grader reconsidered
* biomedical knowledge: MedXpertQA or PubMedQA, Clinical reasoning: HealthBench Professional/ Bedside Bench /ASCENT, Biological discovery: Longevity — genetics regression; LAB-Bench cloning / DNA methylation age comparison, Medical coding: MedConceptsQA — ATC Hard/ MedConceptsQA ICD-10 procedures, hard

