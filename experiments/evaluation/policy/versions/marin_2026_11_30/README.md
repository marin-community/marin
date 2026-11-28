# Eval policy `marin-2026-11-30`

The November 2026 post-training eval policy, proposed in
[#12156](https://github.com/marin-community/marin/issues/12156). It replaces the October policy
([#10197](https://github.com/marin-community/marin/issues/10197)) with Mochi's comparison models and a benchmark set
weighted toward factuality and math and science reasoning. [`policy.py`](policy.py) declares it; print it with

```bash
uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-30
```

## Baselines

Every baseline is served at 131,072 tokens: SWE-bench Pro's protocol context, and the largest window that every
baseline's checkpoint declares (Muse Glimmer 30B and Granite 4.2 30B declare exactly that).

| Tier | Model | Catalog key |
| --- | --- | --- |
| 0 | Marin Snowball v0.1 | `marin-snowball-67b-a2b-v0.1` |
| 1 | Muse Glimmer 30B | `muse-glimmer-30b` |
| 1 | Nemotron 3.5 Super VL | `nvidia-nemotron-3.5-super-vl-120b-a12b-bf16` |
| 1 | Qwen 3.8 27B | `qwen3.8-27b` |
| 1 | Gemma 4 31B-it | `gemma-4-31b-it` |
| 1 | Granite 4.2 30B | `granite-4.2-30b` |
| 2 | K2 Horizon 375B-A23B | `k2-horizon-375b-a23b` |
| 2 | Command A+ | `command-a-plus-05-2026-bf16` |
| 2 | Mistral Large 4 | `mistral-large-4-1t-a52b` |
| 2 | Arcee Trinity Large Thinking | `trinity-large-thinking` |
| 3 | Beam 1.0 | `beam-1.0-501b-a23b` |
| 3 | Inkling 2 Small | `inkling-2-small` |
| 3 | GLM-5.3-Flash | `glm-5.3-flash` |
| 3 | DeepSeek V4.1 Flash (1017) | `deepseek-v4.1-flash-1017` |

Solar Open 2 is not a baseline: vLLM supports it only through an unmerged upstream PR or Upstage's own fork.

## Reasoning settings

Each baseline runs thinking-on, agent and Pi evaluations at the reasoning setting its developer reports its own
benchmarks at, which is its template default except where noted. Level names are not comparable across models
("high" is effort 75 of 100 for DeepSeek, 128 of 512 for Beam, 0.9 for Inkling), so the policy does not pin one shared
level.

| Model | Thinking-on, agent and Pi evaluations | Thinking-off evaluations |
| --- | --- | --- |
| Marin Snowball v0.1 | `/think` (default) | `/nothink`, which the model often ignores |
| Muse Glimmer 30B | reasoning strength high (default) | reasoning strength low |
| Nemotron 3.5 Super VL | thinking (default) | thinking off |
| Qwen 3.8 27B | thinking (default) | thinking off |
| Gemma 4 31B-it | thinking on | thinking off |
| Granite 4.2 30B | thinking (default) | thinking off |
| K2 Horizon | reasoning effort high (default) | reasoning effort low |
| Command A+ | reasoning on (default) | reasoning effort none |
| Mistral Large 4 | reasoning effort high (default) | reasoning effort low |
| Trinity Large Thinking | always reasons | always reasons |
| Beam 1.0 | reasoning effort max (default) | reasoning effort minimal |
| Inkling 2 Small | reasoning effort high (default) | reasoning effort none |
| GLM-5.3-Flash | reasoning effort max (default) | reasoning effort low |
| DeepSeek V4.1 Flash | reasoning effort max, set by the catalog entry; the template defaults to high | thinking off |

Trinity Large Thinking samples at its card's temperature 0.6, top_p 0.95 and top_k 50, which its catalog entry sets in
place of the checkpoint's `generation_config.json`.

## Reporting

A report of this policy's results states, for each model, the reasoning setting in the table above. Thinking-off rows
of models that cannot turn reasoning off (Glimmer, K2, Mistral, Trinity, Beam, GLM, and Snowball, which often ignores
`/nothink`) are labelled with the setting they ran at, not as non-thinking. Each score is reported with the mean
output tokens per item, so differences in reasoning compute stay visible. Reports name each agent evaluation's
harness, and note that chat evaluations cap replies at 65,536 tokens, below the output budgets some developers
report with.

## Evaluations

| Kind | Evaluations |
| --- | --- |
| Knowledge and reasoning | GPQA Diamond, MMLU-Pro, SuperGPQA (fixed 2,500-question sample), ARC-AGI-2, WinoGrande, BoolQ, PIQA, TruthfulQA (v1 MC2 and v2 binary) |
| Factuality | SimpleQA Verified (thinking off), FACTS Parametric, FACTS Grounding v2 |
| Math | AIME24, MATH-500, GSM8K (0-shot), NUPA, OlympiadBench, IMO-AnswerBench, IMO-ProofBench |
| Science | ChemBench, GenomeQA, PDBThink, PHYSICS-Verified, FrontierScience (Olympiad and Research) |
| Code | HumanEval+, MBPP+, CRUXEval, LiveCodeBench (release v6, and the subset absent from hero training data), LiveCodeBench Pro |
| Instruction following and professional | IFBench, MultiChallenge, EconEvals (synthetic occupations, sampled, against o3) |
| Long context | MRCR (default, 32k and 65k), LongBench v2 (thinking), GraphWalks |
| Agentic | Terminal-Bench 2.1, OT-TBLite, DS-1000, SWE-bench Multilingual (100) and SWE-bench Pro (stratified 200 and HARD-51), each with its Terminus 2 or mini-swe-agent harness and again with Pi; BFCL (Pi), BixBench (Pi), SWE-bench Science (Pi) |

Pi runs at the context the benchmark's other harness gets, with BixBench's Pi profile for that context (65,536 or
32,768 tokens), 16,384-token replies, and Python 3 and Pi installed over each task's image.

Judged evaluations use Together's `openai/gpt-oss-120b`, except IMO-ProofBench and PHYSICS-Verified, which use
Together's `zai-org/GLM-5.3-Flash`. GLM-5.3-Flash is also a baseline, so its scores on those two are graded by
itself.

## Differences from October

Removed: SWE-bench Verified and FinanceBench, whose test sets appear in pretraining data. τ³-bench and SOTOPIA-hard
are not run because their Harbor agents were removed from `main`.

Run, with a known difference:

| Benchmark | October | This policy |
| --- | --- | --- |
| Every baseline | served at 73,728 tokens | served at 131,072 |
| Chat benchmarks without their own reply cap | replies up to the 73,728-token window | replies up to 65,536 tokens |
| Terminal-Bench 2.1, OT-TBLite, BixBench, DS-1000, BFCL | every agent 1,800 s and verifier 300 s | each package's own limits: up to 12,000 s on Terminal-Bench 2.1, 600–3,600 s on OT-TBLite, 3,600 s on BixBench, 900 s on DS-1000, 300 s on BFCL |
| BixBench | MiniMax-M3 judge served beside the candidate | Together `openai/gpt-oss-120b` |
| GPQA Diamond, MRCR, IFBench, OlympiadBench and the other chat benchmarks of [#12221](https://github.com/marin-community/marin/pull/12221) | reasoning and answer graded together | final answer graded |
| SWE-bench Multilingual | agent with network access | agent offline, graded on a separate machine with network |
