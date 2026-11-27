# Eval policy `marin-2026-11-30`

The November 2026 post-training eval policy, proposed in
[#12156](https://github.com/marin-community/marin/issues/12156). It replaces the October policy
([#10197](https://github.com/marin-community/marin/issues/10197)) with Mochi's comparison models and a benchmark set
weighted toward factuality and math and science reasoning. [`policy.py`](policy.py) declares it; print it with

```bash
uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-30
```

## Baselines

Every baseline is served at 147,456 tokens, the window SWE-bench Pro's protocol needs (131,072 tokens of agent input
plus 16,384-token replies), except Muse Glimmer 30B and Granite 4.2 30B. Their checkpoints declare 131,072, so they
are served at that.

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

Models whose templates cannot turn reasoning off run thinking-off benchmarks at their lowest reasoning setting
(`show`'s `think` column). Trinity Large Thinking always reasons.

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
| Agentic | Terminal-Bench 2.1, OT-TBLite, DS-1000, BFCL (Pi), BixBench (Pi), SWE-bench Multilingual (100), SWE-bench Pro (stratified 200 and HARD-51), SWE-bench Science |

Judged evaluations use Together's `openai/gpt-oss-120b`, except IMO-ProofBench and PHYSICS-Verified, which use
Together's `zai-org/GLM-5.3-Flash`. GLM-5.3-Flash is also a baseline, so its scores on those two are graded by
itself.

## Differences from October

Removed: SWE-bench Verified and FinanceBench, whose test sets appear in pretraining data. τ³-bench and SOTOPIA-hard
are not run because their Harbor agents were removed from `main`.

Run, with a known difference:

| Benchmark | October | This policy |
| --- | --- | --- |
| Every baseline | served at 73,728 tokens | served at 147,456 (Glimmer and Granite at 131,072) |
| Terminal-Bench 2.1, OT-TBLite, BixBench | every agent 1,800 s and verifier 300 s | each package's own limits: up to 12,000 s on Terminal-Bench 2.1, 600–3,600 s on OT-TBLite, 3,600 s on BixBench |
| BixBench | MiniMax-M3 judge served beside the candidate | Together `openai/gpt-oss-120b` |
| GPQA Diamond, MRCR, IFBench, OlympiadBench and the other chat benchmarks of [#12221](https://github.com/marin-community/marin/pull/12221) | reasoning and answer graded together | final answer graded |
| SWE-bench Multilingual | agent with network access | agent offline, graded on a separate machine with network |

DS-1000 and BFCL keep the October campaign's 1,800 s agent limit: their packages declare 900 s and 300 s.
