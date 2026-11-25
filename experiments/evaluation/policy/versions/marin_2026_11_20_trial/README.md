# Eval policy `marin-2026-11-20-trial`

A trial policy for exercising the policy launcher, not a released policy. It is the closest approximation of the October 2026 eval policy ([#10197](https://github.com/marin-community/marin/issues/10197))
that `main` can run. October ran from Marin commit `75a5690b` (2026-10-16) through the
[unlabeled campaign](../../../campaigns/eval-campaign-09-25-unlabeled/README.md): single-turn
benchmarks on vendored Evalchemy, agentic ones on an external Harbor fork. This policy runs
both as TaskSpec entries of the [benchmark catalog](../../../catalog.py); the agentic ones are the
campaign's curated protocols in [`benchmarks/agentic.py`](../../../benchmarks/agentic.py), which pin
the same Harbor packages, agents and limits. [`policy.py`](policy.py) declares it; print it with

```bash
uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-20-trial
```

## Running

```bash
uv run python -m experiments.evaluation.policy.cli launch --policy marin-2026-11-20-trial \
  --candidate qwen3.6-35b-a3b --priority batch [--evaluations a,b] [--limit 20] [--dry-run]
uv run python -m experiments.evaluation.policy.cli status --policy marin-2026-11-20-trial
uv run python -m experiments.evaluation.policy.cli verify --policy marin-2026-11-20-trial RUN_ID...
```

`launch` and `status` take the models to run in one of three ways:

| Option | Runs |
| --- | --- |
| `--candidate NAME`, repeatable | the named candidates |
| `--all-candidates` | every candidate (the default for `status`) |
| `--model KEY` or `--model-config FILE` | a model outside the policy, such as a training checkpoint, on every evaluation; its window is its `serve.max_model_len`, else its checkpoint's |

- `launch` resolves every model's launch before submitting any job. It leaves out evaluations whose
  record already succeeded, and starts no server for a group with none left; the dry run lists them.
  Failed runs rerun at their old paths. A run that succeeded but is invalid (for example, another
  server window) is not rerun; `status` says so for each such run.
- `status` shows each evaluation's full and smoke run: valid, invalid, not run, or differing from
  this checkout (run at another commit with other inputs). A ⚠ marks a run
  whose replies often hit their output cap or whose attempts often went unscored.
- `verify` checks any run against the policy and names each field that differs. A run of a model
  outside the policy is checked against the model its record names.

A policy marked `released` launches only from a clean checkout of its tag, `eval-policy-<version>`,
and `status` and `verify` mark its runs from any other commit invalid. A fix to a released policy is
a new version, so its runs get new paths. This trial policy is not released and launches from any
commit.

All three need CoreWeave credentials.

## Candidates

| Candidate | Revision | Served window | Native window |
| --- | --- | --- | --- |
| Muse Glimmer 30B | `cec94fc` | 73,728 | 131,072 |
| Nemotron 3.5 Super VL 120B-A12B | `fb0ee0b` | 73,728 | 1,048,576 |
| Qwen3.6-35B-A3B | `995ad96` | 73,728 | 262,144 |

Qwen3.6 is served as October served it: 4×H100 (TP2×DP2), 320 GB host memory, and the `qwen3`
reasoning and `qwen3_coder` tool parsers.

## Evaluations

Every candidate runs every evaluation. `show` prints each one's context, seed, protocol, trials,
output cap, judge and metric.

| Kind | Evaluations |
| --- | --- |
| Knowledge and reasoning | GPQA Diamond, MMLU-Pro, TriviaQA, TruthfulQA, BoolQ, PIQA, WinoGrande |
| Math | AIME24, MATH-500, OlympiadBench, GSM8K (0-shot), NUPA |
| Code | HumanEval+, MBPP+, CruxEval |
| Long context | MRCR (default, 32k and 65k) |
| Instruction following and finance | IFBench, FinanceBench |
| Agentic | Terminal-Bench 2.1, OT-TBLite, SWE-bench Verified (mini-SWE-agent), DS-1000, BFCL (Pi) |

## Differences from October

Not run:

| Benchmark | Reason |
| --- | --- |
| τ³-bench, SOTOPIA-hard | [#11080](https://github.com/marin-community/marin/pull/11080) removed their Harbor agents (ACP relay, SOTOPIA agent). |
| BixBench (Pi) | Its package grader asks a MiniMax-M3 judge served beside the candidate, which this policy does not declare yet. |

Run, with a known difference:

| Benchmark | October | This policy |
| --- | --- | --- |
| CruxEval | no output cap; published accuracy on the output-direction items | protocol cap 2,048 tokens; primary metric is accuracy over both directions |
| FinanceBench | no output cap | protocol cap 4,096 tokens |
| NUPA | no output cap | protocol cap 256 tokens |
| OlympiadBench | no output cap | protocol cap 32,768 tokens |
| FinanceBench, OlympiadBench judge | `MiniMaxAI/MiniMax-M3-MXFP8` (`c5454eb`) on 8×H100 with a 262,144-token window, in offline regrades of the sealed replies | Together `openai/gpt-oss-120b`, during the run |
| Agentic benchmarks | external Harbor fork | curated TaskSpec corpora, with prebuilt and Dockerfile base images pinned by digest, run by Marin's rollout engine ([#11664](https://github.com/marin-community/marin/pull/11664)) |
| TB2, OT-TBLite, SWE-bench Verified | Daytona snapshots | declarative Daytona image builds, which create no snapshots |

The same as October: the 20 single-turn selections (thinking mode, repeats, few-shot count,
MRCR and MMLU-Pro windows) and seed 42 on AIME24 and OlympiadBench. Agent input and reply
limits, trials and concurrency match the October campaign files.
