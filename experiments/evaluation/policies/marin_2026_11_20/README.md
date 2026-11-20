# Eval policy `marin-2026-11-20`

The closest approximation of the October 2026 eval policy ([#10197](https://github.com/marin-community/marin/issues/10197))
that `main` can run. October ran from Marin commit `75a5690b` (2026-10-16) through the
[unlabeled campaign](../../campaigns/eval-campaign-09-25-unlabeled/README.md): single-turn
benchmarks on vendored Evalchemy, agentic ones on an external Harbor fork. This policy runs
single-turn benchmarks as TaskSpec entries of the [benchmark catalog](../../catalog.py) and
agentic ones on vendored Harbor. [`policy.py`](policy.py) declares it; print it with

```bash
uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-20
```

## Candidates

| Candidate | Revision | Served window | Native window |
| --- | --- | --- | --- |
| Muse Glimmer 30B | `cec94fc` | 73,728 | 131,072 |
| Nemotron 3.5 Super VL 120B-A12B | `fb0ee0b` | 73,728 | 1,048,576 |
| Qwen3.6-35B-A3B | `995ad96` | 73,728 | 262,144 |

Qwen3.6 is served as October served it: 4×H100 (TP2×DP2), 320 GB host memory, and the `qwen3`
reasoning and `qwen3_coder` tool parsers. Snowball v0.1 (Smartie RL `.32` step 14) joins once it has
a catalog entry; it will be served with the `grug` reasoning parser, which October did not use.

## Differences from October

Not run, because they do not run on `main`:

| Benchmark | Reason |
| --- | --- |
| τ³-bench, SOTOPIA-hard | [#11080](https://github.com/marin-community/marin/pull/11080) removed their Harbor agents (ACP relay, SOTOPIA agent). |
| BixBench, BFCL (Pi) | Harbor requires Pi tasks to declare a prepared image with Pi installed (`environment/shellbox-images.json`, [#10628](https://github.com/marin-community/marin/pull/10628)); the registry tasks install Pi at runtime. |

Run, with a known difference:

| Benchmark | October | This policy |
| --- | --- | --- |
| CruxEval | no output cap | protocol cap 2,048 tokens |
| FinanceBench | no output cap | protocol cap 4,096 tokens |
| NUPA | no output cap | protocol cap 256 tokens |
| OlympiadBench | no output cap | protocol cap 32,768 tokens |
| TB2, OT-TBLite | external Harbor fork | vendored Harbor; retry list adds the sandbox errors classified since October |
| SWE-bench Verified, DS-1000 | registry at `016dd78` | registry at `38742b0` |
| SWE-bench Verified | 32,768-token budget in the campaign file | same; the campaign README says the policy ran 131,072, unconfirmed |
| OT-TBLite | registry version `2.0` | same; not commit-pinned |

The same as October: the 20 single-turn selections (thinking mode, repeats, few-shot count,
MRCR and MMLU-Pro windows), seed 42 on AIME24 and OlympiadBench, and the Together
`openai/gpt-oss-120b` judge for FinanceBench and OlympiadBench. Harbor agent budgets match the
October campaign files.

Not yet compared: prompt text, item counts and dataset revisions against October's Evalchemy
tasks; task changes between the two Harbor registry pins.
