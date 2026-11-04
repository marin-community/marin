---
license: apache-2.0
pretty_name: Marin Thoughtcrime
language:
- en
task_categories:
- text-generation
tags:
- reasoning
- tool-use
- evaluation
- agents
size_categories:
- n<1K
configs:
- config_name: default
  data_files: rows.parquet
---

# Marin Thoughtcrime

Thoughtcrime is an evaluation set of 328 agent conversations. In each one, a Marin model's next reply came out with malformed reasoning markup. A *double-think* reply either closes its reasoning span a second time (a stray `<|end_think|>`) or opens a second span. The model under test is given the conversation and the tools, and samples one assistant turn. That turn scores 1 when it has at most one closed reasoning span and no stray closer.

The prompts come from [pi](https://github.com/badlogic/pi-mono) agent trajectories that Marin models produced on two benchmarks: [BixBench](https://github.com/Future-House/BixBench) and the [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla), both run through [Harbor](https://github.com/laude-institute/harbor). The rows are built and scored by the code in [marin-community/marin#10907](https://github.com/marin-community/marin/pull/10907).

## Selection

- A trajectory contributes only its first malformed reply, and only when that reply is a double-think, so no prompt contains an earlier markup failure.
- Each benchmark task contributes one row: the qualifying trajectory with the lowest `digest`.
- Prompts are capped at 12,288 tokens.
- The source runs are every Marin-model bixbench-pi, bfclparity-pi and tau3-pi run before 2026-11-04. That's 616 runs and 35,260 trajectories, of which 7,755 first fail with a double-think. tau3 contributes no rows because its runs archive no pi transcript. Transcripts with structured `thinking` content are skipped.

| | Rows |
|---|---|
| BixBench (every task) | 205 |
| BFCL parity subset | 123 |
| Recorded stray close | 212 |
| Recorded second span | 116 |
| Source models | 43 |

`report.json` holds the mining counts and a token-length audit.

## Fields

| Field | Description |
|---|---|
| `digest` | Stable row id |
| `messages` | JSON list of chat messages before the recorded reply |
| `tools` | JSON list of the tool schemas pi advertised |
| `recorded_violation` | `stray_end` or `second_span` |
| `prompt_tokens` | Prompt length reported by the serving provider |
| `benchmark`, `task` | Source benchmark and task name |
| `model`, `run`, `trial` | Model and eval run that produced the trajectory |
| `position` | Index of the recorded reply in the transcript's events |
| `reply` | The malformed reply as recorded |

## Baseline

`open-athena/Marin-Snowball-67B-A2B-v0.1`, with 4 samples per prompt, 8,192 max tokens, thinking enabled and special tokens kept, scores 0.796 overall: 0.807 on BixBench and 0.776 on BFCL. Of its 1,312 replies, 268 are malformed: 140 stray closes, 99 second spans and 29 unclosed spans.

## Licence and attribution

The dataset is released under Apache-2.0. It contains material from:

- **BixBench** (Future-House, Apache-2.0): questions, plus excerpts of the capsule data that agents printed in tool outputs. The bixbench prompts use the corrected task files in `penfever/harbor-datasets`.
- **Berkeley Function Calling Leaderboard** (Gorilla, Apache-2.0): prompts and tool schemas, using the task files in `laude-institute/harbor-datasets`.
- **pi** (Mario Zechner, MIT): the agent's system prompt and tool definitions.
- **Harbor** (Laude Institute, Apache-2.0): the task adapters.

Model replies come from Marin, Grug and Snowball checkpoints.
