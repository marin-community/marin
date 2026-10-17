---
license: other
pretty_name: GLM 5.3 TMax and SWE-smith k=8 agent traces (2026.10)
configs:
- config_name: attempts
  default: true
  data_files: data/attempts/*/*.parquet
- config_name: relay
  data_files: data/relay/*/*.parquet
tags:
- agent-traces
- terminal-bench
- swe-smith
---

# GLM 5.3 TMax and SWE-smith k=8 agent traces (2026.10)

Agent trajectories from GLM 5.3 on two task families, generated in Harbor with Daytona sandboxes for the Marin SFT shootout ([marin-community/marin#10213](https://github.com/marin-community/marin/issues/10213)).

- **TMax**: terminal tasks run by the Terminus-2 agent.
- **SWE-smith**: repository bug-fix tasks run by mini-swe-agent 2.1.0.

The repo grows as shards finish. `provenance/` holds the settings of every run in it.

## Configs

| Config | What it holds |
|---|---|
| `attempts` (default) | Plain runs: GLM 5.3 drives every turn. 8 attempts per task. |
| `relay` | Relay runs: a student model drives the agent; GLM 5.3 takes over when a trigger fires (below). |

One row is one conversation with the model. A trial usually has one conversation. A relay trial can have more (`n_conversations`).

## Columns

| Column | Meaning |
|---|---|
| `family`, `agent` | `tmax`/`swesmith`; `terminus-2`/`mini-swe-agent`. |
| `run`, `shard`, `round` | Source run. `round` is set for relay rows only. |
| `task_id`, `heldout` | Pool task id. `heldout` marks the pool's held-out tasks (5% per family); keep them out of training. |
| `trial_id`, `trial_name`, `sample`, `retry` | Harbor trial. `retry` is the final retry index after infrastructure retries. |
| `reward`, `passed` | Verifier reward; `passed` is `reward == 1.0`. Null when the trial had no verifier result. |
| `exception` | Harbor exception type, if the trial raised one. |
| `trial_seconds`, `agent_seconds` | Wall time of the trial and of the agent phase. |
| `conversation`, `n_conversations` | Index of this conversation in the trial, and the trial's count. |
| `triggers` | Relay triggers fired in this conversation. |
| `problems` | Rebuild problems. An empty list means the conversation was rebuilt cleanly. |
| `messages` | JSON string: the chat messages, with each assistant turn's reasoning restored. |
| `tools` | JSON string: tool definitions sent with the requests. |
| `tokens` | JSON string: turns and prompt, completion and reasoning tokens per model. |
| `dropped_replies` | JSON string: model replies the agent discarded (for example, unparsable ones). |
| `package_revision` | Task package revision. |

`messages`, `tools`, `tokens` and `dropped_replies` are JSON strings because their shapes differ between agents. Decode them with `json.loads`.

## Generation settings

- **Teacher**: GLM 5.3 (tokenizer `zai-org/GLM-5.3` at `ba32f8c19c8738e2c57c71798c601d477a7bf844`). Clients send only the model and `max_tokens`, so service defaults apply. The chat template renders "Reasoning Effort: Max".
- **Harbor**: pin `016dd78b`. Each family runs under its eval-campaign Harbor policy: `tb2-recovery.yaml` for TMax and `swebench-verified.yaml` for SWE-smith. These set the retries, timeouts, verifier and agent settings.
- **SWE-smith input window**: 30,000 tokens at generation (the policy says 32,768). This keeps every trajectory within 32,768 tokens in the student's chat template.
- **Max output tokens per turn**: 8,192 in plain runs; 16,384 in relay runs.
- **TMax**: the policy's Terminus-2 settings are unchanged.
- **Agent network (SWE-smith)**: during the agent phase, a sandbox can reach only the model endpoint's host. Setup and the verifier follow the task policy. Terminus-2 runs outside the sandbox.

## Task pool (v5)

- TMax: 2,015 tasks, 101 held out. Four are dropped from this dataset: `task_000601_e01cc3c2` and `task_001780_dcd2e021` (broken verifiers), and `task_000441_9c3e397f` and `task_001986_dd88e877` (verifiers that hang).
- SWE-smith: 2,100 tasks, 105 held out. This is pool v4 with 7 broken-verifier tasks replaced by 7 new tasks; the new tasks are in shard `v5`.
- The SWE-smith pool parquet was rewritten in place from v4 to v5. The 2,093 tasks kept from v4 have the same bytes in both versions, so runs started before the rewrite used the same tasks.

## Relay runs

- **Student**: Snowball v0 (`open-athena-Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12`, revision `4eeb756b404db25bc04e4daf49dfb9817bceb741`).
- **Triggers**:
  - `format_error`: the student's reply cannot be parsed. GLM writes one repair turn, then the student continues.
  - `context_budget`: the conversation passes 75% of the input window, counted with the student tokenizer (22,500 tokens for SWE-smith). GLM finishes the trial.
  - `completion_claim`: the student says it is done. Its reply is withheld and GLM finishes the trial.
- **Rounds**: round 1 runs 2 attempts per task. Later rounds run 1 attempt for each non-held-out task that has fewer than 4 passing episodes with GLM turns and fewer than 5 runs.

## Notes

- **Earlier reasoning in TMax**: Terminus-2 does not resend earlier reasoning to the model, because `interleaved_thinking` is off (Harbor `016dd78b`, `src/harbor/agents/terminus_2/terminus_2.py:415`). At generation, GLM saw only its current turn's reasoning. The rows restore the reasoning of every turn.
- **Endpoint URLs**: the model endpoint was a capability URL, one per shard. Agent commands in the SWE-smith sandboxes could read it (from the command line in early runs, from environment variables later). Every published file was scanned for capability tokens by decoding them, and every token found was removed. Each endpoint was unregistered when its shard ended; an unregistered endpoint answers 404.
- **Rebuild problems**: rows with a non-empty `problems` list are kept so that per-trial counts stay complete. Filter on `problems == []` for training.
