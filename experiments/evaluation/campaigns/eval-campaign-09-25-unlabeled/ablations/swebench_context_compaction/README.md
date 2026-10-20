# SWE-bench context-compaction ablation

This 4 × 4 grid evaluates SWE-bench Verified with mini-swe-agent 2.1.0. It varies total context,
per-response output capacity, and agent-history summarization while holding every other model and
benchmark setting fixed.

| Arm | Total context | Output cap | Summarization |
|---|---:|---:|---|
| `65k-16k-no-summary` | 65,536 | 16,384 | off |
| `131k-16k-no-summary` | 131,072 | 16,384 | off |
| `32k-8k-summary` | 32,768 | 8,192 | 8,192 accumulated history characters |
| `65k-16k-summary` | 65,536 | 16,384 | 8,192 accumulated history characters |

The summary threshold counts the provider-facing serialized history after the original system and
task messages. Compaction preserves the newest complete turns that fit half of the threshold and
uses the other half for a rolling summary. The saved trajectory retains the full unabridged message
history and records compaction telemetry under `info.context_compaction`.

`generate_configs.py` deterministically materializes all exact model-serving and Harbor configs.
The generated files are checked in and are the launch inputs; `launch.py` does not alter serving or
benchmark semantics.
