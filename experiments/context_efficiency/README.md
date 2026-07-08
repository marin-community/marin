# context_efficiency

Measure how much agent **token budget** a shared wiki/memory, semantic code index
(RAG), better docs, better tools, or result compaction would save — by analyzing your
own Claude Code session transcripts (`~/.claude/projects/*/*.jsonl`).

Read [`REPORT.md`](REPORT.md) for the findings. Headline over 3,251 of one engineer's
sessions: tool output touches ≥59% of the token budget, but half of it is irreducible
(git/PR inspection, test runs, reading the agent's own scratch output). The realizable
saving from all "supermemory" options together is ~6–9% of the budget (Sonnet-calibrated),
dominated by an automatic semantic code index plus better tool defaults plus result
compaction. A per-fact wiki nets ~0.3%; subsystem architecture docs are worth more but
only at subsystem granularity. Any persistent memory must be retrieval-gated — the
always-on prelude is 16.9% of the budget, re-read every turn.

## Run it on your own sessions

The analysis is a single [`ArtifactStep`](../../lib/marin/src/marin/execution/lazy.py)
pipeline. Each stage reads the previous stage's structured output; outputs are
content-addressed, so a re-run reuses finished stages and already-labeled batches.

```sh
MARIN_PREFIX=~/scratch/ce python -m experiments.context_efficiency.pipeline \
    --agent-command 'claude -p --model haiku' \
    --val-agent-command 'claude -p --model sonnet' \
    --agents-md AGENTS.md \
    --memory-md ~/.claude/projects/<your-project>/memory/MEMORY.md
```

Outputs land under `$MARIN_PREFIX/context_efficiency/*` (or `/tmp/marin` if unset). The
final artifact is `analysis/dev/semantic_analysis.json` — the full analysis behind the
report. Useful flags: `--glob '-home-you-code-repo*'` to scope to one repo's sessions,
`--n` for the target labeled-episode count, `--limit` to cap files for a quick trial,
`--val-fraction 0` to skip the stronger-model calibration pass.

The labeling and clustering stages shell out to a headless agent. `claude -p` is the
default; any headless CLI that reads a prompt on stdin and prints a reply works
(`--agent-command 'codex exec'`). Nothing else needs the network or a cluster.

## Stages

| Module | Reads | Writes |
|---|---|---|
| `transcripts.py` | `~/.claude/*.jsonl` | `blocks.parquet`, `turns.parquet` |
| `accounting.py` | blocks/turns | `token_accounting.json`, `session_amplifier.parquet`, `budget_decomposition.json` |
| `episodes.py` | blocks + amplifier | PPS-weighted `episodes_sampled.parquet` + `episode_batches/` |
| `labeling.py` | batches → `claude -p` | per-episode semantic labels; `topic_clusters.json` |
| `analysis.py` | labels + budget anchor | `semantic_analysis.json` (final) |
| `pipeline.py` | — | wires the DAG; the entry point |
| `schema.py` | — | the label vocabulary shared by the prompt and the analysis |

`accounting.py` runs as two steps (token accounting and budget decomposition);
`labeling.py` as three (bulk label, stronger-model validation subset, topic clustering).

## Documents

- [`REPORT.md`](REPORT.md) — the writeup and recommendations.
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — milestone log, including the cost-model
  correction, the redo after the ground-truth accounting fix, and the pipeline packaging.
- [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md) — cited literature survey.
- [`PLAN.md`](PLAN.md) — the analysis plan (codex-reviewed before build).
- [`CODEX_REVIEW.md`](CODEX_REVIEW.md) — the peer review and adopted fixes.

## Cost model

Savings are in base-price input-equivalents under Anthropic prompt-cache pricing (write
1.25×, read 0.10×, input 1.0×, output 5.0×). The budget is
`1.25·cache_creation + 0.10·cache_read + 1.0·input`, taken from usage records; the chars/4
content proxy undercounts real billed tokens ~8.6× (truncated tool results, stripped
thinking, image tokens), so every headline number is anchored on usage, not the proxy. A
saved chunk of `C` tokens in a session with observed amplifier `A = cache_read/cache_creation`
saves `C·(1.25 + 0.10·min(remaining, A))`. The amplifier is measured per session (aggregate
34×, median 2.15×).
