# context_efficiency

Tools that measure how much agent token budget a shared wiki/memory, semantic code
index (RAG), better docs, better tools, or result compaction would save, by
analyzing our own Claude Code session transcripts (`~/.claude/projects/*/*.jsonl`).

Read [`REPORT.md`](REPORT.md) for the findings. Headline over 3,251 sessions: tool
output touches ≥59% of the token budget, but half of it is irreducible
(git/PR inspection, test runs, reading the agent's own scratch output). The
realizable saving from all "supermemory" options together is ~6–9% of the budget
(Sonnet-calibrated), dominated by an automatic semantic code index plus better
tool defaults plus result compaction. A per-fact wiki nets ~0.3%; subsystem
architecture docs are worth more but only at subsystem granularity. Any persistent
memory must be retrieval-gated — the always-on prelude is 16.9% of the budget,
re-read every turn.

## Pipeline

```bash
uv run parse_sessions.py        # transcripts -> _data/{blocks,turns}.parquet (normalized)
uv run token_accounting.py      # denominators, output-fidelity diagnostic, per-session amplifier
uv run budget_decomposition.py  # ground-truth by-class split, prelude decomposition, eviction
uv run sample_episodes.py       # PPS-weighted tool-call episode sample -> _data/episode_batches/
#   -> label the episodes with sub-agents (haiku bulk + sonnet validation) -> _data/labels/
uv run semantic_analysis.py     # labels -> uplift by intervention, realizability, wiki-topic yield
```

The labeling step fans out one sub-agent per 15-episode batch to judge, per
episode, whether a wiki / semantic index / persistent memory / better tool /
result compaction would have served the same need with a smaller answer. Each
analyzer writes a JSON to `_data/` (gitignored). Standalone `uv` scripts; no
pytest per repo convention — validated against the real transcript corpus.

## Documents

- [`REPORT.md`](REPORT.md) — the writeup and recommendations.
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — milestone log, including the probes
  that caught classification bugs, the cost-model correction, and the redo after
  the ground-truth accounting fix.
- [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md) — cited literature survey.
- [`PLAN.md`](PLAN.md) — the analysis plan (codex-reviewed before build).
- [`CODEX_REVIEW.md`](CODEX_REVIEW.md) — the peer review and adopted fixes.

## Cost model

Savings are in base-price input-equivalents under Anthropic prompt-cache pricing
(write 1.25×, read 0.10×, input 1.0×, output 5.0×). The budget is
`1.25·cache_creation + 0.10·cache_read + 1.0·input`, taken from usage records; the
chars/4 content proxy undercounts real billed tokens ~8.6× (truncated tool
results, stripped thinking, image tokens), so every headline number is anchored on
usage, not the proxy. A saved chunk of `C` tokens in a session with observed
amplifier `A = cache_read/cache_creation` saves `C·(1.25 + 0.10·min(remaining, A))`.
The amplifier is measured per session (aggregate 34×, median 2.15×).
