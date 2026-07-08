# context_efficiency

Tools that measure how much agent token budget a shared wiki/memory, RAG code
search, or better docs would save, by analyzing our own Claude Code session
transcripts (`~/.claude/projects/*/*.jsonl`).

Read [`REPORT.md`](REPORT.md) for the findings. Headline: memory/wiki/RAG/docs
save ~0.2–0.5% of the measurable token budget; the addressable ceiling is 7.8%;
the real levers are trimming the always-on prelude (14.8% of budget) and managing
long sessions.

## Pipeline

```bash
uv run parse_sessions.py     # transcripts -> _data/{blocks,turns}.parquet (normalized)
uv run token_accounting.py   # denominators, cost-model validation, prelude residual, calibration
uv run redundancy.py         # cross-session read + exploration pools, content-stability tiers
uv run uplift.py             # per-intervention net uplift, heaviest-session ablation, sensitivity
```

Each analyzer writes a JSON to `_data/` (gitignored). Standalone `uv` scripts;
no pytest per repo convention — validated against the real transcript corpus.

## Documents

- [`REPORT.md`](REPORT.md) — the writeup and recommendations.
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — milestone log (M0–M5), including the
  probes that caught classification bugs and the cost-model correction.
- [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md) — cited literature survey.
- [`PLAN.md`](PLAN.md) — the analysis plan (codex-reviewed before build).
- [`CODEX_REVIEW.md`](CODEX_REVIEW.md) — the peer review and adopted fixes.

## Cost model

Savings are in base-price input-equivalents under Anthropic prompt-cache pricing
(write 1.25×, read 0.10×, input 1.0×, output 5.0×). A saved content chunk of `C`
tokens in a session with observed amplifier `A` saves `C·(1.25 + 0.10·min(remaining, A))`.
The amplifier is measured per session, not assumed (the full-retention model
overcounts re-reads ~12×; see `token_accounting.py`).
