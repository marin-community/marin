# Agent MoE report data

`agent-moe-experiments.jsonl` is the source of truth for the public
[Agent MoE experiment digest](../agent-moe-experiments.md). The Markdown page is
generated and must not be edited directly.

Agents recreating or refreshing the report should first read
`.agents/docs/agent-moe-report.md`.

## Record types

- `metadata` defines the GitHub repository, tracker issue, title prefix,
  snapshot date, schema version, expected issue count, and page-level summary.
- `experiment` records one tracker sub-issue. `outcome` and `summary` are
  editorial fields. `model_flops_speedup` holds the loss-only
  equivalent-compute gain. `wall_clock_speedup` includes the measured
  throughput ratio. A short qualifier replaces a number when the issue does
  not support a precise comparison. `state` and `source_updated_at` identify
  the GitHub version that was reviewed.
- `foundation` records a related issue outside the tracker scope.

Experiment records are ordered by category, section, and preferred reading
order. The renderer preserves that order.

## Refresh workflow

Audit the snapshot against GitHub:

```bash
uv run scripts/pm/render_agent_moe_report.py --audit-github
```

Scheduled jobs can request machine-readable output:

```bash
uv run scripts/pm/render_agent_moe_report.py --audit-github --json
```

The audit reports:

- new matching sub-issues that need categorization and summaries;
- issues removed from the tracker;
- issues whose state or `updatedAt` changed after review.

Review only the reported issues, update their structured records and the
metadata `snapshot_date`, then render and validate:

```bash
uv run scripts/pm/render_agent_moe_report.py
uv run scripts/pm/render_agent_moe_report.py --check
uv run mkdocs build --strict
```

The scheduled automation should use the same sequence. A GitHub change is a
review queue entry, not an automatic scientific conclusion: the automation must
read the new issue evidence before changing `outcome`, `summary`, or the
page-level TL;DR.
