# Agent MoE report playbook

Use this playbook to create or refresh the public
[Agent MoE experiment digest](../../docs/reports/agent-moe-experiments.md).
It is deliberately stored outside `.agents/skills/`; agents should read it
only when the report is in scope.

## Objective

Maintain a durable summary of the `Agent MoE Experiment:` issues attached to
[tracker #4281](https://github.com/marin-community/marin/issues/4281). Each
experiment needs:

- one category and optional subsection;
- an editorial outcome;
- model-FLOPs and wall-clock speedup summaries;
- a one-sentence result with the decisive evidence or caveat;
- the GitHub issue state and `updatedAt` value used for the review.

The checked-in JSON Lines snapshot is the source of truth. The Markdown report
is generated.

## Authoritative files

- Data:
  `docs/reports/data/agent-moe-experiments.jsonl`
- Data schema and refresh commands:
  `docs/reports/data/README.md`
- Renderer and GitHub drift audit:
  `scripts/pm/render_agent_moe_report.py`
- Generated report:
  `docs/reports/agent-moe-experiments.md`
- Behavioral tests:
  `tests/infra/test_agent_moe_report.py`
- Experiment protocol:
  `experiments/grug/moe/agent.md`
- Baseline measurements:
  `experiments/grug/moe/README.md`

## Scope

Include issues that satisfy both conditions:

1. They are sub-issues of #4281.
2. Their title starts with `Agent MoE Experiment:`.

Do not silently add similarly named issues from repository-wide search. Add a
`foundation` record when an out-of-scope issue is necessary to interpret an
included experiment.

## Start with the drift audit

Run:

```bash
uv run scripts/pm/render_agent_moe_report.py --audit-github --json
```

The command returns:

- `new_issues`: add structured records after reviewing their evidence;
- `changed_issues`: re-read and update existing records;
- `removed_issue_numbers`: confirm removal from #4281 before deleting a record.

If `has_drift` is false, do not rewrite summaries only to vary wording.

## Review issue evidence

For every new or changed issue, read:

1. The managed `experiment-tldr` block when present.
2. The latest conclusion or verdict comments.
3. Result tables containing loss, throughput, and speedup.
4. Earlier comments only when the latest conclusion depends on a correction,
   discarded run, changed baseline, or superseded implementation.
5. Linked W&B runs or reports when the issue does not contain enough evidence.

Treat the issue and linked measurements as ground truth. Do not infer success
from open or closed state.

Record the issue's current `state` and `updatedAt` as `source_updated_at` after
the editorial review is complete.

## Outcome vocabulary

Use exactly one of these values:

| Outcome | Use when |
|---|---|
| `Worked` | The experiment met its recorded gate or narrower success criterion. |
| `Promising` | Evidence is positive, but scale coverage or isolation is incomplete. |
| `Mixed` | The result changes by scale or metric, or a strict projection fails. |
| `Did not work` | A completed comparison shows no net benefit. |
| `Not evaluated` | No usable comparison was recorded. |
| `In progress` | An active experiment has measurements but no final verdict. |

State adoption or supersession in the summary. A result can pass its gate and
still be excluded from the recipe because of inference cost, memory, or a later
compound experiment.

## Speedup fields

`model_flops_speedup` and `wall_clock_speedup` are separate evidence fields.

Model-FLOPs speedup is the loss-only equivalent-compute gain:

```python
model_flops_speedup = (
    (baseline_loss - 1.6) / (variant_loss - 1.6)
) ** (1 / 0.0941)
```

Wall-clock speedup includes measured throughput:

```python
wall_clock_speedup = model_flops_speedup * variant_tps / baseline_tps
```

Use the baseline and variant from the same scale. Prefer the issue's published
values when it already reports both metrics.

Formatting rules:

- Use a range when several scales or variants matter.
- Prefix reconstructed or rounded values with `≈`.
- Use `<1x`, `>1x`, or `Mixed` when that is the strongest supported statement.
- Use `N/A (eval)`, `N/A (serving)`, or another short qualifier when the metric
  does not apply.
- Use `—` when the issue lacks enough data. Do not manufacture precision.
- Label modeled systems results as `Modeled`.

## Summary rules

Each `summary` should fit in one table cell and answer:

1. What changed?
2. Did it work?
3. What number or caveat determines the answer?

Lead with the result. Include the relevant scale range when it changes the
interpretation. Keep negative results and projection failures. Remove run
submission history, debugging narration, and implementation inventory unless a
bug invalidated an earlier result.

## Categorization

Reuse existing categories and subsections when possible. Keep related
experiment families adjacent so their progression is visible:

- partial RoPE and PKO;
- attention reuse and residuals;
- QK scaling and normalization;
- expert count and routing;
- Muon and MuonH;
- activations and output-head changes;
- data, evaluation, and serving;
- compound recipes.

Add a new category only when an experiment cannot be understood in an existing
one. Record order in the JSON Lines file is the rendered order.

## Render and validate

After editing the structured data and metadata `snapshot_date`, run:

```bash
uv run scripts/pm/render_agent_moe_report.py
uv run scripts/pm/render_agent_moe_report.py --check
uv run pytest tests/infra/test_agent_moe_report.py
uv run mkdocs build --strict
uv run python infra/check_docs_source_links.py
./infra/pre-commit.py --changed-files --fix
```

Run the GitHub audit again. It should report no drift:

```bash
uv run scripts/pm/render_agent_moe_report.py --audit-github --json
```

Do not edit `docs/reports/agent-moe-experiments.md` directly. The freshness test
will reject a generated page that differs from the structured snapshot.

## Scheduled automation contract

A scheduled agent should:

1. Read this playbook.
2. Run the JSON drift audit.
3. Stop without a commit when `has_drift` is false.
4. Review only new, changed, or removed issues.
5. Update the structured snapshot and page-level TL;DR when conclusions change.
6. Render and validate the report.
7. Open or update a PR with the evidence-changing issues linked in the body.

GitHub `updatedAt` is a review trigger, not proof that the scientific
conclusion changed.
