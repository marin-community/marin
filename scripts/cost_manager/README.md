# cost_manager

A daily job that pulls spend from our cost/billing providers and records it to
finelog, so budget burn is visible next to job/throughput stats instead of in
side services on personal VMs. It is the data-collection half of the
"SREbots / cost-management" idea ([#6550], [#6464]): a follow-up agent reads the
`cost.events` table (and can re-run this script) to flag unusual spend.

## What it does

For each enabled provider it fetches a trailing window of daily cost, normalizes
every line item into a `CostEvent`, and appends the rows to the finelog
`cost.events` namespace.

```
CostEvent:
  ts            UTC midnight of the usage day (finelog ordering timestamp)
  usage_date    "YYYY-MM-DD" UTC usage day
  provider      openai | anthropic | gcp | coreweave
  category      provider-natural grouping (api / a GCP service / compute / storage / ...)
  detail        finer grain: model, line item, SKU, region, instance type
  cost          amount in `currency`
  currency      ISO code, e.g. USD
  amount_kind   "billed" (from a cost API) or "estimated" (usage × rate card)
  collected_ts  when this row was produced (one value per run)
```

### Re-runs and the "latest snapshot" read

finelog is append-only and the current UTC day is always partial, so each run
re-fetches a trailing window (`lookback_days`) and writes **fresh** rows;
earlier partial days get corrected by later runs. Readers therefore take the
newest row per logical key. The canonical query (also what the agent should
use):

```sql
SELECT usage_date, provider, category, detail, cost, currency, amount_kind
FROM (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY usage_date, provider, category, detail ORDER BY seq DESC
  ) AS rn
  FROM "cost.events"
)
WHERE rn = 1
ORDER BY usage_date DESC, provider, cost DESC;
```

## Providers and required secrets

Secrets are passed via the environment only — `config.yaml` holds the env-var
*names* (`*_env`), never values.

| Provider | Source | Secret (env var) | Status |
|----------|--------|------------------|--------|
| **openai** | Costs API `GET /v1/organization/costs` | `OPENAI_ADMIN_KEY` | Works. Needs an **org Admin key** with the dashboard "Usage" permission — a project `sk-proj-…` key is rejected. |
| **anthropic** | Admin Cost Report `GET /v1/organizations/cost_report` | `ANTHROPIC_ADMIN_KEY` | Works. Needs an **Admin key** (`sk-ant-admin01-…`). Amounts arrive in cents → converted to USD. |
| **gcp** | BigQuery billing export (`bq query`) | none (ADC / runner SA) | Disabled by default. The Cloud Billing API exposes no spend; detailed cost lives only in the BigQuery export. The `hai-gcp-models` billing account is Stanford-managed, so its export dataset may be unreadable from CI — point `billing_export_table` at a readable dataset and set `enabled: true`. |
| **coreweave** | Prometheus usage API (`observe.coreweave.com`) × rate card | `COREWEAVE_API_TOKEN` | Disabled by default, **estimate only**. CoreWeave has no dollar API; cost is `usage × rate_card` and tagged `amount_kind=estimated`. Fill in real `unit_rate`s and a token with the Observability Viewer role. |

Adding a provider: drop a `fetch(config, window) -> list[CostEvent]` module in
`backends/`, register it in `backends/__init__.py`, and add a block to
`config.yaml`.

## Running it

```bash
# Local smoke — fetch and print, never connect to finelog. Providers without a
# key fail loudly (and only for themselves); the process exits non-zero.
uv run python -m scripts.cost_manager.run --dry-run

# One provider, custom window, against a finelog server you already tunneled to:
uv run python -m scripts.cost_manager.run \
  --provider openai --lookback-days 7 \
  --finelog-url http://127.0.0.1:10001

# Production shape — open the SSH/k8s tunnel from the 'marin' finelog config:
OPENAI_ADMIN_KEY=… ANTHROPIC_ADMIN_KEY=… \
  uv run python -m scripts.cost_manager.run
```

A single provider failing (missing key, auth/permission error) does not abort
the run: the other backends still record, and the process exits non-zero so CI
surfaces the failure.

## In CI

`.github/workflows/ops-cost-report.yaml` runs this daily at 15:00 UTC. It
authenticates to GCP with the CI service account and opens a `gcloud` SSH tunnel
to the finelog VM (the same mechanism as the storage report), then writes to
`cost.events`. Required GitHub Actions secrets:

- Tunnel (already configured for other ops jobs): `IRIS_CI_GCP_SA_KEY`,
  `GCP_PROJECT_ID`, `IRIS_CI_GCP_SSH_KEY`, `IRIS_CI_GCP_SSH_KEY_PUB`.
- Provider keys (add these): `OPENAI_ADMIN_KEY`, `ANTHROPIC_ADMIN_KEY`,
  `COREWEAVE_API_TOKEN`.

[#6550]: https://github.com/marin-community/marin/issues/6550
[#6464]: https://github.com/marin-community/marin/issues/6464
