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
  provider      openai | anthropic | gcp | coreweave | together
  category      provider-natural grouping (api / a GCP service / compute / storage / ...)
  detail        finer grain: model, line item, SKU, region, instance type
  cost          amount in `currency`
  currency      ISO code, e.g. USD
  amount_kind   "billed" (from a cost API) or "estimated" (usage × rate card)
  region        provider region, when available
  usage_amount  provider usage gauge, when available
  usage_unit    unit for usage_amount, e.g. bytes
  collected_ts  when this row was produced (one value per run)
```

### Re-runs and the "latest snapshot" read

finelog is append-only and the current UTC day is always partial, so each run
re-fetches a trailing window (`lookback_days`) and writes **fresh** rows;
earlier partial days get corrected by later runs. Readers therefore take the
newest row per logical key. The canonical query (also what the agent should
use):

```sql
SELECT usage_date, provider, category, region, detail,
       cost, currency, amount_kind, usage_amount, usage_unit
FROM (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY usage_date, provider, category, region, detail ORDER BY seq DESC
  ) AS rn
  FROM "cost.events"
)
WHERE rn = 1
ORDER BY usage_date DESC, provider, cost DESC;
```

CoreWeave storage rows use `detail` for the bucket and `region` for the provider region.
The usage amount is the last byte sample for that UTC day. The cost integrates
all hourly samples for the day.

## Slack threshold alerts (optional)

The `alerts` block in `config.yaml` defines spend ceilings. Each rule sums a
cost slice — an optional `provider`/`category` filter — over a `window`
(`latest_day`, the most recent complete UTC day, or `window_total`, the whole
fetch window) and fires when that sum exceeds `max_usd`:

```yaml
alerts:
  webhook_url_env: SLACK_WEBHOOK_URL   # env var holding the incoming-webhook URL
  rules:
    - name: total-daily          # no provider -> all providers combined
      window: latest_day
      max_usd: 500
    - name: openai-daily
      provider: openai
      window: latest_day
      max_usd: 200
```

Alerting is best-effort and fully optional: a breach is always logged at
WARNING, but the Slack POST is skipped on `--dry-run` and when the webhook env
var is unset, so local runs and environments without the secret stay silent. A
POST failure is logged and never fails the run. The webhook (a standard Slack
incoming webhook, same `{"text": …}` contract as the repo's `notify-slack`
action) determines the channel — point `webhook_url_env` at a webhook bound to
`#marin-eng`. Remove the `alerts` block (or its `rules`) to disable.

## Providers and required secrets

Secrets are passed via the environment only — `config.yaml` holds the env-var
*names* (`*_env`), never values.

| Provider | Source | Secret (env var) | Status |
|----------|--------|------------------|--------|
| **openai** | Costs API `GET /v1/organization/costs` | `OPENAI_ADMIN_KEY` | Works. Needs an **org Admin key** with the dashboard "Usage" permission — a project `sk-proj-…` key is rejected. |
| **anthropic** | Admin Cost Report `GET /v1/organizations/cost_report` | `ANTHROPIC_ADMIN_KEY` | Works. Needs an **Admin key** (`sk-ant-admin01-…`). Amounts arrive in cents → converted to USD. |
| **gcp** | BigQuery billing export (`bq query`) | none (ADC / runner SA) | Disabled by default. The Cloud Billing API exposes no actual spend; detailed cost lives only in the BigQuery export. Enable once `billing_export_table` points at an export dataset the runner's service account can read. |
| **coreweave** | Prometheus usage API (`observe.coreweave.com`) × rate card | `COREWEAVE_API_TOKEN` | Configured for object storage and activated when the token is set. Cost uses the public hot-storage rate and has `amount_kind=estimated`. |
| **together** | none yet | `TOGETHER_API_KEY` (when available) | Disabled by default, **scaffold only**. Together has no programmatic cost API: spend lives only in the cookie-authenticated billing dashboard (Usage / draft invoice), and the API key is inference-only. `fetch` raises until Together ships a usage/cost endpoint; enabling it before then fails loudly. |

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

## CoreWeave storage dashboard

Grafana provisions the `Storage` dashboard from
`infra/grafana/dashboards/storage.json`. The dashboard shows the latest daily
byte value for each CoreWeave bucket.

Grafana also provisions the `CoreWeaveStorageCapacity` alert. The rule reads
the newest Finelog gauge for each bucket and sends a Slack warning above 80 TiB.
The value must stay above the limit for five minutes. A 36-hour freshness limit
prevents an alert from an expired value. The alert stays normal until the first
CoreWeave row exists.

The collector uses the public hot-storage rate of $0.06/GiB-month. Divide this
rate by 730 hours for the `unit_rate` value in `config.yaml`. Replace the value
when the contracted rate is available.

The 80 TiB ceiling is 80 percent of CoreWeave's default 100 TiB capacity quota
for each availability zone.

`COREWEAVE_API_TOKEN` controls provider activation. If it is unset, the runner
skips CoreWeave and completes the other providers. Use a token with the
Observability Viewer role. The Grafana Kubernetes read token does not include
that role.

## In CI

`.github/workflows/ops-cost-report.yaml` runs this daily at 15:00 UTC. It
authenticates to GCP with the CI service account and opens a `gcloud` SSH tunnel
to the finelog VM (the same mechanism as the storage report), then writes to
`cost.events`. Required GitHub Actions secrets:

- Tunnel (already configured for other ops jobs): `IRIS_CI_GCP_SA_KEY`,
  `GCP_PROJECT_ID`, `IRIS_CI_GCP_SSH_KEY`, `IRIS_CI_GCP_SSH_KEY_PUB`.
- Provider keys (add these): `OPENAI_ADMIN_KEY`, `ANTHROPIC_ADMIN_KEY`, and
  `COREWEAVE_API_TOKEN`. CoreWeave stays inactive until its token is set.
- Optional, for threshold alerts: `SLACK_WEBHOOK_URL` (already configured for
  other ops jobs). Unset → threshold breaches are logged but not posted.

[#6550]: https://github.com/marin-community/marin/issues/6550
[#6464]: https://github.com/marin-community/marin/issues/6464
