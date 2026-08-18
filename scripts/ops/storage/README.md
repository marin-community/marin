# Marin storage tooling

Tooling for Marin's GCS and S3-compatible buckets.

## Parquet recompression

`recompress_parquet.py` uses one Zephyr task per Parquet object to replace
Snappy or uncompressed files with zstd level 3 and page indexes. Each task
writes a temporary sibling, validates its schema, row count, codecs, and page
indexes, checks that the source did not change while it was read, and only then
replaces the source. Reruns skip zstd objects that already have page indexes,
making an interrupted migration resumable. Outputs that are not smaller are
discarded unless an existing zstd object still needs page indexes; page-index
overhead is retained in that case.

Run the job on a cluster local to the bucket. Omit `--apply` first to inventory
the candidate objects and bytes. Start with one dataset prefix; bucket-wide
globs can contain millions of objects and produce a correspondingly large
Zephyr plan.

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 2GB --disk 8GB --priority batch \
  --target-cluster cw-us-east-02a -- \
  python scripts/ops/storage/recompress_parquet.py \
  's3://marin-us-east-02a/marin/normalized/example/**/*.parquet' \
  --workers 16

# After reviewing the dry-run counters, repeat with --apply.
```

The command refuses `tmp/ttl=Nd` inputs because replacement would restart the
object's lifecycle clock. It also refuses to publish an output when the source
fingerprint changes during the rewrite. At 16 workers, additional storage is
bounded to approximately 16 temporary shards plus object-store copy overhead.

## Weekly storage report

`generate_report.py` is a one-command orchestrator: it opens a tunnel to the
Iris cluster, submits the compute as Iris jobs, then publishes from the laptop
(or CI runner).

```
scan_fs (Iris) ─> dedup (Zephyr) ─> render_report (DuckDB) ─> gist + Discord
```

```bash
# Full run, publish a public gist (default)
uv run scripts/ops/storage/generate_report.py

# What the weekly automation runs (secret gist + Discord summary)
uv run scripts/ops/storage/generate_report.py --gist secret --discord internal-discuss

# Reuse prior stages (cheap iteration): skip scan / dedup / report as needed
uv run scripts/ops/storage/generate_report.py --skip-scan --skip-dedup --skip-report
```

Key flags: `--gist {public,secret,none}`, `--discord <channel>`, `--workers N`,
`--history-dir`, `--run-id` (defaults to the UTC date; keeps weekly Iris job
names unique), `--change-threshold-gib`, `--dry-run`.

**Week-over-week diff.** Each run archives a compact per-`(bucket, dir_prefix)`
snapshot (prefixes ≥ 1 GiB, ~1 MiB) to `--history-dir`. The next run flags
prefixes whose size moved ≥ 100 GiB since the prior snapshot, split into
**Increases** (shown first) and **Decreases**. The first run establishes a
baseline. Snapshots are dated UTC; a run never diffs against a same-date
snapshot.

**Modules:** `scan_fs.py` (distributed object scan over GCS, CoreWeave, and R2
via `rigging.filesystem`), `render_report.py` (DuckDB
rollup + diff + markdown), `generate_report.py` (orchestrate + publish).

**Automation.** `.github/workflows/ops-storage-report.yaml` runs it weekly
(Mondays 14:00 UTC) and on manual `workflow_dispatch`. There is no
cluster-native cron — the schedule lives in GitHub Actions, which tunnels into
the controller (SA + SSH key, like `marin-canary-ferry`) and runs
`generate_report`. Discord posting uses the `internal-discuss` webhook (no bot).

**Prereqs:** `gh` (for `--gist`), `gcloud` + ADC (fetch `report.md`), a
reachable `marin` controller, and the channel webhook for `--discord`.

## CoreWeave storage telemetry

`coreweave_usage.py` reads two CoreWeave metrics and writes their current
values to the Finelog `storage.usage` table:

- `billing:object_storage_used_bytes:total` gives the used bytes for each
  bucket, zone, and storage class.
- `cwobject_quota_info` gives the active quota for each zone and storage class.

Each row has these fields:

```text
provider       coreweave
metric         used_bytes or quota_bytes
zone           CoreWeave availability zone
bucket         bucket name for used_bytes; empty for quota_bytes
storage_class  CoreWeave storage class
value_bytes    raw metric value in bytes
observed_at    time of the CoreWeave metric sample
collected_at   time of the collector run
```

The collector does not calculate storage cost. It keeps the source byte values
so Grafana can compare used storage with the current CoreWeave quota. It fails
if a used zone has no quota series. It does not use a fixed quota value.

Run a local check with a CoreWeave token that has the Observability Viewer
role:

```bash
COREWEAVE_API_TOKEN=... \
  uv run python -m scripts.ops.storage.coreweave_usage --dry-run
```

`.github/workflows/ops-coreweave-storage.yaml` runs the collector each hour.
The workflow writes to the `marin` Finelog server through the standard GCP SSH
tunnel. It needs the repository secret `COREWEAVE_API_TOKEN` and fails without
it, because a green run that collects nothing hides a frozen dashboard.

The Grafana `Storage` dashboard shows bucket bytes and quota use for each zone.
The `CoreWeaveStorageCapacity` rule pages after quota use stays above 80 percent
for five minutes. The critical notification reaches Slack and opens the Loom ops
agent on that alert thread. For example, a 900 TiB live quota starts the alert
above 720 TiB. The rule reads the quota metric, so it also follows a later quota
change.

The `CoreWeaveStorageTelemetryStale` rule sends a Slack warning when a known
storage series is more than three hours old. For a stale-data warning, check the
hourly workflow and the token first. For a quota warning, check the zone values
in the Storage dashboard and in the [CoreWeave quota page].

Both rules read the `storage.usage` namespace, which exists only after the
collector writes its first rows. Until then the query fails, and because both
rules set `execErrState: Alerting`, each one pages with `[no value]` labels. Run
the collector once when you add a rule that reads a new namespace.

[CoreWeave quota page]: https://docs.coreweave.com/products/storage/object-storage/manage-quotas
