# Marin storage tooling

Storage reporting for the `marin-*` GCS buckets, telemetry for CoreWeave
object storage, and bulk copies between CoreWeave prefixes.

## Storage report (on demand)

> **The weekly automation is retired (2026-09).** The `marin-*` buckets are
> scanned daily by [gcs.oa.dev](https://gcs.oa.dev) ([Open-Athena/marin-gcs-usage](https://github.com/Open-Athena/marin-gcs-usage)),
> which posts a running monthly digest thread (weekly bullets in the opening
> post, one reply per scan, owners per prefix) to Discord `#gcs-usage`. Use
> the tooling below for a manual run or a cross-check.

`generate_report.py` is a one-command orchestrator: it opens a tunnel to the
Iris cluster, submits the compute as Iris jobs, then publishes from the laptop
(the same tooling ran from a GitHub Actions runner until 2026-09).

```
scan_fs (Iris) ─> dedup (Zephyr) ─> render_report (DuckDB) ─> gist + Discord
```

```bash
# Full run, publish a public gist (default)
uv run scripts/ops/storage/generate_report.py

# Secret gist plus a Discord summary
uv run scripts/ops/storage/generate_report.py --gist secret --discord internal-discuss

# Reuse prior stages (cheap iteration): skip scan / dedup / report as needed
uv run scripts/ops/storage/generate_report.py --skip-scan --skip-dedup --skip-report
```

Key flags: `--gist {public,secret,none}`, `--discord <channel>`, `--workers N`,
`--history-dir`, `--run-id` (defaults to the UTC date; keeps Iris job names
unique across runs), `--change-threshold-gib`, `--dry-run`.

**Week-over-week diff.** Each run archives a compact per-`(bucket, dir_prefix)`
snapshot (prefixes ≥ 1 GiB, ~1 MiB) to `--history-dir`. The next run flags
prefixes whose size moved ≥ 100 GiB since the prior snapshot, split into
**Increases** (shown first) and **Decreases**. The first run establishes a
baseline. Snapshots are dated UTC; a run never diffs against a same-date
snapshot.

**Modules:** `scan_fs.py` (distributed object scan over GCS, CoreWeave, and R2
via `rigging.filesystem`), `render_report.py` (DuckDB
rollup + diff + markdown), `generate_report.py` (orchestrate + publish).

**Automation.** None. The `ops-storage-report` workflow that ran this weekly
from GitHub Actions was deleted in 2026-09 (see above); run `generate_report.py`
from a laptop with a reachable `marin` controller. Discord posting uses the
channel webhook resolved by `scripts/ops/discord.py` (no bot).

**Prereqs:** `gh` (for `--gist`), `gcloud` + ADC (fetch `report.md`), a
reachable `marin` controller, and the channel webhook for `--discord`.

## Bulk copy between CoreWeave prefixes

`copy_prefix.py` copies every object below one CoreWeave prefix to another. It
submits an Iris job through the `marin` hub, which hands it off to the
CoreWeave cluster in the source's region (`--run-near destination` picks the
destination's). Zephyr workers stream the files:

```bash
export CW_KEY_ID=<key-id> CW_KEY_SECRET=<key-secret>
uv run scripts/ops/storage/copy_prefix.py \
  --from s3://marin-us-east-02a/marin/<prefix> \
  --to s3://marin-us-east-08a/marin/<prefix> \
  --dry-run                       # list the source and shard records from this machine; submits no job
```

The job packs source files, in key order, into 32 GiB shards and splits any
shard with more than 2,000 files. Each Zephyr
worker copies one shard eight files at a time with the per-object checks of
`fsutil verified-copy`. Uploads use fixed 50 MiB parts and publish only when
complete. Each upload's ETag is checked against the bytes read, and objects
smaller than one part are read back and hashed. After every file in a shard
passes, the worker writes the shard's record under
`<destination>.copy-status/shards/`. When all shards are done, the job checks
every destination object's size and ETag against the records and writes
`<destination>.copy-status/complete.json` last.

Rerunning the same command resumes the copy one whole shard at a time. A shard
is skipped when its record exists and every destination object still matches
it; every other shard is copied again in full. A
shard's ID covers its files' keys, sizes, and source ETags, so a changed source
file is copied again. Destination files outside the source are left untouched
and unchecked. `--max-workers` (default 50, at 2 CPUs and 8 GB each) bounds the
Zephyr pool so the copy fits in the CPU node pool.

CoreWeave rejects `CopyObject` between buckets in different regions
(`InvalidRegion`), so every byte passes through a worker. Only CoreWeave
prefixes are accepted; Hugging Face buckets and R2 are not supported yet.

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

`.github/workflows/ops-coreweave-storage.yaml` schedules the collector hourly at
minute 17. The workflow writes to the `marin` Finelog server through the
standard GCP SSH tunnel. It needs the repository secret
`COREWEAVE_API_TOKEN` and fails without it, because a green run that collects
nothing hides a frozen dashboard. The collector does not run on an Iris
CoreWeave controller or worker, so restarting those services does not restart
collection.

GitHub scheduled runs are best effort. Production runs have been 4–10 hours
apart despite the hourly cron; see the
[collection-gap incident](https://marina.oa.dev/echo/wiki/1036).

The Grafana `Storage` dashboard shows bucket bytes and quota use for each zone.
The `CoreWeaveStorageQuotaExceeded` rule pages after usage stays above the live
zone quota for five minutes. The critical notification reaches Slack and opens
the Loom ops agent on that alert thread. The rule reads the quota metric, so it
also follows a later quota change.

`CoreWeaveStorageHeadroomLow` warns when a zone and storage class use more
than 80% of their current quota or are projected to do so within 24 hours.
Its alert links the [CoreWeave size-over-time viewer](https://cw-s3.oa.dev/?v=time)
and opens a Loom storage investigation. Compare the viewer's scan timestamp,
bucket and prefix growth with the live quota samples in the Storage dashboard.
Trace growing prefixes to jobs or experiment records before attributing writes
to a person. Mention an owner in the alert thread only after verifying both the
evidence and their Slack user ID; otherwise report the likely source and the
uncertainty without a mention.
Grafana evaluates it every five minutes and delivers notifications to
`#marin-alerts` between 15:00 and 16:00
`America/Los_Angeles` each day, leaving at least an hour before a 17:00 finish.
The notification repeats after 23 hours, so the next afternoon is eligible
even when daylight saving time begins. Changed alert groups can notify again
within the window. The `storage-afternoon` interval and `storage-daily` policy
in `infra/grafana/provisioning/alerting/` own this schedule.

`CoreWeaveStorageQuotaPredicted` warns at any time of day when recent growth
projects usage above the current quota within 24 hours of the evaluation.
It uses the faster positive linear regression over the last 24 or 72 hours,
measured at the source `observed_at` timestamps. The 24-hour estimate needs at
least three distinct observations spanning eight hours; the 72-hour estimate
needs at least six spanning twenty-four hours. These windows tolerate the
observed collection cadence. The projection includes the age of the latest
observation, so its horizon remains 24 hours from the evaluation time.
Repeated collections of the same source timestamp count once. Both warnings
use this projection, with
current usage as a floor when history is insufficient or has no positive
trend. The exhaustion warning repeats every four hours while it fires.

The critical page and both warnings share one query that groups bucket bytes
by zone and storage class, deduplicates retried collection rows, and uses the
newest snapshot for each group. The critical page reads measured quota use;
the warnings read the projection. They require a collection and source
observation from the last twelve hours. Their five-minute pending period
suppresses brief changes between evaluations.
Forecasts assume recent growth continues; collection gaps, unexpected new
writes, and cleanup limit their accuracy. The daily headroom warning provides
a working-hours reminder even when growth has stopped. Check the Storage
dashboard's bucket curves to identify the active writers and arrange cleanup
or redirect new writes before leaving for the day.

The `CoreWeaveStorageTelemetryMissing` rule sends one Slack warning when the
newest collector timestamp is more than twelve hours old. It reads `collected_at`,
so the warning reports whether the workflow delivered telemetry without
creating one alert for every bucket and quota series. Check the workflow
history first:

```bash
gh run list --workflow ops-coreweave-storage.yaml --limit 10
```

If GitHub did not create a run during the window, restore collection with
`gh workflow run ops-coreweave-storage.yaml`. If a run failed, inspect it with
`gh run view <run-id> --log-failed`. `collected_at` is when the workflow fetched
a row, while `observed_at` is the CoreWeave metric timestamp. Compare them in
Finelog when the workflow succeeds but the Storage dashboard appears frozen.

For a quota warning, check the zone values in the Storage dashboard and in the
[CoreWeave quota page].

These rules read the `storage.usage` namespace, which exists only after the
collector writes its first rows. Until then Finelog reports the namespace as
missing, and the rules alert on that non-retryable query error. Run the
collector once when you add a rule that reads a new namespace. A retryable
Finelog service failure returns no data and stays normal; `FinelogFleetUnhealthy`
reports the outage instead.

[CoreWeave quota page]: https://docs.coreweave.com/products/storage/object-storage/manage-quotas
