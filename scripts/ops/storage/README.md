# Marin GCS storage tooling

Tooling for the `marin-*` GCS buckets.

## Weekly storage report

`generate_report.py` is a one-command orchestrator: it opens a tunnel to the
Iris cluster, submits the compute as Iris jobs, then publishes from the laptop
(or CI runner).

```
scan_gcs (Iris) ─> dedup (Zephyr) ─> render_report (DuckDB) ─> gist + Discord
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

**Modules:** `scan_gcs.py` (distributed GCS object scan), `render_report.py`
(DuckDB rollup + diff + markdown), `generate_report.py` (orchestrate + publish).

**Automation.** `.github/workflows/ops-storage-report.yaml` runs it weekly
(Mondays 14:00 UTC) and on manual `workflow_dispatch`. There is no
cluster-native cron — the schedule lives in GitHub Actions, which tunnels into
the controller (SA + SSH key, like `marin-canary-ferry`) and runs
`generate_report`. Discord posting uses the `internal-discuss` webhook (no bot).

**Prereqs:** `gh` (for `--gist`), `gcloud` + ADC (fetch `report.md`), a
reachable `marin` controller, and the channel webhook for `--discord`.
