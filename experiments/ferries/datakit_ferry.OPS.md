# Datakit Ferry Operations

Ad-hoc run/monitor/stop/validate for `experiments/ferries/datakit_ferry.py`.
The ferry runs download → normalize → dedup (fuzzy document) → consolidate →
tokenize on FineWeb-Edu `sample/10BT`. Normally triggered nightly by the
`Marin - Datakit Smoke` GitHub Actions workflow
(`.github/workflows/marin-datakit-smoke.yaml`); the commands below are for
manual experimentation from a dev box.

## Prerequisites

- GCP auth active: `gcloud auth list` shows an authorized account (e.g.
  `rav-agent@hai-gcp-models.iam.gserviceaccount.com`).
- SSH key in agent: `ssh-add -l` shows your `google_compute_engine` key.
  Re-add with `ssh-add ~/.ssh/google_compute_engine` if missing.
- Repo checked out on the branch you want to test — Iris bundles the
  workspace at submit time, so local code == what the workers run.

## Submit

```bash
SMOKE_RUN_ID="datakit-smoke-manual-$(date +%Y%m%d-%H%M%S)"
echo "Run ID: $SMOKE_RUN_ID"

uv run iris --cluster=marin job run --no-wait \
  --memory=2G --disk=4G --cpu=1 --extra=cpu \
  -e SMOKE_RUN_ID "$SMOKE_RUN_ID" \
  -- python -m experiments.ferries.datakit_ferry
```

- `--no-wait` returns immediately; the command prints the Iris job ID
  (`/<user>/iris-run-job-YYYYMMDD-HHMMSS`). Export it as `JOB_ID` for the
  monitor commands below.
- `SMOKE_RUN_ID` is required by the ferry; it namespaces outputs under
  `$MARIN_PREFIX/datakit-smoke/$SMOKE_RUN_ID/{download,normalize,dedup,consolidate,tokens}`.
- `MARIN_PREFIX` defaults to `marin_temp_bucket(ttl_days=1)`
  (`gs://marin-tmp-<region>/ttl=1d/...`). Override with `-e MARIN_PREFIX gs://...`
  for persistence or a specific bucket.
- Use `--cluster=marin` (prod), not `--config=lib/iris/examples/marin-dev.yaml`
  — the dev config needs OS Login impersonation that dev SAs typically lack.

## Monitor

Overall job state + per-task resource usage:

```bash
uv run iris --cluster=marin job summary --json $JOB_ID
```

List the child jobs spawned by each step (one Zephyr coordinator per step,
plus multiple pipelines for dedup and tokenize):

```bash
uv run iris --cluster=marin job list --json --prefix $JOB_ID | \
  python3 -c "import sys,json; [print(f'{j[\"job_id\"]:80s} {j[\"state\"]}') for j in json.load(sys.stdin)]"
```

Per-step wall time from the controller logs (emitted by `StepRunner`
instrumentation on `main`):

```bash
uv run iris --cluster=marin job logs --max-lines 10000 $JOB_ID | \
  grep -E "Step .* succeeded in|Datakit ferry total wall"
```

For per-stage (Zephyr) timing, fetch logs and grep for stage completion —
note log streaming can lag, and the buffer caps at 1000 lines per call
(use `--max-lines`).

Per child-job durations (useful when log streaming is flaky):

```bash
for j in $(uv run iris --cluster=marin job list --json --prefix $JOB_ID | \
  python3 -c "import sys,json; [print(j['job_id']) for j in json.load(sys.stdin) if 'workers' not in j['job_id'] and j['job_id'].count('/')==3]"); do
  uv run iris --cluster=marin job summary --json $j | \
    python3 -c "import sys,json,os; t=json.load(sys.stdin)['tasks'][0]; print(f'{os.environ[\"J\"][:60]:60s} {t[\"duration_ms\"]/1000:.0f}s')" J=$j
done
```

## Stop

```bash
uv run iris --cluster=marin job stop $JOB_ID
```

Terminates the entrypoint job and its Zephyr children.

## Validate output

After success:

```bash
MARIN_PREFIX=gs://marin-tmp-us-central1/ttl=1d \
SMOKE_RUN_ID=$SMOKE_RUN_ID \
  uv run python scripts/datakit/validate_ferry_outputs.py
```

Confirms row counts and dedup fraction across stages.

## Notes / gotchas

- SSH tunnel to the controller is established automatically on every
  `iris` invocation; if commands hang, check
  `ps aux | grep "gcloud.*ssh.*iris-controller"` and kill stale tunnels
  with `pkill -f "gcloud.*ssh.*iris-controller"`.
- TPU worker preemption can kill the Zephyr coordinator mid-run; the
  pipeline auto-retries up to ~100 times. Watch for
  `attempt 0 failed ... retrying` in the logs — as long as attempts are
  progressing, let it run.
- Skip steps by setting their status on the output path to `SUCCESS`, or
  rely on `StepRunner`'s built-in caching (already-succeeded steps are
  skipped by matching `name_with_hash` + `output_path`).
- `datakit-smoke/download` has a fixed output path
  (`$MARIN_PREFIX/datakit-smoke/download`) so it can be cached across
  runs; all other steps are per-`SMOKE_RUN_ID`.
