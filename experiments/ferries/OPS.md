# Datakit Ferry Operations

Ad-hoc run/stop/validate for `experiments/ferries/datakit_ferry.py`.
The ferry runs download → normalize → dedup (fuzzy document) → consolidate →
tokenize on FineWeb-Edu `sample/10BT`. Normally triggered nightly by the
`Marin - Datakit Smoke` GitHub Actions workflow
(`.github/workflows/marin-datakit-smoke.yaml`); the commands below are for
manual experimentation from a dev box.

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
  stop command below.
- `SMOKE_RUN_ID` is required by the ferry; it namespaces outputs under
  `$MARIN_PREFIX/datakit-smoke/$SMOKE_RUN_ID/{download,normalize,dedup,consolidate,tokens}`.
- `MARIN_PREFIX` defaults to `marin_temp_bucket(ttl_days=1)`
  (`gs://marin-<region>/tmp/ttl=1d/...`). Override with `-e MARIN_PREFIX gs://...`
  for persistence or a specific bucket.
- Use `--cluster=marin` (prod), not `--config=lib/iris/examples/marin-dev.yaml`
  — the dev config needs OS Login impersonation that dev SAs typically lack.

## Stop

```bash
uv run iris --cluster=marin job stop $JOB_ID
```

Terminates the entrypoint job and its Zephyr children.

## Validate output

After success:

```bash
MARIN_PREFIX=gs://marin-us-central1/tmp/ttl=1d \
SMOKE_RUN_ID=$SMOKE_RUN_ID \
  uv run python scripts/datakit/validate_ferry_outputs.py
```

Confirms row counts and dedup fraction across stages.
