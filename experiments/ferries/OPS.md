# Datakit Ferry Operations

Ad-hoc run/stop/validate for `experiments/ferries/datakit_ferry.py`.
The ferry runs download → normalize → minhash → fuzzy dedup → consolidate →
tokenize on FineWeb-Edu `sample/10BT`. It normally runs daily from the
`Marin - Canary - Datakit - Tier 1` GitHub Actions workflow
(`.github/workflows/marin-canary-datakit-tier1.yaml`). The commands below are
for manual runs.

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
- `SMOKE_RUN_ID` is required by the ferry. The driver writes outputs under
  `marin_temp_bucket(ttl_days=1, prefix=f"datakit-smoke/{SMOKE_RUN_ID}")` and
  records that absolute prefix in `FERRY_STATUS_PATH` when configured.
- Leave `MARIN_PREFIX` unset. Iris derives the region-local stable prefix used
  by the download cache; the per-run outputs use the one-day temp prefix above.
- Use `--cluster=marin` (prod), not `--config=lib/iris/config/marin-dev.yaml`
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
  uv run python experiments/datakit/scripts/validate_ferry_outputs.py
```

Confirms row counts and dedup fraction across stages.
