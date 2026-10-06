# Datakit Ferry Operations

Ad-hoc run/stop/validate for the datakit ferries. Each ferry runs the reference
Datakit DAG (`experiments/datakit/reference_pipeline.py`) through
`experiments/ferries/datakit_reference_ferry.py`:

| Tier | Ferry | Source | Region | Workflow |
|---|---|---|---|---|
| 1 | `datakit_ferry.py` | FineWeb-Edu `sample/10BT` | us-west4 | `marin-canary-datakit-tier1.yaml` (daily) |
| 2 | `datakit_tier2_skewed_ferry.py` | skewed synthetic, 46 GiB | us-west4 | `marin-canary-datakit-tier2.yaml` (daily) |
| 3 | `datakit_nemotron_ferry.py` | Nemotron-CC high, 1,000 files | europe-west4 | `marin-canary-datakit-tier3.yaml` (weekly) |

The commands below are for manual runs.

## Region prerequisites

The ferry reads two inputs from the region-local `MARIN_PREFIX`. They must be
present in the region where the ferry runs:

- the quality model: `datakit/models/quality/pooled_junkgate2`
- the decontamination eval corpus: `datakit/decontam/evals/<EVAL_CORPUS_VERSION>`

Both are present in `gs://marin-us-west4` and `gs://marin-eu-west4`. To run a
ferry in a different region, copy them there first. They are about 250 MB in total.

## Submit

```bash
SMOKE_RUN_ID="datakit-smoke-manual-$(date +%Y%m%d-%H%M%S)"
echo "Run ID: $SMOKE_RUN_ID"

uv run iris --cluster=marin job run --no-wait \
  --region=us-west4 --memory=2G --disk=4G --cpu=1 --extra=cpu \
  -e SMOKE_RUN_ID "$SMOKE_RUN_ID" \
  -- python -m experiments.ferries.datakit_ferry
```

- `--no-wait` returns immediately; the command prints the Iris job ID
  (`/<user>/iris-run-job-YYYYMMDD-HHMMSS`). Export it as `JOB_ID` for the
  stop command below.
- `SMOKE_RUN_ID` is required by the ferry. Every step output goes under
  `marin_temp_bucket(ttl_days=1, prefix=f"<ferry>/{SMOKE_RUN_ID}")`, and the
  driver records that absolute prefix in `FERRY_STATUS_PATH` when configured.
  Submitting again with the same `SMOKE_RUN_ID` resumes the run: completed steps
  are skipped.
- Leave `MARIN_PREFIX` unset. Iris derives the region-local stable prefix, which
  the ferry uses for its inputs.
- Keep each worker's disk request at or below 64 GB. Every Marin VM has a 100 GB
  disk, so a larger request is unschedulable and Zephyr retries it until the
  workflow times out.
- Use `--cluster=marin` (prod), not `--config=lib/iris/config/marin-dev.yaml`
  — the dev config needs OS Login impersonation that dev SAs typically lack.

## Cancel

```bash
uv run iris --cluster=marin job cancel $JOB_ID
```

Cancels the entrypoint job and its Zephyr children.

## Performance report

```bash
uv run python scripts/ci/collect_perf_metrics.py --job-id $JOB_ID --no-task-wall-time
```

`stage_wall_seconds` sums the `Step … succeeded in …` times that StepRunner
logs in the driver, one entry for each stage of the reference DAG.

## Validate output

After a successful tier-1 run, validate in the ferry's region. Artifact paths are
relative to the regional `MARIN_PREFIX`, and a local run trips the cross-region
transfer budget on the 2 GB download shards:

```bash
uv run iris --cluster=marin job run --region=us-west4 --memory=2G --cpu=1 --extra=cpu \
  -e FERRY_OUTPUT_PREFIX "gs://marin-us-west4/tmp/ttl=1d/datakit-smoke/$SMOKE_RUN_ID" \
  -- python experiments/datakit/scripts/validate_ferry_outputs.py
```

Confirms the download and normalize row counts and the document counts of the
final store.
