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
- `SMOKE_RUN_ID` is required by the ferry; it namespaces outputs under
  `$MARIN_PREFIX/datakit-smoke/$SMOKE_RUN_ID/{download,normalize,minhash,fuzzy_dups,consolidate,tokens}`.
- `MARIN_PREFIX` defaults to `marin_temp_bucket(ttl_days=1)`
  (`gs://marin-<region>/tmp/ttl=1d/...`). Override with `-e MARIN_PREFIX gs://...`
  for persistence or a specific bucket.
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

Confirms row counts across stages and checks that fuzzy dedup's sparse
`dup_doc=True` markers exactly match its accepted-verification counter and the
rows removed by consolidation. Candidate counts, rejection reasons, exact-score
histograms, per-source acceptance rates, and the LSH collision curve are in the
dedup stage report; the full accepted/rejected evidence is under
`$MARIN_PREFIX/datakit-smoke/$SMOKE_RUN_ID/fuzzy_dups/metadata/decisions/`.
