# Ferry Operations

## Datakit ferry

Ad-hoc run/stop/validate for the datakit ferries. Each ferry runs the reference
Datakit DAG (`experiments/datakit/reference_pipeline.py`) through
`experiments/ferries/datakit_reference_ferry.py`:

| Tier | Ferry | Source | Region | Workflow |
|---|---|---|---|---|
| 1 | `datakit_ferry.py` | FineWeb-Edu `sample/10BT` | us-west4 | `marin-canary-datakit-tier1.yaml` (daily) |
| 2 | `datakit_tier2_skewed_ferry.py` | skewed synthetic, 46 GiB | us-west4 | `marin-canary-datakit-tier2.yaml` (daily) |
| 3 | `datakit_nemotron_ferry.py` | Nemotron-CC high, 1,000 files | europe-west4 | `marin-canary-datakit-tier3.yaml` (weekly) |

The commands below are for manual runs.

### Region prerequisites

The ferry reads two inputs from the region-local `MARIN_PREFIX`. They must be
present in the region where the ferry runs:

- the quality model: `datakit/models/quality/pooled_junkgate2_marin` (the marin-tokenizer retrain; the e5-era `pooled_junkgate2` dir is refused by the quality step)
- the decontamination eval corpus: `datakit/decontam/evals/<EVAL_CORPUS_VERSION>`

Both are present in `gs://marin-us-west4` and `gs://marin-eu-west4`. To run a
ferry in a different region, copy them there first. They are about 250 MB in total.

### Submit

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

### Cancel

```bash
uv run iris --cluster=marin job cancel $JOB_ID
```

Cancels the entrypoint job and its Zephyr children.

### Performance report

```bash
uv run python scripts/ci/collect_perf_metrics.py --job-id $JOB_ID --no-task-wall-time
```

`stage_wall_seconds` sums the `Step … succeeded in …` times that StepRunner
logs in the driver, one entry for each stage of the reference DAG.

### Validate output

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

## Canary ferry: bisect an MFU change

A scheduled canary run tests whatever `main` was when GitHub started it, so a change in MFU between two runs narrows only to the commits between them. Each run's commit is the head SHA of its GitHub Actions run, joined on `RUN_ID` (`canary-tpu-<Actions run id>-<attempt>`); runs from Oct 3, 2026 on also record it as `git_commit` in the W&B run config. To find the commit, re-run the canary configuration at commits inside that range.

This covers the TPU canary. The H100 canary's `iris-ci` controller is redeployed by every scheduled canary and CoreWeave PR smoke test, and submitting to it needs the CoreWeave kubeconfig at `~/.kube/coreweave-iris`; that path is untested.

### Rules

- Get the requester's approval before each run, as for any ferry launch.
- Never start bisection runs with `gh workflow run`. The TPU canary workflow's job runs in concurrency group `canary-ferry` with `cancel-in-progress: true`, so a dispatched run cancels the scheduled canary or is cancelled by it.
- Submit to Iris at `--priority batch`. The canary's TPU training job inherits its parent's band, so both yield to every other job. Expect queueing and preemption ([#5942](https://github.com/marin-community/marin/issues/5942)).
- Keep the benchmark fixed. Use the `CANARY_*` values from `.github/workflows/marin-canary-ferry.yaml` at the commit under test. If `experiments/ferries/canary_ferry.py` or those values change inside the range, the two ends are different benchmarks.

### List the suspects

Commits that touch training code, oldest first:

```bash
git log --reverse --first-parent --format='%h %s' <before>..<after> -- \
  lib/levanter/src lib/haliax/src experiments/grug experiments/ferries/canary_ferry.py \
  uv.lock pyproject.toml lib/levanter/pyproject.toml
```

Test the middle suspect. A result near the after MFU puts the change at or before that commit; a result near the before MFU puts it later. Repeat until one suspect remains. Ten suspects need at most four runs.

### Submit one run

Run from a checkout of current `main`:

```bash
COMMIT=<sha to test>
CURRENT=$(pwd)
WORKTREE=$(mktemp -d)/marin-bisect
git worktree add --detach "$WORKTREE" "$COMMIT"
cd "$WORKTREE"

export WANDB_API_KEY="$(python3 -c 'import netrc; print(netrc.netrc().authenticators("api.wandb.ai")[2])')"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
RUN_ID="canary-bisect-tpu-${COMMIT:0:10}-$(date -u +%m%d%H%M)"
# The scheduled canary's settings at the tested commit.
WORKFLOW=.github/workflows/marin-canary-ferry.yaml
BATCH_SIZE=$(grep -m1 'CANARY_BATCH_SIZE: "' "$WORKFLOW" | sed 's/.*"\(.*\)".*/\1/')
TARGET_TOKENS=$(grep -m1 'CANARY_TARGET_TOKENS: "' "$WORKFLOW" | sed 's/.*"\(.*\)".*/\1/')

uv run --project "$CURRENT" iris --config=lib/iris/config/marin.yaml job run --no-wait \
  --timeout 21600 --memory=2G --disk=4G --cpu=1 --extra=cpu \
  --priority batch --preemptible --reserve v6e-4 \
  -e RUN_ID "$RUN_ID" \
  -e CANARY_ACCELERATOR tpu \
  -e CANARY_BATCH_SIZE "$BATCH_SIZE" \
  -e CANARY_TARGET_TOKENS "$TARGET_TOKENS" \
  -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin \
  -- python -m experiments.ferries.canary_ferry
echo "https://wandb.ai/marin-community/marin/runs/$RUN_ID"
```

- `iris job run` bundles the working directory, so the job runs the tested commit's code.
- `--project "$CURRENT"` runs the current `iris` CLI. The controller rejects a root submission from a client whose build date, the last commit touching `lib/iris`, is more than 14 days older than its own; child jobs submitted from inside the bundle are not checked.
- Iris passes `WANDB_API_KEY` and `HF_TOKEN` from the submitting shell. Without `WANDB_API_KEY`, the coordinator raises an error before it submits the TPU job.
- The coordinator job is `/<user>/iris-run-job-<timestamp>`; the TPU child is `grug-train-<RUN_ID>`. A v6e-4 run trains 238 steps in about 8 minutes; environment setup and queueing for capacity add to that.

After the run finishes, go back to the main checkout and remove this one with `cd "$CURRENT" && git worktree remove "$WORKTREE"`.

### Read the result

Compare the median `throughput/mfu` over steps 10 and up with the same statistic for the scheduled runs before and after the change. Check that `throughput/tokens_per_second` moves the same way; if only MFU moves, the FLOP estimate changed, not the speed.

```bash
uv run python - "$RUN_ID" <<'PY'
import statistics, sys
import wandb
run = wandb.Api().run(f"marin-community/marin/{sys.argv[1]}")
rows = [r for r in run.scan_history(keys=["_step", "throughput/mfu", "throughput/tokens_per_second"]) if r["_step"] >= 10]
print("median MFU", statistics.median(r["throughput/mfu"] for r in rows))
print("median tokens/s", statistics.median(r["throughput/tokens_per_second"] for r in rows))
PY
```

If the last suspect's parent also measures near the after MFU, the change is in a commit outside the suspect paths; bisect every commit in the range instead.
