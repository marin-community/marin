# Ferry Operations

## Datakit ferry

Ad-hoc run/stop/validate for `experiments/ferries/datakit_ferry.py`.
The ferry runs download → normalize → MinHash candidate search → full-text
verification → consolidate → tokenize on FineWeb-Edu `sample/10BT`. It
normally runs daily from the
`Marin - Canary - Datakit - Tier 1` GitHub Actions workflow
(`.github/workflows/marin-canary-datakit-tier1.yaml`). The commands below are
for manual runs.

### Submit

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

### Cancel

```bash
uv run iris --cluster=marin job cancel $JOB_ID
```

Cancels the entrypoint job and its Zephyr children.

### Validate output

After success:

```bash
MARIN_PREFIX=gs://marin-us-central1/tmp/ttl=1d \
SMOKE_RUN_ID=$SMOKE_RUN_ID \
  uv run python experiments/datakit/scripts/validate_ferry_outputs.py
```

Confirms row counts and dedup fraction across stages.

## Canary ferry: bisect an MFU change

A scheduled canary run tests whatever `main` was when GitHub started it, so a change in MFU between two runs narrows only to the commits between them. Each run's commit is the head SHA of its GitHub Actions run, joined on `RUN_ID` (`canary-tpu-<Actions run id>-<attempt>`); runs from Oct 2, 2026 on also record it as `git_commit` in the W&B run config. To find the commit, re-run the canary configuration at commits inside that range.

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
- Iris passes `WANDB_API_KEY` and `HF_TOKEN` from the submitting shell. Without the W&B key the run logs nothing to W&B.
- The coordinator job is `/<user>/iris-run-job-<timestamp>`; the TPU child is `grug-train-<RUN_ID>`. A v6e-4 run trains 237 steps in about 8 minutes; environment setup and queueing for capacity add to that.

After the run finishes, remove the checkout with `git worktree remove "$WORKTREE"`.

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
