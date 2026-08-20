---
name: run-ferries
description: Launch, monitor, and seal Marin canary and daily ferry runs.
---

# Ferries

Canary is the stable, low-cost health check (`experiments/ferries/canary_ferry.py`);
daily is the larger bounded-change run (`experiments/ferries/daily.py`). Both
use `nemotron_mix`, default `us-central1`/`us-central1-a`, and the daily log
`docs/experiments/daily-ferry-log.md`. Follow `babysit-job` to terminal state.
Daily baseline is `llama_150m`, sequence length 4096, batch size 512, and about
`1e19` FLOPs unless the launch explicitly overrides it.

Every launch requires explicit requester approval unless the requester explicitly
waived it. Never mutate/restart a cluster without in-thread consent. Report
launch, first eval, major incident, and terminal state. Daily completion gets a
tag `ferry/daily/YYYYMMDD/<run_slug>` and a log-only PR with labels `ferry`,
`ferry-daily`, `ferry-log-only`, `ferry-sealed`.

## Daily

Before editing, inspect the latest log and collect last ferry issue/commit/W&B/
Iris links, the human objective, and the interval since the last ferry. Treat
GitHub-tagged ferry records as source of truth; ask if the objective is unclear.
Bound edits to 1–2 knobs, keep one intentional change, update run naming, and
record delta, rationale, risk, fallback, and approval in the run issue. Push the
launch commit (no proposal PR by default).

```bash
git log --oneline <last_ferry_sha>..HEAD -- experiments/ lib/ scripts/
gh issue list --label experiment --search "updated:>=<last_ferry_date>" --limit 100
uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \
  -- python -m experiments.ferries.daily
```

Capture job ID, cluster, timestamp, and W&B link. To make a deterministic rerun:

```bash
uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \
  -e FERRY_DATE "$(date +%Y%m%d-%H%M%S)-daily-ferry" \
  -- python -m experiments.ferries.daily
```

Monitor to `SUCCEEDED`, `FAILED`, or `STOPPED`; use one bounded fix attempt for
a failure, then escalate. Close the issue with final status, metrics, job/W&B,
experiment link, and next action. Extract log metrics with:

```bash
uv run python scripts/ferries/daily_analysis.py --run <wandb_run_url_or_path> --format markdown
```

Tag the exact launch commit, open a PR changing only the daily log, and keep
debug/retry narrative in the issue. Follow `commit` for PR text and monitoring.

## Canary

Do not run the daily proposal/PR loop for an unchanged canary. Launch as-is:

```bash
uv run iris --config=lib/iris/config/marin.yaml job run --memory=16G --disk=16G --cpu=1 --extra=tpu -- python -m experiments.ferries.canary_ferry
uv run iris --cluster=cw-us-east-02a job run --memory=16G --disk=16G --cpu=1 --extra=cpu \
  -e MARIN_PREFIX s3://marin-na/marin -e CANARY_ACCELERATOR gpu \
  -- python -m experiments.ferries.canary_ferry
```

Canary normally has no seal tag or closure PR. On failure, identify the cause,
then make a focused change only if needed, relaunch, and monitor to terminal
state. For profiling, check warmup count before steady-state conclusions;
`exclusive_per_track` can hide overlapping stalls, so use `exclusive_global`
when investigating stalls. See `profile-training` for raw-trace reanalysis.

Promotion requires broadly better eval losses/soft metrics with no reliability
regression; update the template and this skill in a follow-up PR with before/
after metrics.
