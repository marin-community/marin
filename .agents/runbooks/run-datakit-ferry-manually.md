---
name: run-datakit-ferry-manually
description: Run, stop, and validate a datakit ferry smoke run by hand from a dev box.
---

# Runbook: Run, stop, and validate a datakit ferry smoke run by hand

**When you're here:** You want to reproduce the nightly datakit smoke ferry
(`experiments/ferries/datakit_ferry.py`) on demand — to test a pipeline change,
chase a canary failure, or warm a cache — instead of waiting for the GitHub
Actions schedule. The ferry runs download → normalize → dedup (fuzzy document)
→ consolidate → tokenize on FineWeb-Edu `sample/10BT`.

**TL;DR:**
- **Submit** to the `marin` prod cluster (not `marin-dev`) with a unique
  `SMOKE_RUN_ID` and the CPU-only sizing flags. It returns a `JOB_ID`.
- **Stop** with `iris job stop $JOB_ID` — kills the entrypoint and its Zephyr
  children.
- **Validate** with `scripts/datakit/validate_ferry_outputs.py` once the job
  succeeds — it checks per-stage row counts and the dedup fraction.

## Before you touch anything

- **This is cheap but not free.** The ferry downloads ~9.7M rows of FineWeb-Edu
  and writes five intermediate stages to GCS. Default output goes to a
  1-day-TTL temp bucket (see below), so it self-cleans; if you override
  `MARIN_PREFIX` to a persistent bucket, you own the cleanup.
- **Submit against `--cluster=marin` (prod), not the dev config.** The dev
  config (`lib/iris/config/marin-dev.yaml`) needs OS Login impersonation that
  dev service accounts typically lack — the job will fail to schedule. This is
  the one ferry-specific gotcha that bites; the rest of the job-run flag
  semantics are generic (see lib/iris/OPS.md:50).
- **No human-approval gate here** — this is an additive smoke run, not a
  destructive cluster op. Nothing you do in this runbook touches the controller
  or other people's jobs.

## Steps

### 1. Submit

Mint a unique `SMOKE_RUN_ID` (it namespaces the outputs, so collisions corrupt
a prior run's stages) and submit:

```bash
SMOKE_RUN_ID="datakit-smoke-manual-$(date +%Y%m%d-%H%M%S)"
echo "Run ID: $SMOKE_RUN_ID"

uv run iris --cluster=marin job run --no-wait \
  --memory=2G --disk=4G --cpu=1 --extra=cpu \
  -e SMOKE_RUN_ID "$SMOKE_RUN_ID" \
  -- python -m experiments.ferries.datakit_ferry
```

- The entrypoint is **CPU-only by design**: it's an `executor_main` parent that
  submits the real work as Zephyr/Fray children. Sizing it with an accelerator
  would deadlock the node (lib/iris/OPS.md:60). Hence `--cpu=1 --memory=2G`.
- Flag semantics — `--memory` (not `--ram`), `-e KEY VALUE` as two quoted
  positionals, `--extra` (the Python dependency extra) vs hardware requests —
  are generic to every iris job and live in lib/iris/OPS.md:50 ("job run
  gotchas"). Read that once; this runbook does not restate them.
- `--no-wait` returns immediately and prints the Iris job ID
  (`/<user>/iris-run-job-YYYYMMDD-HHMMSS`). Capture it:

```bash
export JOB_ID=/<user>/iris-run-job-...
```

**Where the output lands.** `SMOKE_RUN_ID` namespaces outputs under
`$MARIN_PREFIX/datakit-smoke/$SMOKE_RUN_ID/{download,normalize,dedup,consolidate,tokens}`.
`MARIN_PREFIX` defaults to `marin_temp_bucket(ttl_days=1)`
(`gs://marin-<region>/tmp/ttl=1d/...`), which auto-expires. Override it for a
persistent or pinned bucket with `-e MARIN_PREFIX gs://...` on the submit.
**Note the resolved `MARIN_PREFIX`** — you need the exact same value at the
validate step.

### 2. Stop (if you need to abort)

```bash
uv run iris --cluster=marin job stop $JOB_ID
```

Terminates the entrypoint and its Zephyr children together. Use this to abort a
wedged run before it finishes; a successful run stops itself.

### 3. Validate

After the job reaches success, point the validator at the **same**
`MARIN_PREFIX` and `SMOKE_RUN_ID` the run used:

```bash
MARIN_PREFIX=gs://marin-us-central1/tmp/ttl=1d \
SMOKE_RUN_ID=$SMOKE_RUN_ID \
  uv run python scripts/datakit/validate_ferry_outputs.py
```

The script
(`scripts/datakit/validate_ferry_outputs.py`, docstring lines 4–16) walks the
full chain — download → normalize → fuzzy_dups → consolidate → tokenize — and
asserts each stage's file count and row count against the expected envelope,
plus that consolidate drops the deduped rows and the tokenize cache ledger row
count matches consolidate. The per-stage thresholds are constants in that file;
don't transcribe them here, they drift with the dataset.

## Verify

- **Job succeeded:** `iris job status $JOB_ID` (or `iris job summary $JOB_ID
  --json`) shows the entrypoint and all children at success, not FAILED/PENDING.
  Command reference: lib/iris/OPS.md:38.
- **Outputs are correct, not just present:** `validate_ferry_outputs.py` exits 0.
  A non-zero exit names the first stage whose counts fell outside the expected
  envelope — that's your regression signal, not a flake. If the script can't
  find files at all, you almost certainly passed a different `MARIN_PREFIX` /
  `SMOKE_RUN_ID` than the run wrote to.

## Why this happens / Notes

- The ferry is normally driven nightly by a GitHub Actions datakit smoke
  workflow; these manual commands exist for dev-box experimentation outside that
  schedule. The canonical flag list and `MARIN_PREFIX` semantics are the
  reference in `experiments/ferries/OPS.md`, which points here for the
  walkthrough.
- The `--cluster=marin` caveat is the load-bearing detail: the dev cluster looks
  like the obvious target for a throwaway run, but its impersonation requirement
  makes scheduling fail in a way that looks unrelated to the ferry. Use prod.

## See also

- `experiments/ferries/OPS.md` — the ferry's flags, `MARIN_PREFIX` default, and
  validate invocation reference; the one-line pointer back to this runbook.
- lib/iris/OPS.md:50 "job run gotchas" — `--memory` vs `--ram`, `-e KEY VALUE`
  two-positional, `--extra` vs hardware, and the CPU-only-parent rule
  (lib/iris/OPS.md:60) that explains the sizing flags above.
- lib/iris/OPS.md:38 — `job status` / `job summary` / `job stop` syntax.
- `run-ferries` skill — owns the scheduled canary/daily ferry loop; this runbook
  is the by-hand analog.
- `triage-canary` skill — when a *scheduled* datakit canary goes red, start
  there; reproduce by hand with this runbook.
- `stand-up-coreweave-cluster` — the GPU canary analog (manual smoke on
  different hardware).
- `evaluate-zephyr-perf` skill — Gate 1 runs this manual ferry to sanity-check a
  zephyr-internals change before the perf gate.
