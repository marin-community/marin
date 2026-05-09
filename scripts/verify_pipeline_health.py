# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify SWE-ZERO data-gen pipeline is actually producing rollouts.

File-size growth alone is misleading: cross-region seed copies (Option C),
retry loops, and various crashes inflate file sizes without producing real
new rollouts. This script reports three INDEPENDENT signals so a misleading
file-size delta can be cross-checked:

1. **Imported-summary count**: number of `imported_summary_*.json` files in
   source per_pr/ dirs. Each one is written by an imported worker on
   successful (rc=0) iter completion — zero = no successful imported iters
   have happened.

2. **Recent inner-worker log evidence**: scan a sample of recent
   `_swarm_worker_*_iter_*.log` files for "Checkpoint: saved N/M rollouts"
   lines. These are emitted only by the inner subprocess during real rollout
   generation (NOT during seed/copy operations).

3. **Audit `_done` delta** (optional, expensive ~10 min): re-runs the audit
   script and reports done-count change vs a prior cached snapshot.

Usage::

    python scripts/verify_pipeline_health.py
    python scripts/verify_pipeline_health.py --include-audit  # full audit
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys

import fsspec

logger = logging.getLogger(__name__)
REGIONS = ["us-east5", "us-east1", "us-west4", "us-central1"]
CHECKPOINT_RE = re.compile(r"Checkpoint: saved (\d+)/(\d+) rollouts")


def count_imported_summaries(fs: fsspec.AbstractFileSystem) -> dict[str, int]:
    """Count imported_summary_*.json files per source region."""
    out: dict[str, int] = {}
    for region in REGIONS:
        pattern = f"marin-{region}/experiments/swe_zero_100b/per_pr/shard_*/imported_summary_*.json"
        try:
            hits = fs.glob(pattern)
        except Exception:
            hits = []
        out[region] = len(hits)
    return out


def sample_recent_inner_logs(fs: fsspec.AbstractFileSystem, n_per_region: int = 3) -> dict[str, list[str]]:
    """Grab the N most-recent _swarm_worker_*_iter_*.log files per region
    and report the last 'Checkpoint: saved' line in each, if any. Empty
    list = inner subprocesses are NOT producing rollouts (or none recently)."""
    out: dict[str, list[str]] = {}
    for region in REGIONS:
        signals: list[str] = []
        # Scan the most-recently-mutated shard dirs first by listing all
        # claim files (small, recently-touched on every heartbeat).
        try:
            shard_dirs = fs.glob(f"marin-{region}/experiments/swe_zero_100b/per_pr/shard_*/_swarm_claim.json")
        except Exception:
            shard_dirs = []
        # Sort by mtime desc using info()
        timed = []
        for path in shard_dirs[:50]:  # bound the scan
            try:
                info = fs.info(path)
                mtime = info.get("mtime") or info.get("updated")
                if mtime:
                    ts = mtime.timestamp() if hasattr(mtime, "timestamp") else float(mtime)
                else:
                    ts = 0
                shard_dir = path.rsplit("/", 1)[0]
                timed.append((ts, shard_dir))
            except Exception:
                continue
        timed.sort(reverse=True)
        for _ts, shard_dir in timed[:n_per_region]:
            # Find the most recent log file in this shard dir
            try:
                logs = fs.glob(f"{shard_dir}/_swarm_worker_*_iter_*.log")
                if not logs:
                    continue
                # Pick the lexicographically last (highest iter)
                latest_log = sorted(logs)[-1]
                with fsspec.open(f"gs://{latest_log}", "rb") as f:
                    text = f.read(64 * 1024).decode("utf-8", errors="replace")  # last 64 KB is plenty
                ckpts = CHECKPOINT_RE.findall(text)
                if ckpts:
                    last_done, last_total = ckpts[-1]
                    signals.append(f"shard={shard_dir.rsplit('/', 1)[-1]} {last_done}/{last_total} rollouts")
            except Exception:
                continue
        out[region] = signals
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-audit", action="store_true", help="Also run the full audit (~10 min)")
    args = parser.parse_args()

    fs = fsspec.filesystem("gs")

    print("=" * 70)
    print("SWE-ZERO PIPELINE HEALTH VERIFICATION")
    print("=" * 70)
    print()
    print("Signal 1: imported_summary_*.json count per source region")
    print("-" * 70)
    summaries = count_imported_summaries(fs)
    total_summaries = sum(summaries.values())
    for region, n in summaries.items():
        marker = "✗" if n == 0 else "✓"
        print(f"  {marker} {region}: {n} summaries")
    print(f"  TOTAL: {total_summaries}")
    if total_summaries == 0:
        print("  ⚠️  Zero summaries means no imported iter has completed cleanly.")
        print("      If imported workers are running, they may be crash-looping.")
    print()

    print("Signal 2: recent inner-worker 'Checkpoint: saved' evidence")
    print("-" * 70)
    log_signals = sample_recent_inner_logs(fs)
    for region, signals in log_signals.items():
        if signals:
            print(f"  ✓ {region}:")
            for s in signals:
                print(f"      {s}")
        else:
            print(f"  ✗ {region}: no recent Checkpoint evidence")
    if not any(log_signals.values()):
        print("  ⚠️  No recent inner-worker logs show Checkpoint lines.")
        print("      Workers may be crashing before rollout generation begins.")
    print()

    if args.include_audit:
        print("Signal 3: full audit (running ~10 min)...")
        print("-" * 70)
        try:
            r = subprocess.run(
                [sys.executable, "scripts/audit_stale_done.py"],
                capture_output=True,
                text=True,
                timeout=900,
            )
            print(r.stdout[-2000:] if r.stdout else "(no stdout)")
            if r.returncode != 0:
                print(f"  AUDIT FAILED rc={r.returncode}")
                print(r.stderr[-500:])
        except subprocess.TimeoutExpired:
            print("  AUDIT TIMED OUT after 15 min")
    else:
        print("Signal 3: full audit skipped (pass --include-audit to run)")
        print("-" * 70)

    print()
    print("=" * 70)
    if total_summaries == 0 and not any(log_signals.values()):
        print("VERDICT: ⚠️  PIPELINE NOT PRODUCING — check launcher logs for crashes")
        return 1
    print("VERDICT: ✓  pipeline is producing rollouts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
