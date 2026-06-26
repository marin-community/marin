# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Incremental W&B exporter for the 36-cell Delphi midtraining sweep.

Writes into the existing ``midtrain_wandb_data/`` schema:

    runs/<run_id>/
        metadata.json       run id, name, state, created_at, url, tags, ...
        config.json         run.config
        summary.json        run.summary._json_dict
        history.jsonl       full scan_history rows, one JSON object per line
        history_meta.json   {"row_count": N, "keys": [...]}
        files_index.json    [{"name", "size", "md5", "url"}, ...]
        files/              W&B run files (config.yaml, output.log, ...) <=200MB

Plus:

    runs_index.csv / runs_index.json   appended (deduped by id)
    download_manifest.json             updated, preserving entity/project/contents

The target list is hard-coded and refreshes/fills the historical sweep cells
per ``.agents/logbooks/midtraining_delphi.md`` line 8344+. The rows labeled
``1e20`` are retained under their W&B names for lookup compatibility, but they
used the deprecated v5-isoflop base, not canonical Delphi.
Idempotent: skips runs whose local ``summary.json._step`` already meets the
expected final step for the scale.

Usage:
    uv run python scripts/analysis/download_midtrain_wandb.py
    uv run python scripts/analysis/download_midtrain_wandb.py --dry-run
    uv run python scripts/analysis/download_midtrain_wandb.py --force-refresh
    uv run python scripts/analysis/download_midtrain_wandb.py --only <run-id>
    uv run python scripts/analysis/download_midtrain_wandb.py --no-files
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path

import wandb
from tqdm.auto import tqdm

logger = logging.getLogger("download_midtrain_wandb")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "midtrain_wandb_data"
RUNS_DIR = DATA_DIR / "runs"

ENTITY = "marin-community"
PROJECT = "delphi-midtraining"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

EXPECTED_FINAL_STEP = {"1e20": 9412, "1e21": 4410, "1e22": 7646}
COMPLETION_TOLERANCE = 5
MAX_FILE_BYTES = 200 * 1024 * 1024

# Hard-coded target list. Order: scale -> mix -> lr.
# Pulled from logbook "Aggregate scoreboard" (line 8344+) and bottom ledger
# tables. Excludes the 8 fossil hashes the logbook flags as stale-broken-recovery
# namespaces ("ignore them in analysis", line 8367).
HISTORICAL_SWEEP_DONE = [
    # Historical v5-isoflop 3e20 rows previously labeled "1e20" (6) -
    # p50m50 mix and all lr0.83.
    "delphi-1e20-p33m67-4p94b-lr0.83-2a22e0",
    "delphi-1e20-p50m50-4p94b-lr0.33-9a74fa",
    "delphi-1e20-p50m50-4p94b-lr0.5-3475fa",
    "delphi-1e20-p50m50-4p94b-lr0.67-554fb6",
    "delphi-1e20-p50m50-4p94b-lr0.83-95e10d",
    "delphi-1e20-p67m33-4p94b-lr0.83-1965f3",
    # 1e21 missing (5)
    "delphi-1e21-p50m50-9p25b-lr0.33-bccff4",
    "delphi-1e21-p50m50-9p25b-lr0.5-973c46",
    "delphi-1e21-p50m50-9p25b-lr0.67-7e82b3",
    "delphi-1e21-p50m50-9p25b-lr0.83-f9edd2",
    "delphi-1e21-p67m33-9p25b-lr0.83-a1a261",
    # 1e22 missing (3)
    "delphi-1e22-p50m50-32p07b-lr0.33-c43ada",
    "delphi-1e22-p50m50-32p07b-lr0.83-3c9f70",
    "delphi-1e22-p67m33-32p07b-lr0.83-d35daa",
]

# Already-in-dump historical sweep cells whose local _step is short of final.
# Refresh required to capture post-2026-05-02 progress.
HISTORICAL_SWEEP_INCOMPLETE = [
    "delphi-1e21-p67m33-9p25b-lr0.5-114e49",
    "delphi-1e21-p67m33-9p25b-lr0.67-ecbd27",
    "delphi-1e22-p33m67-32p07b-lr0.33-e9132105",
    "delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d",
    "delphi-1e22-p33m67-32p07b-lr0.67-54770ae7",
    "delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7",
    "delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a",
    "delphi-1e22-p67m33-32p07b-lr0.67-3c17740e",
]

# 1e22 cells that died capacity-blocked before final step. Pull for prefix analysis.
PARTIAL_PREFIX = [
    "delphi-1e22-p33m67-32p07b-lr0.83-78fd44",
    "delphi-1e22-p50m50-32p07b-lr0.5-ecfa99",
    "delphi-1e22-p50m50-32p07b-lr0.67-e78260",
]

# 1e21 p33m67 lr0.83 was a fresh start at logbook end. Hash unknown. Discovered
# via project-wide regex lookup (see resolve_unknown_targets).
UNKNOWN_TARGETS = [
    ("delphi-1e21-p33m67-9p25b-lr0.83-", "1e21 p33m67 lr0.83 fresh"),
]

NAME_PATTERN = re.compile(
    r"^delphi-(?P<scale>1e2[012])-(?P<mix>p33m67|p50m50|p67m33)-"
    r"(?P<budget>4p94b|9p25b|32p07b)-lr(?P<lr>0\.33|0\.5|0\.67|0\.83)-(?P<hash>.+)$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Print actions, write nothing")
    p.add_argument("--force-refresh", action="store_true", help="Re-download even if local dump looks complete")
    p.add_argument("--only", default=None, help="Restrict to a single run id")
    p.add_argument("--no-files", action="store_true", help="Skip the files/ download (faster)")
    p.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Skip the project-wide lookup for the 1e21 p33m67 lr0.83 fresh hash",
    )
    return p.parse_args()


def expected_step_for(run_id: str) -> int | None:
    m = NAME_PATTERN.match(run_id)
    if m is None:
        return None
    return EXPECTED_FINAL_STEP.get(m["scale"])


def local_summary_step(run_dir: Path) -> int | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    step = data.get("_step")
    return int(step) if isinstance(step, (int, float)) else None


def is_local_complete(run_id: str, run_dir: Path) -> bool:
    expected = expected_step_for(run_id)
    if expected is None:
        return False
    step = local_summary_step(run_dir)
    return step is not None and step >= expected - COMPLETION_TOLERANCE


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def export_run(run, run_dir: Path, *, download_files: bool) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": run.id,
        "name": run.name,
        "entity": run.entity,
        "project": run.project,
        "path": list(run.path),
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
        "tags": list(run.tags or []),
        "group": run.group,
        "job_type": run.job_type,
        "notes": run.notes,
        "sweep": run.sweep.id if run.sweep else None,
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config.json", dict(run.config))

    summary_dict = dict(run.summary._json_dict) if hasattr(run.summary, "_json_dict") else dict(run.summary)
    write_json(run_dir / "summary.json", summary_dict)

    expected_rows = expected_step_for(run.name) or 0
    history_path = run_dir / "history.jsonl"
    keys: set[str] = set()
    row_count = 0
    with (
        history_path.open("w") as f,
        tqdm(
            total=expected_rows or None,
            desc=f"  history {run.name[:48]}",
            unit="row",
            leave=False,
            smoothing=0.05,
        ) as bar,
    ):
        for row in run.scan_history(page_size=2000):
            f.write(json.dumps(row, default=str) + "\n")
            keys.update(row.keys())
            row_count += 1
            bar.update(1)
    write_json(
        run_dir / "history_meta.json",
        {"row_count": row_count, "keys": sorted(keys)},
    )

    files_payload = []
    files = list(run.files())
    for wf in files:
        files_payload.append(
            {
                "name": wf.name,
                "size": int(wf.size),
                "md5": wf.md5,
                "url": wf.url,
            }
        )
    write_json(run_dir / "files_index.json", files_payload)

    if download_files:
        files_dir = run_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        eligible = [wf for wf in files if int(wf.size) <= MAX_FILE_BYTES]
        skipped = [wf for wf in files if int(wf.size) > MAX_FILE_BYTES]
        for wf in skipped:
            logger.info("    skip %s (size %.1fMB > %.0fMB cap)", wf.name, wf.size / 1e6, MAX_FILE_BYTES / 1e6)
        for wf in tqdm(
            eligible,
            desc=f"  files   {run.name[:48]}",
            unit="file",
            leave=False,
        ):
            try:
                wf.download(root=str(files_dir), replace=True)
            except Exception as e:
                logger.warning("    file download failed for %s: %s", wf.name, e)

    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
        "group": run.group or "",
        "job_type": run.job_type or "",
        "tags": ",".join(run.tags or []),
        "row_count": row_count,
        "final_step": summary_dict.get("_step"),
    }


def update_runs_index(new_rows: list[dict]) -> None:
    if not new_rows:
        return
    csv_path = DATA_DIR / "runs_index.csv"
    json_path = DATA_DIR / "runs_index.json"

    existing: dict[str, dict] = {}
    columns = ["id", "name", "state", "created_at", "url", "group", "job_type", "tags"]
    if csv_path.exists():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing[r["id"]] = r

    for r in new_rows:
        existing[r["id"]] = {k: r.get(k, "") for k in columns}

    rows_sorted = sorted(existing.values(), key=lambda r: r["id"])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_sorted)

    if json_path.exists():
        prev = json.loads(json_path.read_text())
    else:
        prev = []
    prev_by_id = {r.get("id"): r for r in prev if isinstance(r, dict)}
    for r in new_rows:
        prev_by_id[r["id"]] = {k: r.get(k, "") for k in columns}
    json_path.write_text(json.dumps(sorted(prev_by_id.values(), key=lambda r: r["id"]), indent=2))


def update_manifest(now_epoch: float) -> None:
    manifest_path = DATA_DIR / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"entity": ENTITY, "project": PROJECT, "project_path": PROJECT_PATH}
    manifest["last_incremental_at_epoch"] = now_epoch
    manifest["run_count"] = len(list(RUNS_DIR.glob("*")))
    manifest_path.write_text(json.dumps(manifest, indent=2))


def resolve_unknown_targets(api: wandb.Api) -> list[str]:
    resolved = []
    for prefix, label in UNKNOWN_TARGETS:
        try:
            runs = list(
                api.runs(
                    PROJECT_PATH,
                    filters={"display_name": {"$regex": f"^{re.escape(prefix)}"}},
                    per_page=20,
                )
            )
        except Exception as e:
            logger.warning("Lookup failed for %s: %s", label, e)
            continue
        if not runs:
            logger.info("No run found matching prefix %s (%s) — skipping", prefix, label)
            continue
        for r in runs:
            logger.info("Resolved %s → %s (state=%s)", label, r.name, r.state)
            resolved.append(r.name)
    return resolved


def build_target_list(api: wandb.Api, args: argparse.Namespace) -> list[str]:
    if args.only:
        return [args.only]
    targets = list(HISTORICAL_SWEEP_DONE) + list(HISTORICAL_SWEEP_INCOMPLETE) + list(PARTIAL_PREFIX)
    if not args.skip_unknown:
        targets += resolve_unknown_targets(api)
    seen, dedup = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    api = wandb.Api(timeout=60)
    targets = build_target_list(api, args)
    logger.info("Target list: %d run id(s)", len(targets))
    for t in targets:
        logger.info("  - %s", t)

    new_rows: list[dict] = []
    skipped, downloaded, missing, failed = 0, 0, 0, 0

    iterator = tqdm(
        list(enumerate(targets, 1)),
        total=len(targets),
        desc="runs",
        unit="run",
    )
    for i, run_id in iterator:
        iterator.set_postfix_str(run_id[-32:])
        run_dir = RUNS_DIR / run_id
        prefix = f"[{i}/{len(targets)}] {run_id}"

        if not args.force_refresh and is_local_complete(run_id, run_dir):
            logger.info("%s ✓ already complete (skipping)", prefix)
            skipped += 1
            continue

        if args.dry_run:
            local_step = local_summary_step(run_dir)
            expected = expected_step_for(run_id) or "?"
            logger.info("%s would download (local _step=%s/%s)", prefix, local_step, expected)
            continue

        try:
            run = api.run(f"{PROJECT_PATH}/{run_id}")
        except wandb.errors.CommError as e:
            logger.warning("%s NOT FOUND on W&B: %s", prefix, e)
            missing += 1
            continue
        except Exception as e:
            logger.error("%s lookup error: %s", prefix, e)
            failed += 1
            continue

        try:
            t0 = time.time()
            row = export_run(run, run_dir, download_files=not args.no_files)
            elapsed = time.time() - t0
            logger.info(
                "%s ✓ downloaded (%d rows, final _step=%s, %.1fs)",
                prefix,
                row["row_count"],
                row["final_step"],
                elapsed,
            )
            new_rows.append(row)
            downloaded += 1
        except Exception as e:
            logger.exception("%s export failed: %s", prefix, e)
            failed += 1

    if not args.dry_run and new_rows:
        update_runs_index(new_rows)
        update_manifest(time.time())

    logger.info(
        "Summary: downloaded=%d skipped=%d missing=%d failed=%d",
        downloaded,
        skipped,
        missing,
        failed,
    )


if __name__ == "__main__":
    main()
