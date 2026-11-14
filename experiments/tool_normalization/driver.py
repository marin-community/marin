# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a frozen source-file manifest with durable dispositions and bounded workers."""

import argparse
import fcntl
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from experiments.tool_normalization.model import CapacityUnavailable, ModelConfig, initialize_ledger
from experiments.tool_normalization.review_real import digest, process
from experiments.tool_normalization.verify import verify_adapter


def run(config_path: Path, manifest_path: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(config_path, manifest_path, root)


def run_locked(config_path: Path, manifest_path: Path, root: Path) -> None:
    started = time.time()
    cfg = json.loads(config_path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != cfg:
        raise ValueError("Existing run has a different configuration")
    frozen.write_text(json.dumps(cfg, indent=2))
    frozen_manifest = root / "sources.jsonl"
    if frozen_manifest.exists() and frozen_manifest.read_bytes() != manifest_path.read_bytes():
        raise ValueError("Existing run has a different source manifest")
    frozen_manifest.write_bytes(manifest_path.read_bytes())
    budget = cfg["artifact_budget_bytes"]
    reservation = cfg["max_record_bytes"]
    if not 0 < reservation <= budget <= 1_000_000_000_000:
        raise ValueError("Invalid artifact budget")
    model_cfg = cfg["model"].copy()
    for key in ("api_key_file", "capacity_file", "ledger"):
        model_cfg[key] = Path(model_cfg[key])
    model = ModelConfig(**model_cfg)
    initialize_ledger(model)
    allocation = json.loads(model.capacity_file.read_text())
    if not 1 <= cfg["workers"] <= allocation["conversion_slots"]:
        raise ValueError("Worker count exceeds allocated capacity")
    for evidence in cfg["adapter_evidence"]:
        if digest(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("Adapter evidence changed")
    tokenizer = AutoTokenizer.from_pretrained(cfg["serving_tokenizer"], local_files_only=True)
    queue = root / "queue.sqlite3"
    with sqlite3.connect(queue) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS rows "
            "(id TEXT PRIMARY KEY, source_file TEXT, status TEXT, detail TEXT, bytes INTEGER)"
        )
        db.execute("UPDATE rows SET status='interrupted_not_retried' WHERE status='running'")
        for line in manifest_path.read_text().splitlines():
            row = json.loads(line)
            existing = db.execute("SELECT source_file FROM rows WHERE id=?", (row["id"],)).fetchone()
            if existing and existing[0] != row["source_file"]:
                raise ValueError("Source identity changed in resumed manifest")
            db.execute("INSERT OR IGNORE INTO rows VALUES (?,?,'pending','',0)", (row["id"], row["source_file"]))

    stop_dispatch = threading.Event()

    def worker() -> None:
        while not stop_dispatch.is_set() and not (root / "STOP").exists():
            with sqlite3.connect(queue, timeout=60) as db:
                db.execute("BEGIN IMMEDIATE")
                used, active = db.execute(
                    "SELECT coalesce(sum(bytes),0),coalesce(sum(status='running'),0) FROM rows"
                ).fetchone()
                if used + (active + 1) * reservation > budget:
                    return
                row = db.execute("SELECT id,source_file FROM rows WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    return
                db.execute("UPDATE rows SET status='running' WHERE id=?", (row[0],))
            try:
                result = process(
                    Path(row[1]),
                    root / "records" / row[0],
                    model,
                    tokenizer,
                    cfg,
                )
                if result["status"] in {
                    "converted_adapter_checked",
                    "rule_converted_adapter_checked",
                    "header_normalized_no_calls",
                }:
                    result["verification"] = verify_adapter(root / "records" / row[0], cfg)
            except CapacityUnavailable as error:
                stop_dispatch.set()
                result = {"status": "capacity_deferred", "reason": str(error)}
            except Exception as error:
                # The durable row remains visible; interrupted POSTs are never replayed automatically.
                status = "length_deferred" if str(error).startswith("length_deferred:") else "infrastructure_deferred"
                result = {"status": status, "reason": f"{type(error).__name__}: {error}"}
            record = root / "records" / row[0]
            record.mkdir(parents=True, exist_ok=True)
            (record / "result.json").write_text(json.dumps(result, indent=2))
            with sqlite3.connect(queue, timeout=60) as db:
                size = sum(path.stat().st_size for path in (root / "records" / row[0]).rglob("*") if path.is_file())
                db.execute(
                    "UPDATE rows SET status=?,detail=?,bytes=? WHERE id=?",
                    (result["status"], json.dumps(result), size, row[0]),
                )

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(worker) for _ in range(cfg["workers"])]
        for future in futures:
            future.result()
    with sqlite3.connect(queue) as db:
        counts = dict(db.execute("SELECT status,count(*) FROM rows GROUP BY status"))
    timing_path = root / "timing.json"
    first_start = json.loads(timing_path.read_text())["started_at_unix"] if timing_path.exists() else started
    timing_path.write_text(json.dumps({"started_at_unix": first_start, "completed_at_unix": time.time()}, indent=2))
    (root / "summary.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.manifest, args.root)


if __name__ == "__main__":
    main()
