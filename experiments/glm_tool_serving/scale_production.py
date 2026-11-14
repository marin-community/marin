# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Feed retained source shards through conversion with durable batch cursors."""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import time
from collections import Counter
from pathlib import Path

import fsspec
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials
from tokenizers import Tokenizer

from experiments.glm_tool_serving.ortet_production import read_bulk_token, resolve_relay_base_url
from experiments.tool_normalization.acquire import acquire_shard
from experiments.tool_normalization.driver import run
from experiments.tool_normalization.preflight import classify

DESTINATION = "s3://marin-us-east-08a/marin/users/benfeuer/tool-normalization/h100-scale-20261103"


def resume_cursor(prefix: str, predecessor: str | None, catalog_hash: str) -> tuple[dict, str, str]:
    """Continue from durable progress without moving or rewriting the previous run."""
    cursor_url = prefix + "/cursor.json"
    fs, _ = filesystem_for(cursor_url)
    cursor_source = cursor_url
    read_prefix = prefix
    if not fs.exists(cursor_url) and predecessor:
        cursor_source = predecessor + "/cursor.json"
        read_prefix = predecessor
        fs, _ = filesystem_for(cursor_source)
        if not fs.exists(cursor_source):
            raise ValueError("Predecessor cursor missing; refusing to restart at row zero")
    cursor = {"shard": 0, "row": 0, "batches": 0, "artifact_bytes": 0, "counts": {}, "catalog_sha256": catalog_hash}
    if fs.exists(cursor_source):
        with fsspec.open(cursor_source) as stream:
            cursor = json.load(stream)
        if cursor["catalog_sha256"] != catalog_hash:
            raise ValueError("Resume catalog differs from the saved cursor")
    return cursor, read_prefix, cursor_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--catalog", default="scale-catalog.json")
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--predecessor", help="Read-only previous run prefix; used only when no new cursor exists")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--artifact-budget", type=int, default=250_000_000_000)
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    if not 1 <= args.replica <= args.replicas <= 8 or args.slots * args.replicas > 32:
        raise ValueError("Invalid replica ownership or aggregate admission allocation")
    configure_coreweave_s3()
    os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"] = s3_credentials(StoreType.COREWEAVE)
    os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = data_buckets()["marin-us-east-02a"].signing_region
    data = Path(__file__).resolve().parent / "production_data"
    catalog = json.loads((data / args.catalog).read_text())
    catalog_hash = hashlib.sha256((data / args.catalog).read_bytes()).hexdigest()
    recovery_path = data / f"recovery-{args.replica}.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    shards = catalog["shards"][args.replica - 1 :: args.replicas]
    skip = set(catalog["skip_ids"])
    root = Path(f"/tmp/tool-scale-r{args.replica}")
    root.mkdir(exist_ok=False)
    prefix = f"{args.destination}/r{args.replica}"
    cursor_url = prefix + "/cursor.json"
    fs, _ = filesystem_for(cursor_url)
    predecessor = f"{args.predecessor.rstrip('/')}/r{args.replica}" if args.predecessor else None
    cursor, read_prefix, cursor_source = resume_cursor(prefix, predecessor, catalog_hash)
    cfg = json.loads((data / f"config-{args.replica}.json").read_text())
    cfg["workers"] = args.slots
    if "rule_approval" in cfg:
        cfg["rule_approval"]["path"] = str(data / cfg["rule_approval"]["path"])
    for key in ("serving_tokenizer", "sft_tokenizer"):
        cfg[key] = str(data / cfg[key])
    for evidence in cfg["adapter_evidence"]:
        evidence["path"] = str(data / evidence["path"])
    cfg["model"]["generation"] = f"{args.destination.rstrip('/').rsplit('/', 1)[-1]}-r{args.replica}"
    # The dedicated per-replica serving pool is retired (four of its eight endpoints died on
    # 2026-11-13); resolve the shared glm-5.3 relay like the Ortet fleet and authenticate with
    # the sanctioned bulk token instead of the anonymous placeholder.
    token = read_bulk_token()
    cfg["model"]["base_url"] = resolve_relay_base_url(token)
    for key, filename in (
        ("api_key_file", "api-key.txt"),
        ("capacity_file", "capacity.json"),
        ("ledger", "ledger.sqlite3"),
    ):
        cfg["model"][key] = str(root / filename)
    (root / "api-key.txt").write_text(token)
    (root / "capacity.json").write_text(
        json.dumps(
            {
                "generation": cfg["model"]["generation"],
                "conversion_slots": args.slots,
                "science_slots": 0,
                "expires_at_unix": time.time() + 30 * 3600,
            }
        )
    )
    (root / "lineage.json").write_text(
        json.dumps({"cursor_source": cursor_source, "initial_cursor": cursor, "destination": prefix}, indent=2)
    )
    config_path = root / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    environment_path = Path(__file__).resolve().parents[1] / "tool_normalization/environments.json"
    packages = json.loads(environment_path.read_text())[cfg["execution"]["image"]]["packages"]
    tokenizer = Tokenizer.from_file(cfg["sft_tokenizer"])
    counts = Counter(cursor["counts"])
    deadline = time.monotonic() + args.hours * 3600
    for shard_index in range(cursor["shard"], len(shards)):
        shard = shards[shard_index]
        start = max(cursor["row"] if shard_index == cursor["shard"] else 0, shard.get("start_row", 0))
        for first in range(start, shard["rows"], 128):
            if (
                (root / "STOP").exists()
                or time.monotonic() >= deadline
                or cursor["artifact_bytes"] >= args.artifact_budget
            ):
                print(json.dumps({"phase": "allocation_complete", "cursor": cursor}), flush=True)
                # No batch is open here; avoid library teardown keeping a completed allocation alive.
                os._exit(0)
            stop = min(first + 128, shard["rows"])
            name = f"shard-{shard_index:05d}-rows-{first:08d}-{stop:08d}"
            preserved = {}
            recovered_bytes = 0
            if recovery is not None and recovery["batch"] == name:
                if cursor != recovery["cursor"]:
                    raise ValueError("Recovery cursor changed; reconcile again")
                old_archive = read_prefix + "/batches/" + name + ".tar.gz"
                with fsspec.open(old_archive, "rb") as stream:
                    archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if archive_hash != recovery["archive_sha256"]:
                    raise ValueError("Stopped archive changed; reconcile again")
                preserved = recovery["preserved"]
                recovered_bytes = recovery["archive_bytes"]
                name += "-recovery-" + archive_hash[:12]
            batch = root / name
            batch.mkdir()
            if preserved:
                (batch / "reconciliation.json").write_text(json.dumps(recovery, indent=2))
            source_rows = [
                {"source": shard["source"], "shard": shard["shard"], "row": row, "ledger_sha256": shard["ledger_sha256"]}
                for row in range(first, stop)
            ]
            # Claim before acquisition/inference; an interrupted batch is never silently replayed.
            claim_url = prefix + "/claims/" + name + ".json"
            prior_claim = predecessor + "/claims/" + name + ".json" if predecessor else claim_url
            prior_fs, _ = filesystem_for(prior_claim)
            if fs.exists(claim_url) or prior_fs.exists(prior_claim):
                archive = read_prefix + "/batches/" + name + ".tar.gz"
                if not fs.exists(archive):
                    # Killed before any upload: nothing was committed, so the claim can be
                    # cleared and the batch replayed from the cursor without reconciliation.
                    claim_data = fs.open(claim_url, "rb").read() if fs.exists(claim_url) else b"{}"
                    stale = {
                        "replica": args.replica,
                        "batch": name,
                        "claim_evidence": json.loads(claim_data),
                        "archive_absent": True,
                        "decision": "self-healed: cleared archive-less claim; replaying from the cursor",
                        "reconciled_at": time.time(),
                    }
                    with fs.open(prefix + "/reconciliations/" + name + "-selfheal.json", "wb") as stream:
                        stream.write(json.dumps(stale, indent=2).encode())
                    if fs.exists(claim_url):
                        with fs.open(prefix + "/reconciled-claims/" + name, "wb") as stream:
                            stream.write(claim_data)
                        fs.rm(claim_url)
                else:
                    raise RuntimeError(f"Uncommitted prior batch requires reconciliation: {claim_url}")
            with fsspec.open(claim_url, "w") as stream:
                json.dump({"started_at": time.time(), "source": shard, "first": first, "stop": stop}, stream)
            index = acquire_shard(source_rows, tokenizer, batch / "sources")
            selected = []
            dispositions = []
            for item in index:
                source_file = batch / "sources" / item["id"] / "source.json"
                source = json.loads(source_file.read_text())
                result = {"status": "previously_processed"} if item["id"] in skip else classify(source, packages)
                # Open-SWE needs outcome provenance before entering review; never infer success.
                if item["source"].startswith("open_swe_traces/"):
                    result = {"status": "requires_resolved_outcome_provenance"}
                dispositions.append({**item, **result})
                counts[result["status"]] += 1
                if item["id"] in preserved:
                    counts["conversion_" + preserved[item["id"]]] += 1
                    shutil.rmtree(source_file.parent)
                elif result["status"] in {"mapped_calls", "glm_mapping_residue"}:
                    if preserved and item["id"] not in recovery["eligible_unsubmitted_ids"]:
                        raise ValueError("Recovery selected a row without an unsubmitted receipt check")
                    selected.append({"id": item["id"], "source_file": str(source_file)})
                else:
                    shutil.rmtree(source_file.parent)
            (batch / "dispositions.json").write_text(json.dumps(dispositions))
            manifest = batch / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
            if selected:
                run(config_path, manifest, batch / "run")
                summary = json.loads((batch / "run/summary.json").read_text())
                counts.update({"conversion_" + key: value for key, value in summary.items()})
            occupied = 0
            if Path(cfg["model"]["ledger"]).exists():
                with sqlite3.connect(cfg["model"]["ledger"]) as ledger:
                    occupied = ledger.execute(
                        "SELECT count(*) FROM requests WHERE state IN ('active','uncertain')"
                    ).fetchone()[0]
                    snapshot = sqlite3.connect(batch / "ledger.sqlite3")
                    try:
                        ledger.backup(snapshot)
                    finally:
                        snapshot.close()
            archive = root / (name + ".tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(batch, arcname=name)
                tar.add(config_path, arcname="config.json")
                tar.add(root / "lineage.json", arcname="lineage.json")
            with archive.open("rb") as source, fsspec.open(prefix + "/batches/" + archive.name, "wb") as target:
                shutil.copyfileobj(source, target)
            if occupied or (selected and (summary.get("pending", 0) or summary.get("capacity_deferred", 0))):
                raise RuntimeError("Saved stopped batch requires admission reconciliation; cursor not advanced")
            cursor.update(
                shard=shard_index,
                row=stop,
                batches=cursor["batches"] + 1,
                artifact_bytes=cursor["artifact_bytes"] + archive.stat().st_size + recovered_bytes,
                counts=dict(counts),
                updated_at=time.time(),
            )
            with fsspec.open(cursor_url, "w") as stream:
                json.dump(cursor, stream)
            print(json.dumps({"phase": "batch_committed", "name": name, "cursor": cursor}), flush=True)
            shutil.rmtree(batch)
            archive.unlink()
    print(json.dumps({"phase": "catalog_complete", "cursor": cursor}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
