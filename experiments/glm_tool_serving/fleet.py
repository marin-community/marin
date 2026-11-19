# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Feed retained source shards through conversion with durable batch cursors."""

import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fsspec
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials
from tokenizers import Tokenizer

from experiments.glm_tool_serving.assets import provision_tokenizers
from experiments.glm_tool_serving.catalog_loader import load_catalog
from experiments.glm_tool_serving.provenance import emit_receipt, write_provenance
from experiments.glm_tool_serving.route_gate import gate_result, load_route_gate
from experiments.tool_normalization.acquire import acquire_shard
from experiments.tool_normalization.driver import run
from experiments.tool_normalization.preflight import classify
from experiments.tool_normalization.relay import RELAY_JOB, read_bulk_token, resolve_relay_base_url
from experiments.tool_normalization.scope import terminus_source


def resume_cursor(prefix: str, predecessor: str | None, catalog_hash: str) -> tuple[dict, str]:
    """Continue from durable progress without moving or rewriting the previous run."""
    cursor_url = prefix + "/cursor.json"
    fs, _ = filesystem_for(cursor_url)
    cursor_source = cursor_url
    if not fs.exists(cursor_url) and predecessor:
        cursor_source = predecessor + "/cursor.json"
        fs, _ = filesystem_for(cursor_source)
        if not fs.exists(cursor_source):
            raise ValueError("Predecessor cursor missing; refusing to restart at row zero")
    cursor = {"shard": 0, "row": 0, "batches": 0, "artifact_bytes": 0, "counts": {}, "catalog_sha256": catalog_hash}
    if fs.exists(cursor_source):
        with fsspec.open(cursor_source) as stream:
            cursor = json.load(stream)
        if cursor["catalog_sha256"] != catalog_hash:
            raise ValueError("Resume catalog differs from the saved cursor")
    return cursor, cursor_source


@dataclass(frozen=True)
class FleetConfig:
    replica: int
    replicas: int
    catalog: str
    destination: str
    predecessor: str | None
    slots: int
    artifact_budget: int
    hours: float
    root: Path
    config_file: str
    recovery_file: str | None
    generation: str
    capacity_hours: float
    tail_catalog: str | None


def assigned_shards(catalog: dict, catalog_hash: str, replica: int, replicas: int, tail: dict | None) -> list[dict]:
    """Keep cursor indices stable while stopping before the tail fleet's ownership."""
    shards = catalog["shards"]
    if tail is not None:
        migration = tail["migration"]
        if catalog_hash == migration["parent_catalog_sha256"]:
            boundary = migration["boundary_catalog_index"]
        else:
            tail_keys = {(s["source"], s["shard"]) for s in tail["shards"]}
            boundary = next((i for i, s in enumerate(shards) if (s["source"], s["shard"]) in tail_keys), len(shards))
        shards = shards[:boundary]
    return shards[replica - 1 :: replicas]


def reconcile_claims(prefix: str, predecessor: str | None, name: str, replica: int) -> None:
    """Refuse replay if either claim owns an archive; preserve archive-less claim evidence."""
    claims = []
    for owner in dict.fromkeys(p for p in (prefix, predecessor) if p is not None):
        claim = f"{owner}/claims/{name}.json"
        fs, _ = filesystem_for(claim)
        if not fs.exists(claim):
            continue
        if fs.exists(f"{owner}/batches/{name}.tar.gz"):
            raise RuntimeError(f"Uncommitted prior batch requires reconciliation: {claim}")
        with fs.open(claim, "rb") as stream:
            claims.append((owner, claim, stream.read()))
    for owner, claim, raw in claims:
        evidence = {
            "replica": replica,
            "batch": name,
            "claim_url": claim,
            "claim_evidence": json.loads(raw),
            "archive_absent": True,
            "decision": "self-healed: replaying archive-less claim from the cursor",
            "reconciled_at": time.time(),
        }
        tag = "current" if owner == prefix else "predecessor"
        with fsspec.open(f"{prefix}/reconciliations/{name}-{tag}-selfheal.json", "w") as stream:
            json.dump(evidence, stream, indent=2)
        with fsspec.open(f"{prefix}/reconciled-claims/{name}-{tag}", "wb") as stream:
            stream.write(raw)
        if owner == prefix:
            fs, _ = filesystem_for(claim)
            fs.rm(claim)


def recovery_archive(prefix: str, predecessor: str | None, name: str, expected_sha256: str) -> str:
    """Locate the explicitly approved stopped archive, including the new run's first batch."""
    matches = []
    for owner in dict.fromkeys(p for p in (prefix, predecessor) if p is not None):
        archive = f"{owner}/batches/{name}.tar.gz"
        fs, _ = filesystem_for(archive)
        if fs.exists(archive):
            with fs.open(archive, "rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
                    raise ValueError("Stopped archive changed; reconcile again")
            matches.append(archive)
    if not matches:
        raise ValueError("Stopped archive missing; reconcile again")
    return matches[0]


def run_fleet(config: FleetConfig) -> None:
    configure_coreweave_s3()
    os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"] = s3_credentials(StoreType.COREWEAVE)
    os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = data_buckets()["marin-us-east-02a"].signing_region
    data = Path(__file__).resolve().parent / "production_data"
    catalog, catalog_hash = load_catalog(data, config.catalog)
    recovery_path = data / config.recovery_file if config.recovery_file else None
    recovery = json.loads(recovery_path.read_text()) if recovery_path and recovery_path.exists() else None
    tail = load_catalog(data, config.tail_catalog)[0] if config.tail_catalog else None
    shards = assigned_shards(catalog, catalog_hash, config.replica, config.replicas, tail)
    skip = set(catalog["skip_ids"])
    root = config.root
    root.mkdir(exist_ok=False)
    prefix = f"{config.destination}/r{config.replica}"
    cursor_url = prefix + "/cursor.json"
    predecessor = f"{config.predecessor.rstrip('/')}/r{config.replica}" if config.predecessor else None
    cursor, cursor_source = resume_cursor(prefix, predecessor, catalog_hash)
    cfg = json.loads((data / config.config_file).read_text())
    cfg["workers"] = config.slots
    if "rule_approval" in cfg:
        cfg["rule_approval"]["path"] = str(data / cfg["rule_approval"]["path"])
    provision_tokenizers(data)
    for key in ("serving_tokenizer", "sft_tokenizer"):
        cfg[key] = str(data / cfg[key])
    for evidence in cfg["adapter_evidence"]:
        evidence["path"] = str(data / evidence["path"])
    cfg["model"]["generation"] = config.generation
    cfg["model"]["relay_job"] = RELAY_JOB
    cfg["model"]["deadline_unix"] = time.time() + config.hours * 3600
    cfg["model"]["stop_file"] = str(root / "STOP")
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
                "conversion_slots": config.slots,
                "science_slots": 0,
                "expires_at_unix": time.time() + config.capacity_hours * 3600,
            }
        )
    )
    (root / "lineage.json").write_text(
        json.dumps({"cursor_source": cursor_source, "initial_cursor": cursor, "destination": prefix}, indent=2)
    )
    config_path = root / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    provenance = write_provenance(Path(__file__).resolve().parents[2], root, catalog_hash)
    route_gate = load_route_gate(root)
    environment_path = Path(__file__).resolve().parents[1] / "tool_normalization/environments.json"
    packages = json.loads(environment_path.read_text())[cfg["execution"]["image"]]["packages"]
    tokenizer = Tokenizer.from_file(cfg["sft_tokenizer"])
    counts = Counter(cursor["counts"])
    deadline = cfg["model"]["deadline_unix"]
    for shard_index in range(cursor["shard"], len(shards)):
        shard = shards[shard_index]
        start = max(cursor["row"] if shard_index == cursor["shard"] else 0, shard.get("start_row", 0))
        for first in range(start, shard["rows"], 128):
            if (root / "STOP").exists() or time.time() >= deadline or cursor["artifact_bytes"] >= config.artifact_budget:
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
                recovery_archive(prefix, predecessor, name, recovery["archive_sha256"])
                archive_hash = recovery["archive_sha256"]
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
            reconcile_claims(prefix, predecessor, name, config.replica)
            with fsspec.open(claim_url, "w") as stream:
                json.dump({"started_at": time.time(), "source": shard, "first": first, "stop": stop}, stream)
            index = acquire_shard(source_rows, tokenizer, batch / "sources")
            selected = []
            dispositions = []
            terminus = terminus_source(shard.get("source", ""))
            for item in index:
                source_file = batch / "sources" / item["id"] / "source.json"
                source = json.loads(source_file.read_text())
                gated = gate_result(route_gate, item)
                if terminus:
                    result = {
                        "status": "out_of_scope_terminus",
                        "reason": (
                            "terminus-2-class terminal-JSON trace; out of scope per the 2026-11-15 campaign refocus"
                        ),
                    }
                elif gated is not None:
                    # Deterministic replacement exists for this exact row (shard+row+input
                    # hash verified); the row converts on the CPU path, not here.
                    result = gated
                else:
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
                tar.add(root / "provenance.json", arcname="provenance.json")
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
            emit_receipt(prefix, name, archive, catalog_hash, cursor, provenance)
            shutil.rmtree(batch)
            archive.unlink()
    print(json.dumps({"phase": "catalog_complete", "cursor": cursor}), flush=True)
    os._exit(0)
