# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded original JSONL component sampling for the pinned Nemotron Ultra blends."""

import hashlib
import importlib
import json
import random
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import requests
from taskcompendium.pipeline.datasets.nemotron_ultra_catalog import NEMOTRON_MODULES

from experiments.post_training.task_curation_nemotron_placeholders import placeholder_sources
from experiments.post_training.task_curation_prefix_sampling import (
    READ_BLOCK_BYTES,
    BudgetedHfFileSystem,
    TransferBudget,
)

DATASET = "nvidia/Nemotron-RL-Ultra-Training-Blends"
REVISION = "482392c14c6418e26804ea2e5d10359df9877df4"


def component_catalog() -> dict:
    result = {}
    for name in NEMOTRON_MODULES:
        leaf = importlib.import_module("taskcompendium.pipeline.datasets." + name)
        result[name] = {
            "atlas_id": leaf.ATLAS_ID,
            "blend": leaf.BLEND,
            "selector": leaf.SELECTOR,
            "family": leaf.FAMILY,
            "component": leaf.COMPONENT,
            "upstream": leaf.UPSTREAM,
        }
    return result


COMPONENTS = component_catalog()


def row_selector(row: dict) -> str:
    return row.get("dataset") or "agent:" + row["agent_ref"]["name"]


def matches_component(row: dict, component: dict, swe_gym_ids: frozenset[str]) -> bool:
    if row_selector(row) != component["selector"]:
        return False
    if component["family"] != "swe-repo":
        return True
    instance = row["metadata"]["instance_id"]
    belongs_to_gym = instance in swe_gym_ids
    return belongs_to_gym == component["component"].endswith("/SWE-Gym/SWE-Gym")


def response_chunks(response: requests.Response, maximum: int) -> Iterator[bytes]:
    """Read decoded payload bytes without consuming beyond the caller's remaining budget."""
    remaining = maximum
    while remaining > 0:
        chunk = response.raw.read(min(READ_BLOCK_BYTES, remaining), decode_content=True)
        if not chunk:
            return
        remaining -= len(chunk)
        yield chunk


def range_bytes(url: str, start: int, end: int) -> tuple[bytes, int]:
    """Validate Range headers before reading a bounded response body."""
    with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=60) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError("Server ignored bounded Range request")
        interval, total = response.headers["Content-Range"].removeprefix("bytes ").split("/")
        if interval != f"{start}-{end}":
            raise ValueError("Server returned a different byte interval than requested")
        expected = end - start + 1
        data = b"".join(response_chunks(response, expected))
        if len(data) != expected:
            raise ValueError("Server returned a truncated Range response")
        return data, int(total)


def sample_blend(
    blend: str,
    output: Path,
    count: int,
    seed: int,
    *,
    max_transfer_bytes: int = 64 * 1024 * 1024,
    max_range_bytes: int = 64 * 1024 * 1024,
    swe_gym_inventory: Path | None = None,
) -> dict:
    """Persist bounded original component probes with exact row or byte positions.

    A curriculum prefix is supplemented by seeded stratified byte windows. This
    is a smoke sample, not a population-uniform sample. Incomplete components
    and unresolved SWE subcorpus membership are explicit in their manifests.
    """
    selected = {name: item for name, item in COMPONENTS.items() if item["blend"] == blend}
    inventory = frozenset(swe_gym_inventory.read_text().splitlines()) if swe_gym_inventory else frozenset()
    retained = {name: [] for name in selected}
    seen_offsets = {name: set() for name in selected}

    def retain(row: dict, position: dict) -> None:
        for name, component in selected.items():
            if len(retained[name]) >= count:
                continue
            if component["family"] == "swe-repo" and swe_gym_inventory is None:
                continue
            if not matches_component(row, component, inventory):
                continue
            offset = position["source_byte_offset"]
            if offset in seen_offsets[name]:
                continue
            seen_offsets[name].add(offset)
            retained[name].append({**row, **position, "path": f"{name}/byte-{offset}"})

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{blend}.jsonl"
    transferred = 0
    rows_scanned = 0
    size = None
    buffer = b""
    source_offset = 0
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        if response.headers.get("Content-Length"):
            size = int(response.headers["Content-Length"])
        for chunk in response_chunks(response, max_transfer_bytes):
            transferred += len(chunk)
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    retain(json.loads(line), {"sample_index": rows_scanned, "source_byte_offset": source_offset})
                    rows_scanned += 1
                source_offset += len(line) + 1
            if all(len(rows) >= count for rows in retained.values()):
                break
        else:
            if buffer.strip() and (transferred < max_transfer_bytes or transferred == size):
                retain(json.loads(buffer), {"sample_index": rows_scanned, "source_byte_offset": source_offset})
                rows_scanned += 1
    windows = []
    size_probe_bytes = 0
    if any(len(rows) < count for rows in retained.values()) and max_range_bytes:
        if size is None:
            probe, size = range_bytes(url, 0, 0)
            size_probe_bytes = len(probe)
        generator = random.Random(f"{seed}:{blend}")
        window_count = 16
        width = (max_range_bytes - size_probe_bytes) // window_count
        if width <= 0:
            raise ValueError("Range budget cannot hold the requested byte windows")
        for section in range(window_count):
            start = int((section + generator.random()) * size / window_count)
            end = min(size - 1, start + width - 1)
            data, observed_size = range_bytes(url, start, end)
            if observed_size != size:
                raise ValueError("Pinned blend size changed between requests")
            windows.append({"start": start, "end": end, "bytes": len(data)})
            lines = data.splitlines(keepends=True)
            offset = start + len(lines[0])
            # Boundary fragments are not complete source records.
            for line in lines[1:-1]:
                retain(json.loads(line), {"source_byte_offset": offset})
                offset += len(line)
            if all(len(rows) >= count for rows in retained.values()):
                break
    manifests = {}
    for name, component in selected.items():
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        rows, placeholder_manifest = placeholder_sources(retained[name], output / "placeholder-source-rows")
        serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        snapshot = directory / "sample.jsonl"
        snapshot.write_text(serialized)
        unresolved = component["family"] == "swe-repo" and swe_gym_inventory is None
        manifest = {
            **component,
            "name": name,
            "dataset": DATASET,
            "revision": REVISION,
            "config": f"{blend}/{component['component']}",
            "hf_config": blend,
            "split": "train",
            "seed": seed,
            "sample_rows": len(rows),
            "sampling": "Curriculum prefix then seeded stratified byte windows; not population-uniform",
            "snapshot_path": str(snapshot),
            "snapshot_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "prefix_rows_scanned": rows_scanned,
            "prefix_bytes": transferred,
            "prefix_budget_bytes": max_transfer_bytes,
            "range_windows": windows,
            "range_budget_bytes": max_range_bytes,
            "range_bytes": size_probe_bytes + sum(w["bytes"] for w in windows),
            "range_size_probe_bytes": size_probe_bytes,
            "selector_attribution": "requires SWE-Gym membership" if unresolved else "verified source selection",
            "swe_gym_inventory": str(swe_gym_inventory) if component["family"] == "swe-repo" else None,
            "swe_attribution_basis": (
                "Exact SWE-Gym membership; complement attributed to SWE-rebench only under "
                "the pinned Ultra dataset card exhaustive two-source composition"
                if component["family"] == "swe-repo"
                else None
            ),
            "complete_probe": len(rows) == count and not unresolved,
            "intended_use": "train",
            "placeholder_acquisition": placeholder_manifest,
        }
        (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifests[name] = manifest
    return manifests


def swe_gym_inventory(output: Path) -> Path:
    """Acquire only the pinned instance-id column needed for component attribution."""
    directory = output / "swe-membership"
    directory.mkdir(parents=True, exist_ok=True)
    inventory = directory / "instance-ids.txt"
    manifest_path = directory / "manifest.json"
    pin = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
    if inventory.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] == pin and hashlib.sha256(inventory.read_bytes()).hexdigest() == manifest["sha256"]:
            return inventory
    budget = TransferBudget(8 * 1024 * 1024)
    filesystem = BudgetedHfFileSystem(budget=budget)
    shard = "data/train-00000-of-00001.parquet"
    with filesystem.open(f"datasets/SWE-Gym/SWE-Gym@{pin}/{shard}", cache_type="none") as stream:
        table = pq.ParquetFile(stream).read(columns=["instance_id"])
    serialized = "\n".join(sorted(row["instance_id"] for row in table.to_pylist())) + "\n"
    inventory.write_text(serialized)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "SWE-Gym/SWE-Gym",
                "revision": pin,
                "shard": shard,
                "columns": ["instance_id"],
                "rows": table.num_rows,
                "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "transferred_bytes": budget.transferred,
            },
            indent=2,
        )
        + "\n"
    )
    return inventory


def sample_source(name: str, output: Path, count: int, seed: int) -> dict:
    manifest_path = output / name / "sample-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        snapshot = output / name / "sample.jsonl"
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if (
            manifest["revision"] == REVISION
            and manifest["sample_rows"] == count
            and manifest["seed"] == seed
            and manifest["snapshot_sha256"] == digest
            and manifest["complete_probe"]
        ):
            return manifest
    manifest = sample_blend(COMPONENTS[name]["blend"], output, count, seed, swe_gym_inventory=swe_gym_inventory(output))[
        name
    ]
    if not manifest["complete_probe"]:
        raise ValueError(
            f"Incomplete bounded component probe for {name}: {manifest['sample_rows']}/{count}; "
            f"{manifest['selector_attribution']}"
        )
    return manifest
