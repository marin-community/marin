# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cheap checks around the torch reference job: the source and inputs before the GPU load, the result after.

``source_preflight(weights, expect)`` reads no weights: only the files named in ``expect`` (config, generation config,
safetensors index, hashed files and inputs). It returns every mismatch; an empty list is a pass.
``expect`` (JSON)::

    {"files": {"config.json": SHA256, ...},          # files under ``weights``
     "config": {"qk_mult": 1.75, ...},               # config.json fields
     "eos_token_id": [128001, 128009],               # generation_config.json
     "index_keys": 502,                              # model.safetensors.index.json weight_map entries
     "inputs": [{"path": P, "sha256": S, "sequences": 465, "tokens": 30999}]}  # counts only for a positions.pt

``check_router_rounding_result(result, positions, positions_sha256, layers, k)`` checks a worker result directory:
``done.json`` and no ``error.json``, the input hash the job recorded, and that ``routes.pt`` and the totals cover
every sequence, position and layer of ``positions``. It returns every failure. A worker exit code alone is not a pass.
"""

import argparse
import hashlib
import io
import json

import torch
from rigging.filesystem.storage_path import prefix_join

from experiments.interp.snowball_identity.logit_check import read
from experiments.interp.snowball_identity.routing import VARIANTS


def _exists(path: str) -> bool:
    try:
        read(path)
    except FileNotFoundError:
        return False
    return True


def source_preflight(weights: str, expect: dict) -> list[str]:
    """Every mismatch between the source/inputs and ``expect``; empty is a pass."""
    failures = []
    for name, sha in expect.get("files", {}).items():
        actual = hashlib.sha256(read(prefix_join(weights, name))).hexdigest()
        if actual != sha:
            failures.append(f"{name}: sha256 {actual}, expected {sha}")
    config = json.loads(read(prefix_join(weights, "config.json")))
    for field, value in expect.get("config", {}).items():
        if config.get(field) != value:
            failures.append(f"config.json {field}: {config.get(field)!r}, expected {value!r}")
    if "eos_token_id" in expect:
        eos = json.loads(read(prefix_join(weights, "generation_config.json"))).get("eos_token_id")
        if eos != expect["eos_token_id"]:
            failures.append(f"generation_config.json eos_token_id: {eos!r}, expected {expect['eos_token_id']!r}")
    if "index_keys" in expect:
        keys = len(json.loads(read(prefix_join(weights, "model.safetensors.index.json")))["weight_map"])
        if keys != expect["index_keys"]:
            failures.append(f"index: {keys} tensors, expected {expect['index_keys']}")
    for entry in expect.get("inputs", []):
        data = read(entry["path"])
        actual = hashlib.sha256(data).hexdigest()
        if actual != entry["sha256"]:
            failures.append(f"{entry['path']}: sha256 {actual}, expected {entry['sha256']}")
            continue
        if "sequences" in entry or "tokens" in entry:
            stored = torch.load(io.BytesIO(data), map_location="cpu")
            counts = {"sequences": len(stored), "tokens": sum(len(r["tokens"]) for r in stored.values())}
            for name, value in counts.items():
                if name in entry and value != entry[name]:
                    failures.append(f"{entry['path']}: {value} {name}, expected {entry[name]}")
    return failures


def check_router_rounding_result(result: str, positions: str, positions_sha256: str, layers: int, k: int) -> list[str]:
    """Every reason the ``router_rounding`` result in ``result`` is not a complete run on ``positions``."""
    if _exists(prefix_join(result, "error.json")):
        return [f"error.json: {json.loads(read(prefix_join(result, 'error.json'))).get('error')}"]
    if not _exists(prefix_join(result, "done.json")):
        return ["no done.json"]
    failures = []
    report = json.loads(read(prefix_join(result, "routing.json")))
    if report.get("positions_sha256") != positions_sha256:
        failures.append(f"routing.json positions_sha256 {report.get('positions_sha256')}, expected {positions_sha256}")
    stored = torch.load(io.BytesIO(read(positions)), map_location="cpu")
    routes = torch.load(io.BytesIO(read(prefix_join(result, "routes.pt"))), map_location="cpu")
    if set(routes) != set(stored):
        failures.append(f"routes.pt has {len(routes)} sequences, positions.pt {len(stored)}")
    for sequence_id, record in stored.items():
        shape = tuple(routes[sequence_id].shape) if sequence_id in routes else None
        if shape != (len(record["tokens"]), layers, k):
            failures.append(f"{sequence_id}: routes shape {shape}, expected {(len(record['tokens']), layers, k)}")
    expected = sum(len(r["tokens"]) for r in stored.values()) * layers
    for variant in VARIANTS:
        covered = report.get("totals", {}).get(variant, {}).get("positions")
        if covered != expected:
            failures.append(f"{variant}: {covered} position-layers, expected {expected}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    source = commands.add_parser("source", help="check the source and inputs before the load")
    source.add_argument("--weights", required=True)
    source.add_argument("--expect", required=True, help="expectations JSON (s3:// or local)")
    result = commands.add_parser("result", help="check a router_rounding worker result")
    result.add_argument("--result", required=True, help="the worker's results/<id> directory")
    result.add_argument("--positions", required=True)
    result.add_argument("--positions-sha256", required=True)
    result.add_argument("--layers", type=int, default=26)
    result.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    if args.action == "source":
        failures = source_preflight(args.weights, json.loads(read(args.expect)))
    else:
        failures = check_router_rounding_result(args.result, args.positions, args.positions_sha256, args.layers, args.k)
    print(json.dumps({"failures": failures}, indent=1))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
