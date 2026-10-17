# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble admitted real tasks and disjoint structural tasks without changing payloads."""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.spec import LoweredTaskSpec


def build_mixture(
    real: StoragePath,
    structural: Path,
    output: StoragePath,
    *,
    source_routes: Mapping[str, str],
    domain_weights: Mapping[str, float],
) -> dict:
    """Validate source manifests and publish both splits with sampling-domain labels."""
    if output.exists():
        raise FileExistsError(f"Mixture output must be new: {output}")
    real_manifest_bytes = (real / "manifest.json").read_bytes()
    structural_manifest_bytes = (structural / "manifest.json").read_bytes()
    real_manifest = json.loads(real_manifest_bytes)
    structural_manifest = json.loads(structural_manifest_bytes)
    if real_manifest["failures"]:
        raise ValueError("Real-task conversion contains failures")
    sources = real_manifest["sources"]
    if set(source_routes) != {source["root"] for source in sources}:
        raise ValueError("Route selection must name every admitted real source exactly once")
    routes_by_id = {}
    for source in sources:
        route = source_routes[source["root"]]
        if not route or route == "structural":
            raise ValueError(f"Invalid real-task route: {route!r}")
        for split in ("train", "validation"):
            for task_id in source[f"{split}_ids"]:
                if task_id in routes_by_id:
                    raise ValueError(f"Duplicate source task identity: {task_id}")
                routes_by_id[task_id] = route
    if set(domain_weights) != set(source_routes.values()) | {"structural"}:
        raise ValueError("Sampling weights must match the real and structural routes")
    if any(not math.isfinite(weight) or weight <= 0 for weight in domain_weights.values()):
        raise ValueError("Sampling weights must be positive finite numbers")
    tables = {}
    all_ids = set()
    evidence = {}
    for split in ("train", "validation"):
        payloads = []
        routes = []
        evidence[split] = {}
        for route, path in (("real", real / f"{split}.parquet"), ("structural", structural / f"{split}.parquet")):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                stream.seek(0)
                rows = pq.read_table(stream)["lowered_task_spec"].to_pylist()
            ids = [LoweredTaskSpec.model_validate_json(row).task.id for row in rows]
            if not rows or len(set(ids)) != len(ids) or all_ids.intersection(ids):
                raise ValueError(f"Empty or overlapping task identities in {route}/{split}")
            if route == "real":
                if set(ids) != set(real_manifest[f"{split}_ids"]):
                    raise ValueError(f"Real {split} identities differ from the manifest")
            else:
                expected = structural_manifest["outputs"][split]
                if digest != expected["native_sha256"] or len(rows) != expected["rows"]:
                    raise ValueError(f"Structural {split} differs from the disjoint manifest")
            all_ids.update(ids)
            payloads.extend(rows)
            routes.extend([routes_by_id[task_id] for task_id in ids] if route == "real" else [route] * len(rows))
            evidence[split][route] = {"rows": len(rows), "sha256": digest}
        if set(routes_by_id) != set(real_manifest["train_ids"]) | set(real_manifest["validation_ids"]):
            raise ValueError("Source task identities differ from the real-task manifest")
        tables[split] = pa.table({"lowered_task_spec": payloads, "teacher_route": routes})
        evidence[split]["routes"] = dict(sorted(Counter(routes).items()))
    report = {
        "real_source": str(real),
        "real_manifest_sha256": hashlib.sha256(real_manifest_bytes).hexdigest(),
        "structural_manifest_sha256": hashlib.sha256(structural_manifest_bytes).hexdigest(),
        "inputs": evidence,
        "sampling": {
            "kind": "domain-weighted",
            "source_routes": dict(source_routes),
            "domain_weights": dict(domain_weights),
        },
    }
    output.mkdirs(exist_ok=False)
    for split, table in tables.items():
        with (output / f"{split}.parquet").open("wb") as stream:
            pq.write_table(table, stream)
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


@click.command()
@click.option("--real", required=True)
@click.option("--structural-repo", required=True)
@click.option("--structural-revision", required=True)
@click.option("--output", required=True)
@click.option("--routing", type=click.Path(exists=True, path_type=Path), required=True)
def main(real: str, structural_repo: str, structural_revision: str, output: str, routing: Path) -> None:
    paths = [
        Path(hf_hub_download(structural_repo, f"structural/{name}", repo_type="dataset", revision=structural_revision))
        for name in ("train.parquet", "validation.parquet", "manifest.json")
    ]
    routes = json.loads(routing.read_text())
    report = build_mixture(
        StoragePath(real),
        paths[0].parent,
        StoragePath(output),
        source_routes=routes["source_routes"],
        domain_weights=routes["domain_weights"],
    )
    click.echo(json.dumps(report))


if __name__ == "__main__":
    main()
