# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-cloud-storage>=3.8.0"]
# ///

"""Compare East5 and Europe TPP40 evaluation payloads by object CRC32C."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from google.cloud import storage
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix.audit_delphi_tpp40_europe_runtime_caches import shard_ledger_invariants
from experiments.domain_phase_mix.delphi_tpp40_evaluation_identity import (
    table9_request_set_identity,
    validation_payload_identity,
)
from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_evaluation_caches import evaluation_steps

REGIONS = ("us-east5", "europe-west4")
REGION_LABELS = {"us-east5": "east5", "europe-west4": "europe"}
UNCHEATABLE_RAW_PREFIX = "raw/uncheatable_eval/2026.06.28/"
EXCLUDED_VALIDATION_OBJECT_SUFFIXES = ("shard_ledger.json",)
TABLE9_REQUEST_SET_RELATIVE_PATH = "raw/eval-datasets/olmo_base_eval_table9/v2"


@dataclass(frozen=True)
class ObjectMetadata:
    size: int
    crc32c: str


def _split_gcs_path(path: str) -> tuple[str, str]:
    bucket, separator, prefix = path.removeprefix("gs://").partition("/")
    if not path.startswith("gs://") or not separator or not prefix:
        raise ValueError(f"Expected a GCS path, got {path!r}")
    return bucket, prefix.rstrip("/") + "/"


def _validation_metadata(client: storage.Client, path: str) -> dict[str, ObjectMetadata]:
    bucket, prefix = _split_gcs_path(path)
    validation_prefix = prefix + "validation/"
    metadata: dict[str, ObjectMetadata] = {}
    for blob in client.list_blobs(bucket, prefix=validation_prefix):
        relative_path = blob.name.removeprefix(validation_prefix)
        if not relative_path or relative_path.endswith(EXCLUDED_VALIDATION_OBJECT_SUFFIXES):
            continue
        if blob.crc32c is None:
            raise ValueError(f"Missing CRC32C metadata for {path}/{relative_path}")
        metadata[relative_path] = ObjectMetadata(size=int(blob.size or 0), crc32c=blob.crc32c)
    if ".stats.json" not in metadata:
        raise ValueError(f"Evaluation cache lacks validation/.stats.json: {path}")
    return metadata


def _cache_paths(region: str) -> dict[str, str]:
    os.environ["MARIN_PREFIX"] = marin_prefix_for_region(region)
    paths = {name: step.path() for name, step in evaluation_steps(region=region).items()}
    required_prefix = marin_prefix_for_region(region).rstrip("/") + "/"
    if not all(path.startswith(required_prefix) for path in paths.values()):
        raise ValueError(f"Evaluation cache path escaped {region}: {paths}")
    return paths


def evaluation_paths_sha256(paths: dict[str, str]) -> str:
    payload = [{"name": name, "path": path} for name, path in sorted(paths.items())]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _root_shard_ledger(client: storage.Client, path: str) -> dict[str, object]:
    bucket, prefix = _split_gcs_path(path)
    return json.loads(client.bucket(bucket).blob(prefix + "validation/shard_ledger.json").download_as_text())


def _uncheatable_plain_hashes(client: storage.Client, region: str) -> dict[str, str]:
    bucket = marin_prefix_for_region(region).removeprefix("gs://")
    hashes: dict[str, str] = {}
    for blob in client.list_blobs(bucket, prefix=UNCHEATABLE_RAW_PREFIX):
        if not blob.name.endswith(".jsonl.gz"):
            continue
        name = blob.name.removeprefix(UNCHEATABLE_RAW_PREFIX)
        hashes[name] = hashlib.sha256(gzip.decompress(blob.download_as_bytes())).hexdigest()
    if len(hashes) != 14:
        raise ValueError(f"Expected 14 Uncheatable raw files in {region}, got {len(hashes)}")
    return hashes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = storage.Client()
    paths = {region: _cache_paths(region) for region in REGIONS}
    if paths["us-east5"].keys() != paths["europe-west4"].keys():
        raise ValueError("Regional evaluation cache names differ")

    results: list[dict[str, object]] = []
    total_objects = 0
    total_bytes = 0
    for name in sorted(paths["us-east5"]):
        regional_metadata = {region: _validation_metadata(client, paths[region][name]) for region in REGIONS}
        if regional_metadata["us-east5"] != regional_metadata["europe-west4"]:
            changed = sorted(
                set(regional_metadata["us-east5"]) ^ set(regional_metadata["europe-west4"])
                | {
                    object_name
                    for object_name in set(regional_metadata["us-east5"]) & set(regional_metadata["europe-west4"])
                    if regional_metadata["us-east5"][object_name] != regional_metadata["europe-west4"][object_name]
                }
            )
            raise ValueError(f"Evaluation payload differs for {name}: {changed[:20]}")
        ledgers = {region: _root_shard_ledger(client, paths[region][name]) for region in REGIONS}
        if shard_ledger_invariants(ledgers["us-east5"]) != shard_ledger_invariants(ledgers["europe-west4"]):
            raise ValueError(f"Evaluation shard-ledger row layout differs for {name}")
        metadata = regional_metadata["us-east5"]
        total_objects += len(metadata)
        total_bytes += sum(item.size for item in metadata.values())
        results.append(
            {
                "name": name,
                "paths": {REGION_LABELS[region]: paths[region][name] for region in REGIONS},
                "validation_objects": len(metadata),
                "validation_bytes": sum(item.size for item in metadata.values()),
                "stats": asdict(metadata[".stats.json"]),
            }
        )

    raw_hashes = {region: _uncheatable_plain_hashes(client, region) for region in REGIONS}
    if raw_hashes["us-east5"] != raw_hashes["europe-west4"]:
        raise ValueError("Decompressed Uncheatable raw files differ across regions")
    validation_identities = {
        region: validation_payload_identity(
            paths[region],
            excluded_suffixes=EXCLUDED_VALIDATION_OBJECT_SUFFIXES,
        )
        for region in REGIONS
    }
    validation_payload_sha256 = {
        REGION_LABELS[region]: identity["payload_sha256"] for region, identity in validation_identities.items()
    }
    if len(set(validation_payload_sha256.values())) != 1:
        raise ValueError(f"Tokenized validation payload identity differs across regions: {validation_payload_sha256}")
    uncheatable_validation_identities = {
        region: validation_payload_identity(
            {name: path for name, path in paths[region].items() if name.startswith("uncheatable_eval/")},
            excluded_suffixes=EXCLUDED_VALIDATION_OBJECT_SUFFIXES,
        )
        for region in REGIONS
    }
    uncheatable_validation_payload_sha256 = {
        REGION_LABELS[region]: identity["payload_sha256"]
        for region, identity in uncheatable_validation_identities.items()
    }
    if len(set(uncheatable_validation_payload_sha256.values())) != 1:
        raise ValueError(
            "English Uncheatable tokenized payload identity differs across regions: "
            f"{uncheatable_validation_payload_sha256}"
        )
    table9_request_set_dirs = {
        region: f"{marin_prefix_for_region(region).rstrip('/')}/{TABLE9_REQUEST_SET_RELATIVE_PATH}" for region in REGIONS
    }
    table9_identities = {
        region: table9_request_set_identity(request_set_dir)
        for region, request_set_dir in table9_request_set_dirs.items()
    }
    table9_payload_sha256 = {
        REGION_LABELS[region]: identity["payload_sha256"] for region, identity in table9_identities.items()
    }
    if len(set(table9_payload_sha256.values())) != 1:
        raise ValueError(f"Table-9 request-set identity differs across regions: {table9_payload_sha256}")
    report = {
        "schema_version": 2,
        "status": "evaluation_payload_equivalent",
        "evaluation_caches": len(results),
        "evaluation_paths_sha256": {REGION_LABELS[region]: evaluation_paths_sha256(paths[region]) for region in REGIONS},
        "validation_objects": total_objects,
        "validation_bytes": total_bytes,
        "excluded_non_payload_object_suffixes": list(EXCLUDED_VALIDATION_OBJECT_SUFFIXES),
        "uncheatable_raw_files": len(raw_hashes["us-east5"]),
        "uncheatable_plain_sha256": raw_hashes["us-east5"],
        "validation_payload_sha256": validation_payload_sha256,
        "uncheatable_validation_payload_sha256": uncheatable_validation_payload_sha256,
        "table9_request_set_dirs": {
            REGION_LABELS[region]: request_set_dir for region, request_set_dir in table9_request_set_dirs.items()
        },
        "table9_payload_sha256": table9_payload_sha256,
        "table9_identity": table9_identities["us-east5"],
        "caches": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"caches", "uncheatable_plain_sha256"}},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
