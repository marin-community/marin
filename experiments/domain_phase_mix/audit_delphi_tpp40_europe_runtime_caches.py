# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-cloud-storage>=3.8.0"]
# ///

"""Compare every Europe TPP40 runtime cache with its frozen East5 source."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from google.cloud import storage
from levanter.store.cache import CACHE_LAYOUT_CONSOLIDATED, CACHE_LAYOUT_SHARDED
from marin.execution.artifact import read_record

from experiments.domain_phase_mix.compare_delphi_tpp40_runtime_digests import (
    ACCEPTANCE_MODE,
    compare_digest_reports,
    read_digest_artifact,
)
from experiments.domain_phase_mix.delphi_tpp40_europe_runtime_caches import (
    EUROPE_HISTORICAL_STACK_INPUT_PREFIX,
    EUROPE_HISTORICAL_STACK_MERGED_PREFIX,
    EUROPE_SOURCE_RUNTIME_CACHE_PATHS,
    EXPECTED_STACK_ELEMENTS,
    EXPECTED_STACK_TOKENS,
)
from experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache import (
    RuntimeObjectManifest,
    runtime_object_manifest,
    runtime_object_manifest_binding,
)
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    DOLMA3_AVAILABLE_TOKEN_COUNTS,
    TOP_LEVEL_DOMAIN_PARTITIONS,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)
from experiments.domain_phase_mix.launch_delphi_augmented_swarm_tpp40 import (
    _runtime_cache_paths,
    _runtime_paths_sha256,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DOMAIN_NAMES,
    PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION,
    PREFERRED_MERGED_RUNTIME_DOMAIN_NAMES,
    _source_tokenized_runtime_cache_path,
)

EAST5_PREFIX = "gs://marin-us-east5"
EUROPE_PREFIX = "gs://marin-eu-west4"
STACK_COMPONENT = "dolma3_stack_edu"
EAST5_STACK_INPUT_PREFIX = f"{EAST5_PREFIX}/tokenized/dolma3_pool/"
NON_PAYLOAD_TRAIN_OBJECTS = frozenset({"shard_ledger.json", "shard_ledger.json.bak"})
NON_PAYLOAD_TRAIN_PREFIXES = ("___temp/",)
NON_PAYLOAD_TRAIN_TOP_LEVEL_SUFFIXES = (".tmp",)
SHARD_LEDGER_LOGICAL_INVARIANT_FIELDS = ("total_num_rows", "is_finished")
SHARD_LEDGER_LAYOUT_INVARIANT_FIELDS = ("shard_rows",)
SHARD_LEDGER_INVARIANT_FIELDS = SHARD_LEDGER_LOGICAL_INVARIANT_FIELDS + SHARD_LEDGER_LAYOUT_INVARIANT_FIELDS
DIGEST_ARTIFACT_PREFIXES = {
    "east5": f"{EAST5_PREFIX}/experiments/domain_phase_mix/delphi_tpp40_multiregion_runtime_digests_v4/",
    "europe": f"{EUROPE_PREFIX}/experiments/domain_phase_mix/delphi_tpp40_multiregion_runtime_digests_v4/",
}
DIGEST_ARTIFACT_PATH_OVERRIDES = {
    ("europe", "dolmino_stem_heavy_crawl"): (
        f"{EUROPE_PREFIX}/experiments/domain_phase_mix/"
        "delphi_tpp40_multiregion_runtime_digests_v4_stem_metadata_repair/"
        "dolmino_stem_heavy_crawl.json"
    ),
}
DIGEST_VERIFIED_COMPONENTS = frozenset(
    {
        "finemath_3plus",
        "dolmino_stem_heavy_crawl",
        "synth_instruction/dolmino_flan",
        "synth_math/dolmino_math",
        "synth_math/verifiable_o4mini",
        "synth_qa/wiki_to_rcqa",
        "synth_thinking/code_meta_reasoning",
        "synth_thinking/math_meta_reasoning",
        "synth_thinking/program_verifiable",
    }
)
DIGEST_EXPECTED_COUNTS = {
    "finemath_3plus": (21_405_610, 34_001_855_418),
    "dolmino_stem_heavy_crawl": (5_160_830, 5_213_753_236),
    "synth_instruction/dolmino_flan": (56_099_440, 16_442_404_921),
    "synth_math/dolmino_math": (20_961_626, 10_708_619_773),
    "synth_math/verifiable_o4mini": (173_680, 73_921_022),
    "synth_qa/wiki_to_rcqa": (22_340_366, 4_254_057_981),
    "synth_thinking/code_meta_reasoning": (910_921, 1_267_452_019),
    "synth_thinking/math_meta_reasoning": (984_610, 1_051_507_567),
    "synth_thinking/program_verifiable": (273_431, 391_614_940),
}
assert set(DIGEST_EXPECTED_COUNTS) == DIGEST_VERIFIED_COMPONENTS
# The frozen East5 runtime cache contains 163 tokens beyond the declared source count.
# Payload equivalence follows the bytes consumed by the historical run; mixture weights
# continue to use the declared availability table.
assert DOLMA3_AVAILABLE_TOKEN_COUNTS["finemath_3plus"] == 34_001_855_255
assert DIGEST_EXPECTED_COUNTS["finemath_3plus"][1] == 34_001_855_418


@dataclass(frozen=True)
class CachePair:
    domain: str
    component: str
    east5_path: str
    europe_path: str


@dataclass(frozen=True)
class ObjectMetadata:
    size: int
    crc32c: str


def digest_comparison_filename(component: str) -> str:
    return component.replace("/", "_") + ".json"


def digest_artifact_path(*, region: str, component: str) -> str:
    return DIGEST_ARTIFACT_PATH_OVERRIDES.get(
        (region, component),
        DIGEST_ARTIFACT_PREFIXES[region] + digest_comparison_filename(component),
    )


def validate_digest_comparison(
    pair: CachePair,
    comparison: dict[str, object],
    *,
    current_manifests: dict[str, RuntimeObjectManifest],
) -> None:
    expected_paths = {"east5": pair.east5_path, "europe": pair.europe_path}
    if comparison.get("mode") != "acceptance":
        raise ValueError(f"Logical digest for {pair.component!r} is not an acceptance comparison")
    if comparison.get("status") != "equivalent" or comparison.get("equivalent") is not True:
        raise ValueError(f"Logical digest for {pair.component!r} is not equivalent")
    if comparison.get("payload_matches") is not True or comparison.get("exclusion_gate_passes") is not True:
        raise ValueError(f"Logical digest for {pair.component!r} did not pass payload and exclusion gates")
    if comparison.get("cache_paths") != expected_paths:
        raise ValueError(
            f"Logical digest cache paths differ for {pair.component!r}: "
            f"{comparison.get('cache_paths')!r} != {expected_paths!r}"
        )
    expected_rows, expected_tokens = DIGEST_EXPECTED_COUNTS[pair.component]
    expected_row_counts = {"east5": expected_rows, "europe": expected_rows}
    expected_token_counts = {"east5": expected_tokens, "europe": expected_tokens}
    if comparison.get("selected_rows") != expected_row_counts:
        raise ValueError(
            f"Logical digest row counts differ for {pair.component!r}: "
            f"{comparison.get('selected_rows')!r} != {expected_row_counts!r}"
        )
    if comparison.get("selected_tokens") != expected_token_counts:
        raise ValueError(
            f"Logical digest token counts differ for {pair.component!r}: "
            f"{comparison.get('selected_tokens')!r} != {expected_token_counts!r}"
        )

    artifacts = comparison.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"east5", "europe"}:
        raise ValueError(f"Logical digest comparison for {pair.component!r} lacks both artifact bindings")
    reports: dict[str, dict[str, object]] = {}
    for region in ("east5", "europe"):
        artifact = artifacts[region]
        if not isinstance(artifact, dict):
            raise ValueError(f"Logical digest artifact binding for {pair.component!r}/{region} is malformed")
        path = artifact.get("path")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(path, str) or not isinstance(expected_sha256, str):
            raise ValueError(f"Logical digest artifact binding for {pair.component!r}/{region} is incomplete")
        expected_path = digest_artifact_path(region=region, component=pair.component)
        if path != expected_path:
            raise ValueError(
                f"Logical digest artifact path differs for {pair.component!r}/{region}: {path!r} != {expected_path!r}"
            )
        report, actual_sha256 = read_digest_artifact(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Logical digest artifact SHA-256 differs for {pair.component!r}/{region}")
        binding = report.get("binding")
        if not isinstance(binding, dict):
            raise ValueError(f"Logical digest artifact binding for {pair.component!r}/{region} is malformed")
        if binding.get("runtime_object_manifest") != runtime_object_manifest_binding(current_manifests[region]):
            raise ValueError(f"Logical digest artifact for {pair.component!r}/{region} is stale")
        reports[region] = report

    recomputed = compare_digest_reports(reports["east5"], reports["europe"], mode=ACCEPTANCE_MODE)
    recomputed["artifacts"] = artifacts
    if comparison != recomputed:
        raise ValueError(f"Logical digest comparison for {pair.component!r} does not match its bound artifacts")


def load_digest_comparisons(directory: Path) -> dict[str, dict[str, object]]:
    comparisons: dict[str, dict[str, object]] = {}
    for component in sorted(DIGEST_VERIFIED_COMPONENTS):
        path = directory / digest_comparison_filename(component)
        if not path.is_file():
            raise ValueError(f"Missing logical digest comparison for {component!r}: {path}")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Logical digest comparison is not a JSON object: {path}")
        comparisons[component] = value
    return comparisons


def cache_pairs(
    *,
    source_cache_resolver: Callable[[str, tuple[str, ...]], str | None] | None = None,
) -> tuple[CachePair, ...]:
    source_cache_resolver = source_cache_resolver or _source_tokenized_runtime_cache_path
    pairs: list[CachePair] = []
    east5_merged = PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION["us-east5"]
    europe_merged = PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION["europe-west4"]
    for domain in DOMAIN_NAMES:
        if domain in PREFERRED_MERGED_RUNTIME_DOMAIN_NAMES:
            pairs.append(
                CachePair(
                    domain=domain,
                    component=domain,
                    east5_path=east5_merged[domain],
                    europe_path=europe_merged[domain],
                )
            )
            continue
        for component in TOP_LEVEL_DOMAIN_PARTITIONS[domain]:
            europe_path = EUROPE_SOURCE_RUNTIME_CACHE_PATHS[component]
            east5_path = source_cache_resolver(component, ("us-east5",))
            if east5_path is None:
                raise ValueError(f"No frozen East5 runtime cache resolved for {component!r}")
            pairs.append(
                CachePair(
                    domain=domain,
                    component=component,
                    east5_path=east5_path,
                    europe_path=europe_path,
                )
            )
    return tuple(pairs)


def _split_gcs_path(path: str) -> tuple[str, str]:
    if not path.startswith("gs://"):
        raise ValueError(f"Expected a GCS path, got {path!r}")
    bucket, separator, prefix = path.removeprefix("gs://").partition("/")
    if not separator or not prefix:
        raise ValueError(f"Expected an object prefix, got {path!r}")
    return bucket, prefix.rstrip("/") + "/"


def _training_object_metadata(
    client: storage.Client,
    path: str,
) -> dict[str, ObjectMetadata]:
    bucket, prefix = _split_gcs_path(path)
    training_prefix = prefix + "train/"
    blobs = client.list_blobs(
        bucket,
        prefix=training_prefix,
        fields="items(name,size,crc32c),nextPageToken",
    )
    metadata: dict[str, ObjectMetadata] = {}
    for blob in blobs:
        relative_path = blob.name.removeprefix(training_prefix)
        if not relative_path or blob.crc32c is None:
            raise ValueError(f"Missing CRC32C metadata for {path}/{relative_path}")
        if _is_non_payload_training_object(relative_path):
            continue
        metadata[relative_path] = ObjectMetadata(size=int(blob.size or 0), crc32c=blob.crc32c)
    if not metadata:
        raise ValueError(f"Cache has no training payload objects: {path}")
    return metadata


def _is_non_payload_training_object(relative_path: str) -> bool:
    if relative_path in NON_PAYLOAD_TRAIN_OBJECTS or relative_path.startswith(NON_PAYLOAD_TRAIN_PREFIXES):
        return True
    top_level_name = relative_path.partition("/")[0]
    return top_level_name.endswith(NON_PAYLOAD_TRAIN_TOP_LEVEL_SUFFIXES)


def compare_metadata(
    east5: dict[str, ObjectMetadata],
    europe: dict[str, ObjectMetadata],
) -> tuple[str, ...]:
    differences: list[str] = []
    for object_name in sorted(set(east5) | set(europe)):
        if east5.get(object_name) != europe.get(object_name):
            differences.append(object_name)
    return tuple(differences)


def _shard_payload_metadata(metadata: dict[str, ObjectMetadata]) -> dict[str, ObjectMetadata]:
    payload = {
        name: object_metadata
        for name, object_metadata in metadata.items()
        if name.startswith("part-") and not name.endswith("/shard_ledger.json")
    }
    if not payload:
        raise ValueError("Tokenized cache has no immutable shard payload objects")
    return payload


def _object_manifest_sha256(metadata: dict[str, ObjectMetadata]) -> str:
    payload = [{"name": name, "size": value.size, "crc32c": value.crc32c} for name, value in sorted(metadata.items())]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_training_json(client: storage.Client, path: str, name: str) -> dict[str, object]:
    bucket, prefix = _split_gcs_path(path)
    return json.loads(client.bucket(bucket).blob(prefix + "train/" + name).download_as_text())


def shard_ledger_invariants(
    ledger: dict[str, object],
    fields: tuple[str, ...] = SHARD_LEDGER_INVARIANT_FIELDS,
) -> dict[str, object]:
    return {field: ledger.get(field) for field in fields}


def cache_layout(ledger: dict[str, object]) -> str:
    layout = ledger.get("layout", CACHE_LAYOUT_CONSOLIDATED)
    if layout not in {CACHE_LAYOUT_CONSOLIDATED, CACHE_LAYOUT_SHARDED}:
        raise ValueError(f"Unknown cache layout: {layout!r}")
    return layout


def shard_ledger_runtime_layout(ledger: dict[str, object]) -> dict[str, object]:
    """Return the ledger fields that affect runtime row order and reachability."""
    layout = cache_layout(ledger)
    invariants = {
        "layout": layout,
        **shard_ledger_invariants(ledger, SHARD_LEDGER_LAYOUT_INVARIANT_FIELDS),
    }
    if layout == CACHE_LAYOUT_SHARDED:
        invariants["finished_shards"] = ledger.get("finished_shards")
    return invariants


def finished_shards(ledger: dict[str, object]) -> tuple[str, ...]:
    value = ledger.get("finished_shards", [])
    if not isinstance(value, list):
        raise ValueError("Runtime cache ledger has malformed finished_shards")
    return tuple(str(name) for name in value)


def _artifact_input_cache_dirs(path: str) -> dict[str, str]:
    record = read_record(path)
    if record is None or not isinstance(record.result, dict):
        raise ValueError(f"Cache lacks an artifact result: {path}")
    input_configs = record.result.get("input_configs")
    if not isinstance(input_configs, dict):
        raise ValueError(f"Cache lacks input_configs: {path}")

    cache_dirs: dict[str, str] = {}
    for component, config in input_configs.items():
        if not isinstance(component, str) or not isinstance(config, dict):
            raise ValueError(f"Cache has malformed input config: {path}")
        cache_dir = config.get("cache_dir")
        if not isinstance(cache_dir, str):
            raise ValueError(f"Cache input {component!r} lacks cache_dir: {path}")
        cache_dirs[component] = cache_dir
    return cache_dirs


def _payload_stats(client: storage.Client, path: str) -> dict[str, int]:
    stats = _read_training_json(client, path, ".stats.json")
    total_tokens = stats.get("total_tokens")
    total_elements = stats.get("total_elements")
    if not isinstance(total_tokens, int) or not isinstance(total_elements, int):
        raise ValueError(f"Cache has malformed payload statistics: {path}")
    return {"total_tokens": total_tokens, "total_elements": total_elements}


def validate_stack_payload_stats(
    *,
    east5_merged: dict[str, int],
    europe_merged: dict[str, int],
    east5_inputs: dict[str, dict[str, int]],
    europe_inputs: dict[str, dict[str, int]],
) -> None:
    expected_inputs = frozenset(TOP_LEVEL_DOMAIN_PARTITIONS[STACK_COMPONENT])
    if frozenset(east5_inputs) != expected_inputs or frozenset(europe_inputs) != expected_inputs:
        raise ValueError("Stack-Edu logical audit does not cover the frozen 15-language input set")
    if east5_merged != europe_merged:
        raise ValueError(f"Stack-Edu merged statistics differ: {east5_merged} != {europe_merged}")
    if east5_inputs != europe_inputs:
        raise ValueError("Stack-Edu per-language payload statistics differ")

    expected_merged = {
        "total_tokens": EXPECTED_STACK_TOKENS,
        "total_elements": EXPECTED_STACK_ELEMENTS,
    }
    if east5_merged != expected_merged:
        raise ValueError(
            f"Stack-Edu merged statistics differ from the frozen payload: {east5_merged} != {expected_merged}"
        )

    for partition, stats in east5_inputs.items():
        expected_tokens = DOLMA3_AVAILABLE_TOKEN_COUNTS[partition]
        if stats["total_tokens"] != expected_tokens:
            raise ValueError(
                f"Stack-Edu token count differs from the frozen East5 count for {partition}: "
                f"{stats['total_tokens']} != {expected_tokens}"
            )
    summed = {
        field: sum(stats[field] for stats in east5_inputs.values()) for field in ("total_tokens", "total_elements")
    }
    if summed != east5_merged:
        raise ValueError(f"Stack-Edu input statistics do not sum to the merged statistics: {summed} != {east5_merged}")


def _audit_stack_logical_payload(client: storage.Client, pair: CachePair) -> dict[str, object]:
    if not pair.europe_path.startswith(EUROPE_HISTORICAL_STACK_MERGED_PREFIX):
        raise ValueError(f"Europe Stack merged cache is not in the historical namespace: {pair.europe_path}")
    east5_cache_dirs = _artifact_input_cache_dirs(pair.east5_path)
    europe_cache_dirs = _artifact_input_cache_dirs(pair.europe_path)
    expected_input_order = tuple(TOP_LEVEL_DOMAIN_PARTITIONS[STACK_COMPONENT])
    if tuple(east5_cache_dirs) != expected_input_order or tuple(europe_cache_dirs) != expected_input_order:
        raise ValueError("Stack-Edu merged caches do not preserve the frozen 15-language input order")
    for component, cache_dir in east5_cache_dirs.items():
        if not cache_dir.startswith(EAST5_STACK_INPUT_PREFIX):
            raise ValueError(f"East5 Stack input {component!r} is outside the frozen East5 namespace: {cache_dir}")
    for component, cache_dir in europe_cache_dirs.items():
        if not cache_dir.startswith(EUROPE_HISTORICAL_STACK_INPUT_PREFIX):
            raise ValueError(f"Europe Stack input {component!r} is not in the historical namespace: {cache_dir}")

    input_shard_manifests: dict[str, dict[str, object]] = {}
    for component in expected_input_order:
        east5_metadata = _shard_payload_metadata(_training_object_metadata(client, east5_cache_dirs[component]))
        europe_metadata = _shard_payload_metadata(_training_object_metadata(client, europe_cache_dirs[component]))
        differences = compare_metadata(east5_metadata, europe_metadata)
        if differences:
            raise ValueError(
                f"Stack-Edu immutable input-shard payload differs for {component}: {list(differences[:20])}"
            )
        input_shard_manifests[component] = {
            "objects": len(east5_metadata),
            "bytes": sum(value.size for value in east5_metadata.values()),
            "object_manifest_sha256": _object_manifest_sha256(east5_metadata),
        }

    east5_inputs = {
        component: _payload_stats(client, cache_dir) for component, cache_dir in sorted(east5_cache_dirs.items())
    }
    europe_inputs = {
        component: _payload_stats(client, cache_dir) for component, cache_dir in sorted(europe_cache_dirs.items())
    }
    east5_merged = _payload_stats(client, pair.east5_path)
    europe_merged = _payload_stats(client, pair.europe_path)
    validate_stack_payload_stats(
        east5_merged=east5_merged,
        europe_merged=europe_merged,
        east5_inputs=east5_inputs,
        europe_inputs=europe_inputs,
    )
    evidence: dict[str, object] = {
        "merged": east5_merged,
        "inputs": east5_inputs,
        "ordered_components": list(expected_input_order),
        "exact_input_shard_payloads": input_shard_manifests,
        "europe_input_namespace": EUROPE_HISTORICAL_STACK_INPUT_PREFIX,
        "europe_merged_namespace": EUROPE_HISTORICAL_STACK_MERGED_PREFIX,
    }
    evidence["sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return evidence


def _relative_cache_path(path: str) -> str:
    for prefix in (EAST5_PREFIX, EUROPE_PREFIX):
        if path.startswith(prefix + "/"):
            return path.removeprefix(prefix + "/")
    raise ValueError(f"Cache is outside the audited regional buckets: {path}")


def logical_runtime_contract_sha256(pairs: tuple[CachePair, ...], *, region: str) -> str:
    if region not in {"east5", "europe"}:
        raise ValueError(f"Unknown identity region {region!r}")
    payload = []
    for pair in pairs:
        cache_path = _relative_cache_path(pair.east5_path if region == "east5" else pair.europe_path)
        if pair.component == STACK_COMPONENT:
            cache_path = f"logical://{STACK_COMPONENT}/historical-full-document-v1"
        elif pair.component in DIGEST_VERIFIED_COMPONENTS:
            cache_path = f"logical://{pair.component}/zero-exclusion-runtime-digest-v4"
        payload.append(
            {
                "domain": pair.domain,
                "component": pair.component,
                "cache_path": cache_path,
                "domain_tokens": TOP_LEVEL_DOMAIN_TOKEN_COUNTS[pair.domain],
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digest-comparison-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = storage.Client()
    pairs = cache_pairs()
    pair_components = {pair.component for pair in pairs}
    missing_digest_components = DIGEST_VERIFIED_COMPONENTS - pair_components
    if missing_digest_components:
        raise ValueError(f"Digest-verified components are absent from the runtime contract: {missing_digest_components}")
    digest_comparisons = load_digest_comparisons(args.digest_comparison_dir)
    launcher_paths = {
        "east5": _runtime_cache_paths("us-east5"),
        "europe": _runtime_cache_paths("europe-west4"),
    }
    pair_paths = {
        "east5": tuple(pair.east5_path for pair in pairs),
        "europe": tuple(pair.europe_path for pair in pairs),
    }
    for region in ("east5", "europe"):
        if len(set(pair_paths[region])) != len(pair_paths[region]):
            raise ValueError(f"Audit runtime bindings contain duplicate paths for {region}")
        if set(pair_paths[region]) != set(launcher_paths[region]):
            raise ValueError(f"Audit and launcher runtime bindings differ for {region}")
    east5_identity = logical_runtime_contract_sha256(pairs, region="east5")
    europe_identity = logical_runtime_contract_sha256(pairs, region="europe")
    if east5_identity != europe_identity:
        raise ValueError(f"Regional logical data identities differ: {east5_identity} != {europe_identity}")

    pair_results: list[dict[str, object]] = []
    east5_total_objects = 0
    east5_total_bytes = 0
    europe_total_objects = 0
    europe_total_bytes = 0
    shard_ledger_metadata_differences = 0
    crc_exact_cache_pairs = 0
    digest_exact_cache_pairs = 0
    stack_input_shard_crc_exact_cache_pairs = 0
    stack_logical_payload: dict[str, object] | None = None
    for pair in pairs:
        east5_metadata = _training_object_metadata(client, pair.east5_path)
        europe_metadata = _training_object_metadata(client, pair.europe_path)
        differences = compare_metadata(east5_metadata, europe_metadata)
        uses_digest = pair.component in DIGEST_VERIFIED_COMPONENTS
        if differences and pair.component != STACK_COMPONENT and not uses_digest:
            raise ValueError(
                f"Training payload differs for {pair.domain}/{pair.component}: " f"{list(differences[:20])}"
            )
        east5_ledger = _read_training_json(client, pair.east5_path, "shard_ledger.json")
        europe_ledger = _read_training_json(client, pair.europe_path, "shard_ledger.json")
        east5_layout = cache_layout(east5_ledger)
        europe_layout = cache_layout(europe_ledger)
        cache_layouts_match = east5_layout == europe_layout
        if not cache_layouts_match and pair.component != STACK_COMPONENT and not uses_digest:
            raise ValueError(f"Cache layout differs for {pair.domain}/{pair.component}")
        if pair.component == STACK_COMPONENT and east5_layout != CACHE_LAYOUT_CONSOLIDATED:
            raise ValueError("Stack-Edu merged runtime cache must remain consolidated")
        logical_ledger_invariants_match = shard_ledger_invariants(
            east5_ledger,
            SHARD_LEDGER_LOGICAL_INVARIANT_FIELDS,
        ) == shard_ledger_invariants(europe_ledger, SHARD_LEDGER_LOGICAL_INVARIANT_FIELDS)
        if not logical_ledger_invariants_match:
            raise ValueError(f"Shard-ledger logical row state differs for {pair.domain}/{pair.component}")
        layout_ledger_invariants_match = shard_ledger_runtime_layout(east5_ledger) == shard_ledger_runtime_layout(
            europe_ledger
        )
        if not layout_ledger_invariants_match and pair.component != STACK_COMPONENT and not uses_digest:
            raise ValueError(f"Shard-ledger row layout differs for {pair.domain}/{pair.component}")
        if pair.component == STACK_COMPONENT:
            stack_logical_payload = _audit_stack_logical_payload(client, pair)
            stack_input_shard_crc_exact_cache_pairs += 1
            equivalence_mode = "exact_ordered_input_shard_crc32c_and_merged_stats"
        elif uses_digest:
            current_manifests = {
                "east5": runtime_object_manifest(
                    client,
                    pair.east5_path,
                    layout=east5_layout,
                    finished_shards=finished_shards(east5_ledger),
                ),
                "europe": runtime_object_manifest(
                    client,
                    pair.europe_path,
                    layout=europe_layout,
                    finished_shards=finished_shards(europe_ledger),
                ),
            }
            validate_digest_comparison(
                pair,
                digest_comparisons[pair.component],
                current_manifests=current_manifests,
            )
            digest_exact_cache_pairs += 1
            equivalence_mode = "zero_exclusion_logical_runtime_digest"
        else:
            crc_exact_cache_pairs += 1
            equivalence_mode = "object_crc32c_and_shard_layout"
        ledger_metadata_differs = east5_ledger != europe_ledger
        shard_ledger_metadata_differences += int(ledger_metadata_differs)
        east5_pair_objects = len(east5_metadata)
        east5_pair_bytes = sum(metadata.size for metadata in east5_metadata.values())
        europe_pair_objects = len(europe_metadata)
        europe_pair_bytes = sum(metadata.size for metadata in europe_metadata.values())
        east5_total_objects += east5_pair_objects
        east5_total_bytes += east5_pair_bytes
        europe_total_objects += europe_pair_objects
        europe_total_bytes += europe_pair_bytes
        pair_results.append(
            {
                **asdict(pair),
                "east5_training_objects": east5_pair_objects,
                "east5_training_bytes": east5_pair_bytes,
                "europe_training_objects": europe_pair_objects,
                "europe_training_bytes": europe_pair_bytes,
                "stats_crc32c": east5_metadata[".stats.json"].crc32c if ".stats.json" in east5_metadata else None,
                "shard_ledger_metadata_differs": ledger_metadata_differs,
                "cache_layout": east5_layout,
                "cache_layouts_match": cache_layouts_match,
                "shard_ledger_logical_invariants_match": logical_ledger_invariants_match,
                "shard_ledger_layout_invariants_match": layout_ledger_invariants_match,
                "equivalence_mode": equivalence_mode,
                "physical_layout_difference_count": len(differences),
            }
        )

    if stack_logical_payload is None:
        raise ValueError("Runtime-cache audit did not encounter Stack-Edu")
    expected_crc_pairs = len(pairs) - len(DIGEST_VERIFIED_COMPONENTS) - 1
    if (
        crc_exact_cache_pairs != expected_crc_pairs
        or digest_exact_cache_pairs != len(DIGEST_VERIFIED_COMPONENTS)
        or stack_input_shard_crc_exact_cache_pairs != 1
    ):
        raise ValueError("Runtime-cache audit did not cover the frozen CRC, digest, and Stack equivalence classes")

    report = {
        "status": "training_payload_equivalent",
        "cache_pairs": len(pairs),
        "top_level_domains": len(DOMAIN_NAMES),
        "east5_training_objects": east5_total_objects,
        "east5_training_bytes": east5_total_bytes,
        "europe_training_objects": europe_total_objects,
        "europe_training_bytes": europe_total_bytes,
        "direct_crc_exact_cache_pairs": crc_exact_cache_pairs,
        "logical_digest_exact_cache_pairs": digest_exact_cache_pairs,
        "stack_input_shard_crc_exact_cache_pairs": stack_input_shard_crc_exact_cache_pairs,
        "completion_evidence": (
            f"{crc_exact_cache_pairs} cache pairs use exact object CRC and shard-layout equality; "
            f"{digest_exact_cache_pairs} frozen physical exceptions use zero-exclusion logical runtime digests; "
            "Stack-Edu uses exact ordered per-language shard-object CRC equality plus frozen per-language and "
            "merged token/row statistics because only the redundant merged-cache rechunking differs physically"
        ),
        "excluded_non_payload_objects": sorted(NON_PAYLOAD_TRAIN_OBJECTS),
        "excluded_non_payload_prefixes": list(NON_PAYLOAD_TRAIN_PREFIXES),
        "excluded_non_payload_top_level_suffixes": list(NON_PAYLOAD_TRAIN_TOP_LEVEL_SUFFIXES),
        "shard_ledger_metadata_differences": shard_ledger_metadata_differences,
        "logical_runtime_contract_sha256": east5_identity,
        "runtime_paths_sha256": {region: _runtime_paths_sha256(paths) for region, paths in launcher_paths.items()},
        "stack_logical_payload": stack_logical_payload,
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
