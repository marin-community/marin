# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a content-pinned loop configuration and apply scale-only smoke overrides."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import yaml

from experiments.post_training.long_loop.stage_config import validate_stage_config


@dataclass(frozen=True)
class FilePin:
    path: str
    sha256: str


@dataclass(frozen=True)
class DatasetPins:
    sft: FilePin
    agentic: FilePin | None
    nonagentic: FilePin
    heldout: FilePin | None
    exclusions: FilePin | None
    evaluation: FilePin


@dataclass(frozen=True)
class StorageConfig:
    output_prefix: str
    temporary_prefix: str


@dataclass(frozen=True)
class RunSpec:
    name: str
    marin_revision: str
    datasets: DatasetPins
    sft: FilePin
    rl: FilePin
    evaluation: FilePin
    endpoint: FilePin
    archive: FilePin
    initializer: FilePin
    tokenizer: FilePin
    execution: FilePin
    smoke_overrides: dict[str, int]
    storage: StorageConfig
    adopted_sft: FilePin | None = None
    adopted_handoff: FilePin | None = None
    adopted_endpoint: FilePin | None = None


# Geometry, context, group mix and data are deliberately absent. A smoke must
# exercise the production path; only its duration and evaluation fanout shrink.
SCALE_PATHS = {
    "sft_steps": ("sft", "stop_after_steps"),
    "rl_steps": ("rl", "config", "trainer", "max_steps"),
    "eval_replicas": ("evaluation", "role_plan", "num_inference_engines"),
    "eval_tasks_per_benchmark": ("evaluation", "tasks_per_benchmark"),
}


def read_pin(pin: FilePin, directory: Path) -> Any:
    path = directory / pin.path
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != pin.sha256:
        raise ValueError(f"Pinned file changed: {path}")
    return yaml.safe_load(content)


# rno2a runs keep their intermediates in that region's TTL scratch.
OUTPUT_ROOTS = ("s3://marin-us-east-08a/", "s3://marin-rno2a/tmp/ttl=")


def resolve_spec(path: Path, *, smoke: bool) -> dict:
    """Resolve exact files; missing admission inputs remain explicit blockers."""
    spec = draccus.decode(RunSpec, yaml.safe_load(path.read_text()))
    if len(spec.marin_revision) != 40 or any(c not in "0123456789abcdef" for c in spec.marin_revision):
        raise ValueError("marin_revision must be a full Git commit")
    prefixes = (spec.storage.output_prefix, spec.storage.temporary_prefix)
    roots = {next((root for root in OUTPUT_ROOTS if prefix.startswith(root)), None) for prefix in prefixes}
    if None in roots or len(roots) != 1:
        raise ValueError(f"New long-loop outputs must both use one of {OUTPUT_ROOTS}")
    directory = path.resolve().parent
    result = {
        "storage": vars(spec.storage),
        "name": spec.name,
        "marin_revision": spec.marin_revision,
        "spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "smoke": smoke,
        "datasets": {},
        "blockers": [],
    }
    for name, pin in vars(spec.datasets).items():
        if pin is None:
            result["blockers"].append(f"Missing admitted {name} manifest")
            continue
        result["datasets"][name] = read_pin(pin, directory)
    result["adopted_sft"] = read_pin(spec.adopted_sft, directory) if spec.adopted_sft is not None else None
    result["adopted_handoff"] = read_pin(spec.adopted_handoff, directory) if spec.adopted_handoff is not None else None
    result["adopted_endpoint"] = (
        read_pin(spec.adopted_endpoint, directory) if spec.adopted_endpoint is not None else None
    )
    sft_manifest = result["datasets"]["sft"]
    for field, pin in sft_manifest["sections"].items():
        sft_manifest["data"][field] = read_pin(draccus.decode(FilePin, pin), directory)
    for name in ("sft", "rl", "evaluation", "endpoint", "archive", "initializer", "tokenizer", "execution"):
        result[name] = read_pin(getattr(spec, name), directory)
    for stage in ("rl", "evaluation"):
        validate_stage_config(result[stage]["config"], stage)
        for field in ("rendezvous_timeout", "cluster_join_timeout", "driver_liveness_timeout"):
            if field not in result[stage]["ray_timeouts"]:
                raise ValueError(f"{stage} run spec must explicitly set ray_timeouts.{field}")
    for name, document in {
        **result["datasets"],
        **{key: result[key] for key in ("rl", "evaluation", "archive", "initializer")},
    }.items():
        result["blockers"].extend(f"{name}: {reason}" for reason in document.get("admission_pending", []))
    if result["adopted_sft"] is not None:
        result["blockers"].extend(
            f"adopted_sft: {reason}" for reason in result["adopted_sft"].get("admission_pending", [])
        )
    unknown = set(spec.smoke_overrides) - SCALE_PATHS.keys()
    if unknown:
        raise ValueError(f"Smoke cannot change data or semantics: {sorted(unknown)}")
    result["scale_overrides"] = spec.smoke_overrides if smoke else {}
    if smoke:
        for name, value in spec.smoke_overrides.items():
            if value < 1:
                raise ValueError(f"Smoke scale {name} must be positive")
            target = result
            *parents, key = SCALE_PATHS[name]
            for parent in parents:
                target = target[parent]
            target[key] = value
        if "eval_replicas" in spec.smoke_overrides:
            evaluation = result["evaluation"]
            replicas = spec.smoke_overrides["eval_replicas"]
            evaluation["role_plan"]["policy_num_nodes"] = replicas
            evaluation["config"]["generator"]["num_inference_engines"] = replicas
            placement = evaluation["config"]["trainer"]["placement"]
            placement["policy_num_nodes"] = replicas
            if evaluation["role_plan"]["reference_num_nodes"] is None:
                placement["ref_num_nodes"] = replicas
        if "eval_tasks_per_benchmark" in spec.smoke_overrides:
            result["evaluation"]["expected_rollouts"] = result["evaluation"]["tasks_per_benchmark"] * sum(
                result["evaluation"]["attempts_per_selected_task"].values()
            )
            result["evaluation"]["expected_source_rows"] = result["evaluation"]["tasks_per_benchmark"] * len(
                result["evaluation"]["attempts_per_selected_task"]
            )
            result["evaluation"]["expected_expanded_attempt_count"] = result["evaluation"]["expected_rollouts"]
    heldout = result["datasets"].get("heldout")
    if heldout is not None:
        for callback in result["rl"]["config"]["trainer"].get("callbacks", []):
            if callback["type"] == "best_checkpoint" and callback["expected_tasks"] != heldout["rows"]:
                raise ValueError("Best-checkpoint validation cardinality differs from the pinned held-out manifest")
    # Dataset manifests are the sole data authority, independent of stage files.
    result["sft"]["data"] = copy.deepcopy(result["datasets"]["sft"]["data"])
    result["gpu_peaks"] = allocation_peaks(result)
    result["resolved_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def allocation_peaks(resolved: dict) -> dict[str, int]:
    """Count serialized stage allocations, including the co-hosted eval judge."""
    execution = resolved["execution"]
    for runtime in (execution["coordinator"], resolved["rl"]["execution"], resolved["evaluation"]["execution"]):
        if runtime["priority"] != "production" or runtime["cluster"] != "cw-rno2a":
            raise ValueError("This experiment is authorized only on RNO2A at production priority")
    if resolved["endpoint"]["priority"] != "production":
        raise ValueError("Endpoint priority must remain production")
    coordinator_gpu = execution["coordinator"]["coordinator_gpu"]
    coordinator_count = coordinator_gpu["count"] if coordinator_gpu else 0
    resources = {key: execution[key] for key in ("export", "eval_inputs", "archive")}
    resources["sft"] = resolved["sft"]["resources"]
    peaks = {
        key: value["replicas"] * value["device"]["count"] if value["device"]["kind"] == "gpu" else 0
        for key, value in resources.items()
    }
    for stage in ("rl", "evaluation"):
        plan = resolved[stage]["role_plan"]
        nodes = plan["policy_num_nodes"]
        width = plan["policy_num_gpus_per_node"]
        if not plan["colocate_all"]:
            engine_size = plan["inference_engine_tensor_parallel_size"] * plan["inference_engine_pipeline_parallel_size"]
            engine_size *= plan["inference_engine_data_parallel_size"]
            per_node = width // engine_size
            if not per_node:
                raise ValueError("An inference engine does not fit on one node")
            nodes += (plan["num_inference_engines"] + per_node - 1) // per_node
            nodes += plan["reference_num_nodes"] or 0
        peaks[stage] = nodes * width
    peaks["evaluation"] += resolved["evaluation"]["judge_gpu_count"]
    peaks = {key: value + coordinator_count for key, value in peaks.items()}
    if max(peaks.values()) > 256:
        raise ValueError(f"Run exceeds the authorized 256 H100 allocation: {peaks}")
    return peaks


def stage_blockers(resolved: dict, through: str) -> list[str]:
    """Blockers for the stages up to ``through``; ``sft`` and ``export`` need only the SFT inputs."""
    if through in {"rl", "eval", "archive"}:
        blockers = list(resolved["blockers"])
        if through == "rl":
            blockers = [item for item in blockers if not item.startswith(("archive:", "evaluation:"))]
        if through == "eval":
            blockers = [item for item in blockers if not item.startswith("archive:")]
        return blockers
    documents = {
        "sft": resolved["datasets"]["sft"],
        "initializer": resolved["initializer"],
        "tokenizer": resolved["tokenizer"],
    }
    return [f"{name}: {reason}" for name, value in documents.items() for reason in value.get("admission_pending", [])]
