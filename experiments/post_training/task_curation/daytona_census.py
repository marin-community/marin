# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure declared Daytona demands from complete converted populations, without provider calls."""

import argparse
import base64
import hashlib
import json
import logging
import subprocess
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq
from finestore.rl.curation import DAYTONA_SAFE_SNAPSHOT_LIMIT
from harbor_config.models.task.config import EnvironmentConfig
from huggingface_hub import snapshot_download
from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import UnsupportedMachineSpec
from taskcompendium.grading_environment import GradingEnvironmentStatus, assess_grading_environment
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec, missing_environment
from taskcompendium.pipeline.conversion import ConvertedRow, convert_source_row
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, SourceRecipe
from taskcompendium.pipeline.sources import conversion_context, source_shards, staged_raw_file_rows

from experiments.post_training.task_curation.daytona_identity import (
    count_lowered_spec_identities,
    count_native_spec_identities,
    native_task_toml,
)
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    environment_requirements,
    planned_lock_url,
    source_recipe,
)
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import GRANDFATHERED_MARINSKYRL_SOURCES
from experiments.post_training.task_curation.sources import all_sources

logger = logging.getLogger(__name__)
PROFILE = dict(cpus=None, memory_mb=None, storage_mb=None, gpus=0, user=None, startup_timeout=900, cleanup_timeout=120)


@dataclass(frozen=True)
class DaytonaCensus:
    source_id: str
    source_revision: str | None
    status: str
    rows: int
    converted: int
    rejections: dict[str, int]
    machine_slots: int
    snapshot_count: int | None
    snapshot_sizes: dict[str, int]
    execution_failure_tasks: int
    execution_failures: dict[str, int]
    mapping_failures: dict[str, int]
    verdict: str | None
    basis: str
    error: str | None
    roles: dict
    converter_sha256: str
    admission_rejections: dict[str, int]
    actor_sources: dict[str, int]
    unsafe_reasons: dict[str, str]
    unsafe_counts: dict[str, int]


class EnvelopeCensus(NamedTuple):
    converted: int
    machine_slots: int
    snapshot_sizes: Counter[str]
    execution_failure_tasks: int
    execution_failures: Counter[str]
    mapping_failures: Counter[str]
    roles: dict
    actor_sources: Counter[str]
    unsafe_task_counts: Counter[str]


@cache
def cached_sources():
    return all_sources()


@cache
def converter_sha256() -> str:
    root = Path(__file__).resolve().parents[3]
    directories = (
        root / "experiments/post_training/task_curation",
        root / "lib/taskcompendium/src",
        root / "lib/rolloutengine/src",
        root / "lib/shellbox/src",
    )
    return canonical_sha256(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in directories
            for path in sorted(directory.rglob("*.py"))
        }
    )


def cached_input(reference) -> str:
    return snapshot_download(
        reference.repo,
        repo_type="dataset",
        revision=reference.revision,
        allow_patterns=list(reference.files),
        local_files_only=True,
    )


def declared_shapes(converted: ConvertedRow) -> dict:
    raw = converted.raw.data
    if raw.get("lowered_task_spec") is not None:
        return json.loads(raw["lowered_task_spec"]).get("runtime") or {}
    files = raw.get("files") or {}
    if "task.toml" not in files:
        return {}
    return native_shapes(base64.b64decode(files["task.toml"]))


def native_shapes(task_toml: bytes) -> dict:
    document = tomllib.loads(task_toml.decode())
    actor = EnvironmentConfig.model_validate(document.get("environment") or {})
    verifier = EnvironmentConfig.model_validate(
        (document.get("verifier") or {}).get("environment") or actor.model_dump()
    )
    return {
        role: {
            "cpus": environment.cpus,
            "memory_mb": environment.memory_mb,
            "storage_mb": environment.storage_mb,
            "gpus": environment.gpus or 0,
            "network": "allow" if environment.allow_internet else "deny",
        }
        for role, environment in (("task_machine", actor), ("verifier_machine", verifier))
    }


def measure_source(source_id: str, artifact_root: str) -> DaytonaCensus:
    source = cached_sources()[source_id]
    config = source.config
    if not isinstance(config, CurationRecipe | PrebuiltCohort):
        raise ValueError(
            "No configured converter or converted-spec artifact is registered for this inventory declaration"
        )
    rejections = Counter()
    admission_rejections = Counter()
    result = measure_envelopes(converted_source(source, artifact_root, rejections, admission_rejections))
    converted, machine_slots, sizes, failed_tasks, failures, mapping, roles, actor_sources, _unsafe_task_counts = result
    if isinstance(config, PrebuiltCohort) and converted != config.expected_rows:
        raise ValueError(f"Published cohort has {converted} TaskSpecs, expected {config.expected_rows}")
    rows = converted + sum(rejections.values())
    error = f"Registered converter emitted zero TaskSpecs from {rows} selected inputs" if not converted else None
    verdict = (
        None
        if error
        else (
            "not_applicable"
            if not machine_slots
            else ("unsafe" if len(sizes) > DAYTONA_SAFE_SNAPSHOT_LIMIT or failed_tasks or mapping else "safe")
        )
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return DaytonaCensus(
        source_id,
        source.dataset.revision if source.dataset else None,
        "ingestion_failure" if error else "measured",
        rows,
        converted,
        dict(rejections),
        machine_slots,
        None if error else len(sizes),
        dict(sizes),
        failed_tasks,
        dict(failures),
        dict(mapping),
        verdict,
        f"Registered converter TaskSpec outputs before admission gates (post-18a population); "
        f"actor on Daytona, package-lock grader on LOCAL; "
        f"production Harbor machine_spec -> shellbox snapshot_identity; converter checkout {revision}; "
        f"recipe version {source.version}; allocations from native task.toml or published runtime, absent values null; "
        f"no Daytona API calls",
        error,
        roles,
        converter_sha256(),
        dict(admission_rejections),
        dict(actor_sources),
        daytona_unsafe_reasons(result),
        daytona_unsafe_counts(result),
    )


def measure_envelopes(documents: Iterable[tuple[dict, dict]]) -> EnvelopeCensus:
    """Separate actor Daytona compatibility from each grader's assigned backend."""
    converted = machine_slots = failed_tasks = 0
    sizes, failures, mapping = Counter(), Counter(), Counter()
    identities = {}
    local_checks = {}
    roles = {role: {} for role in ("agent", "grader")}
    actor_sources = Counter()
    unsafe_task_counts = Counter()
    for task, shapes in documents:
        converted += 1
        if converted % 10000 == 0:
            logger.info("Census streamed %d converted tasks", converted)
        native = "harbor" in task.get("tags", ())
        if native:
            shapes = native_shapes(native_task_toml(task))
        runtime = {}
        for role, slot, declared, network in (
            ("agent", "task_machine", task["environment_requirements"], "allow"),
            ("grader", "verifier_machine", task["grader"].get("environment"), "deny"),
        ):
            needs_machine = declared is not None and any(declared.values())
            shape = {**PROFILE, "backend": "daytona", "network": network, **(shapes.get(slot) or {})}
            if role == "grader" and declared and declared.get("packages_lock"):
                shape["backend"] = "local"
            runtime[slot] = shape if needs_machine else None
            if not needs_machine:
                continue
            outcome = roles[role].setdefault(shape["backend"], {"machine_slots": 0, "failures": Counter()})
            outcome["machine_slots"] += 1
            if shape["backend"] == "local":
                local_key = canonical_sha256([declared, shape])
                if local_key not in local_checks:
                    requirements = EnvironmentRequirements.model_validate(declared)
                    try:
                        spec = _machine_spec(requirements, MachineRuntimeSpec.model_validate(shape))
                        # prepare_machine_spec removes the per-process memory limit for LOCAL.
                        check_local_environment(requirements, replace(spec, memory_mb=None))
                    except UnsupportedMachineSpec as error:
                        local_checks[local_key] = str(error)
                    else:
                        local_checks[local_key] = None
                if local_checks[local_key] is not None:
                    outcome["failures"][local_checks[local_key]] += 1
            elif role == "grader":
                grader_runtime = {"task_machine": None, "verifier_machine": shape}
                grader_key = canonical_sha256([declared, grader_runtime])
                if grader_key not in identities:
                    identities[grader_key] = count_lowered_spec_identities(
                        [(task["id"], {"task": task, "runtime": grader_runtime})]
                    )
                grader_measurement = identities[grader_key]
                outcome["failures"].update(reason for _, reason in grader_measurement.execution_failures)
        actor_runtime = {**runtime, "verifier_machine": None}
        if runtime["task_machine"] is not None:
            actor_sources.update(
                name
                for name in ("docker_image", "docker_build", "packages_lock", "package_spec")
                if task["environment_requirements"].get(name) is not None
            )
        envelope = {"task": task, "runtime": actor_runtime}
        key = canonical_sha256([task["environment_requirements"], actor_runtime])
        if key not in identities:
            identities[key] = (
                count_native_spec_identities([(task["id"], task)])
                if native
                else count_lowered_spec_identities([(task["id"], envelope)])
            )
        measurement = identities[key]
        unsafe_task_counts.update(
            {
                actor_failure_reason(detail, task["environment_requirements"])
                for _, detail in (*measurement.execution_failures, *measurement.failures)
            }
        )
        machine_slots += runtime["task_machine"] is not None
        sizes.update(measurement.sizes)
        failed_tasks += bool(measurement.execution_failures)
        failures.update(reason for _, reason in measurement.execution_failures)
        mapping.update(reason for _, reason in measurement.failures)
    if "daytona" in roles["agent"]:
        roles["agent"]["daytona"]["failures"] = dict(failures)
    for backends in roles.values():
        for outcome in backends.values():
            outcome["failures"] = dict(outcome["failures"])
    return EnvelopeCensus(
        converted, machine_slots, sizes, failed_tasks, failures, mapping, roles, actor_sources, unsafe_task_counts
    )


def actor_failure_reason(detail: str, sources: Mapping[str, object]) -> str:
    if sources.get("docker_build") and "Unresolved Docker build context" in detail:
        return "unbuilt_image"
    if sources.get("packages_lock") and "requires a registry image or Docker build context" in detail:
        return "host_image_required"
    return "execution_requirement"


def daytona_unsafe_reasons(census: EnvelopeCensus) -> dict[str, str]:
    """Classify actor refusals and the snapshot budget, preserving their actual details."""
    details: dict[str, set[str]] = {}
    if len(census.snapshot_sizes) > DAYTONA_SAFE_SNAPSHOT_LIMIT:
        details["too_many_snapshots"] = {
            f"{len(census.snapshot_sizes)} actor snapshot identities exceed "
            f"the {DAYTONA_SAFE_SNAPSHOT_LIMIT}-snapshot budget"
        }
    for detail in (*census.execution_failures, *census.mapping_failures):
        details.setdefault(actor_failure_reason(detail, census.actor_sources), set()).add(detail)
    return {reason: "; ".join(sorted(values)) for reason, values in details.items()}


def daytona_unsafe_counts(census: EnvelopeCensus) -> dict[str, int]:
    """Count affected actor task slots once per reason, including the cohort budget."""
    counts = dict(census.unsafe_task_counts)
    if len(census.snapshot_sizes) > DAYTONA_SAFE_SNAPSHOT_LIMIT:
        counts["too_many_snapshots"] = census.machine_slots
    return counts


def converted_source(source, artifact_root: str, rejections: Counter, admission_rejections: Counter):
    config = source.config
    if isinstance(config, PrebuiltCohort):
        root = snapshot_download(
            config.repo,
            repo_type=config.repo_type,
            revision=config.revision,
            allow_patterns=list(config.files),
            local_files_only=True,
        )
        for batch in pq.ParquetFile(Path(root) / config.files[0]).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                envelope = json.loads(row["lowered_task_spec"])
                task = TaskSpec.model_validate(envelope["task"])
                if missing_environment(envelope["task"]) is not None:
                    admission_rejections["source_defect:missing_environment"] += 1
                else:
                    try:
                        assessment = assess_grading_environment(task)
                    except ValueError:
                        admission_rejections["source_defect:grading_environment_unavailable"] += 1
                    else:
                        if assessment.status == GradingEnvironmentStatus.MISMATCH:
                            admission_rejections["source_defect:grading_environment_mismatch"] += 1
                yield envelope["task"], envelope["runtime"]
        return
    root = cached_input(config.source)
    inputs = {name: cached_input(reference) for name, reference in config.inputs.items()}
    grader = None
    if config.grader is not None:
        if config.grader.image is not None:
            grader = environment_requirements(config.grader)
        else:
            with artifact_prefix(artifact_root):
                lock = planned_lock_url(config.grader)
            grader = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=lock)
    recipe = source_recipe(config, inputs, grader)
    context = conversion_context(recipe)
    records = (
        record
        for shard in source_shards(root, recipe.source)
        for record in staged_raw_file_rows(root, shard, recipe.source, context)
    )
    yield from converted_records(records, recipe, rejections, admission_rejections)


def converted_records(
    records: Iterable[dict], recipe: SourceRecipe, rejections: Counter, admission_rejections: Counter
) -> Iterable[tuple[dict, dict]]:
    """Retain emitted specifications and count common admission failures independently."""
    emitted: list[TaskSpec] = []

    def capture(row, context):
        result = recipe.convert(row, context)
        if isinstance(result, NormalizedTask):
            emitted.append(result.task)
        elif isinstance(result, TaskSpec):
            emitted.append(result)
        return result

    census_recipe = replace(recipe, convert=capture)
    for record in records:
        emitted.clear()
        conversion = convert_source_row(record, census_recipe)
        result = conversion.result
        if isinstance(result, ImportRejection):
            counts = admission_rejections if emitted else rejections
            counts[f"{result.kind.value}:{result.reason}"] += 1
        if not emitted:
            continue
        task = emitted[0].model_dump(mode="json")
        yield task, declared_shapes(conversion)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    selected = sorted(set(json.loads(args.ids.read_text())) - GRANDFATHERED_MARINSKYRL_SOURCES)
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(measure_source, source_id, args.artifact_root): source_id for source_id in selected}
        for future in as_completed(pending):
            source_id = pending[future]
            try:
                result = future.result()
            except Exception as error:
                # A source that cannot be measured is a retained ingestion failure.
                result = DaytonaCensus(
                    source_id,
                    None,
                    "ingestion_failure",
                    0,
                    0,
                    {},
                    0,
                    None,
                    {},
                    0,
                    {},
                    {},
                    None,
                    "No Daytona API calls",
                    f"{type(error).__name__}: {error}",
                    {},
                    converter_sha256(),
                    {},
                    {},
                    {},
                    {},
                )
            target = args.output / (canonical_sha256(source_id)[:20] + ".json")
            target.write_text(json.dumps(asdict(result), indent=2) + "\n")
            logger.info("Census %s: %s %s (%d tasks)", source_id, result.status, result.verdict, result.converted)


if __name__ == "__main__":
    main()
