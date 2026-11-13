# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure calibration prerequisites on frozen retained TaskSpec populations."""

import argparse
import json
import logging
import re
import shlex
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.grading_environment import assess_grading_environment
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import DOCKER_IMAGE_PATTERN, TaskSpec
from taskcompendium.pipeline.oracle_filter import has_oracle_solution
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.environment import dockerfile_lines
from experiments.post_training.task_curation.full_qc_review import load_inputs
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.native_quick import source_slug
from experiments.post_training.task_curation.remediation_publish import ReleasePlan, filter_population
from experiments.post_training.task_curation.retention_plan import RetentionPlan

LOGGER = logging.getLogger(__name__)
LEDGER_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("task_sha256", pa.string()),
        ("runtime_sha256", pa.string()),
        ("blockers", pa.list_(pa.string())),
    ]
)


def image_blockers(image: str | None, dockerfile: bytes | None) -> set[str]:
    """Identify absent or mutable image inputs without resolving or replacing them."""
    if image:
        return set() if re.fullmatch(DOCKER_IMAGE_PATTERN, image) else {"mutable_image"}
    if dockerfile is None:
        return {"missing_environment"}
    stages: set[str] = set()
    found = False
    for line in dockerfile_lines(dockerfile.decode()):
        tokens = shlex.split(line)
        if tokens[0].upper() != "FROM":
            continue
        found = True
        fields = [token for token in tokens[1:] if not token.startswith("--")]
        base = fields[0]
        if base != "scratch" and base not in stages and not re.fullmatch(DOCKER_IMAGE_PATTERN, base):
            return {"mutable_docker_base"}
        if len(fields) == 3 and fields[1].upper() == "AS":
            stages.add(fields[2])
    return set() if found else {"missing_docker_base"}


@cache
def packages_lock_exists(path: str) -> bool:
    return StoragePath(path).exists()


def task_blockers(task: TaskSpec) -> set[str]:
    """Record static execution blockers; an empty set supplies no qualification evidence."""
    blockers: set[str] = set()
    resources = {resource.path: resource for resource in task.resources.verifier}
    native = "harbor" in task.tags and "task.toml" in resources
    if native:
        config = tomllib.loads(resource_bytes(resources["task.toml"]).decode())
        environment = config.get("environment", {})
        dockerfile = resources.get("environment/Dockerfile")
        blockers.update(
            image_blockers(environment.get("docker_image"), None if dockerfile is None else resource_bytes(dockerfile))
        )
        oracle_paths = {resource.path for resource in task.resources.oracle}
        if has_oracle_solution(task) and "solution/solve.sh" not in oracle_paths:
            blockers.add("oracle_without_solve_sh")
    else:
        # Exporting these contracts requires a production conversation adapter; the legacy
        # exporter can introduce an actor image and file-answer instructions.
        blockers.add("requires_faithful_harbor_adapter")
        environments = [task.environment_requirements]
        if task.grader.kind != "none":
            environments.append(task.grader.environment)
        for environment in environments:
            if environment is None:
                continue
            lock = environment.packages_lock
            if lock and not packages_lock_exists(lock):
                blockers.add("missing_packages_lock")
    if not has_oracle_solution(task):
        blockers.add("requires_calibration_solution_and_empty_liveness")
    boundary = assess_grading_environment(task)
    if boundary.status != "matched":
        blockers.add("grading_environment_mismatch")
    return blockers


def inventory_source(plan: ReleasePlan, output_root: str, prepared_path: str | None = None) -> dict:
    """Inspect every retained task after checking frozen population and drop accounting."""
    root = StoragePath(output_root) / canonical_sha256(plan.source_id)[:20]
    if root.exists():
        raise FileExistsError(f"Select a new inventory version: {root}")
    population = SourceTextPopulation.raw_load(plan.input_artifact)
    expected = (
        plan.source_id,
        plan.source_dataset_id,
        plan.source_revision,
        plan.converter_version,
        plan.converter_revision,
        plan.task_specs_sha256,
        plan.raw_count,
    )
    actual = (
        str(population.source_id),
        population.dataset_id,
        population.dataset_revision,
        population.source_version,
        population.converter_revision,
        population.task_specs_sha256,
        population.task_count,
    )
    if actual != expected:
        raise ValueError(f"Frozen population differs from retention plan: {plan.source_id}")
    counts: Counter[str] = Counter()
    boundaries: Counter[str] = Counter()
    missing = oracles = count = ready = 0
    with TemporaryDirectory(prefix=f"calibration-{source_slug(plan.source_id)[:24]}-") as scratch:
        cohort = Path(scratch) / "tasks.parquet"
        drop_sha256 = file_sha256(StoragePath(plan.combined_drop_manifest_uri))
        expected_contracts = None
        if prepared_path is not None:
            inputs = load_inputs(prepared_path)
            if inputs.population != population or inputs.retention != RetentionPlan.model_validate(plan.model_dump()):
                raise ValueError(f"Prepared index belongs to another population: {plan.source_id}")
            if inputs.drop_manifest_sha256 != drop_sha256:
                raise ValueError(f"Drop manifest differs from prepared inputs: {plan.source_id}")
            if file_sha256(StoragePath(inputs.index_path)) != inputs.index_sha256:
                raise ValueError(f"Prepared contract index changed: {plan.source_id}")
            with StoragePath(inputs.index_path).open("rb") as stream:
                contracts = pq.read_table(stream).to_pylist()
            expected_contracts = {row["task_id"]: row for row in contracts}
            if len(expected_contracts) != len(contracts) or len(contracts) != plan.retained_count:
                raise ValueError(f"Incomplete prepared contract index: {plan.source_id}")
        filter_population(plan, population.specs_output_dir, cohort, None)
        if file_sha256(StoragePath(plan.combined_drop_manifest_uri)) != drop_sha256:
            raise ValueError(f"Drop manifest changed during inventory: {plan.source_id}")
        ledger_path = root / "tasks.parquet"
        with ledger_path.open("wb", auto_mkdir=True) as stream, pq.ParquetWriter(stream, LEDGER_SCHEMA) as writer:
            for batch in pq.ParquetFile(cohort).iter_batches(batch_size=32):
                rows = []
                for row in batch.to_pylist():
                    task = TaskSpec.model_validate_json(row["task_spec"])
                    contract = {
                        "task_id": task.id,
                        "task_sha256": canonical_sha256(task.model_dump(mode="json")),
                        "runtime_sha256": canonical_sha256(json.loads(row["runtime"])),
                    }
                    if expected_contracts is not None and expected_contracts.get(task.id) != contract:
                        raise ValueError(f"Task or runtime differs from prepared contract: {task.id}")
                    blockers = task_blockers(task)
                    if expected_contracts is None:
                        blockers.add("requires_frozen_contract_index")
                    counts.update(blockers)
                    boundary = assess_grading_environment(task)
                    boundaries[f"{boundary.required}:{boundary.status}"] += 1
                    count += 1
                    oracles += has_oracle_solution(task)
                    missing += task.execution_requirements is None
                    ready += not blockers
                    rows.append({**contract, "blockers": sorted(blockers)})
                writer.write_table(pa.Table.from_pylist(rows, schema=LEDGER_SCHEMA))
    result = {
        "source_id": plan.source_id,
        "plan": plan.model_dump(mode="json"),
        "retained_count": count,
        "missing_execution_requirements": missing,
        "oracle_tasks": oracles,
        "static_prerequisites_satisfied": ready,
        "blockers": dict(counts),
        "grading_environments": dict(boundaries),
        "ledger_path": str(ledger_path),
        "ledger_sha256": file_sha256(ledger_path),
        "drop_manifest_sha256": drop_sha256,
        "qualification_status": "pending",
        "prepared_inputs": prepared_path,
        "contract_verification": "prepared_index" if prepared_path is not None else "observed_only",
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("%s: %d retained, %d static prerequisites satisfied", plan.source_id, count, ready)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-plans", required=True)
    parser.add_argument("--source-ids", required=True)
    parser.add_argument("--prepared-inputs", help="source ID to previously frozen FULL-QC input artifact JSON")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    identifiers = json.loads(StoragePath(args.source_ids).read_text())
    plans = [
        ReleasePlan.model_validate(row)
        for row in json.loads(StoragePath(args.retention_plans).read_text())
        if row["source_id"] in identifiers
    ]
    if len(plans) != len(identifiers) or {plan.source_id for plan in plans} != set(identifiers):
        raise ValueError("Selection must name distinct sources present in the frozen plans")
    prepared = {} if args.prepared_inputs is None else json.loads(StoragePath(args.prepared_inputs).read_text())
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_coreweave_s3) as pool:
        futures = [pool.submit(inventory_source, plan, args.output_root, prepared.get(plan.source_id)) for plan in plans]
        results = [future.result() for future in as_completed(futures)]
    (StoragePath(args.output_root) / "inventory.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
