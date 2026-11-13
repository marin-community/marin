# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census current Atlas grader contracts and freeze deterministic content panels."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import SourceDeclaration
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import NoGrader, TaskSpec
from taskcompendium.pipeline.oracle_filter import has_oracle_solution
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training.task_curation.admission_binding import AdmissionCensusReceipt
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.prebuilt import validate_native_package

LOGGER = logging.getLogger(__name__)
SAMPLE_SIZE = 100
SAMPLE_SEED = 0
PANEL_SCHEMA = pa.schema([("task_id", pa.string()), ("task_spec", pa.large_string()), ("runtime", pa.large_string())])


def grader_contract(task: TaskSpec) -> str:
    """Distinguish executable native packages from ordinary missing graders."""
    if not isinstance(task.grader, NoGrader):
        return task.grader.kind
    if "harbor" not in task.tags:
        return "none"
    validate_native_package(task)
    return "native_harbor"


def inventory_source(declaration: SourceDeclaration, output_root: str, resume_root: str | None) -> dict:
    """Verify the current complete population and retain its seed-zero content panel."""
    root = StoragePath(output_root) / canonical_sha256(declaration.id)[:20]
    if root.exists():
        raise FileExistsError(f"Preserve previous inventory and use a new version: {root}")
    binding = declaration.actor_census_input
    if binding is None:
        raise ValueError(f"Current source lacks its exact input census: {declaration.id}")
    actor = ActorCensus.raw_load(str(StoragePath(binding.evidence_url).parent))
    population = SourceTextPopulation.raw_load(actor.input_artifact)
    expected = (declaration.id, declaration.dataset_id, declaration.dataset_revision, declaration.pipeline.version)
    actual = (population.source_id, population.dataset_id, population.dataset_revision, population.source_version)
    if actual != expected or (
        population.task_count != binding.measured_tasks
        or population.task_specs_sha256 != binding.task_specs_sha256
        or population.task_count != declaration.within_source_dedup.unique_count
    ):
        raise ValueError(f"Population differs from the current declaration: {declaration.id}")
    if resume_root is not None:
        previous = StoragePath(resume_root) / canonical_sha256(declaration.id)[:20] / "summary.json"
        if previous.exists():
            result = json.loads(previous.read_text())
            census_receipt(declaration, result)
            if (
                result["input_artifact"] != population.path
                or result["sample_seed"] != SAMPLE_SEED
                or result["sample_size"] != min(population.task_count, SAMPLE_SIZE)
            ):
                raise ValueError(f"Completed panel belongs to another population or sample design: {declaration.id}")
            if file_sha256(StoragePath(result["panel_path"])) != result["panel_sha256"]:
                raise ValueError(f"Completed sample panel changed: {declaration.id}")
            LOGGER.info("Reused complete admission census and panel: %s", declaration.id)
            return result
    digest = hashlib.sha256()
    seen = set()
    kinds: Counter[str] = Counter()
    missing_reasons: Counter[str] = Counter()
    oracle_count = 0

    def rows():
        nonlocal oracle_count
        for path in sorted((StoragePath(population.specs_output_dir) / "*.parquet").glob(), key=str):
            with path.open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(batch_size=16, columns=["task_spec", "runtime"]):
                    for row in batch.to_pylist():
                        encoded = row["task_spec"]
                        task = TaskSpec.model_validate_json(encoded)
                        if task.id in seen:
                            raise ValueError(f"Duplicate task in current population: {task.id}")
                        seen.add(task.id)
                        digest.update(encoded.encode())
                        kind = grader_contract(task)
                        kinds[kind] += 1
                        if kind == "none":
                            assert isinstance(task.grader, NoGrader)
                            missing_reasons[task.grader.reason] += 1
                        oracle_count += has_oracle_solution(task)
                        yield {"task_id": task.id, "task_spec": encoded, "runtime": row["runtime"]}

    count, panel = seeded_sample(rows(), size=SAMPLE_SIZE, key=lambda row: seeded_order(row["task_id"], SAMPLE_SEED))
    if count != population.task_count:
        raise ValueError(f"Current task count differs from the frozen population: {declaration.id}")
    panel_path = root / "panel.parquet"
    with panel_path.open("wb", auto_mkdir=True) as stream:
        pq.write_table(pa.Table.from_pylist(panel, schema=PANEL_SCHEMA), stream, compression="zstd")
    result = {
        "source_id": declaration.id,
        "dataset_id": declaration.dataset_id,
        "dataset_revision": declaration.dataset_revision,
        "converter_version": declaration.pipeline.version,
        "input_artifact": population.path,
        "task_specs_sha256": population.task_specs_sha256,
        "scanned_task_specs_sha256": digest.hexdigest(),
        "retained_tasks": count,
        "runnable_grader_tasks": count - kinds["none"],
        "no_grader_tasks": kinds["none"],
        "grader_kinds": dict(kinds),
        "missing_grader_reasons": dict(missing_reasons),
        "oracle_tasks": oracle_count,
        "sample_seed": SAMPLE_SEED,
        "sample_size": len(panel),
        "panel_path": str(panel_path),
        "panel_sha256": file_sha256(panel_path),
        "execution_qualification": "not_assessed",
        "evidence_url": str(root / "summary.json"),
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("%s: %d tasks, %d missing graders", declaration.id, count, kinds["none"])
    return result


def census_receipt(declaration: SourceDeclaration, result: dict) -> AdmissionCensusReceipt:
    binding = declaration.actor_census_input
    if binding is None or (
        result["source_id"],
        result["dataset_id"],
        result["dataset_revision"],
        result["converter_version"],
        result["task_specs_sha256"],
    ) != (
        declaration.id,
        binding.dataset_id,
        binding.dataset_revision,
        binding.converter_version,
        binding.task_specs_sha256,
    ):
        raise ValueError(f"Admission census differs from the current input: {declaration.id}")
    return AdmissionCensusReceipt(
        source_id=declaration.id,
        input=binding,
        population_path=result["input_artifact"],
        scanned_task_specs_sha256=result["scanned_task_specs_sha256"],
        retained_tasks=result["retained_tasks"],
        runnable_grader_tasks=result["runnable_grader_tasks"],
        no_grader_tasks=result["no_grader_tasks"],
        grader_kinds=result["grader_kinds"],
        missing_grader_reasons=result["missing_grader_reasons"],
        evidence_url=result["evidence_url"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--resume-root")
    args = parser.parse_args()
    configure_coreweave_s3()
    declarations = [
        SourceDeclaration.model_validate(row) for row in json.loads(StoragePath(args.declarations).read_text()).values()
    ]
    if len({row.id for row in declarations}) != len(declarations):
        raise ValueError("Inventory requires distinct current declarations")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_coreweave_s3) as pool:
        futures = [
            pool.submit(partial(inventory_source, output_root=args.output_root, resume_root=args.resume_root), row)
            for row in declarations
        ]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda row: row["source_id"])
    (StoragePath(args.output_root) / "inventory.json").write_text(json.dumps(results, indent=2) + "\n")
    by_id = {row.id: row for row in declarations}
    receipts = [census_receipt(by_id[row["source_id"]], row).model_dump(mode="json") for row in results]
    (StoragePath(args.output_root) / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
