# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare regional census and seed-zero panel evidence for the facet_v2 repair."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from iris.cluster.client.job_info import get_job_info
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.identity import task_spec_json
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.admission_inventory import inventory_population
from experiments.post_training.task_curation.census_binding import integrity_summary_for_actor, receipt_from_census
from experiments.post_training.task_curation.derived_contract_census import (
    DerivedContractCensusInput,
)
from experiments.post_training.task_curation.derived_contract_census import (
    census_source as derived_census,
)
from experiments.post_training.task_curation.grading_census import GradingCensusInput
from experiments.post_training.task_curation.grading_census import census_source as grading_census
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.lexical_published import PublishedPopulation
from experiments.post_training.task_curation.published_census import CensusConfig, measure_population

SOURCE_ID = "FWeindel:validated-tasks/facet_v2"
REPAIR_REPO = "open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1"
REPAIR_REVISION = "e9099dac47e02dbe53ab559583f67f5353857c3c"
REPAIR_PARQUET_SHA256 = "f026ee23f5eee69a5d191f974a3bd72ed681d66c1f5d75afa9cc85a343d7c34d"
REPAIR_TASKS = 775


def prepare_population(parquet: Path, metadata: PublishedPopulation, output_root: str) -> SourceTextPopulation:
    """Preserve published task contracts and runtime packets in the regional population."""
    with parquet.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != metadata.tasks_parquet_sha256:
            raise ValueError("Repair parquet differs from its published readback hash")
    root = StoragePath(output_root)
    if root.exists():
        raise FileExistsError("Preserve previous preparation and select a new output version")

    def documents():
        for batch in pq.ParquetFile(parquet).iter_batches(
            batch_size=16, columns=["task_spec", "task_sha256", "runtime"]
        ):
            for row in batch.to_pylist():
                encoded = row["task_spec"]
                if hashlib.sha256(encoded.encode()).hexdigest() != row["task_sha256"]:
                    raise ValueError("Published repair task differs from its row hash")
                task = TaskSpec.model_validate_json(encoded)
                if task_spec_json(task) != encoded:
                    raise ValueError("Current serializer would change the published repair task bytes")
                yield json.loads(encoded), json.loads(row["runtime"])

    files = write_text_population(documents(), output_root)
    if files.raw_task_count != metadata.task_count or files.task_count != metadata.task_count:
        raise ValueError("Repair preparation changed the frozen population count")
    population = SourceTextPopulation(
        path=output_root,
        source_id=metadata.source_id,
        dataset_id=metadata.dataset_id,
        dataset_revision=metadata.dataset_revision,
        source_version=metadata.converter_version,
        converter_revision=metadata.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=population.path,
            result_type=result_type_name(type(population)),
            result=population.result_payload(),
        )
    )
    return population


def prepare_evidence(parquet: Path, metadata: PublishedPopulation, output_root: str) -> dict:
    """Measure the complete repair population and freeze its panel without admitting it."""
    root = StoragePath(output_root)
    population = prepare_population(parquet, metadata, str(root / "population"))
    actor = measure_population(CensusConfig(population.path, str(root / "actor")))
    write_record(
        ArtifactRecord(output_path=actor.path, result_type=result_type_name(type(actor)), result=actor.result_payload())
    )
    record = actor.model_dump(mode="json")
    integrity = integrity_summary_for_actor(record, {**record, "summary": actor.integrity_contracts})
    binding = receipt_from_census(record, integrity)
    grading = grading_census(GradingCensusInput(population=population, input=binding.input), str(root / "grading"))
    derived = derived_census(
        DerivedContractCensusInput(population=population, input=binding.input), str(root / "derived")
    )
    assert derived is not None
    panel = inventory_population(population, str(root / "inventory"), None)
    result = {
        "source_id": metadata.source_id,
        "population": population.model_dump(mode="json"),
        "actor": binding.model_dump(mode="json"),
        "grading": grading.model_dump(mode="json"),
        "derived": derived.model_dump(mode="json"),
        "inventory": panel.model_dump(mode="json"),
        "qualification": "pending fresh SAMPLE review, lexical qualification and admission registry publication",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--converter-version", required=True)
    args = parser.parse_args()
    info = get_job_info()
    if info is None or info.worker_region != "us-central2":
        raise ValueError("Facet repair inputs must be prepared by an Iris worker in us-central2")
    if not args.output_root.startswith("gs://marin-us-central2/"):
        raise ValueError("Facet repair outputs must stay in us-central2 GCS")
    configure_coreweave_s3()
    parquet = Path(
        hf_hub_download(
            REPAIR_REPO, "tasks.parquet", repo_type="dataset", revision=REPAIR_REVISION, token=require_secret("HF_TOKEN")
        )
    )
    root = StoragePath(args.output_root)
    metadata = PublishedPopulation(
        source_id=SOURCE_ID,
        dataset_id=REPAIR_REPO,
        dataset_revision=REPAIR_REVISION,
        converter_version=args.converter_version,
        converter_revision=args.converter_revision,
        tasks_parquet_uri=str(parquet),
        tasks_parquet_sha256=REPAIR_PARQUET_SHA256,
        task_count=REPAIR_TASKS,
        converter_rejections={},
        admission_rejections={},
    )
    result = prepare_evidence(parquet, metadata, str(root))
    (root / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "receipt": str(root / "preparation.json"),
                "tasks": result["population"]["task_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
