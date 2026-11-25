# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the repaired MiMo pin: text population, lexical scan, within-source dedup.

The repaired population's public text is the original prompts' text; the measurement
rebuilds the text population from the published repaired release and re-scans it against
the pinned shared eval bloom (atlas-eval-coverage-v2, the same basis the original pin was
measured on), then censuses exact task contracts with the fresh lexical outcome. The
three steps run in one regional job through the maintained modules, writing their
receipts to the release prefix.
"""

import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import LexicalUse
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from marin.execution.lazy import run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.lexical_scan import scan_steps
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.within_source_census import census_source
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

logger = logging.getLogger(__name__)
REGIONAL_PREFIX = "s3://marin-us-east-02a/marin"
SOURCE_ID = "FWeindel:validated-tasks/mimo_v1"
BLOOM_PATH = "s3://marin-us-east-02a/marin/datakit/bloom/_combined_fixed_3a94b52b"
EVAL_CORPUS_VERSION = "atlas-eval-coverage-v2"
POPULATION_SET = "retained"
INPUT_REPO = "open-athena/fweindel-validated-tasks-mimo-v1-pinned-images-20261124-a-v1"
INPUT_REVISION = "729925160824efba40d672c10ee4311169dc1a8b"
CONVERTER_VERSION = "2026.11.23.pinned-images.1"
CONVERTER_REVISION = "831fb168588948a3a9c28ab1d6063d255555f0f4"
EXPECTED_TASKS = 2501
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="4g")


def prepare_population(output_path: str) -> SourceTextPopulation:
    """Build the text population from the published repaired release and verify its pin."""
    tasks_local = Path(hf_hub_download(INPUT_REPO, "tasks.parquet", repo_type="dataset", revision=INPUT_REVISION))

    def documents():
        for batch in pq.ParquetFile(tasks_local).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                yield json.loads(row["task_spec"]), json.loads(row["runtime"]) if row["runtime"] else {}

    files = write_text_population(documents(), output_path)
    if files.task_count != EXPECTED_TASKS:
        raise ValueError(f"the repaired population holds {files.task_count} tasks, expected {EXPECTED_TASKS}")
    return SourceTextPopulation(
        path=output_path,
        source_id=SOURCE_ID,
        dataset_id=INPUT_REPO,
        dataset_revision=INPUT_REVISION,
        source_version=CONVERTER_VERSION,
        converter_revision=CONVERTER_REVISION,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements-root", default=REGIONAL_PREFIX + "/rl-data/atlas/ingest-b/fweindel-mimo-v1/2026.11.23.1"
    )
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    arguments = parser.parse_args()
    configure_coreweave_s3()
    measurements_root = f"{arguments.measurements_root}/measurements-{arguments.artifact_version}"

    runtime = CampaignRuntime()
    with artifact_prefix(f"{arguments.measurements_root}/artifacts-{arguments.artifact_version}"):
        population = prepare_population(f"{measurements_root}/population-v1")
        StoragePath(f"{measurements_root}/populations.json").write_text(
            json.dumps([population.model_dump(mode="json")], indent=1) + "\n"
        )
        report = scan_steps(
            [population],
            BLOOM_PATH,
            EVAL_CORPUS_VERSION,
            POPULATION_SET,
            arguments.artifact_version,
            arguments.max_workers,
            runtime,
        )
        with ExitStack() as cleanup:
            client = LocalClient(max_threads=max(16, arguments.max_workers * 2))
            cleanup.callback(client.shutdown)
            cleanup.enter_context(set_current_client(client))
            with ZephyrContext(client=client, max_workers=arguments.max_workers, resources=WORKER_RESOURCES) as context:
                with runtime.activate(context):
                    result = run(report, max_concurrent=2)[0]
    dump = result.model_dump(mode="json")
    measurement = dump["measurements"][canonical_source_id(SOURCE_ID)]
    StoragePath(f"{measurements_root}/lexical-result.json").write_text(json.dumps(dump, indent=1) + "\n")
    measurement_json = census_source(
        population,
        {
            "contaminated_tasks": measurement["contaminated_tasks"],
            "evidence_url": (measurement.get("drop_set_uris") or [""])[0],
            "measured_tasks": measurement["measured_tasks"],
        },
        f"{measurements_root}/within-source-v1",
        lexical_use=LexicalUse.TRAIN,
    )
    StoragePath(f"{measurements_root}/within-source-measurement.json").write_text(
        json.dumps(measurement_json, indent=1) + "\n"
    )
    logger.info(
        "lexical: %d contaminated of %d; dedup: %d raw, %d unique, %d duplicates",
        measurement["contaminated_tasks"],
        measurement["measured_tasks"],
        measurement_json["raw_count"],
        measurement_json["unique_count"],
        measurement_json["within_source_duplicates"],
    )


if __name__ == "__main__":
    main()
