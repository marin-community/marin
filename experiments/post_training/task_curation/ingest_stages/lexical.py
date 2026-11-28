# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the maintained lexical graph against exactly the sealed deduplicated source."""

from marin.execution.lazy import run as run_artifacts
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage
from experiments.post_training.task_curation.launch_lexical_scan import BLOOM, EVAL_CORPUS_VERSION
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_scan import scan_steps
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


def lexical_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Measure contamination without treating a raw measurement as admission."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    parent_receipt(job.execution, IngestStage.BACKEND_ORACLE, job.inputs.dependencies[IngestStage.BACKEND_ORACLE])
    bloom = StoragePath(BLOOM)
    if bloom.bucket != StoragePath(job.execution.artifact_root).bucket:
        raise ValueError("lexical bloom is outside the source's regional deployment")
    population = SourceTextPopulation.model_validate(read_ref(job.inputs.artifacts["dedup/population"]))
    runtime = CampaignRuntime()
    with local_client() as client:
        with (
            ZephyrContext(
                client=client,
                max_workers=1,
                chunk_storage_prefix=prefix_join(job.output_root, "chunks"),
            ) as context,
            runtime.activate(context),
            artifact_prefix(job.output_root),
        ):
            step = scan_steps(
                [population],
                str(bloom),
                EVAL_CORPUS_VERSION,
                canonical_sha256(job.execution.identity()),
                job.execution.artifact_version,
                1,
                runtime,
            )
            result = run_artifacts(step, max_concurrent=1)[0]
    if result.source_ids != [population.source_id]:
        raise ValueError("lexical report measured a different source set")
    measured = result.measurements[population.source_id]
    if measured["measured_tasks"] != population.task_count:
        raise ValueError("lexical report does not cover the complete declared population")
    outputs = {"report": seal_manifest(result.model_dump(mode="json"), prefix_join(job.output_root, "report.json"))}
    for index, path in enumerate(sorted((StoragePath(result.path) / "**/*.parquet").glob(), key=str)):
        outputs[f"measurement/{index}"] = ArtifactRef(uri=str(path), sha256=file_sha256(path))
    return outputs
