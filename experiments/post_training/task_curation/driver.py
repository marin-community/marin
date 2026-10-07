# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan or execute the Atlas campaign inside a single Iris driver job."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import ArtifactStep
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.direct_transport import MAX_DIRECT_CONCURRENT_REQUESTS
from taskcompendium.pipeline.models import FilterPolicy, HFSource
from taskcompendium.pipeline.review import BatchReviewer, DirectReviewer, Reviewer
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewTransport

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation.campaign import (
    CampaignPool,
    campaign_identity,
    campaign_plan,
    require_matching_sample,
    run_campaign,
)
from experiments.post_training.task_curation.pipeline import (
    RecordedReviewInput,
    SourceRuntime,
    SourceRuntimeConfig,
)
from experiments.post_training.task_curation.sources import rl_data_pipelines

REVIEW_REQUEST_TIMEOUT = 60


def source_input_artifacts(path: Path | None) -> dict[str, ArtifactStep[Artifact]]:
    """Adopt staged files only when their declared input matches the recipe pin."""
    if path is None:
        return {}
    definitions = rl_data_pipelines()
    artifacts = {}
    for name, declaration in json.loads(path.read_text()).items():
        source = HFSource(**declaration["source"])
        if name not in definitions:
            raise ValueError(f"Staged input has unknown source: {name}")
        if source != definitions[name].source:
            raise ValueError(f"Staged input does not match the pinned recipe for {name}")
        metadata = {"source": asdict(source), "path": declaration["path"]}
        identity = fingerprint_hash(canonical_json(metadata))[:16]
        artifacts[name] = ArtifactStep.adopt(
            f"task-curation/input/{name}-{identity}",
            declaration["version"],
            source=declaration["path"],
            kind=Artifact,
            config=metadata,
        )
    return artifacts


def recorded_review_bundles(path: Path | None) -> dict[str, RecordedReviewInput]:
    """Declare exact immutable reviewer evidence for canonical source bindings."""
    if path is None:
        return {}
    definitions = {definition.source_key for definition in rl_data_pipelines().values()}
    bundles = {}
    for name, declaration in json.loads(path.read_text()).items():
        if name not in definitions:
            raise ValueError(f"Recorded review bundle has unknown source binding: {name}")
        bundle = RecordedReviewInput(**declaration)
        if len(bundle.sha256) != 64 or any(character not in "0123456789abcdef" for character in bundle.sha256):
            raise ValueError(f"Recorded review bundle requires an exact SHA-256 digest for {name}")
        bundles[name] = bundle
    return bundles


def normalization_only_sample(path: str) -> bool:
    """Identify admitted samples whose unbound controls withheld every output."""
    source = StoragePath(path)
    report_path = source / "report.json"
    manifest_path = source / "work/verified/manifest.json"
    if not report_path.exists() or not manifest_path.exists():
        return False
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    verification = report["verification"]
    counts = verification["counts"]
    return (
        report["mode"] == "sample"
        and report["status"] in {"sampled", "completed"}
        and report["incomplete_reviews"] == 0
        and manifest["input_rows"] == report["processed_rows"]
        and sum(manifest["dispositions"].values()) == report["processed_rows"]
        and manifest["dispositions"].get("keep", 0) == 0
        and verification["status"] == "inconclusive"
        and counts["unsupported"] > 0
        and counts["infra_error"] == 0
    )


@click.command(help=__doc__)
@click.option("--runtime-manifest", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--staged-inputs", type=click.Path(exists=True, path_type=Path))
@click.option("--recorded-review-bundles", "recorded_review_manifest", type=click.Path(exists=True, path_type=Path))
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--model-revision", required=True)
@click.option("--base-url")
@click.option("--review-cache", required=True)
@click.option("--review-transport", type=click.Choice(["provider-batch", "direct-chat"]), required=True)
@click.option(
    "--review-concurrency",
    type=click.IntRange(min=1, max=MAX_DIRECT_CONCURRENT_REQUESTS),
    default=MAX_DIRECT_CONCURRENT_REQUESTS,
    show_default=True,
)
@click.option("--mode", type=click.Choice(["sample", "full"]), default="sample", show_default=True)
@click.option("--max-workers", type=click.IntRange(min=1), required=True)
@click.option("--coordinator-memory", required=True, help="Explicit RAM budget for the shared coordinator, e.g. 16g.")
@click.option("--normalized-shards", type=click.IntRange(min=1), required=True)
@click.option("--concurrent-sources", type=click.IntRange(min=10), default=10, show_default=True)
@click.option("--worker-image", required=True)
@click.option("--controller-url")
@click.option("--seed", type=int, default=0)
@click.option("--verification-sample-size", type=click.IntRange(min=1), default=100)
@click.option("--report-path", required=True)
@click.option("--sample-report", help="Terminal sample campaign report required before executing full mode.")
@click.option("--source", "sources", multiple=True, help="Canonical source name to execute; repeat to select multiple.")
@click.option("--run", "do_run", is_flag=True)
def main(
    runtime_manifest: Path,
    staged_inputs: Path | None,
    recorded_review_manifest: Path | None,
    model: str,
    model_revision: str,
    base_url: str | None,
    review_cache: str,
    review_transport: str,
    review_concurrency: int,
    mode: str,
    max_workers: int,
    coordinator_memory: str,
    normalized_shards: int,
    concurrent_sources: int,
    worker_image: str,
    controller_url: str | None,
    seed: int,
    verification_sample_size: int,
    report_path: str,
    sample_report: str | None,
    sources: tuple[str, ...],
    do_run: bool,
) -> None:
    catalog = rl_data_pipelines()
    unknown = set(sources) - catalog.keys()
    if unknown:
        raise click.UsageError(f"Unknown source: {', '.join(sorted(unknown))}")
    pipelines = {name: pipeline for name, pipeline in catalog.items() if not sources or name in sources}
    manifest = json.loads(runtime_manifest.read_text())
    runtime = SourceRuntimeConfig(
        images={name: SourceRuntime(**values) for name, values in manifest.items()},
        controller_url=controller_url,
        source_inputs=source_input_artifacts(staged_inputs),
        recorded_reviews=recorded_review_bundles(recorded_review_manifest),
    )
    review = ReviewConfig(model=model, model_revision=model_revision, transport=ReviewTransport(review_transport))
    worker_resources = ResourceConfig(cpu=2, ram="8g", image=worker_image)
    reviewer: Reviewer | None = None
    if do_run:
        if base_url is None:
            raise click.UsageError("--base-url is required with --run")
        if mode == "full":
            if sample_report is None:
                raise click.UsageError("Full execution requires --sample-report")
        token = os.environ[GLM_BULK_TOKEN_ENV]
        batch_client = OpenAIBatchClient(base_url, token, timeout=REVIEW_REQUEST_TIMEOUT, request_attempts=3)
        if review.transport == ReviewTransport.DIRECT_CHAT:
            reviewer = DirectReviewer(
                OpenAIChatClient(base_url, token, timeout=REVIEW_REQUEST_TIMEOUT),
                model,
                model_revision,
                query_cache_root=review_cache,
                max_concurrent=review_concurrency,
                max_batch_bytes=review.max_batch_bytes,
            )
        else:
            reviewer = BatchReviewer(
                batch_client,
                model,
                model_revision,
                query_cache_root=review_cache,
                max_batch_bytes=review.max_batch_bytes,
            )
    config = SourcePipelineConfig(
        mode=SourceProcessingMode(mode),
        quality_policy=SourceQualityPolicy(sample_size=100, seed=seed),
        verification_policy=SourceVerificationPolicy(verification_sample_size, seed, 2, 0.95),
        review=review,
        execution=AuditExecution(
            max_workers=max_workers,
            review_batch_size=64,
            reviewer=reviewer,
            worker_resources=worker_resources,
            # Iris isolates each shard in a process. Admit one review process per
            # worker so its request semaphore enforces the worker's provider cap.
            review_task_resources=worker_resources,
        ),
        filter_policy=FilterPolicy(),
        normalized_shards=normalized_shards,
    )
    steps = [pipeline.bind(config, runtime) for pipeline in pipelines.values()]
    sample_steps = (
        steps
        if mode == "sample"
        else [
            pipeline.bind(replace(config, mode=SourceProcessingMode.SAMPLE), runtime) for pipeline in pipelines.values()
        ]
    )
    sample_identity = campaign_identity(sample_steps, worker_image)
    pool = CampaignPool(
        max_workers,
        concurrent_sources,
        worker_resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram=coordinator_memory, preemptible=False),
    )
    if do_run:
        sample_outcomes = None
        if mode == "full":
            assert sample_report is not None
            sampled = require_matching_sample(
                json.loads(StoragePath(sample_report).read_text()), sample_identity, sample_steps
            )
            verification_inputs = {}
            for definition, sample_step, outcome in zip(pipelines.values(), sample_steps, sampled, strict=True):
                if outcome.status not in {"sampled", "completed"}:
                    continue
                provenance = {
                    "campaign_report": sample_report,
                    "sample_identity": sample_identity,
                    "sample_source": sample_step.name,
                    "sample_fingerprint": sample_step.fingerprint(),
                    "sample_path": outcome.path,
                }
                verification_inputs[definition.source_key] = ArtifactStep.adopt(
                    f"task-curation/verification-input/{definition.source_key}-{fingerprint_hash(canonical_json(provenance))[:16]}",
                    sample_step.version,
                    source=outcome.path,
                    kind=Artifact,
                    config=provenance,
                )
            runtime = replace(runtime, verification_inputs=verification_inputs)
            steps = [
                pipeline.bind(
                    (
                        replace(config, mode=SourceProcessingMode.NORMALIZE_ONLY)
                        if outcome.status in {"sampled", "completed"} and normalization_only_sample(outcome.path)
                        else config
                    ),
                    runtime,
                )
                for pipeline, outcome in zip(pipelines.values(), sampled, strict=True)
            ]
            sample_outcomes = {step.name: outcome for step, outcome in zip(steps, sampled, strict=True)}
        run_campaign(
            steps,
            runtime=runtime.campaign,
            pool=pool,
            report_path=report_path,
            sample_identity=sample_identity,
            mode=mode,
            sample_outcomes=sample_outcomes,
        )
        return
    click.echo(json.dumps(campaign_plan(steps, pool), indent=2))


if __name__ == "__main__":
    main()
