# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own the pinned OpenResearcher index for a long-loop RL attempt."""

import hashlib
from dataclasses import dataclass, replace

import yaml
from fray.types import ResourceConfig, create_environment
from iris.cluster.config import load_config
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import sanitize_job_name
from marin.rl.retrieval_service import RetrievalServiceConfig, RetrievalServiceLaunch, retrieval_service
from marin.rl.skyrl import ArtifactDataSource, IrisSkyRLExecution, SkyRLRun, SkyRLRunConfig, run_skyrl
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import require_regional_coreweave_bucket
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior, require_secret
from rigging.timing import Duration
from skyrlrecipe import RecipePatch
from taskcompendium.openresearcher import OpenResearcherState

from experiments.evaluation.judge_config import MICRO210_JUDGE

JUDGE_SECRET = "OPENRESEARCHER_JUDGE_KEY"

RETRIEVAL_IMAGE = (
    "ghcr.io/marin-community/rl-data-openresearcher@sha256:"
    "6ccb79c34d3c1da34c6bd947b9a0206e852b2e00d79b671efade9af1f853e9cb"
)
RETRIEVAL_STATE = OpenResearcherState(
    implementation_revision="036cceab31e06844720488fd29e504d479777f05",
    corpus_revision="8367703babe6e080899a48782f365c284c32b668",
    index_manifest_sha256="3203f0e6387c2e6a3e7509a5d4261cbce2c8e263abb0b72bf2eb580f1929f177",
    tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
    top_k=5,
    snippet_max_tokens=512,
    include_get_document=True,
)


@dataclass(frozen=True)
class OpenResearcherService:
    """Explicit source selection and regional index for the service-backed RL recipe."""

    dataset_id: str
    dataset_revision: str
    index_uri: str
    parquet_sha256: str


def service_data(source: OpenResearcherService, *, version: str) -> ArtifactDataSource:
    """Select the private service variant at its immutable HF revision."""

    def build_config(ctx: StepContext) -> DownloadConfig:
        return DownloadConfig(
            hf_dataset_id=source.dataset_id,
            revision=source.dataset_revision,
            hf_urls_glob=["train.parquet", "retrieval-service.json"],
            gcs_output_path=ctx.output_path,
        )

    return ArtifactDataSource(
        ArtifactStep(
            name="openresearcher/service-input/" + source.dataset_revision,
            version=version,
            artifact_type=Artifact,
            run=download_hf,
            build_config=build_config,
        ),
        relative_path="train.parquet",
    )


def service_execution(execution: IrisSkyRLExecution) -> IrisSkyRLExecution:
    sources = dict(execution.coordinator_secret_sources)
    if JUDGE_SECRET in sources and sources[JUDGE_SECRET] != MICRO210_JUDGE.credential:
        raise ValueError("OpenResearcher judge credential differs from the canonical policy")
    sources[JUDGE_SECRET] = MICRO210_JUDGE.credential
    return replace(
        execution,
        required_env_vars=tuple(name for name in execution.required_env_vars if name != "TOGETHER_API_KEY"),
        coordinator_secret_sources=tuple(sources.items()),
    )


def service_recipe() -> RecipePatch:
    machine = {
        "backend": "daytona",
        "network": "deny",
        "cpus": 1,
        "memory_mb": 2048,
        "storage_mb": 2048,
        "gpus": 0,
        "user": None,
        "startup_timeout": 600,
        "cleanup_timeout": 120,
    }
    return RecipePatch(
        environment={
            "task_sessions": {
                "daytona": {
                    "api_key_env": "DAYTONA_RL_API_KEY",
                    "eval_api_key_env": "DAYTONA_RL_API_KEY",
                    "ttl_minutes": 30,
                },
                "remote_judge": {"base_url": MICRO210_JUDGE.base_url, "api_key_env": "TOGETHER_API_KEY"},
                "task_spec": {
                    "max_turns_by_dataset": {"OpenResearcher/OpenResearcher-Dataset": 20},
                    "task_machine_by_dataset": {"OpenResearcher/OpenResearcher-Dataset": machine},
                    "image_verifier": {**machine, "network": "allow"},
                },
            }
        },
        extra_env={"VERIFYIT_JUDGE_MODEL": MICRO210_JUDGE.model},
    )


def retrieval_launch(source: OpenResearcherService, execution: IrisSkyRLExecution) -> RetrievalServiceLaunch:
    cluster = load_config(execution.cluster_config)
    if cluster.platform.coreweave is None:
        raise ValueError("The retrieval service requires a CoreWeave run region")
    region = cluster.platform.coreweave.region
    require_regional_coreweave_bucket(source.index_uri, region)
    return RetrievalServiceLaunch(
        config=RetrievalServiceConfig(
            state=RETRIEVAL_STATE,
            index_uri=source.index_uri,
            region=region,
            root="/opt/openresearcher",
            backend_python="/usr/local/bin/python",
            max_concurrent_requests=64,
            endpoint_name="openresearcher-retrieval",
        ),
        resources=ResourceConfig(cpu=8, ram="32GB", disk="128GB", preemptible=False, image=RETRIEVAL_IMAGE),
        environment=create_environment(
            sync_packages=["marin-core"],
            extras=["cpu"],
            env_vars={"MARIN_CLUSTER": "coreweave", "JAVA_HOME": "/opt/java21", "JDK_JAVA_OPTIONS": "-Xmx2048m"},
            secret_behavior=SecretBehavior.ISOLATE,
        ),
        startup_timeout=3600,
        lifetime=Duration.from_hours(execution.coordinator_timeout_hours),
    )


def run_with_retrieval(config: SkyRLRunConfig, *, launch: RetrievalServiceLaunch, parquet_sha256: str) -> SkyRLRun:
    """Inject a ready attempt-owned endpoint and keep its child alive through RL cleanup."""
    document = yaml.safe_load(config.launch_config_yaml)
    (source,) = document["inputs"]["train_data"]
    fs, path = filesystem_for(str(StoragePath(source["uri"]) / source["relative_path"]))
    with fs.open(path, "rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != parquet_sha256:
        raise ValueError("The service-backed OpenResearcher Parquet differs from its pinned SHA-256")
    launch = replace(
        launch,
        config=replace(launch.config, endpoint_name=sanitize_job_name(f"{config.run_id}-{config.attempt_id}-retrieval")),
    )
    with retrieval_service(launch) as endpoint:
        document["skyrl"]["environment"]["task_sessions"]["retrieval_service"] = endpoint
        return run_skyrl(
            replace(config, launch_config_yaml=yaml.safe_dump(document, sort_keys=False)),
            env_vars={"TOGETHER_API_KEY": require_secret(JUDGE_SECRET)},
        )
