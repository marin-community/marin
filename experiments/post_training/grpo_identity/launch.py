# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the identity repair smoke through the standard RL artifact entrypoint."""

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import click
import yaml
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

HERE = Path(__file__).parent
BASE = "open-athena/Marin-Snowball-67B-A2B-v0.1"
BASE_REVISION = "adc0078a19a4383bc998fcdb43867b3360baf5f3"


@dataclass(frozen=True)
class InputFiles:
    output_path: str
    source_path: str
    hashes: dict[str, str]


def stage_inputs(config: InputFiles) -> None:
    for name, digest in config.hashes.items():
        path = Path(config.source_path) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input changed after graph construction: {name}")
        payload = path.read_bytes()
        if name.endswith(".gz"):
            payload = gzip.decompress(payload)
        (StoragePath(config.output_path) / name.removesuffix(".gz")).write_bytes(payload)


def build():
    version = resolve_version("grpo-identity", None)
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    recipe = config["skyrl"]
    if any("UNRESOLVED" in str(value) for value in recipe["identity_repair"].values()):
        raise ValueError("Resolve and verify identity manifests and judge settings before launch")
    model = ArtifactStep(
        name="models/benfeuer/grpo-identity-base",
        version="2026.10.23.1",
        artifact_type=LevanterCheckpoint,
        run=download_hf,
        build_config=lambda ctx: DownloadConfig(
            hf_dataset_id=BASE,
            revision=BASE_REVISION,
            hf_repo_type_prefix="",
            gcs_output_path=ctx.output_path,
            zephyr_max_parallelism=8,
        ),
    )
    inputs = HERE / "inputs"
    hashes = {
        name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        for name in ["train.parquet.gz", "validation.parquet.gz", "requested-facts.json", "facts.json"]
    }
    data = ArtifactStep(
        name="documents/benfeuer/grpo-identity-smoke",
        version="2026.10.23.1",
        artifact_type=Artifact,
        run=stage_inputs,
        build_config=lambda ctx: InputFiles(ctx.output_path, str(inputs), hashes),
    )
    plan = SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=8,
        num_inference_engines=2,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=4,
        inference_engine_expert_parallel_size=4,
        train_batch_size=recipe["trainer"]["train_batch_size"],
        policy_mini_batch_size=recipe["trainer"]["policy_mini_batch_size"],
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=4,
    )
    spec = SkyRLSpec(
        name="checkpoints/benfeuer/grpo-identity/repair-smoke",
        version=version,
        config_yaml=yaml.safe_dump(recipe, sort_keys=False),
        model=ArtifactHfModel(model, BASE, BASE_REVISION, relative_path=""),
        train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
        validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
        topology=SkyRLTopology(6, 8, "H100", plan),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
        seed=20261023,
    )
    return skyrl_step(
        spec,
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=48,
            memory="1000Gi",
            disk="2000Gi",
            priority="production",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config="lib/iris/config/marin.yaml",
            coordinator_timeout_hours=6,
            job_timeout_seconds=21600,
        ),
        export_hf=False,
    )


@click.command()
@rl_build_options
def main():
    return build()


if __name__ == "__main__":
    main()
