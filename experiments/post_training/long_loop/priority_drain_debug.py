# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small H100 control-flow comparison for priority, abort and wait evaluation drains."""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from iris.cluster.client.job_info import get_job_info
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
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
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)

from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision
from experiments.post_training.long_loop.rl import run_checked_rl

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_URI = "s3://marin-us-east-02a/marin/models/Qwen--Qwen3-0-6B--c1899de"
MODEL_CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
MODEL_ARTIFACT_VERSION = "2026.11.23.1"
DATA_VERSION = "2026.11.23.2"
TRAIN_PROMPTS = 1024
EVAL_PROMPTS = 2048
OUTPUT_PREFIX = "s3://marin-us-east-02a/marin/users/benfeuer/mochi-priority-drain-debug"
TEMPORARY_PREFIX = "s3://marin-us-east-02a/tmp"
RECIPE_PATH = Path(__file__).with_name("rounds") / "priority-drain-h100.yaml"
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=1,
    policy_num_gpus_per_node=1,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=1,
    inference_engine_expert_parallel_size=1,
    train_batch_size=4,
    policy_mini_batch_size=4,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=2,
)


@dataclass(frozen=True)
class DebugData:
    output_path: str


def control_rows(split: str, count: int) -> list[dict]:
    """Generate text-only arithmetic tasks with an in-process verifier."""
    return [
        {
            "task_spec": (
                TaskSpec(
                    id=f"{split}-{index}",
                    context=ConversationInput(
                        events=(
                            TextMessage(
                                role="user",
                                content=(
                                    f"Problem {split}-{index}: Calculate {index + 1} + 7. "
                                    "Give a detailed explanation of at least 200 words, "
                                    "then give the answer as \\boxed{integer}."
                                ),
                            ),
                        )
                    ),
                    environment_requirements=EnvironmentRequirements(),
                    answer_type=AnswerType.TEXT,
                    answer_format=PlainText(),
                    grader=VerifyitGrader(mode="exact", parameters={"expected": [str(index + 8)], "substring": True}),
                    source=Source(
                        dataset=f"priority-drain-debug-{split}",
                        revision=DATA_VERSION,
                        row=str(index),
                        importer_revision="1",
                    ),
                ).model_dump_json()
            ),
        }
        for index in range(count)
    ]


def build_data(config: DebugData) -> None:
    """Write the shared small cohort in the regional CPU coordinator."""
    root = StoragePath(config.output_path)
    root.mkdirs(exist_ok=True)
    files = {}
    for split, count in (("train", TRAIN_PROMPTS), ("validation", EVAL_PROMPTS)):
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(control_rows(split, count)), sink)
        payload = sink.getvalue().to_pybytes()
        (root / f"{split}.parquet").write_bytes(payload)
        files[f"{split}.parquet"] = {"rows": count, "sha256": hashlib.sha256(payload).hexdigest()}
    (root / "manifest.json").write_text(json.dumps({"version": DATA_VERSION, "files": files}, indent=2))


def debug_run(drain: str, version: str) -> ArtifactStep:
    """Build the ordinary long-loop validation and SkyRL artifact path on two H100s."""
    data = ArtifactStep(
        name=user_owned_name("documents/mochi-priority-drain-debug"),
        version=DATA_VERSION,
        artifact_type=Artifact,
        run=build_data,
        build_config=lambda context: DebugData(context.output_path),
    )
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    recipe["trainer"]["rollout_buffer"]["eval_drain"] = drain
    step = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"long-loop/priority-drain-debug-{drain}"),
            version=version,
            config_yaml=yaml.safe_dump(recipe, sort_keys=False),
            model=ArtifactHfModel(
                step=ArtifactStep.adopt(
                    user_owned_name("long-loop/priority-drain-debug-model"),
                    MODEL_ARTIFACT_VERSION,
                    source=MODEL_URI,
                    kind=LevanterCheckpoint,
                    config={"hf_revision": MODEL_REVISION, "config_sha256": MODEL_CONFIG_SHA256},
                ),
                relative_path="",
                tokenizer_uri=MODEL,
                tokenizer_revision=MODEL_REVISION,
            ),
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
            topology=SkyRLTopology(
                num_nodes=ROLE_PLAN.num_nodes,
                gpus_per_node=ROLE_PLAN.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=20261123,
        ),
        IrisSkyRLExecution(
            cluster="cw-us-east-02a",
            cluster_config="lib/iris/config/cw-us-east-02a.yaml",
            target_cluster="cw-us-east-02a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            cpu=16,
            memory="128GB",
            disk="64GB",
            priority="interactive",
            max_retries=0,
            coordinator_timeout_hours=4,
            job_timeout_seconds=10800,
            wandb_entity="marin-community",
        ),
        export_hf=False,
    )
    checked = replace(step, run=partial(run_checked_rl, env_vars={}))
    return replace(checked, expected_fingerprint=checked.fingerprint())


@click.command(help=__doc__)
@click.option("--drain", type=click.Choice(["priority", "abort", "wait"]), required=True)
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@rl_build_options
def main(drain: str, source_revision: str, validation_output: Path) -> ArtifactStep:
    configure_coreweave_s3()
    os.environ.update(MARIN_PREFIX=OUTPUT_PREFIX, MARIN_TEMP_PREFIX=TEMPORARY_PREFIX)
    if get_job_info() is None:
        require_clean_revision(Path(__file__).resolve().parents[3], source_revision)
    version = click.get_current_context().params["default_version"]
    terminal = debug_run(drain, version)
    model_config = (StoragePath(MODEL_URI) / "config.json").read_bytes()
    if hashlib.sha256(model_config).hexdigest() != MODEL_CONFIG_SHA256:
        raise ValueError("Regional Qwen model configuration changed")
    validate_graph(terminal, validation_output, OUTPUT_PREFIX, model_config=model_config, check_vocabulary=True)
    return terminal


if __name__ == "__main__":
    main()
