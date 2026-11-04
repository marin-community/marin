# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke the thoughtcrime tasks through MarinSkyRL on a small model.

Converts the pinned double-think rows into ``task_spec`` Parquet, then trains Qwen2.5-0.5B-Instruct
for one synchronous update on them with evaluation before and after, through the ``taskcompendium``
entrypoint the quick RL loop uses. The run proves the tasks lower, roll out without a machine, grade in
process and report per-dataset ``eval/<repository>/avg_score`` metrics; its scores say nothing about Snowball.

    uv run python -m experiments.post_training.thoughtcrime.smoke --version dev
    uv run python -m experiments.post_training.thoughtcrime.smoke --version dev --run
"""

import logging
from dataclasses import dataclass, replace

import click
import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.pi_trajectory import prefix_row_task
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from zephyr.writers import write_parquet_file

from experiments.models import qwen2_5_0_5b_instruct
from experiments.post_training.cat_count_canary.launcher import QWEN_SOURCE, role_plan, training_config
from experiments.post_training.task_curation.datasets.thoughtcrime import ROWS_FILE, sources
from experiments.post_training.task_curation.inventory import (
    TASK_SCHEMA,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    task_records,
)

logger = logging.getLogger(__name__)

CLUSTER = "cw-rno2a"
SEED = 17
BATCH_SIZE = 8
GROUP_SIZE = 4
# Pi prompts run to 12k tokens; leave room for a reasoning span and one function call.
REQUEST_WINDOW_TOKENS = 16_384
MAX_NEW_TOKENS = 1_024


@dataclass(frozen=True)
class PrefixTaskDataConfig:
    output_path: str
    repositories: tuple[tuple[str, str], ...]
    """``(repository, revision)`` of each pinned row file; every row becomes a training and evaluation task."""


def write_prefix_tasks(config: PrefixTaskDataConfig) -> None:
    """Convert the pinned rows the way the curation pipeline would and publish them as ``task_spec`` Parquet."""
    tasks = []
    for repository, revision in config.repositories:
        path = hf_hub_download(repository, ROWS_FILE, repo_type="dataset", revision=revision)
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            source = Source(dataset=repository, revision=revision, row=f"{ROWS_FILE}:{index}", importer_revision="smoke")
            tasks.append(prefix_row_task(RawRow(row["digest"], source, row), ConversionContext({}, None)))
    destination = StoragePath(config.output_path)
    destination.mkdirs()
    for filename in (TRAIN_FILENAME, VALIDATION_FILENAME):
        write_parquet_file(
            task_records(iter(tasks)),
            str(destination / filename),
            schema=TASK_SCHEMA,
        )
    logger.info("Wrote %d prefix tasks to %s", len(tasks), destination)


def prefix_task_data_step(version: str) -> ArtifactStep[Artifact]:
    repositories = tuple(
        (source.config.source.repo, source.config.source.revision) for source in sources() if source.config is not None
    )
    return ArtifactStep(
        name=user_owned_name("documents/thoughtcrime-smoke"),
        version=version,
        artifact_type=Artifact,
        run=write_prefix_tasks,
        build_config=lambda ctx: PrefixTaskDataConfig(output_path=ctx.output_path, repositories=repositories),
    )


def smoke_config(gpus: int) -> tuple[dict, SkyRLRolePlan]:
    """The cat-count canary's dry recipe, pointed at canonical tasks."""
    plan = replace(
        role_plan(batch_size=BATCH_SIZE, group_size=GROUP_SIZE, micro_train_batch_size=1),
        colocate_all=True,
        policy_num_gpus_per_node=gpus,
        num_inference_engines=gpus,
    )
    config = training_config(
        preset="dry",
        lane="sync",
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        micro_train_batch_size=1,
        settings=("trainer.logger=console", "trainer.update_epochs_per_batch=1"),
    )
    _materialize_role_plan_config(config, plan)
    config["entrypoint"] = "taskcompendium"
    config["context_budget"] = {
        "request_window_tokens": REQUEST_WINDOW_TOKENS,
        "max_new_tokens_per_turn": MAX_NEW_TOKENS,
        "max_turns": 1,
    }
    # Machine-free, in-process graded tasks: the default task_spec session settings apply and no
    # Shellbox backend is enabled, so the run needs no privileged container.
    config["environment"] = {"task_sessions": {}}
    config["data"] = {"kind": "parquet", "shuffle": True, "train_data": [], "val_data": []}
    generator = config["generator"]
    # The model's own template renders the advertised tools; vLLM parses the hermes-style calls it emits.
    generator["chat_template"] = None
    generator["engine_init_kwargs"] = {"enable_auto_tool_choice": True, "tool_call_parser": "hermes"}
    generator["gpu_memory_utilization"] = 0.4
    config["trajectory_runner"] = {
        "max_concurrent_tasks": 16,
        "rollout_workers": {"num_workers": 2, "cpus_per_worker": 2},
    }
    trainer = config["trainer"]
    trainer["eval_batch_size"] = 64
    trainer["callbacks"] = [callback for callback in trainer["callbacks"] if callback["type"] != "database_registration"]
    for callback in trainer["callbacks"]:
        if callback["type"] == "evaluation":
            callback["additional_evaluations"] = {}
            callback["metric_groups"] = {}
    return config, plan


def build_smoke(version: str, *, gpus: int) -> ArtifactStep[SkyRLRun]:
    data = prefix_task_data_step(version)
    config, plan = smoke_config(gpus)
    download = qwen2_5_0_5b_instruct.build_config(
        StepContext.for_fingerprint(qwen2_5_0_5b_instruct.runtime_args, qwen2_5_0_5b_instruct.deps)
    )
    model = ArtifactStep.adopt(
        qwen2_5_0_5b_instruct.name, qwen2_5_0_5b_instruct.version, QWEN_SOURCE, kind=qwen2_5_0_5b_instruct.artifact_type
    )
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name("checkpoints/thoughtcrime-smoke"),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=ArtifactHfModel(
                step=model, relative_path="", tokenizer_uri=download.hf_dataset_id, tokenizer_revision=download.revision
            ),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=gpus, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="128GB",
            disk="200GB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=2,
            job_timeout_seconds=2400,
        ),
    )


@click.command(help=__doc__)
@click.option("--gpus", type=click.IntRange(1, 8), default=1, show_default=True, help="H100s on the one node.")
@rl_build_options
def main(gpus: int) -> ArtifactStep[SkyRLRun]:
    return build_smoke(resolve_version("thoughtcrime-smoke", None), gpus=gpus)


if __name__ == "__main__":
    main()
