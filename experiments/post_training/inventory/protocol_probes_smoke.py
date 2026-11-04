# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the protocol probes against Qwen3-0.6B served by vLLM on one H100.

The run trains one dry step on the probes so the trainer evaluates them twice, before and after,
through the same path as the quick RL loop: the taskcompendium entrypoint, ShellSim task machines,
the fetch provider, and the local grading backend for the answer file. Read the labels from the
dumped ``protocol-probes.jsonl`` rows, under ``verification.diagnostics.protocol``.
"""

from dataclasses import replace

import click
import yaml
from iris.rpc import job_pb2
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
from taskcompendium.examples.protocol_probes import PROBE_DATASET, PROBE_TURNS, SHELLSIM_MACHINE

from experiments.post_training.cat_count_canary.launcher import MODELS, role_plan, training_config
from experiments.post_training.inventory.protocol_probes import PROBE_VARIANTS, PROBES_FILENAME, probe_tasks_step

ARTIFACT_NAME = "checkpoints/protocol-probes-smoke"
MODEL = "qwen3-0.6b"
SEED = 17
BATCH_SIZE = 4
GROUP_SIZE = 2
MICRO_BATCH_SIZE = 1
REQUEST_WINDOW_TOKENS = 4096
MAX_NEW_TOKENS = 256
PROBE_ROWS = 4 * PROBE_VARIANTS


def smoke_config() -> tuple[dict, SkyRLRolePlan]:
    """The canary's dry sync recipe, pointed at canonical probe rows on one GPU."""
    plan = replace(
        role_plan(batch_size=BATCH_SIZE, group_size=GROUP_SIZE, micro_train_batch_size=MICRO_BATCH_SIZE),
        colocate_all=True,
        policy_num_gpus_per_node=1,
        num_inference_engines=1,
    )
    config = training_config(
        preset="dry",
        lane="sync",
        model=MODEL,
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        micro_train_batch_size=MICRO_BATCH_SIZE,
        settings=("trainer.logger=console", "trainer.update_epochs_per_batch=1"),
    )
    _materialize_role_plan_config(config, plan)
    config["entrypoint"] = "taskcompendium"
    config["context_budget"] = {
        "request_window_tokens": REQUEST_WINDOW_TOKENS,
        "max_new_tokens_per_turn": MAX_NEW_TOKENS,
        "max_turns": PROBE_TURNS,
    }
    config["environment"] = {
        "task_sessions": {
            # The answer file is graded in the local sandbox, which needs the privileged container profile.
            "local": {},
            "task_spec": {
                "max_turns_by_dataset": {PROBE_DATASET: PROBE_TURNS},
                "task_machine_by_dataset": {PROBE_DATASET: SHELLSIM_MACHINE},
            },
        }
    }
    config["generator"]["gpu_memory_utilization"] = 0.4
    config["trajectory_runner"] = {
        "max_concurrent_tasks": 8,
        "rollout_workers": {"num_workers": 2, "cpus_per_worker": 2},
    }
    trainer = config["trainer"]
    trainer["eval_batch_size"] = PROBE_ROWS
    trainer["callbacks"] = [callback for callback in trainer["callbacks"] if callback["type"] != "database_registration"]
    for callback in trainer["callbacks"]:
        if callback["type"] == "evaluation":
            callback["additional_evaluations"] = {}
            callback["metric_groups"] = {}
    return config, plan


def build_smoke(version: str, cluster: str) -> ArtifactStep[SkyRLRun]:
    config, plan = smoke_config()
    probes = probe_tasks_step()
    choice = MODELS[MODEL]
    download = choice.step.build_config(StepContext.for_fingerprint(choice.step.runtime_args, choice.step.deps))
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(ARTIFACT_NAME),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=ArtifactHfModel(
                step=choice.step,
                relative_path=None,
                tokenizer_uri=download.hf_dataset_id,
                tokenizer_revision=download.revision,
            ),
            train_data=(ArtifactDataSource(probes, relative_path=PROBES_FILENAME),),
            validation_data=(ArtifactDataSource(probes, relative_path=PROBES_FILENAME),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=1, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=16,
            memory="128GB",
            disk="200GB",
            priority="interactive",
            max_retries=0,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=2,
            job_timeout_seconds=2400,
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=False,
    )


@click.command(help=__doc__)
@click.option("--cluster", type=click.Choice(("cw-rno2a", "cw-us-east-02a")), default="cw-rno2a", show_default=True)
@rl_build_options
def main(cluster: str) -> ArtifactStep[SkyRLRun]:
    return build_smoke(resolve_version(user_owned_name(ARTIFACT_NAME), None), cluster)


if __name__ == "__main__":
    main()
