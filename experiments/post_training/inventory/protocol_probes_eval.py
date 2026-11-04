# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the protocol probes against a Snowball checkpoint served by vLLM on one 8xH100 node.

SkyRL's evaluation-only entrypoint serves the model and runs the probes through the quick RL loop's
path (the taskcompendium rows, ShellSim task machines, the fetch provider, the local grading
backend) without loading a policy, so one node is enough for the 67B-A2B model. Read the labels from
the dumped ``protocol-probes.jsonl`` rows, under ``verification.diagnostics.protocol``.
"""

from dataclasses import replace
from types import MappingProxyType

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
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)
from taskcompendium.examples.protocol_probes import PROBE_DATASET, PROBE_TURNS, SHELLSIM_MACHINE

from experiments.models import ModelConfig, download_model
from experiments.post_training.async_rl import SNOWBALL_RECIPE
from experiments.post_training.inventory.protocol_probes import PROBE_VARIANTS, PROBES_FILENAME, probe_tasks_step
from experiments.post_training.inventory.train_quick_loop import training_recipe

ARTIFACT_NAME = "checkpoints/protocol-probes-eval"
MODELS = MappingProxyType(
    {
        "snowball-v0.1": ModelConfig(
            hf_repo_id="open-athena/Marin-Snowball-67B-A2B-v0.1",
            hf_revision="adc0078a19a4383bc998fcdb43867b3360baf5f3",
        ),
    }
)
DEFAULT_MODEL = "snowball-v0.1"
GPUS_PER_NODE = 8
PROBE_ROWS = 4 * PROBE_VARIANTS
REQUEST_WINDOW_TOKENS = 8192
MAX_NEW_TOKENS = 1024
SEED = 17


def eval_config() -> dict:
    """The quick loop's recipe, reduced to serving the model and evaluating the probes on one node."""
    config = training_recipe(steps=1)
    config["entrypoint"] = "generate"
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
    generator = config["generator"]
    # Nothing syncs weights in an evaluation-only run; the expert-block transport would forbid one node.
    generator["weight_sync_transport"] = "broadcast"
    generator["chat_template_kwargs"] = {"enable_thinking": False}
    generator["eval_sampling_params"] = {"temperature": 0.0, "max_generate_length": MAX_NEW_TOKENS}
    # The serving configs of this model family shard the loader across the expert-parallel ranks.
    generator["engine_init_kwargs"]["model_loader_extra_config"] = {"distributed": True}
    trainer = config["trainer"]
    trainer["logger"] = "console"
    trainer["eval_batch_size"] = PROBE_ROWS
    # No policy runs, but the recipe's geometry is validated for the one node this run occupies.
    for role in ("policy", "ref"):
        trainer[role]["megatron_config"]["pipeline_model_parallel_size"] = 1
    return config


def build_eval(model: str, version: str, cluster: str) -> ArtifactStep[SkyRLRun]:
    plan = replace(
        SNOWBALL_RECIPE.role_plan,
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        train_batch_size=PROBE_ROWS,
        policy_mini_batch_size=PROBE_ROWS,
        n_samples_per_prompt=1,
    )
    config = eval_config()
    _materialize_role_plan_config(config, plan)
    checkpoint = download_model(MODELS[model])
    download = checkpoint.build_config(StepContext.for_fingerprint(checkpoint.runtime_args, checkpoint.deps))
    probes = probe_tasks_step()
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"{ARTIFACT_NAME}/{model}"),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=ArtifactHfModel(
                step=checkpoint,
                relative_path=None,
                tokenizer_uri=download.hf_dataset_id,
                tokenizer_revision=download.revision,
            ),
            train_data=(),
            validation_data=(ArtifactDataSource(probes, relative_path=PROBES_FILENAME),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=32,
            memory="512GB",
            disk="500GB",
            priority="interactive",
            max_retries=0,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=3,
            job_timeout_seconds=3600,
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=False,
    )


@click.command(help=__doc__)
@click.option("--model", type=click.Choice(tuple(MODELS)), default=DEFAULT_MODEL, show_default=True)
@click.option("--cluster", type=click.Choice(("cw-rno2a", "cw-us-east-02a")), default="cw-rno2a", show_default=True)
@rl_build_options
def main(model: str, cluster: str) -> ArtifactStep[SkyRLRun]:
    return build_eval(model, resolve_version(user_owned_name(f"{ARTIFACT_NAME}/{model}"), None), cluster)


if __name__ == "__main__":
    main()
