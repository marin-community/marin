# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the agent protocol probes against a checkpoint served by vLLM on one 8xH100 node.

The probes are the ``protocol-probes`` dataset of ``taskcompendium.examples.protocol_probes``, published
as canonical TaskSpec rows whose answer file is graded from the verifyit lock of an environment with no
extra packages. The run starts from the quick RL loop's recipe and switches SkyRL to its
evaluation-only entrypoint: it serves the model and runs the probes through ShellSim task machines, the
fetch provider and the local grading backend without loading a policy. Read the dumped
``protocol-probes.jsonl`` rows: ``score`` is the protocol result on the file, tool and URL probes, and
``messages`` and ``response_token_ids`` show how a failed attempt went wrong.
"""

from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
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
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import Backend, NetworkPolicy
from taskcompendium.examples.protocol_probes import (
    PROBE_DATASET,
    PROBE_TURNS,
    ProtocolCheck,
    locked_grader_environment,
    probe_tasks,
)
from taskcompendium.models import SCHEMA_VERSION
from taskcompendium.runtime.local import LOCK_FILE
from zephyr.writers import write_parquet_file

from experiments.models import ModelConfig, download_model
from experiments.post_training.async_rl import SNOWBALL_RECIPE
from experiments.post_training.inventory.train_quick_loop import CLUSTER, sft_model, training_recipe
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import environment_artifact
from experiments.post_training.task_curation.inventory import TASK_SCHEMA, task_records

PROBES_FILENAME = "probes.parquet"
PROBE_ARTIFACT_NAME = f"rl-data/{PROBE_DATASET}"
PROBE_ARTIFACT_VERSION = "2026.11.04.1"
"""Bump when the probe tasks change; the schema, grader environment and variant count enter the identity."""
PROBE_VARIANTS = 8
"""Tasks per check; one evaluation of all variants gives each check's rate."""
PROBE_ROWS = len(ProtocolCheck) * PROBE_VARIANTS
GRADER_ENVIRONMENT = Environment()
"""verifyit alone; the exact-match grader of the answer file needs nothing else."""

ARTIFACT_NAME = "checkpoints/protocol-probes-eval"
SNOWBALL_V0_1 = ModelConfig(
    hf_repo_id="open-athena/Marin-Snowball-67B-A2B-v0.1",
    hf_revision="adc0078a19a4383bc998fcdb43867b3360baf5f3",
)
DEFAULT_MODEL = "snowball-v0.1"
CLUSTERS = ("cw-us-east-02a", "cw-rno2a")
GPUS_PER_NODE = 8
REQUEST_WINDOW_TOKENS = 8192
MAX_NEW_TOKENS = 1024
SEED = 17
TOOL_PARSER = "hermes"
"""vLLM rejects requests that offer tools unless a tool-call parser is enabled; this family uses Hermes."""
SHELLSIM_MACHINE = MachineRuntimeSpec(
    backend=Backend.SHELLSIM,
    network=NetworkPolicy.DENY,
    cpus=None,
    memory_mb=None,
    storage_mb=None,
    gpus=0,
    user=None,
    startup_timeout=60,
    cleanup_timeout=60,
)
"""The file and tool probes' task machine."""


def write_probe_tasks(output_path: str, grader_environment: str, variants: int, schema_version: str) -> None:
    """Write the probes as canonical TaskSpec rows, graded from the built ``grader_environment``'s lock.

    ``schema_version`` is the TaskSpec schema the rows are written in; it enters the artifact's identity so
    a schema change rebuilds the rows.
    """
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"The probe rows are written in TaskSpec schema {SCHEMA_VERSION}, not {schema_version}")
    grader = locked_grader_environment(str(StoragePath(grader_environment) / LOCK_FILE))
    tasks = probe_tasks(grader_environment=grader, variants=variants)
    destination = str(StoragePath(output_path) / PROBES_FILENAME)
    write_parquet_file(task_records(iter(tasks)), destination, schema=TASK_SCHEMA)


def probe_tasks_step() -> ArtifactStep[Artifact]:
    """The probe rows under ``PROBE_ARTIFACT_VERSION``, for the current grader environment and variant count."""
    return apply(
        PROBE_ARTIFACT_NAME,
        write_probe_tasks,
        version=PROBE_ARTIFACT_VERSION,
        output_path=OUT,
        grader_environment=environment_artifact(GRADER_ENVIRONMENT),
        variants=PROBE_VARIANTS,
        schema_version=SCHEMA_VERSION,
    )


def snowball_v0_1() -> ArtifactHfModel:
    return ArtifactHfModel(
        step=download_model(SNOWBALL_V0_1),
        relative_path=None,
        tokenizer_uri=SNOWBALL_V0_1.hf_repo_id,
        tokenizer_revision=SNOWBALL_V0_1.hf_revision,
    )


MODELS: MappingProxyType[str, Callable[[], ArtifactHfModel]] = MappingProxyType(
    {"datakit-sft": sft_model, "snowball-v0.1": snowball_v0_1}
)
"""Checkpoints of the 67B-A2B family, whose serving settings this run uses: the quick loop's starting
point and the released Snowball model."""


def eval_config(thinking: bool) -> dict:
    """The quick loop's recipe, reduced to serving the model and evaluating the probes on one node.

    ``thinking`` sets the chat template's ``enable_thinking`` for every probe; the thinking probe tests
    only a run with thinking off.
    """
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
                "task_machine_by_dataset": {PROBE_DATASET: SHELLSIM_MACHINE.model_dump(mode="json")},
            },
        }
    }
    generator = config["generator"]
    # Nothing syncs weights in an evaluation-only run; the expert-block transport would forbid one node.
    generator["weight_sync_transport"] = "broadcast"
    generator["chat_template_kwargs"] = {"enable_thinking": thinking}
    generator["eval_sampling_params"] = {"temperature": 0.0, "max_generate_length": MAX_NEW_TOKENS}
    generator["engine_init_kwargs"].update(
        enable_auto_tools=True,
        tool_parser=TOOL_PARSER,
        # The serving configs of this model family shard the loader across the expert-parallel ranks.
        model_loader_extra_config={"distributed": True},
    )
    trainer = config["trainer"]
    trainer["logger"] = "console"
    trainer["eval_batch_size"] = PROBE_ROWS
    # No policy runs, but the recipe's geometry is validated for the one node this run occupies.
    for role in ("policy", "ref"):
        trainer[role]["megatron_config"]["pipeline_model_parallel_size"] = 1
    return config


def build_eval(model: str, version: str, cluster: str, thinking: bool) -> ArtifactStep[SkyRLRun]:
    """Serve ``model`` on one node of ``cluster`` and evaluate every probe once."""
    plan = replace(
        SNOWBALL_RECIPE.role_plan,
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        train_batch_size=PROBE_ROWS,
        policy_mini_batch_size=PROBE_ROWS,
        n_samples_per_prompt=1,
    )
    config = eval_config(thinking)
    _materialize_role_plan_config(config, plan)
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"{ARTIFACT_NAME}/{model}"),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=MODELS[model](),
            train_data=(),
            validation_data=(ArtifactDataSource(probe_tasks_step(), relative_path=PROBES_FILENAME),),
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
@click.option(
    "--cluster",
    type=click.Choice(CLUSTERS),
    default=CLUSTER,
    show_default=True,
    help="The quick loop's cluster holds the SFT weights; another reads them across regions.",
)
@click.option(
    "--thinking/--no-thinking", default=False, show_default=True, help="Serve with the chat template's thinking on."
)
@rl_build_options
def main(model: str, cluster: str, thinking: bool) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(f"{ARTIFACT_NAME}/{model}")
    return build_eval(model, resolve_version(name, None), cluster, thinking)


if __name__ == "__main__":
    main()
