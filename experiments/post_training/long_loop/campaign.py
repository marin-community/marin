# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fifty-update Mochi data campaign, using the .131 recipe and native RL CLI."""

import copy
import dataclasses
import json
import os
from functools import partial
from pathlib import Path

import click
import draccus
import yaml
from fray.types import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, materialized_config
from marin.execution.remote import remote
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLCheckpointStorage, SkyRLPolicyModel, SkyRLRolePlan, SkyRLRunConfig, run_skyrl
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import require_secret
from rigging.timing import Duration
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL

from experiments.evaluation.micro210 import ServingProfileName, micro210_artifact_step, registered_serving_profile
from experiments.post_training.long_loop.campaign_data import CampaignMix, load_mix, prepare_pools
from experiments.post_training.long_loop.campaign_daytona import (
    PURGE_RUNTIME_ARG,
    CampaignSnapshotPurge,
    purge_before_campaign_submission,
)
from experiments.post_training.long_loop.campaign_storage import retain_campaign_checkpoints
from experiments.post_training.long_loop.execution_policy import RlStopping, rl_training_callbacks
from experiments.post_training.long_loop.launch_validation import validate_before_allocation, validate_launch
from experiments.post_training.long_loop.rl_terminal_evaluation import preserved_evaluation
from experiments.post_training.long_loop.round_runner import (
    AdoptedRlRound,
    adopted_round_rl,
    load_round,
)

REFERENCE = Path(__file__).with_name("campaign") / "mochi131.yaml"
DEPLOYED_RL_SPEC = Path(__file__).with_name("specs") / "rl.yaml"
REFERENCE_SOURCE = "fe4695fbcd27e365284dccab649a40d6caa9fb2b"
CAMPAIGN_PIPELINE_STAGES = 16
LOCAL_DISK_ROOT = "/var/lib/shellbox-disk-pool"
LOCAL_REGISTRY_AUTH_SECRET = "GHCR_READ_PULL_DOCKER_CONFIG"
# Sources with one digest-pinned image per task run actor and grader on LOCAL: a Daytona
# snapshot per task image would exceed the RL org's snapshot quota. Maps the campaign source
# label to the TaskSpec source dataset that the placement keys on.
PER_TASK_IMAGE_DATASETS = {
    "RST:laion/rts-1k-rootless-apptainer": "laion/rts-1k-rootless-apptainer",
    "Lego-X:Terminal-Lego-15k": "Lego-X/Terminal-Lego-15k",
}
PER_TASK_IMAGE_MACHINE = {
    "backend": "local",
    "cpus": 1,
    "memory_mb": 2048,
    "storage_mb": None,
    "gpus": 0,
    "network": "allow",
    "user": None,
    "startup_timeout": 600,
    "cleanup_timeout": 60,
}
# The template for the tasks that declare environment_requirements.required_backend = "daytona"
# (the x86-only Lego-X contexts the user ruling routes to Daytona): the task artifact carries the
# routing decision, this map only supplies the machine shape for the declared backend.
DAYTONA_REQUIRED_MACHINE = {
    "backend": "daytona",
    "cpus": 2,
    "memory_mb": 4096,
    "storage_mb": 10240,
    "gpus": 0,
    "network": "allow",
    "user": None,
    "startup_timeout": 600,
    "cleanup_timeout": 60,
}
CAMPAIGN_SECRET_NAMES = (
    "CW_KEY_ID",
    "CW_KEY_SECRET",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "DAYTONA_RL_API_KEY",
    "DAYTONA_EVAL_API_KEY",
    "TOGETHER_API_KEY",
    "GHCR_READ_PULL_DOCKER_CONFIG",
)


@dataclasses.dataclass(frozen=True)
class CampaignGraph:
    training: ArtifactStep
    terminal: ArtifactStep
    plan: dict


def deployed_judge_binding(reference: dict) -> dict:
    """The deployed long-loop judge binding, with its #12374 model substitutions, on .131's endpoint."""
    binding = yaml.safe_load(DEPLOYED_RL_SPEC.read_text())["config"]["environment"]["task_sessions"]["remote_judge"]
    if {key: binding[key] for key in reference} != reference:
        raise ValueError("The deployed judge binding no longer uses the .131 judge endpoint")
    return binding


def place_per_task_images(recipe: dict, mix: CampaignMix) -> None:
    """Run per-task-image sources' actor and grader on LOCAL; other mixes keep .131's placements."""
    per_task = [dataset for label, dataset in PER_TASK_IMAGE_DATASETS.items() if label in mix.sources]
    if not per_task:
        return
    task_spec = recipe["environment"]["task_sessions"]["task_spec"]
    for placement in ("task_machine_by_dataset", "verifier_machine_by_dataset"):
        task_spec[placement] = {
            **task_spec.get(placement, {}),
            **{dataset: dict(PER_TASK_IMAGE_MACHINE) for dataset in per_task},
        }
    # Tasks that declare required_backend = "daytona" (the routed x86-only Lego-X contexts)
    # resolve their machines from this backend template; it is inert for every other task.
    task_spec["task_machine_by_backend"] = {
        **task_spec.get("task_machine_by_backend", {}),
        "daytona": dict(DAYTONA_REQUIRED_MACHINE),
    }


def campaign_round(
    mix: CampaignMix, version: str, *, max_steps: int, coordinator_memory: str | None = None
) -> AdoptedRlRound:
    """Apply only the campaign's mix, step stop, identities and storage policy to .131."""
    if max_steps != 50:
        raise ValueError("The controlled campaign requires --max-steps 50")
    reference = load_round(REFERENCE)
    if not isinstance(reference, AdoptedRlRound):
        raise TypeError(".131 is not an adopted RL round")
    recipe = copy.deepcopy(reference.rl.recipe)
    recipe["trainer"]["callbacks"] = rl_training_callbacks(
        recipe["trainer"]["callbacks"],
        groups_per_step=reference.rl.role_plan["train_batch_size"],
        stopping=RlStopping.MAX_STEPS,
    )
    # User-authorized 2026-11-27 protocol: all rows train, final is the deliverable.
    trainer = recipe["trainer"]
    trainer["callbacks"] = [
        {**callback, "save_steps": 5, "save_on_train_end": True} if callback["type"] == "checkpoint" else callback
        for callback in trainer["callbacks"]
        if callback["type"] not in {"evaluation", "best_checkpoint"}
    ]
    trainer.update(ckpt_interval=5, eval_interval=-1, hf_export_checkpoint="latest")
    recipe["trainer"]["rollout_buffer"]["source_routes"] = {label: pin["route"] for label, pin in mix.sources.items()}
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    # Stored TaskRolloutSpecs carry the catalog's per-task harness and machine sizing.
    recipe["environment"]["task_sessions"]["task_spec"]["harness_by_dataset"] = {}
    place_per_task_images(recipe, mix)
    # Retained lock-based verifiers declare LOCAL storage, which only the quota pool enforces.
    # LOCAL actors (Math-TIR) pull digest-pinned private GHCR images with the read-only pull credential.
    recipe["environment"]["task_sessions"]["local"] = {
        "disk_root": LOCAL_DISK_ROOT,
        "registry_auth_secret": LOCAL_REGISTRY_AUTH_SECRET,
    }
    recipe["environment"]["task_sessions"]["remote_judge"] = deployed_judge_binding(
        recipe["environment"]["task_sessions"]["remote_judge"]
    )
    megatron = {**reference.rl.megatron, "pipeline_model_parallel_size": CAMPAIGN_PIPELINE_STAGES}
    trainer_gpus = (
        megatron["pipeline_model_parallel_size"]
        * megatron["context_parallel_size"]
        * megatron["tensor_model_parallel_size"]
    )
    role_plan = {
        **reference.rl.role_plan,
        "policy_num_nodes": trainer_gpus // reference.rl.role_plan["policy_num_gpus_per_node"],
    }
    spec = dataclasses.replace(
        reference,
        name=f"rl-data-campaign/{mix.id}",
        slug=f"mochi-campaign-{mix.id}",
        version=version,
        evaluation_checkpoints="final",
        rl=dataclasses.replace(
            reference.rl,
            recipe=recipe,
            role_plan=role_plan,
            megatron=megatron,
            max_steps=max_steps,
            retained_checkpoints=2,
            stopping=RlStopping.MAX_STEPS,
            native_checkpoint_storage=SkyRLCheckpointStorage.TEMPORARY,
            execution={
                **reference.rl.execution,
                "coordinator_name": f"mochi-campaign-{mix.id}-{version}",
                "coordinator_secret_sources": tuple((name, f"env:{name}") for name in CAMPAIGN_SECRET_NAMES),
            },
            pools={"uri": prefix_join(reference.storage.output_prefix, f"rl-data-campaign/{mix.id}/pools/{version}")},
        ),
    )
    return apply_coordinator_memory(spec, coordinator_memory)


def apply_coordinator_memory(spec: AdoptedRlRound, memory: str | None) -> AdoptedRlRound:
    """Raise only the training coordinator container's memory request.

    Sizing evidence (coordinator ruling 2026-11-28): the .712 driver task peaked at
    721,078 MiB against its 704 GiB request and was OOMKilled; the GB200 node has 960 GB,
    so the request is not at the node maximum. Applied to the training round only — the pool
    lowering reference keeps the reference execution, so qualified pool fingerprints and their
    adoption certificates are unaffected.
    """
    if memory is None:
        return spec
    if memory == spec.rl.execution.get("memory"):
        return spec
    return dataclasses.replace(spec, rl=dataclasses.replace(spec.rl, execution={**spec.rl.execution, "memory": memory}))


def run_validated(config: SkyRLRunConfig):
    """Use the existing real-agent CPU gate before the native GPU allocation."""
    receipt = validate_before_allocation(
        config.launch_config_yaml,
        secrets={name: require_secret(name) for name in CAMPAIGN_SECRET_NAMES},
        check_task_inputs=True,
        check_vocabulary=True,
    )
    (StoragePath(config.output.resolved_config_uri).parent / "campaign-preflight.json").write_text(
        json.dumps(receipt, indent=2)
    )
    return run_skyrl(config, env_vars={})


def campaign_graph(mix: CampaignMix, spec: AdoptedRlRound, *, pool_version: str | None = None) -> CampaignGraph:
    """Prepare all training data regionally, then resolve the native launch config."""
    pool_version = spec.version if pool_version is None else pool_version
    pool_round = campaign_round(mix, pool_version, max_steps=spec.rl.max_steps)
    spec = dataclasses.replace(spec, rl=dataclasses.replace(spec.rl, pools=pool_round.rl.pools))
    # The .131 scaffold validates model/role/optimizer controls before new pools exist.
    reference = load_round(REFERENCE)
    # Pool lowering bakes each row's machines, so per-task-image placement must reach this document too.
    lowering_recipe = copy.deepcopy(reference.rl.recipe)
    place_per_task_images(lowering_recipe, mix)
    reference = dataclasses.replace(
        reference, rl=dataclasses.replace(reference.rl, execution=spec.rl.execution, recipe=lowering_recipe)
    )
    pool_reference = dataclasses.replace(
        reference, rl=dataclasses.replace(reference.rl, execution=pool_round.rl.execution)
    )
    scaffold = adopted_round_rl(reference, verify_pool_at_runtime=True)
    native = materialized_config(scaffold, spec.storage.output_prefix)
    pools = apply(
        f"{spec.name}/pools",
        remote(
            prepare_pools,
            resources=ResourceConfig.with_cpu(
                cpu=16,
                ram="128GB",
                disk="500GB",
            ),
            timeout=Duration.from_hours(4),
        ),
        version=pool_version,
        output=OUT,
        mix=mix,
        pool_version=pool_version,
        launch_document=native.launch_config_yaml,
    )
    pool_build = pools.build_config

    def build_pools(context: StepContext):
        if context.is_fingerprint:
            # The concrete launch carries an ephemeral attempt ID. Its complete
            # authored recipe, model pins and mix carry the preparation identity.
            return {
                "reference": dataclasses.asdict(pool_reference),
                "training_protocol": "no-validation-final-2026-11-27",
                "mix": dataclasses.asdict(mix),
                "version": pool_version,
            }
        return pool_build(context)

    pools = dataclasses.replace(pools, build_config=build_pools)

    # apply reserves 'version'; the work function's version travels in a distinct key.
    def build(context: StepContext):
        if context.is_fingerprint:
            return {
                "round": dataclasses.asdict(spec),
                "mix": dataclasses.asdict(mix),
                "pools": context.artifact_path(pools),
            }
        uri = context.artifact_path(pools)
        actual = dataclasses.replace(spec, rl=dataclasses.replace(spec.rl, pools={"uri": uri}))
        inner = adopted_round_rl(actual)
        return inner.build_config(
            StepContext.for_run(
                context.output_path,
                context.prefix,
                region=context.region,
                deps=inner.deps,
                runtime_args=inner.runtime_args,
            )
        )

    training = dataclasses.replace(
        scaffold,
        name=spec.name,
        version=spec.version,
        build_config=build,
        run=run_validated,
        deps=(*[dep for dep in scaffold.deps if dep.adopt_source == reference.adopted_model.uri], pools),
        expected_fingerprint=None,
    )
    training = dataclasses.replace(training, expected_fingerprint=training.fingerprint())
    retained = apply(
        f"{spec.name}/storage",
        remote(
            retain_campaign_checkpoints,
            resources=ResourceConfig.with_cpu(cpu=8, ram="32GB", disk="64GB"),
            timeout=Duration.from_hours(6),
        ),
        version=spec.version,
        training=training,
        output=OUT,
    )
    profile = registered_serving_profile(ServingProfileName(spec.evaluation_profile))
    profile = dataclasses.replace(
        profile, execution=dataclasses.replace(profile.execution, coordinator_name=spec.rl.execution["coordinator_name"])
    )
    evaluation = micro210_artifact_step(
        SkyRLPolicyModel(training),
        name=f"{spec.name}/micro210",
        version=spec.version,
        serving_profile=profile,
    )
    evaluated = preserved_evaluation(evaluation)
    # User order 2026-11-27 13:53/14:01: the Micro210 eval launches when training commits its
    # final export; snapshot retention and storage bookkeeping run in parallel and gate only the
    # preservation commit, never the eval.
    evaluated = dataclasses.replace(evaluated, deps=(*evaluated.deps, retained))
    roles = draccus.decode(SkyRLRolePlan, spec.rl.role_plan)
    plan = {
        "round": dataclasses.asdict(spec),
        "reference_source": REFERENCE_SOURCE,
        "mix": dataclasses.asdict(mix),
        "training_identity": training.fingerprint(),
        "terminal_identity": evaluated.fingerprint(),
        "pool_uri": pools.path(spec.storage.output_prefix),
        "pool_version": pool_version,
        "data_preflight": (
            "regional pool hashes, full-population training and actual-agent gate required before GPU allocation"
        ),
        "learning_stop": {"max_steps": 50, "wall_time_budget": None},
        "gpu_footprint": {
            "trainer": roles.policy_num_nodes * roles.policy_num_gpus_per_node,
            "rollout": roles.num_inference_engines * roles.engine_gpus,
            "total": roles.num_nodes * roles.gpus_per_node,
        },
        "operational_timeout_seconds": spec.rl.execution["job_timeout_seconds"],
    }
    return CampaignGraph(training, evaluated, plan)


@click.command()
@click.option("--mix-id", required=True)
@click.option("--max-steps", type=int, default=50, show_default=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@click.option("--snapshot-reference-registry", multiple=True, help="Current complete live-job protection registry URI.")
@click.option("--daytona-target", envvar="DAYTONA_TARGET", help="Daytona target used by this campaign.")
@click.option("--daytona-api-url", envvar="DAYTONA_API_URL", default=DEFAULT_DAYTONA_API_URL, show_default=True)
@click.option(
    "--snapshot-demand",
    type=click.Choice(["daytona", "zero"]),
    default="daytona",
    show_default=True,
    help="Declares the mix's Daytona snapshot demand. 'zero' (coordinator-authorized, recorded in the "
    "purge receipt) skips the purge/usage gate for mixes that provision no Daytona snapshots.",
)
@click.option(
    "--pool-version", default=None, help="Reuse this mix's immutable prepared-pool version for a fresh RL attempt."
)
@click.option(
    "--coordinator-memory",
    default=None,
    help="Raise only the training coordinator container's memory request (e.g. 810GiB) after a "
    "measured OOM; pool lowering keeps the reference request so pool adoption is unaffected.",
)
@partial(rl_build_options, before_submit=purge_before_campaign_submission)
def main(
    mix_id: str,
    max_steps: int,
    validation_output: Path,
    snapshot_reference_registry: tuple[str, ...],
    daytona_target: str | None,
    daytona_api_url: str,
    snapshot_demand: str,
    coordinator_memory: str | None,
    pool_version: str | None,
) -> ArtifactStep:
    """Build the campaign graph; --run is reserved for the coordinator's rated queue."""
    configure_coreweave_s3()
    mix = load_mix(mix_id)
    version = resolve_version(f"rl-data-campaign/{mix.id}", None)
    spec = campaign_round(mix, version, max_steps=max_steps, coordinator_memory=coordinator_memory)
    os.environ.update(MARIN_PREFIX=spec.storage.output_prefix, MARIN_TEMP_PREFIX=spec.storage.temporary_prefix)
    graph = campaign_graph(mix, spec, pool_version=pool_version)
    terminal, plan = graph.terminal, graph.plan
    validation_output.mkdir(parents=True, exist_ok=True)
    # This checks the recipe/model/geometry before the candidate's regional data exists.
    # It is explicitly not the candidate's actual-agent/data admission receipt.
    reference = load_round(REFERENCE)
    controls = dataclasses.replace(
        reference,
        rl=dataclasses.replace(
            reference.rl,
            role_plan=spec.rl.role_plan,
            megatron=spec.rl.megatron,
            max_steps=max_steps,
            stopping=RlStopping.MAX_STEPS,
            native_checkpoint_storage=SkyRLCheckpointStorage.TEMPORARY,
        ),
    )
    if (StoragePath(plan["pool_uri"]) / "manifest.json").exists():
        rendered = materialized_config(graph.training, spec.storage.output_prefix)
        scope = "candidate pools and authored protocol; real-agent admission still precedes GPU allocation"
    else:
        rendered = materialized_config(
            adopted_round_rl(controls, verify_pool_at_runtime=True), spec.storage.output_prefix
        )
        scope = "controls on .131 pools only; candidate data and authored protocol await pool materialization"
    plan["native_recipe_preflight"] = validate_launch(rendered.launch_config_yaml, validation_output / "native-controls")
    plan["native_recipe_preflight_scope"] = scope
    (validation_output / "plan.json").write_text(json.dumps(plan, indent=2))
    return dataclasses.replace(
        terminal,
        runtime_args={
            **terminal.runtime_args,
            PURGE_RUNTIME_ARG: CampaignSnapshotPurge(
                registry_uris=snapshot_reference_registry,
                target=daytona_target,
                api_url=daytona_api_url,
                receipt_path=str(validation_output / "snapshot-purge.json"),
                snapshot_demand=snapshot_demand,
            ),
        },
    )


if __name__ == "__main__":
    main()
