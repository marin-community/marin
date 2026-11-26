# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fifty-update Mochi data campaign, using the .131 recipe and native RL CLI."""

import copy
import dataclasses
import json
import os
from pathlib import Path

import click
import draccus
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

from experiments.evaluation.micro210 import ServingProfileName, micro210_artifact_step, registered_serving_profile
from experiments.post_training.long_loop.campaign_data import CampaignMix, load_mix, prepare_pools
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
REFERENCE_SOURCE = "fe4695fbcd27e365284dccab649a40d6caa9fb2b"
CAMPAIGN_SECRET_NAMES = (
    "CW_KEY_ID",
    "CW_KEY_SECRET",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "DAYTONA_RL_API_KEY",
    "DAYTONA_EVAL_API_KEY",
    "TOGETHER_API_KEY",
)


@dataclasses.dataclass(frozen=True)
class CampaignGraph:
    training: ArtifactStep
    terminal: ArtifactStep
    plan: dict


def campaign_round(mix: CampaignMix, version: str, *, max_steps: int) -> AdoptedRlRound:
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
    recipe["trainer"]["rollout_buffer"]["source_routes"] = {label: pin["route"] for label, pin in mix.sources.items()}
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    # Stored TaskRolloutSpecs carry the catalog's per-task harness and machine sizing.
    recipe["environment"]["task_sessions"]["task_spec"]["harness_by_dataset"] = {}
    return dataclasses.replace(
        reference,
        name=f"rl-data-campaign/{mix.id}",
        slug=f"mochi-campaign-{mix.id}",
        version=version,
        evaluation_checkpoints="best",
        rl=dataclasses.replace(
            reference.rl,
            recipe=recipe,
            max_steps=max_steps,
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


def campaign_graph(mix: CampaignMix, spec: AdoptedRlRound) -> CampaignGraph:
    """Prepare data regionally, then resolve a native config against its real holdout."""
    # The existing .131 bindings validate model/role/optimizer controls without inventing
    # future holdout identities. Only the pool producer may supply the campaign holdout.
    reference = load_round(REFERENCE)
    reference = dataclasses.replace(reference, rl=dataclasses.replace(reference.rl, execution=spec.rl.execution))
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
        version=spec.version,
        output=OUT,
        mix=mix,
        pool_version=spec.version,
        launch_document=native.launch_config_yaml,
    )
    pool_build = pools.build_config

    def build_pools(context: StepContext):
        if context.is_fingerprint:
            # The concrete launch carries an ephemeral attempt ID. Its complete
            # authored recipe, model pins and mix carry the preparation identity.
            return {"reference": dataclasses.asdict(reference), "mix": dataclasses.asdict(mix), "version": spec.version}
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
    evaluation = dataclasses.replace(evaluation, deps=(*evaluation.deps, retained))
    evaluated = preserved_evaluation(evaluation)
    roles = draccus.decode(SkyRLRolePlan, spec.rl.role_plan)
    plan = {
        "round": dataclasses.asdict(spec),
        "reference_source": REFERENCE_SOURCE,
        "mix": dataclasses.asdict(mix),
        "training_identity": training.fingerprint(),
        "terminal_identity": evaluated.fingerprint(),
        "pool_uri": pools.path(spec.storage.output_prefix),
        "data_preflight": "regional pool hashes, holdout and actual-agent gate required before GPU allocation",
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
@rl_build_options
def main(mix_id: str, max_steps: int, validation_output: Path) -> ArtifactStep:
    """Build the campaign graph; --run is reserved for the coordinator's rated queue."""
    configure_coreweave_s3()
    mix = load_mix(mix_id)
    version = resolve_version(f"rl-data-campaign/{mix.id}", None)
    spec = campaign_round(mix, version, max_steps=max_steps)
    os.environ.update(MARIN_PREFIX=spec.storage.output_prefix, MARIN_TEMP_PREFIX=spec.storage.temporary_prefix)
    graph = campaign_graph(mix, spec)
    terminal, plan = graph.terminal, graph.plan
    validation_output.mkdir(parents=True, exist_ok=True)
    # This checks the recipe/model/geometry before the candidate's regional data exists.
    # It is explicitly not the candidate's actual-agent/data admission receipt.
    reference = load_round(REFERENCE)
    controls = dataclasses.replace(
        reference,
        rl=dataclasses.replace(
            reference.rl,
            max_steps=max_steps,
            stopping=RlStopping.MAX_STEPS,
            native_checkpoint_storage=SkyRLCheckpointStorage.TEMPORARY,
        ),
    )
    rendered = materialized_config(adopted_round_rl(controls, verify_pool_at_runtime=True), spec.storage.output_prefix)
    plan["native_recipe_preflight"] = validate_launch(rendered.launch_config_yaml, validation_output / "native-controls")
    plan["native_recipe_preflight_scope"] = (
        "controls on existing .131 pools; candidate data/agent gate is a dependency before GPU allocation"
    )
    (validation_output / "plan.json").write_text(json.dumps(plan, indent=2))
    return terminal


if __name__ == "__main__":
    main()
