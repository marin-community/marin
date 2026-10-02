# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Snowball under one trainer-vLLM numerics setup, with everything else held fixed across setups.

Every run trains the 67B-A2B Snowball SFT export on the E6.2 Snowball training selection with 64 prompts and eight
answers per update, PPO with token-level truncated importance sampling at staleness 4, and the async launcher's
optimizer, and it evaluates on the held-out MATH level 3-5 set. The MarinSkyRL commit that this branch pins decides
the numerics; ``--routing replay`` replays vLLM's expert routes in the trainer. Only the RL stage is built: the in-run
evaluation scores the held-out set.

Plan or run::

    python -m experiments.post_training.snowball_numerics_ab --version 2026.10.02.1 --preset gate --routing native
    python -m experiments.post_training.snowball_numerics_ab --version 2026.10.02.1 --preset full \\
        --routing replay --run
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from enum import StrEnum
from types import MappingProxyType

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)

from experiments.post_training.async_rl import (
    DEFAULT,
    RETENTION,
    SNOWBALL_RECIPE,
    AsyncPreset,
    apply_settings,
    snowball_config,
)
from experiments.post_training.curriculum_rl.launch import GPU_VARIANT, GPUS_PER_NODE, SNOWBALL_MODEL, SNOWBALL_POLICY

EXPERIMENT_NAME = "snowball-numerics-ab"
WANDB_PROJECT = f"marin-{EXPERIMENT_NAME}"
DATA_ROOT = "s3://marin-us-east-02a/marin/users/ahmad/documents/math-eval-pool/1.0.0-candidate1"
# The E6.2 Snowball training selection: 1,853 rows of GSM8K, MATH levels 1-3 and reasoning-gym chain sums.
TRAIN_DATA_URI = f"{DATA_ROOT}/buckets/e62-resolved-reviewed-v1/snowball"
TRAIN_FILENAME = "train.parquet"
# MATH test levels 3-5 without MATH-500: 3,294 rows graded by the aime env, none of them in the training selection.
HELDOUT_DATA_URI = f"{DATA_ROOT}/batteries/e9-heldout-v1/snowball"
HELDOUT_FILENAME = "heldout-math-l345.parquet"
PROMPTS_PER_UPDATE = 64
# Eight answers per prompt keep 512 sequences per update, the async launcher's default load.
ANSWERS_PER_PROMPT = 8
# Prompts per evaluation request batch; four batches score the held-out set.
EVAL_BATCH_PROMPTS = 1024
# The E9 held-out protocol's sampling.
EVAL_SAMPLING = MappingProxyType({"temperature": 0.6, "top_p": 0.95})

NUMERICS_AB_RECIPE = replace(
    SNOWBALL_RECIPE,
    role_plan=replace(
        SNOWBALL_RECIPE.role_plan,
        train_batch_size=PROMPTS_PER_UPDATE,
        policy_mini_batch_size=PROMPTS_PER_UPDATE,
        n_samples_per_prompt=ANSWERS_PER_PROMPT,
    ),
)

# 100 updates, evaluating every 10. 96 prompt groups in flight hold the default's 768 sequences.
FULL = replace(DEFAULT, label="full", max_in_flight_groups=96)
# The full run's settings for two updates.
GATE = replace(FULL, label="gate", max_steps=2)
PRESETS: Mapping[str, AsyncPreset] = MappingProxyType({preset.label: preset for preset in (GATE, FULL)})


class Routing(StrEnum):
    """Whether the trainer routes tokens itself (``native``) or replays the experts vLLM chose (``replay``)."""

    NATIVE = "native"
    REPLAY = "replay"


def numerics_ab_config(preset: AsyncPreset, routing: Routing, settings: tuple[str, ...] = ()) -> dict:
    """Render the shared A/B config for one routing mode, then apply ``--set`` changes."""
    config = snowball_config(preset, NUMERICS_AB_RECIPE)
    trainer = config["trainer"]
    trainer["project_name"] = WANDB_PROJECT
    trainer["eval_batch_size"] = EVAL_BATCH_PROMPTS
    algorithm = trainer["algorithm"]
    # PPO's clipped ratio against the trainer's own old log-probabilities.
    algorithm["policy_loss_type"] = "regular"
    # Each token's loss weight is min(exp(old - rollout), 2) over the rollout's log-probabilities.
    algorithm["off_policy_correction"] = "tis"
    # Each batch trains the groups leased for it, so a step's prompts do not depend on generation speed.
    trainer["rollout_buffer"]["batch_policy"] = "full_batch"
    replay = routing is Routing.REPLAY
    trainer["policy"]["megatron_config"]["moe_router_replay"] = replay
    generator = config["generator"]
    generator["engine_init_kwargs"]["enable_return_routed_experts"] = replay
    generator["eval_sampling_params"] = dict(EVAL_SAMPLING)
    # GSM8K rows earn reward only for a completed response whose last line is "#### <number>", as their prompt asks.
    config["environment"]["skyrl_gym"] = {"gsm8k": {"reward_method": "final_line"}}
    # Seeded passes over the training rows in a fresh trainer.seed order each pass.
    config["data"]["shuffle"] = True
    config["data"]["sampling"] = {"kind": None}
    return apply_settings(config, settings)


def build_run(
    preset: AsyncPreset, routing: Routing, version: str | None, settings: tuple[str, ...] = ()
) -> ArtifactStep[SkyRLRun]:
    """Assemble the RL step for one preset and routing mode under this branch's MarinSkyRL commit."""
    recipe = NUMERICS_AB_RECIPE
    policy = SNOWBALL_POLICY
    config = numerics_ab_config(preset, routing, settings)
    train = ArtifactStep.adopt(
        user_owned_name(f"documents/{EXPERIMENT_NAME}/e62-snowball-train"), "2026.09.12", TRAIN_DATA_URI
    )
    heldout = ArtifactStep.adopt(
        user_owned_name(f"documents/{EXPERIMENT_NAME}/heldout-math-l345"), "2026.09.14", HELDOUT_DATA_URI
    )
    # The run's address names the MarinSkyRL commit and routing mode; a --set run also carries a hash of its settings.
    changes = "\n".join(settings)
    suffix = f"-set-{fingerprint_hash(changes)}" if settings else ""
    base_name = f"checkpoints/{EXPERIMENT_NAME}/skyrl-{MARIN_SKYRL.commit[:9]}-{routing}-{preset.label}{suffix}"
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            runtime=SkyRLRuntime(),
            model=ArtifactHfModel(
                step=SNOWBALL_MODEL,
                tokenizer_uri=policy.tokenizer_uri,
                tokenizer_revision=policy.tokenizer_revision,
                relative_path=policy.model_relative_path,
            ),
            train_data=(ArtifactDataSource(train, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(heldout, relative_path=HELDOUT_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=recipe.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=recipe.role_plan,
            ),
            retention=RETENTION,
            seed=config["trainer"]["seed"],
        ),
        IrisSkyRLExecution(
            cluster=policy.cluster,
            cluster_config=f"lib/iris/config/{policy.cluster}.yaml",
            cpu=16,
            memory=recipe.host_memory,
            disk="2TB",
            priority="interactive",
            # One automatic retry, then fail; a healthy run resumes from its latest checkpoint on resubmission.
            max_retries=1,
            target_cluster=policy.cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            # The W&B key decides the entity.
            wandb_entity=None,
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), required=True)
@click.option("--routing", type=click.Choice([mode.value for mode in Routing]), required=True)
@click.option(
    "--set",
    "settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Change one setting of the rendered RL config (dotted key; prefix + to add a new key).",
)
@rl_build_options
def main(preset: str, routing: str, settings: tuple[str, ...]) -> dict[str, ArtifactStep]:
    mode = Routing(routing)
    return {f"skyrl-{MARIN_SKYRL.commit[:9]}-{mode}-{preset}": build_run(PRESETS[preset], mode, None, settings)}


if __name__ == "__main__":
    main()
