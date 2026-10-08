# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Change only the learning objective for a fixed native-DPO chosen exposure."""

from dataclasses import replace

import click
from levanter.main.train_lm import TrainLmConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION
from experiments.post_training.bfcl_rl.matched_sft_data import MatchedChosenCache, matched_sft_data_config
from experiments.post_training.bfcl_rl.native_optimize import native_optimizer_step
from experiments.post_training.bfcl_rl.offline_optimize import dispatch_offline_training


def matched_sft_optimizer_step(
    control: ArtifactStep[LevanterCheckpoint], projection: ArtifactStep[MatchedChosenCache]
) -> ArtifactStep[LevanterCheckpoint]:
    """Reuse the complete DPO optimizer/model/resource configuration for chosen-only SFT."""
    name = user_owned_name("models/bfcl-rl-matched-chosen-sft")

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        reference_config = control.build_config(ctx)
        dpo = reference_config.train_config
        if ctx.is_fingerprint:
            cache_path = prefix_join(ctx.artifact_path(projection), "train")
        else:
            chosen = ctx.resolved(projection)
            expected_source = dpo.data.components["bfcl_complement"].cache_dir.removesuffix("/train")
            if chosen.source_cache_path != expected_source or chosen.tokenizer != dpo.data.tokenizer:
                raise ValueError("SFT exposure does not derive from the configured DPO cache")
            if chosen.max_length != dpo.train_seq_len or chosen.seed != dpo.trainer.seed:
                raise ValueError("SFT exposure differs from the DPO context or sampling seed")
            if chosen.presentations != dpo.trainer.num_train_steps * dpo.trainer.train_batch_size:
                raise ValueError("SFT exposure count differs from the DPO optimizer's presentations")
            cache_path = chosen.cache_path
        train = TrainLmConfig(
            data=matched_sft_data_config(cache_path, dpo.data.tokenizer),
            trainer=dpo.trainer,
            model=dpo.model,
            optimizer=dpo.optimizer,
            train_seq_len=dpo.train_seq_len,
            data_seed=dpo.data_seed,
            initialize_from_hf=dpo.initialize_from_hf,
            use_hf_model_config=dpo.use_hf_model_config,
            hf_save_steps=dpo.hf_save_steps,
            hf_save_dtype=dpo.hf_save_dtype,
        )
        return TrainLmOnPodConfig(
            train,
            reference_config.resources,
            output_path=ctx.output_path,
            auto_build_caches=False,
            env_vars=reference_config.env_vars,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=LevanterCheckpoint,
        run=dispatch_offline_training,
        build_config=build_config,
        deps=(*control.deps, projection),
    )


@click.command(help=__doc__)
@click.option("--preference-name", required=True)
@click.option("--preference-version", required=True)
@click.option("--projection-version", required=True)
@click.option("--recovery-version", required=True)
@click.option("--policy-export-version", required=True)
@click.option("--policy-checkpoint-step", type=click.IntRange(min=0), required=True)
@click.option("--num-train-steps", type=click.IntRange(min=1), required=True)
@click.option("--learning-rate", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--hf-save-steps", type=click.IntRange(min=1), required=True)
@rl_build_options
def main(
    preference_name: str,
    preference_version: str,
    projection_version: str,
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
    num_train_steps: int,
    learning_rate: float,
    hf_save_steps: int,
) -> ArtifactStep:
    control = native_optimizer_step(
        preference_version,
        preference_name,
        recovery_version,
        policy_export_version,
        policy_checkpoint_step,
        num_train_steps,
        learning_rate,
        hf_save_steps,
    )
    name = user_owned_name("data/bfcl-rl-matched-chosen")
    projection = ArtifactStep.adopt(
        name + "-input", projection_version, f"{name}/{projection_version}", kind=MatchedChosenCache
    )
    step = matched_sft_optimizer_step(control, projection)
    return replace(step, runtime_args={"execution": COLLECTION_EXECUTION})


if __name__ == "__main__":
    main()
