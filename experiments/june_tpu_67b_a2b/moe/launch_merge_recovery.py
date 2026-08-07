# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Central2-only calibration and functional matching for the June 67B-A2B checkpoint."""

import dataclasses
from dataclasses import dataclass

import click
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem import prefix_join

from experiments.grug.moe.merge_artifacts import ExpertCalibrationArtifact, ExpertMatchingArtifact
from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import (
    _MIXTURE_BLOCK_SIZE,
    _datakit_components,
    _phase_weights,
)
from experiments.june_tpu_67b_a2b.moe.merge_jobs import (
    JuneCalibrationJobConfig,
    JuneMatchingJobConfig,
    JuneSourceCheckpointConfig,
    run_calibration,
    run_matching,
)
from experiments.june_tpu_67b_a2b.moe.moe_67b_a2b_d2560_cooldown_step39k_seq64k_bs1024_rep8_muon_10T import (
    _model as _SOURCE_MODEL,
)
from experiments.marin_tokenizer import marin_tokenizer

_RESOURCES_KEY = "merge_resources"
_EXPERIMENT_REGION = "us-central2"
_CENTRAL2_PREFIX = "gs://marin-us-central2"
_CALIBRATION_RESOURCES = ResourceConfig.with_tpu("v4-256", regions=[_EXPERIMENT_REGION], preemptible=False)
_MATCHING_RESOURCES = ResourceConfig.with_tpu("v4-64", regions=[_EXPERIMENT_REGION], preemptible=False)
_SOURCE_CHECKPOINT = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step102k-3dac46/"
    "checkpoints/step-105149/"
)
_SOURCE_COMMIT = "b450a9d590"
_CALIBRATION_SEQUENCE_LENGTH = 8_192


@dataclass(frozen=True)
class JuneMergeMatchingPipeline:
    calibration: ArtifactStep[ExpertCalibrationArtifact]
    matching: ArtifactStep[ExpertMatchingArtifact]


def _central2_component(component: DatasetComponent | ConcatDatasetComponent):
    if isinstance(component, ConcatDatasetComponent):
        return dataclasses.replace(
            component,
            children={name: _central2_component(child) for name, child in component.children.items()},
        )
    return dataclasses.replace(component, cache_dir=prefix_join(_CENTRAL2_PREFIX, component.cache_dir))


def _calibration_data() -> LmDataConfig:
    components = {name: _central2_component(component) for name, component in _datakit_components().items()}
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=components,
        train_weights=[(0, _phase_weights(1))],
        auto_build_caches=False,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
    )


def build_merge_matching_pipeline(*, version: str | None = None) -> JuneMergeMatchingPipeline:
    source_model = dataclasses.replace(
        _SOURCE_MODEL,
        max_seq_len=_CALIBRATION_SEQUENCE_LENGTH,
        qk_mult=1.57,
        expert_bank_for_layer=tuple(range(_SOURCE_MODEL.num_layers)),
    )
    source = JuneSourceCheckpointConfig(
        model=source_model,
        checkpoint_dir=_SOURCE_CHECKPOINT,
        source_commit=_SOURCE_COMMIT,
    )

    calibration_name = "grug/expert_merge/june67b/calibration-layers-12-13"
    calibration_version = resolve_version(calibration_name, version)

    def calibration_config(ctx: StepContext) -> JuneCalibrationJobConfig:
        return JuneCalibrationJobConfig(
            source=source,
            data=_calibration_data(),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-calibration-june67b-l12-l13",
        )

    calibration = ArtifactStep(
        name=user_namespaced_name(calibration_name, calibration_version),
        version=calibration_version,
        artifact_type=ExpertCalibrationArtifact,
        run=run_calibration,
        build_config=calibration_config,
        runtime_args={_RESOURCES_KEY: _CALIBRATION_RESOURCES},
    )

    matching_name = "grug/expert_merge/june67b/matching-layers-12-13"
    matching_version = resolve_version(matching_name, version)

    def matching_config(ctx: StepContext) -> JuneMatchingJobConfig:
        return JuneMatchingJobConfig(
            source=source,
            calibration_path=ctx.artifact_path(calibration),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-matching-june67b-l12-l13",
        )

    matching = ArtifactStep(
        name=user_namespaced_name(matching_name, matching_version),
        version=matching_version,
        artifact_type=ExpertMatchingArtifact,
        run=run_matching,
        build_config=matching_config,
        deps=(calibration,),
        runtime_args={_RESOURCES_KEY: _MATCHING_RESOURCES},
    )
    return JuneMergeMatchingPipeline(calibration=calibration, matching=matching)


def build(*, version: str | None = None) -> ArtifactStep[ExpertMatchingArtifact]:
    """Build the central2-only June 67B calibration and matching graph."""
    return build_merge_matching_pipeline(version=version).matching


@click.command()
@click.option("--stage", type=click.Choice(["calibration", "matching"]), default="matching", show_default=True)
@build_options
def main(stage: str) -> ArtifactStep[ExpertCalibrationArtifact] | ArtifactStep[ExpertMatchingArtifact]:
    pipeline = build_merge_matching_pipeline()
    return pipeline.calibration if stage == "calibration" else pipeline.matching


if __name__ == "__main__":
    main()
