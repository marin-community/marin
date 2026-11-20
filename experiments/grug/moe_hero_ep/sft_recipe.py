# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Mochi SFT training and export its final checkpoint for policy initialization."""

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fray.types import ResourceConfig
from levanter.checkpoint import discover_checkpoint_candidates
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import ArtifactHfModel
from marin.training.run_environment import add_run_env_variables
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.hero_recipe import HERO_MODEL_CONFIG, HeroThroughputResult, with_transport_remat_mode
from experiments.grug.moe_hero_ep.launch_hero_sft import (
    build_sft_run,
    checkpoint_step,
    permanent_checkpoint_dir,
    sft_resources,
)
from experiments.grug.moe_hero_ep.ops.export_cpu import CpuExportConfig, export
from experiments.grug.moe_hero_ep.ops.upload_hf import write_tokenizer_metadata
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.train import RouterBiasMode


@dataclass(frozen=True, kw_only=True)
class MochiSFT:
    run_id: str
    initialize_from: str
    muon_lr: float
    adam_lr: float
    max_grad_norm: float | None
    replay_fraction: float
    dp_racks: int
    router_bias_mode: RouterBiasMode
    capacity_factor: float
    version: str
    source_revision: str
    export_resources: ResourceConfig
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoints: Literal["periodic", "final", "none"]
    stop_after_steps: int | None = None
    eval_every: int = 0
    expert_placement: ExpertPlacement | None = None

    def trained(self, resources: ResourceConfig) -> ArtifactHfModel:
        recipe = replace(self, checkpoints="final") if self.checkpoints == "none" else self
        prefix = None if self.checkpoints == "periodic" else self.scratch_prefix()
        return recipe.export_model(recipe.training_step(resources, prefix=prefix))

    def scratch_prefix(self) -> str:
        return marin_temp_bucket(3, source_prefix=self.initialize_from)

    def training_step(
        self,
        resources: ResourceConfig,
        *,
        prefix: str | None = None,
    ) -> ArtifactStep[HeroThroughputResult]:
        allocation = sft_resources(self.dp_racks)
        if resources != replace(allocation, regions=resources.regions, target_cluster=resources.target_cluster):
            raise ValueError("Mochi SFT resources must match the GB200 allocation derived from dp_racks")
        run_id = self.run_id
        if prefix is not None:
            length = f"{self.stop_after_steps}-steps" if self.stop_after_steps is not None else "full"
            suffix = "-export" if self.checkpoints == "final" else ""
            run_id = f"{run_id}-{length}{suffix}"
        step = build_sft_run(
            run_id=run_id,
            initialize_from=self.initialize_from,
            start_step=checkpoint_step(self.initialize_from),
            muon_lr=self.muon_lr,
            adam_lr=self.adam_lr,
            max_grad_norm=self.max_grad_norm,
            replay_fraction=self.replay_fraction,
            dp_racks=self.dp_racks,
            router_bias_mode=self.router_bias_mode,
            capacity_factor=self.capacity_factor,
            stop_after_steps=self.stop_after_steps,
            eval_every=self.eval_every,
            expert_placement=self.expert_placement,
            version=self.version,
            resources=resources,
            checkpoints=self.checkpoints,
        )
        if prefix is None:
            return step
        return replace(step, name=user_owned_name(f"grug/{run_id}"), output_path_prefix=prefix)

    def export_model(self, step: ArtifactStep[HeroThroughputResult]) -> ArtifactHfModel:
        training = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))
        relative_path = "hf"
        source_revision = self.source_revision
        if step.output_path_prefix is not None:
            source_revision = add_run_env_variables({}).get("GIT_COMMIT", source_revision)
        export_step = apply(
            f"{step.name}-hf",
            remote(_export_final_sft, resources=self.export_resources, env_vars={"TMPDIR": "/cache"}),
            version=step.version,
            output=OUT,
            trained=step,
            expected_step=training.stop_step,
            placement=self.expert_placement,
            capacity_factor=self.capacity_factor,
            source_revision=source_revision,
            relative_path=relative_path,
            tokenizer_uri=self.tokenizer_uri,
            tokenizer_revision=self.tokenizer_revision,
        )
        return ArtifactHfModel(
            step=replace(export_step, output_path_prefix=step.output_path_prefix), relative_path=relative_path
        )


def _export_final_sft(
    *,
    output: str,
    trained: str,
    expected_step: int,
    placement: ExpertPlacement | None,
    capacity_factor: float,
    source_revision: str,
    relative_path: str,
    tokenizer_uri: str,
    tokenizer_revision: str,
) -> None:
    configure_coreweave_s3()
    directory = permanent_checkpoint_dir(trained, None, placement)
    candidates = discover_checkpoint_candidates(directory)
    final = next(
        (candidate for candidate in reversed(candidates) if candidate.metadata.get("is_temporary") is False), None
    )
    if final is None:
        raise ValueError(f"Mochi SFT has no permanent checkpoint in {directory}")
    if final.step != expected_step:
        raise ValueError(f"Mochi SFT expected final step {expected_step}, found newest permanent step {final.step}")
    model = with_transport_remat_mode(replace(HERO_MODEL_CONFIG, capacity_factor=capacity_factor))
    destination = StoragePath(output) / relative_path
    config = CpuExportConfig(final.path, digest(final.metadata), model, str(destination), source_revision)
    with TemporaryDirectory() as directory:
        metadata = Path(directory)
        write_tokenizer_metadata(metadata, repo_id=tokenizer_uri, revision=tokenizer_revision)
        export(config, accept_completed=True)
        for path in metadata.rglob("*"):
            relative = path.relative_to(metadata)
            if path.is_file() and ".cache" not in relative.parts:
                (destination / relative.as_posix()).upload_from(str(path))
