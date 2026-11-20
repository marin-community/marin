# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the final permanent Mochi SFT checkpoint for policy initialization."""

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from fray.types import ResourceConfig
from levanter.checkpoint import discover_checkpoint_candidates
from marin.execution.lazy import OUT, StepContext, apply
from marin.execution.remote import remote
from marin.rl.skyrl import ArtifactHfModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.hero_recipe import HERO_MODEL_CONFIG, with_transport_remat_mode
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
from experiments.sft.launcher import sft_child_resources


@dataclass(frozen=True, kw_only=True)
class GrugHeroSFT:
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
    stop_after_steps: int | None = None
    eval_every: int = 0
    expert_placement: ExpertPlacement | None = None

    def trained(self, resources: ResourceConfig, *, coordinator_cluster: str | None = None) -> ArtifactHfModel:
        allocation = sft_resources(self.dp_racks)
        if resources != replace(allocation, regions=resources.regions, target_cluster=resources.target_cluster):
            raise ValueError("Mochi SFT resources must match the GB200 allocation derived from dp_racks")
        step = build_sft_run(
            run_id=self.run_id,
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
            resources=sft_child_resources(resources, coordinator_cluster),
        )
        training = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))
        relative_path = "hf"
        export_step = apply(
            f"{step.name}-hf",
            remote(
                _export_final_sft,
                resources=sft_child_resources(self.export_resources, coordinator_cluster),
                env_vars={"TMPDIR": "/cache"},
            ),
            version=step.version,
            output=OUT,
            trained=step,
            expected_step=training.stop_step,
            placement=self.expert_placement,
            capacity_factor=self.capacity_factor,
            source_revision=self.source_revision,
            relative_path=relative_path,
            tokenizer_uri=self.tokenizer_uri,
            tokenizer_revision=self.tokenizer_revision,
        )
        return ArtifactHfModel(step=export_step, relative_path=relative_path)


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
