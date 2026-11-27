# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Mochi SFT training and export its final checkpoint for policy initialization."""

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
from typing import Literal

from fray.types import ResourceConfig
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.infra.cli_helpers import get_git_commit
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import ArtifactHfModel
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.launch_hero_sft import (
    SFT_TOKENIZER,
    build_sft_run,
    checkpoint_step,
    permanent_checkpoint_dir,
    sft_model_config,
    sft_resources,
)
from experiments.grug.moe_hero_ep.ops.export_cpu import (
    DEFAULT_EXPORT_HOST_BUDGET_BYTES,
    DEFAULT_READ_HOST_BUDGET_BYTES,
    CpuExportConfig,
    export,
    full_expert_bank_read_bytes,
)
from experiments.grug.moe_hero_ep.ops.upload_hf import write_tokenizer_metadata
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.train import RouterBiasMode
from experiments.sft.launcher import PreparedData, sft_child_resources


@dataclass(frozen=True, kw_only=True)
class MochiSFT:
    """Train a Mochi mixture on the given number of GB200 racks and export its final checkpoint."""

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
    prepared_data: PreparedData | None = None

    def with_data_mix(self, data: PreparedData) -> "MochiSFT":
        """Bind a versioned LM data manifest while retaining the training recipe."""
        return replace(self, run_id=f"{self.run_id}-mix-{data.mix_fingerprint()}", prepared_data=data)

    def mix_tokenizer(self) -> str:
        return SFT_TOKENIZER

    def with_steps(self, steps: int, *, version: str) -> "MochiSFT":
        """Set the SFT update budget and keep its final native checkpoint."""
        return replace(self, stop_after_steps=steps, checkpoints="final", version=version)

    def trained(
        self,
        resources: ResourceConfig,
        *,
        steps: int | None = None,
        prefix: str | None = None,
        coordinator_cluster: str | None = None,
        coordinator_prefix: str | None = None,
    ) -> ArtifactHfModel:
        if steps is not None or self.stop_after_steps is not None or self.checkpoints != "periodic":
            prefix = prefix or coordinator_prefix
        recipe, prefix = self._run_recipe(steps=steps, export=True, prefix=prefix)
        training = recipe._training_artifact(sft_child_resources(resources, coordinator_cluster), prefix=prefix)
        return recipe.export_model(training, prefix=prefix, coordinator_cluster=coordinator_cluster)

    def train_artifact(
        self, resources: ResourceConfig, *, steps: int | None = None, prefix: str | None = None
    ) -> ArtifactStep[HeroThroughputResult]:
        """Build the native training artifact."""
        recipe, prefix = self._run_recipe(steps=steps, export=False, prefix=prefix)
        return recipe._training_artifact(resources, prefix=prefix)

    def _run_recipe(self, *, steps: int | None, export: bool, prefix: str | None) -> tuple["MochiSFT", str | None]:
        stop = self.stop_after_steps if steps is None else steps
        if stop is None and self.checkpoints == "periodic":
            return self, prefix
        recipe = replace(self, stop_after_steps=stop, checkpoints="final" if export else "none")
        if prefix is None:
            prefix = marin_temp_bucket(3, source_prefix=self.initialize_from)
        return recipe, prefix

    def _training_artifact(
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
            prepared_data=self.prepared_data,
        )
        if prefix is None:
            return step
        return replace(step, name=user_owned_name(f"grug/{run_id}"), output_path_prefix=prefix)

    def export_model(
        self,
        step: ArtifactStep[HeroThroughputResult],
        *,
        prefix: str | None,
        coordinator_cluster: str | None = None,
    ) -> ArtifactHfModel:
        expected_step = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps)).stop_step
        relative_path = "hf"
        source_revision = self.source_revision
        export_resources = self.export_resources
        export_fn: Callable[..., None] = _export_final_sft
        if self.checkpoints == "final":
            try:
                source_revision = os.environ.get("GIT_COMMIT") or get_git_commit()
            except (CalledProcessError, OSError) as error:
                raise ValueError("Set GIT_COMMIT to the source revision when exporting a new Mochi SFT run") from error
            if not source_revision:
                raise ValueError("A final Mochi SFT export requires the submission's source revision")
            # A final-only checkpoint can store a whole expert bank in one native chunk.
            gib = 1024**3
            reservation_increment = 64 * gib
            cache_bytes = full_expert_bank_read_bytes(sft_model_config(self.capacity_factor))
            read_budget = (cache_bytes + reservation_increment - 1) // reservation_increment * reservation_increment
            host_bytes = read_budget + DEFAULT_EXPORT_HOST_BUDGET_BYTES + 32 * gib
            export_resources = replace(export_resources, cpu=32, ram=f"{host_bytes // gib}GiB")
            export_fn = partial(_export_final_sft, read_host_budget_bytes=read_budget)
        export_step = apply(
            f"{step.name}-hf",
            remote(
                export_fn,
                resources=sft_child_resources(export_resources, coordinator_cluster),
                env_vars={"TMPDIR": "/cache"},
            ),
            version=step.version,
            output=OUT,
            trained=step,
            expected_step=expected_step,
            placement=self.expert_placement,
            capacity_factor=self.capacity_factor,
            source_revision=source_revision,
            relative_path=relative_path,
            tokenizer_uri=self.tokenizer_uri,
            tokenizer_revision=self.tokenizer_revision,
        )
        return ArtifactHfModel(step=replace(export_step, output_path_prefix=prefix), relative_path=relative_path)


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
    read_host_budget_bytes: int = DEFAULT_READ_HOST_BUDGET_BYTES,
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
    model = sft_model_config(capacity_factor)
    destination = StoragePath(output) / relative_path
    config = CpuExportConfig(
        final.path, digest(final.metadata), model, str(destination), source_revision, read_host_budget_bytes
    )
    with TemporaryDirectory() as directory:
        metadata = Path(directory)
        write_tokenizer_metadata(metadata, repo_id=tokenizer_uri, revision=tokenizer_revision)
        export(config, accept_completed=True)
        for path in metadata.rglob("*"):
            relative = path.relative_to(metadata)
            if path.is_file() and ".cache" not in relative.parts:
                (destination / relative.as_posix()).upload_from(str(path))
