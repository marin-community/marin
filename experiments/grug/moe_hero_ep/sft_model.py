# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the final permanent Mochi SFT checkpoint for policy initialization."""

import json
from dataclasses import dataclass, replace

import draccus
from fray.types import ResourceConfig
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.compat.hf_export import HFShardRecord, HFShardResume
from marin.execution.lazy import OUT, apply
from marin.execution.remote import remote
from marin.rl.skyrl import ArtifactHfModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.hero_recipe import HERO_MODEL_CONFIG, with_transport_remat_mode
from experiments.grug.moe_hero_ep.launch_hero_sft import _placement_label, build_sft_run, sft_resources
from experiments.grug.moe_hero_ep.ops.export_cpu import CpuExportConfig, export
from experiments.grug.moe_hero_ep.ops.export_metadata import (
    EXPORT_VERSION,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    REQUEST_FILENAME,
    artifact_config,
    bf16_hf_config,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.train import RouterBiasMode


@dataclass(frozen=True, kw_only=True)
class MochiSft:
    run_id: str
    initialize_from: str
    start_step: int
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
    checkpoint_root: str | None = None
    expert_placement: ExpertPlacement | None = None

    def trained(self, resources: ResourceConfig) -> ArtifactHfModel:
        if resources != sft_resources(self.dp_racks):
            raise ValueError("Mochi SFT resources must match the GB200 allocation derived from dp_racks")
        step = build_sft_run(
            run_id=self.run_id,
            initialize_from=self.initialize_from,
            start_step=self.start_step,
            muon_lr=self.muon_lr,
            adam_lr=self.adam_lr,
            max_grad_norm=self.max_grad_norm,
            replay_fraction=self.replay_fraction,
            dp_racks=self.dp_racks,
            router_bias_mode=self.router_bias_mode,
            capacity_factor=self.capacity_factor,
            stop_after_steps=self.stop_after_steps,
            eval_every=self.eval_every,
            checkpoint_root=self.checkpoint_root,
            expert_placement=self.expert_placement,
            version=self.version,
        )
        export_step = apply(
            f"{step.name}-hf",
            remote(_export_final_sft, resources=self.export_resources, env_vars={"TMPDIR": "/cache"}),
            version=step.version,
            output=OUT,
            trained=step,
            checkpoint_root=self.checkpoint_root,
            placement_label=None if self.expert_placement is None else _placement_label(self.expert_placement),
            capacity_factor=self.capacity_factor,
            source_revision=self.source_revision,
        )
        return ArtifactHfModel(
            step=export_step,
            tokenizer_uri=self.tokenizer_uri,
            tokenizer_revision=self.tokenizer_revision,
            relative_path="hf",
        )


def _export_final_sft(
    *,
    output: str,
    trained: str,
    checkpoint_root: str | None,
    placement_label: str | None,
    capacity_factor: float,
    source_revision: str,
) -> None:
    configure_coreweave_s3()
    root = trained if checkpoint_root is None else checkpoint_root
    if placement_label is not None:
        root = prefix_join(root, placement_label)
    directory = prefix_join(root, "checkpoints")
    candidates = discover_checkpoint_candidates(directory)
    final = next(
        (candidate for candidate in reversed(candidates) if candidate.metadata.get("is_temporary") is False), None
    )
    if final is None:
        raise ValueError(f"Mochi SFT has no permanent checkpoint in {directory}")
    model = with_transport_remat_mode(replace(HERO_MODEL_CONFIG, capacity_factor=capacity_factor))
    config = CpuExportConfig(final.path, digest(final.metadata), model, prefix_join(output, "hf"), source_revision)
    try:
        export(config)
    except FileExistsError:
        root = StoragePath(config.destination)
        if not (root / MANIFEST_FILENAME).exists():
            raise
        _verify_completed_export(root, config)


def _verify_completed_export(root: StoragePath, config: CpuExportConfig) -> None:
    """Accept a completed export after an interrupted artifact commit."""
    request = {
        "export_version": EXPORT_VERSION,
        "implementation": "cpu",
        "config": artifact_config(draccus.encode(config)),
        "hf_config": bf16_hf_config(config.model),
    }
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    export_id = digest(request)
    if (
        export_id != manifest["export_id"]
        or digest(manifest["request"]) != export_id
        or digest(json.loads((root / REQUEST_FILENAME).read_text())) != export_id
    ):
        raise FileExistsError(f"Completed Mochi export differs from the selected checkpoint: {root}")
    if digest(json.loads((root / "config.json").read_text())) != digest(request["hf_config"]):
        raise ValueError(f"Completed Mochi export has an inconsistent HF config: {root}")

    records = [draccus.decode(HFShardRecord, item) for item in manifest["shards"]]
    index = json.loads((root / INDEX_FILENAME).read_text())
    weight_map = {name: record.filename for record in records for name in record.tensor_names}
    if not weight_map or index["weight_map"] != weight_map:
        raise ValueError(f"Completed Mochi export index disagrees with its shard manifests: {root}")
    resume = HFShardResume(root, export_id)
    for record in records:
        if resume.completed(record.filename, record.tensor_names) != record:
            raise ValueError(f"Completed Mochi export has an inconsistent shard receipt: {record.filename}")
