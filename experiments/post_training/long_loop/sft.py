# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the tested 262k SFT reproduction and frozen-router HF handoff."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import draccus
import equinox as eqx
import jax
import yaml
from fray.types import ResourceConfig
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.checkpointing import load_grug_checkpoint
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig  # noqa: F401 -- register optimizer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.execution_policy import DEFAULT_TASK_TIMEOUT_SECONDS
from experiments.post_training.long_loop.runtime_timing import timed
from experiments.post_training.long_loop.serving import SNOWBALL_CONTEXT_LIMIT
from experiments.post_training.long_loop.sft_runtime.export_writer import ExportConfig, export_weights
from experiments.post_training.long_loop.sft_runtime.import_hf import ImportConfig, convert
from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig, Transformer
from experiments.post_training.long_loop.sft_runtime.train import GrugRunConfig, run_grug

BASE_MODEL = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
BASE_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
TOKENIZER_REPO = "marin-community/marin-tokenizer-tools"
TOKENIZER_REVISION = "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09"
TOKENIZER_URI = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"
SMOKE_CONFIG = Path(__file__).with_name("sft-smoke.yaml")


def read_json(uri: str) -> dict:
    fs, key = filesystem_for(uri)
    return json.loads(fs.cat_file(key))


def import_base(*, output: str, model: dict, revision: str) -> None:
    configure_coreweave_s3()
    convert(ImportConfig(BASE_MODEL, revision, draccus.decode(GrugModelConfig, model), prefix_join(output, "native")))


def fresh_sft_config(*, output: str, initializer: str, recipe: dict, run_id: str) -> GrugRunConfig:
    """Build a fresh weights-only start; this entrypoint never implicitly resumes."""
    # A fresh decoded copy preserves the identity-bearing template supplied by the artifact.
    config = json.loads(json.dumps(recipe))
    config["resources"]["target_cluster"] = None  # Already running inside the RNO2A coordinator.
    trainer = config["trainer"]["trainer"]
    watchdog = trainer.setdefault("progress_watchdog", {})
    for field in ("process_timeout", "startup_timeout", "step_timeout"):
        watchdog.setdefault(field, f"{DEFAULT_TASK_TIMEOUT_SECONDS}s")
    trainer["checkpointer"].setdefault("async_timeout_seconds", DEFAULT_TASK_TIMEOUT_SECONDS)
    trainer["id"] = run_id
    config["native_timing_root"] = prefix_join(output, "timing/native-sft")
    trainer["initialize_from"] = prefix_join(initializer, "native/step-0")
    trainer["load_checkpoint"] = False
    trainer["load_checkpoint_path"] = []
    trainer["checkpointer"]["base_path"] = prefix_join(output, "checkpoints")
    trainer["checkpointer"]["temporary_base_path"] = prefix_join(output, "temporary-checkpoints")
    trainer["tracker"].update(id=run_id, name=run_id)
    return draccus.decode(GrugRunConfig, config)


def sft_run(*, output: str, initializer: str, recipe: dict, run_id: str) -> None:
    configure_coreweave_s3()
    verification = read_json(prefix_join(initializer, "native/verification.json"))
    if not verification["exact_parity"] or len(verification["tensors"]) != 502:
        raise ValueError("Base import did not verify all 502 tensors")
    config = fresh_sft_config(output=output, initializer=initializer, recipe=recipe, run_id=run_id)
    (StoragePath(output) / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    run_grug(config)
    final_step = config.stop_after_steps or config.trainer.trainer.num_train_steps
    metadata = read_json(prefix_join(output, f"checkpoints/step-{final_step}/metadata.json"))
    if metadata.get("step") != final_step or metadata.get("is_temporary") is not False:
        raise ValueError(f"SFT did not commit permanent step {final_step}")
    (StoragePath(output) / "selected.json").write_text(
        json.dumps(
            {
                "checkpoint": prefix_join(output, f"checkpoints/step-{final_step}"),
                "step": final_step,
                "metadata_digest": digest(metadata),
            }
        )
    )


def restore_frozen(
    checkpoint: str, metadata_digest: str, config: GrugModelConfig, mesh: jax.sharding.Mesh
) -> Transformer:
    metadata = read_json(prefix_join(checkpoint, "metadata.json"))
    if (
        digest(metadata) != metadata_digest
        or metadata.get("step") != int(checkpoint.rsplit("step-", 1)[1])
        or metadata.get("is_temporary") is not False
    ):
        raise ValueError("SFT handoff metadata changed or is not the selected permanent step")
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.key(0))
    state = load_grug_checkpoint(state={"params": template}, candidate=checkpoint, mesh=mesh, allow_partial=False)
    return jax.block_until_ready(state["params"])


def validate_handoff(root: str) -> None:
    """Reject schema declarations that would cause Bridge to omit expert weights."""
    config = read_json(prefix_join(root, "hf/config.json"))
    weight_map = read_json(prefix_join(root, "hf/model.safetensors.index.json"))["weight_map"]
    if config["grugmoe_artifact_schema_version"] != 2:
        raise ValueError("Long-loop handoff requires split-expert schema 2")
    for layer in range(config["num_hidden_layers"]):
        for expert in range(config["num_experts"]):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                if name not in weight_map:
                    raise ValueError(f"Invalid SFT export: schema-2 tensor missing: {name}")


def export_sft(
    *,
    output: str,
    trained: str,
    model: dict,
    source_revision: str,
    tokenizer_uri: str = TOKENIZER_URI,
    context_limit: int = SNOWBALL_CONTEXT_LIMIT,
) -> None:
    configure_coreweave_s3()
    selected = read_json(prefix_join(trained, "selected.json"))
    config = ExportConfig(
        selected["checkpoint"],
        selected["metadata_digest"],
        draccus.decode(GrugModelConfig, model),
        prefix_join(output, "hf"),
        source_revision,
        max_position_embeddings=context_limit,
    )
    export_weights(config, restore_frozen, weights_kind="snowball_sft_frozen_router")
    fs, prefix = filesystem_for(tokenizer_uri)
    destination_fs, destination = filesystem_for(prefix_join(output, "hf"))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        content = fs.cat_file(prefix_join(prefix, name))
        destination_fs.pipe_file(prefix_join(destination, name), content)
    destination_fs.pipe_file(
        prefix_join(destination, "generation_config.json"),
        json.dumps(
            {
                "bos_token_id": 128000,
                "eos_token_id": [128001, 128009],
                "pad_token_id": 128001,
            }
        ).encode(),
    )
    manifest = read_json(prefix_join(output, "hf/export-manifest.json"))
    for shard in manifest["shards"]:
        with destination_fs.open(prefix_join(destination, shard["filename"]), "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != shard["sha256"]:
            raise ValueError(f"Export hash mismatch: {shard['filename']}")
    validate_handoff(output)


def sft_steps(
    version: str,
    source_revision: str,
    prerequisites: tuple[ArtifactStep, ...] = (),
    *,
    recipe: dict | None = None,
    run_kind: str = "plumbing",
    initializer: ArtifactStep | None = None,
) -> tuple[ArtifactStep, ArtifactStep[LevanterCheckpoint]]:
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text()) if recipe is None else recipe
    if initializer is None:
        initializer = apply(
            "long-loop/base",
            remote(
                import_base,
                name=f"long-loop-base-{version}",
                resources=ResourceConfig.with_gpu(
                    "H100", count=8, cpu=32, ram="512GB", disk="500GB", target_cluster=None
                ),
            ),
            version=version,
            output=OUT,
            model=recipe["model"],
            revision=BASE_REVISION,
        )
        initializer = timed(replace(initializer, deps=(*initializer.deps, *prerequisites)))
    trained = apply(
        "long-loop/sft",
        sft_run,
        version=version,
        output=OUT,
        initializer=initializer,
        recipe=recipe,
        run_id=f"long-loop-{run_kind}-sft-{version}",
    )
    trained = timed(trained)
    return trained, sft_export_step(version, source_revision, trained, recipe["model"])


def sft_export_step(
    version: str, source_revision: str, trained: ArtifactStep, model: dict
) -> ArtifactStep[LevanterCheckpoint]:
    """Export an existing committed native SFT checkpoint into a fresh handoff."""
    exported = apply(
        "long-loop/sft-handoff",
        remote(
            export_sft,
            name=f"long-loop-export-{version}",
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512GB", disk="256GB", target_cluster=None),
            env_vars={"JAX_PLATFORMS": "cpu"},
        ),
        version=version,
        artifact_type=LevanterCheckpoint,
        output=OUT,
        trained=trained,
        model=model,
        source_revision=source_revision,
    )
    return timed(exported)
