# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Base-start worker, extracted unchanged from recipe_screen without launch code."""

import hashlib
import json
import os

import draccus
import jax
from iris.jax.compile_cache import configure_jax_compilation_cache
from levanter.checkpoint import load_checkpoint_metadata
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.sft_runtime.train import GrugRunConfig, _run_grug_local


def train_arm(config: GrugRunConfig, runtime: dict) -> None:
    """Run the existing Grug worker after checking its frozen runtime environment."""
    configure_coreweave_s3()
    configure_jax_compilation_cache()
    for name, expected in runtime["worker_environment"].items():
        if os.environ.get(name) != expected:
            raise ValueError(f"Worker runtime differs: {name}")
    if os.environ.get("XLA_FLAGS") != runtime["xla_flags"]:
        # Iris adds its node-local autotune-cache directory to the frozen flags.
        actual = [
            f
            for f in os.environ.get("XLA_FLAGS", "").split()
            if not f.startswith("--xla_gpu_per_fusion_autotune_cache_dir=")
        ]
        if actual != runtime["xla_flags"].split():
            raise ValueError("Worker XLA flags differ from the reviewed E12 profile")
    output = StoragePath(config.trainer.trainer.checkpointer.base_path).parent
    if int(os.environ["IRIS_MULTIGPU_PROCESS_INDEX"]) == 0:
        (output / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    _run_grug_local(config)
    rank = int(os.environ["IRIS_MULTIGPU_PROCESS_INDEX"])
    (output / f"allocator-end-rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "platform": jax.default_backend(),
                "devices": [{"id": device.id, "stats": device.memory_stats()} for device in jax.local_devices()],
            },
            indent=2,
        )
    )
    step = config.stop_after_steps
    checkpoint = str(output / "checkpoints" / f"step-{step}")
    metadata = load_checkpoint_metadata(checkpoint)
    if metadata["step"] != step or metadata["is_temporary"]:
        raise ValueError("Recipe arm did not commit its selected final checkpoint")
    if int(os.environ["IRIS_MULTIGPU_PROCESS_INDEX"]) == 0:
        (output / "selected.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "checkpoint": checkpoint,
                    "metadata_digest": digest(metadata),
                    "metadata_sha256": (
                        hashlib.sha256((StoragePath(checkpoint) / "metadata.json").read_bytes()).hexdigest()
                    ),
                },
                indent=2,
            )
        )
