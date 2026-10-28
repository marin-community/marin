# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initialize online tracking and verify hardware for TPU operator checks."""

import dataclasses
import os
from pathlib import Path

import jax
import levanter.tracker as tracker
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer


def initialize_operator_check() -> tuple[Path, str, str]:
    """Start online tracking; return the local output directory, operator TPU type and reference TPU type."""
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    reference_topology = os.environ["BELAY_TPU"]
    operator_topology = os.environ.get("BELAY_OPERATOR_TPU", reference_topology)
    config = build_config(
        run_id,
        str(StoragePath(DATA_ROOT) / "kaiyuew/belay-ckpt1-20261025" / run_id),
        verified_tokenizer(),
        reference_topology,
        0,
    )
    config.trainer.trainer.tracker = [
        (
            dataclasses.replace(
                item,
                tags=[tag for tag in item.tags if not tag.startswith("v4-")]
                + [f"operator-{operator_topology}", f"reference-{reference_topology}"],
            )
            if isinstance(item, WandbConfig)
            else item
        )
        for item in config.trainer.trainer.tracker
    ]
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Operator qualification requires TPU")
    expected_devices = {"v4-8": 4, "v4-32": 16, "v4-64": 32}[operator_topology]
    if jax.device_count() != expected_devices:
        raise RuntimeError("Operator topology differs from its declared allocation")
    tracker.log_summary(
        {
            "kernel_check/optimizer_updates": 0,
            "kernel_check/operator_topology": operator_topology,
            "kernel_check/reference_topology": reference_topology,
            "kernel_check/device_count": jax.device_count(),
        }
    )
    return Path(os.environ["IRIS_OUTPUT_DIR"]), operator_topology, reference_topology
