# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trainer-only qualification uses the same loss normalization as the RL driver."""

import hashlib
from argparse import Namespace

import pytest
import yaml


@pytest.mark.parametrize("variant", ["chunked", "offload"])
def test_probe_derives_native_loss_context_before_worker_start(tmp_path, variant):
    pytest.importorskip("hydra", reason="Probe admission requires the native SkyRL CPU environment")
    from cloud.iris.launch_config import load_launch_config  # noqa: PLC0415 - optional SkyRL runtime
    from cloud.iris.tests.launch_config_fixtures import raw_launch_config  # noqa: PLC0415 - optional SkyRL runtime
    from omegaconf import OmegaConf  # noqa: PLC0415 - optional SkyRL runtime
    from skyrl_train.config.driver_validation import validate_cfg  # noqa: PLC0415 - optional SkyRL runtime

    from experiments.post_training.long_loop.trainer_memory_probe_config import (  # noqa: PLC0415 - optional SkyRL runtime
        compiled_config,
    )

    source = tmp_path / "launch.yaml"
    source.write_text(yaml.safe_dump(raw_launch_config()))
    reference = load_launch_config(source).skyrl
    assert OmegaConf.select(reference, "trainer.algorithm.max_seq_len") is None
    validate_cfg(reference, check_runtime_resources=False)
    config = compiled_config(
        Namespace(
            config_uri=str(source),
            config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output_prefix=str(tmp_path / variant),
            variant=variant,
        )
    )
    assert config.trainer.algorithm.max_seq_len == reference.trainer.algorithm.max_seq_len == 1024
    assert config.trainer.algorithm.resolved_group_advantage == reference.trainer.algorithm.resolved_group_advantage
