# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU admission for the trainer-only memory qualification."""

import hashlib
import tempfile
from pathlib import Path

from cloud.iris.launch_config import load_launch_config
from omegaconf import open_dict
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from skyrl_train.config.driver_validation import validate_cfg


def compiled_config(args):
    fs, path = filesystem_for(args.config_uri)
    raw = fs.cat(path)
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError("Resolved source launch SHA mismatch")
    with tempfile.TemporaryDirectory(prefix="trainer-sizing-config-") as directory:
        source = Path(directory) / "launch.yaml"
        source.write_bytes(raw)
        launch = load_launch_config(source)
    cfg = launch.skyrl
    with open_dict(cfg):
        cfg.trainer.run_name = args.output_prefix.rstrip("/").split("/")[-1]
        cfg.trainer.logger = "console"
        cfg.trainer.export_path = args.output_prefix
        cfg.trainer.native_telemetry_root = prefix_join(args.output_prefix, "telemetry")
        cfg.trainer.policy.model.path = "/tmp/trainer-profile-model-metadata"
        cfg.trainer.policy.model.tokenizer_path = launch.inputs.model.tokenizer_uri
        cfg.trainer.policy.model.tokenizer_revision = launch.inputs.model.tokenizer_revision
        megatron = cfg.trainer.policy.megatron_config
        megatron.output_projection_chunk_size = 1024 if args.variant == "chunked" else None
        megatron.optimizer_config_kwargs.optimizer_cpu_offload = args.variant == "offload"
        megatron.optimizer_config_kwargs.optimizer_offload_fraction = 1.0 if args.variant == "offload" else 0.0
        megatron.memory_profile = {
            "enabled": True,
            "max_sequence_length": 262144,
            "valid_token_counts": [85557],
            "minimum_headroom_fraction": 0.1,
        }
    validate_cfg(cfg, check_runtime_resources=False)
    return cfg
