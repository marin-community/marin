# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the selected checkpoint's context using the pinned vLLM ModelConfig."""

import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import vllm
from vllm.config import ModelConfig


def main(root: Path, overrides_path: Path, requested: int, destination: Path) -> None:
    release = json.loads(Path(__file__).with_name("vllm-release.json").read_text())
    if version("vllm") != release["distribution"]["version"]:
        raise ValueError(f"Context resolver has wrong vLLM version: {vllm.__version__}")
    if os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
        raise ValueError("Long-context validation cannot use VLLM_ALLOW_LONG_MAX_MODEL_LEN")
    config_bytes = (root / "config.json").read_bytes()
    overrides = json.loads(overrides_path.read_text())
    model = ModelConfig(
        model=str(root),
        tokenizer=str(root),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=requested,
        hf_overrides=overrides,
    )
    if model.max_model_len < requested:
        raise ValueError("Pinned vLLM resolved a shorter context than requested")
    destination.write_text(
        json.dumps(
            {
                "vllm_commit": release["source"]["fork_commit"],
                "vllm_version": version("vllm"),
                "vllm_source_version": vllm.__version__,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "overrides": overrides,
                "requested_window": requested,
                "max_model_len": model.max_model_len,
                "accepted": True,
                "resolver": "vllm.config.ModelConfig",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
