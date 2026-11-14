# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate tensors in local sharded or single-file Hugging Face checkpoints."""

import json
from pathlib import Path

from safetensors import safe_open


def checkpoint_map(directory: Path) -> dict[str, str]:
    index = directory / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as reader:
        return dict.fromkeys(reader.keys(), "model.safetensors")
