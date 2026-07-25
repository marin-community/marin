# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

from rigging.filesystem import StoragePath


def discover_hf_checkpoints(base_path: str) -> list[str]:
    """Discover Hugging Face checkpoints under ``base_path``, sorted by mtime ascending (most recent last).

    A directory counts as a checkpoint when it contains both ``config.json``
    (matched by the glob) and ``tokenizer_config.json``.
    """
    config_paths = [str(m) for m in StoragePath(os.path.join(base_path, "**/config.json")).glob()]
    config_paths.sort(key=lambda path: StoragePath(path).mtime())
    return [
        os.path.dirname(path)
        for path in config_paths
        if StoragePath(os.path.join(os.path.dirname(path), "tokenizer_config.json")).exists()
    ]
