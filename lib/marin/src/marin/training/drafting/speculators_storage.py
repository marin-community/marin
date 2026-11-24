# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Speculators with a reader for indexed activation files in object storage."""

from argparse import Namespace

import torch
from activation_store import OBJECT_STORE_BACKEND, activation_path  # pyrefly: ignore[missing-import]
from hs_connectors import FileBackend, HiddenStatesBackend, HiddenStatesTransfer  # pyrefly: ignore[missing-import]
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from safetensors.torch import load


class ObjectStoreTransfer(HiddenStatesTransfer):
    """Read existing safetensors directly into the trainer's batch workers."""

    def __init__(self, path: str):
        self.root = StoragePath(path)

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        fs, key = filesystem_for(str(activation_path(self.root, file_idx)))
        try:
            payload = fs.cat_file(key)
        except FileNotFoundError:
            return None
        return load(payload)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        raise ValueError(f"Object-store training expects prefilled activations, got a generated handle {handle!r}")


@HiddenStatesBackend.register(OBJECT_STORE_BACKEND)
class ObjectStoreBackend(HiddenStatesBackend):
    """Use the local file connector for prefill and object storage for training."""

    add_train_args = staticmethod(FileBackend.add_train_args)
    add_launch_args = staticmethod(FileBackend.add_launch_args)
    build_kv_transfer_config = staticmethod(FileBackend.build_kv_transfer_config)

    @staticmethod
    def from_train_args(args: Namespace, _data_path: str) -> ObjectStoreTransfer:
        if not args.hidden_states_path:
            raise ValueError("--hidden-states-path is required for the object-store backend")
        return ObjectStoreTransfer(args.hidden_states_path)


if __name__ == "__main__":
    # The config parser snapshots backend arguments at import time.
    from speculators.train.cli import main  # pyrefly: ignore[missing-import]
    from speculators.train.config import TrainConfig  # pyrefly: ignore[missing-import]

    main(TrainConfig.resolve())
