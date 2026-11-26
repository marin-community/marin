"""Bounded local metadata views of remote Megatron checkpoints."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

import torch
from torch.optim import Optimizer

from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from skyrl_train.io import io


@contextmanager
def remote_checkpoint_metadata(checkpoint_dir: str, *, optimizer_classes: tuple[type[Optimizer], ...]):
    """Stage Megatron control files while leaving DCP rank tensors remote."""
    with tempfile.TemporaryDirectory(prefix="megatron-metadata-") as directory:
        root = Path(directory).resolve()
        for filename in io.find_files(checkpoint_dir):
            relative = relative_resource_path(checkpoint_dir, filename)
            if relative.endswith(".distcp"):
                continue
            destination = root / relative
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Invalid checkpoint metadata path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(io.read_bytes(join_resource_path(checkpoint_dir, relative)))
        # HybridDeviceOptimizer stores its CPU/GPU optimizer classes in common.pt.
        # Keep restricted Torch loading, with only the runtime's known classes allowed
        # for this restore; unrelated checkpoint globals must still fail closed.
        with torch.serialization.safe_globals(list(optimizer_classes)):
            yield directory
