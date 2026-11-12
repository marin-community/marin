"""Bounded local metadata views of remote Megatron checkpoints."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

import torch

from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from skyrl_train.io import io


@contextmanager
def remote_checkpoint_metadata(checkpoint_dir: str, load_strategy):
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
        if not (root / "common.pt").exists():
            from megatron.core.dist_checkpointing.mapping import ShardedObject

            # Core 0.19 stores common state as one DCP record. Read only that
            # record through S3, then expose the legacy control file to Core's
            # local metadata loader without staging any rank tensor shards.
            common = load_strategy.load(
                {"common_state": ShardedObject("common_state", None, (1,), (0,))}, root
            )
            torch.save(common["common_state"], root / "common.pt")
        yield directory
