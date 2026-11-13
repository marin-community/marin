"""Bounded local metadata views of remote Megatron checkpoints."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from skyrl_train.io import io


@contextmanager
def remote_checkpoint_metadata(checkpoint_dir: str):
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
        yield directory
