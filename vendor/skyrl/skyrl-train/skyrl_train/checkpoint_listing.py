"""Torch-free helpers for enumerating checkpoint directories; safe to import from launcher environments."""

import os

from loguru import logger
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX

from skyrl_train.io import io


def extract_step_from_path(path: str) -> int:
    basename = os.path.basename(path)
    if basename.startswith(GLOBAL_STEP_PREFIX):
        return int(basename.split(GLOBAL_STEP_PREFIX)[1])
    return -1


def list_checkpoint_dirs(checkpoint_base_path: str) -> list[str]:
    """
    List all checkpoint directories in the base path.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored

    Returns:
        list[str]: List of checkpoint directory names
    """
    if not io.exists(checkpoint_base_path):
        return []

    try:
        all_items = io.list_dir(checkpoint_base_path)

        # Filter for directories that match the global_step_* pattern
        checkpoint_dirs = []
        for item in all_items:
            # Get just the basename for pattern matching
            basename = os.path.basename(item)
            if basename.startswith(GLOBAL_STEP_PREFIX) and io.isdir(os.path.join(checkpoint_base_path, basename)):
                checkpoint_dirs.append(basename)

        return sorted(checkpoint_dirs)
    except Exception as e:
        logger.warning(f"Failed to list checkpoint directories from {checkpoint_base_path}: {e}")
        return []
