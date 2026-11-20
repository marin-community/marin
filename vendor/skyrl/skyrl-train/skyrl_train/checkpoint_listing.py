"""Torch-free helpers for finding and retaining native checkpoints."""

import json
import os

from loguru import logger
from marinskyrl.checkpoint_paths import BEST_CHECKPOINT_FILENAME, GLOBAL_STEP_PREFIX

from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
from skyrl_train.io import io

ROLLOUT_STATE_FILENAME = "data.pt"


def extract_step_from_path(path: str) -> int:
    basename = os.path.basename(path)
    if basename.startswith(GLOBAL_STEP_PREFIX) and (suffix := basename.removeprefix(GLOBAL_STEP_PREFIX)).isdecimal():
        return int(suffix)
    return -1


def list_checkpoint_dirs(checkpoint_base_path: str, *, strict: bool = False) -> list[str]:
    """
    List all checkpoint directories in the base path.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored
        strict: Propagate storage failures for retention decisions on a committed run

    Returns:
        list[str]: List of checkpoint directory names
    """
    if not strict and not io.exists(checkpoint_base_path):
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
        if strict:
            raise
        logger.warning(f"Failed to list checkpoint directories from {checkpoint_base_path}: {e}")
        return []


def best_checkpoint_step(checkpoint_root: str) -> int | None:
    best_path = os.path.join(checkpoint_root, BEST_CHECKPOINT_FILENAME)
    if io.exists(best_path):
        with io.open_file(best_path, "r") as stream:
            return int(json.load(stream)["step"])
    return None


def periodic_checkpoint_steps(
    checkpoint_root: str, keep_every: int | None, latest_step: int, *, strict: bool = False
) -> set[int]:
    if keep_every is None:
        return set()
    return committed_checkpoint_steps(checkpoint_root, latest_step, every=keep_every, strict=strict)


def committed_checkpoint_steps(
    checkpoint_root: str, latest_step: int, *, every: int = 1, strict: bool = False
) -> set[int]:
    return {
        step
        for directory in list_checkpoint_dirs(checkpoint_root, strict=strict)
        if 0 < (step := extract_step_from_path(directory)) <= latest_step
        and step % every == 0
        and io.exists(os.path.join(checkpoint_root, directory, TRAINER_STATE_FILENAME))
        and io.exists(os.path.join(checkpoint_root, directory, ROLLOUT_STATE_FILENAME))
    }


def cleanup_old_checkpoints(
    checkpoint_base_path: str, max_checkpoints: int, protected_steps: set[int] | None = None
) -> None:
    """
    Keep the most recent `max_checkpoints` and protected checkpoints; remove the rest.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored
        max_checkpoints: Maximum number of recent checkpoints to keep
        protected_steps: Additional checkpoint steps retained until external work completes
    """
    if max_checkpoints < 0:
        return

    checkpoint_dirs = list_checkpoint_dirs(checkpoint_base_path)

    if len(checkpoint_dirs) <= max_checkpoints:
        return

    checkpoint_dirs.sort(key=extract_step_from_path)

    protected_steps = protected_steps or set()
    recent = set(checkpoint_dirs[-max_checkpoints:]) if max_checkpoints > 0 else set()
    dirs_to_remove = [
        directory
        for directory in checkpoint_dirs
        if directory not in recent and extract_step_from_path(directory) not in protected_steps
    ]

    for dir_name in dirs_to_remove:
        full_path = os.path.join(checkpoint_base_path, dir_name)
        try:
            io.remove(full_path)
            step_num = extract_step_from_path(dir_name)
            logger.info(f"Cleaned up old checkpoint: global_step_{step_num} at {full_path}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {full_path}: {e}")
