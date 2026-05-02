# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from collections.abc import Callable, Sequence
from typing import TypeVar

import fsspec
import jax
from fsspec import AbstractFileSystem
from levanter.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")


def _get_fs_and_plain_path(path: str) -> tuple[AbstractFileSystem, str]:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    return fs, plain_path


def _checkpoint_candidates(checkpoint_search_paths: Sequence[str]) -> list[str]:
    candidates: list[tuple[int, str, str]] = []
    for search_path in checkpoint_search_paths:
        candidates.extend(_scan_checkpoint_root(search_path))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered_candidates = [candidate for _, _, candidate in candidates]

    for search_path in checkpoint_search_paths:
        if search_path not in ordered_candidates:
            ordered_candidates.append(search_path)
    return ordered_candidates


def _scan_checkpoint_root(root_path: str) -> list[tuple[int, str, str]]:
    """Scan a single root path and return (step, timestamp, path) tuples."""
    fs, plain_path = _get_fs_and_plain_path(root_path)
    base_path_protocol = urllib.parse.urlparse(root_path).scheme

    def maybe_unstrip_protocol(path: str) -> str:
        if base_path_protocol != "" and urllib.parse.urlparse(path).scheme == "":
            return f"{base_path_protocol}://{path}"
        return path

    checkpoint_dirs = [maybe_unstrip_protocol(d) for d in fs.glob(os.path.join(plain_path, "*")) if fs.isdir(d)]
    checkpoint_dirs.append(root_path)

    results: list[tuple[int, str, str]] = []
    for candidate in checkpoint_dirs:
        metadata_path = os.path.join(candidate, "metadata.json")
        if not fs.exists(metadata_path):
            continue

        try:
            with fs.open(metadata_path) as metadata_in:
                metadata = json.load(metadata_in)
        except Exception:
            logger.warning("Skipping unreadable checkpoint metadata at %s", metadata_path, exc_info=True)
            continue

        step = metadata.get("step")
        try:
            step_num = int(step)
        except (TypeError, ValueError):
            step_num = -1

        timestamp = metadata.get("timestamp")
        timestamp_key = str(timestamp) if timestamp is not None else ""
        results.append((step_num, timestamp_key, candidate))

    return results


def restore_grug_state_from_checkpoint(
    state: StateT,
    *,
    checkpoint_search_paths: Sequence[str],
    load_checkpoint_setting: bool | None,
    mesh: jax.sharding.Mesh | None,
    allow_partial: bool,
    _load_fn: Callable[..., StateT] = load_checkpoint,
) -> StateT:
    if not checkpoint_search_paths:
        if load_checkpoint_setting:
            raise FileNotFoundError("load_checkpoint=True but no checkpoint search paths are configured.")
        return state

    if load_checkpoint_setting is False:
        return state

    candidates = _checkpoint_candidates(checkpoint_search_paths)
    last_error: FileNotFoundError | None = None

    for candidate in candidates:
        try:
            loaded = _load_candidate_state(
                state=state,
                candidate=candidate,
                mesh=mesh,
                allow_partial=allow_partial,
                load_fn=_load_fn,
            )
            if candidate not in checkpoint_search_paths:
                logger.info("Loaded checkpoint from %s while searching %s", candidate, checkpoint_search_paths)
            return loaded
        except FileNotFoundError as exc:
            last_error = exc
            logger.warning(
                "Checkpoint candidate %s could not be loaded (%s). Trying an older checkpoint.", candidate, exc
            )

    if load_checkpoint_setting is True:
        search_path_summary = ", ".join(checkpoint_search_paths)
        attempted = ", ".join(candidates)
        if last_error is None:
            raise FileNotFoundError(f"Could not find checkpoint under any of: {search_path_summary}")
        raise FileNotFoundError(
            f"Could not load a checkpoint from search paths {search_path_summary}. Attempted: {attempted}"
        ) from last_error

    logger.info("Checkpoint not found under %s. Starting from scratch.", checkpoint_search_paths)
    return state


def _load_candidate_state(
    *,
    state: StateT,
    candidate: str,
    mesh: jax.sharding.Mesh | None,
    allow_partial: bool,
    load_fn: Callable[..., StateT],
) -> StateT:
    try:
        return load_fn(
            state,
            candidate,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=allow_partial,
        )
    except FileNotFoundError:
        # Backward compatibility: older grug runs saved {"train_state": state}.
        wrapped = load_fn(
            {"train_state": state},
            candidate,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=allow_partial,
        )
        logger.info("Loaded legacy wrapped grug checkpoint format from %s", candidate)
        return wrapped["train_state"]  # type: ignore[index]
