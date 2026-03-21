# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from collections.abc import Callable
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


def _checkpoint_candidates(checkpoint_path: str) -> list[str]:
    fs, plain_path = _get_fs_and_plain_path(checkpoint_path)
    base_path_protocol = urllib.parse.urlparse(checkpoint_path).scheme

    def maybe_unstrip_protocol(path: str) -> str:
        if base_path_protocol != "" and urllib.parse.urlparse(path).scheme == "":
            return f"{base_path_protocol}://{path}"
        return path

    checkpoint_dirs = [maybe_unstrip_protocol(d) for d in fs.glob(os.path.join(plain_path, "*")) if fs.isdir(d)]
    checkpoint_dirs.append(checkpoint_path)

    candidates: list[tuple[int, str, str]] = []
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
        candidates.append((step_num, timestamp_key, candidate))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered_candidates = [candidate for _, _, candidate in candidates]
    if checkpoint_path not in ordered_candidates:
        ordered_candidates.append(checkpoint_path)

    return ordered_candidates


def restore_grug_state_from_checkpoint(
    state: StateT,
    *,
    checkpoint_path: str | None,
    load_checkpoint_setting: bool | None,
    mesh: jax.sharding.Mesh | None,
    allow_partial: bool,
    _load_fn: Callable[..., StateT] = load_checkpoint,
) -> StateT:
    if checkpoint_path is None:
        if load_checkpoint_setting:
            raise FileNotFoundError("load_checkpoint=True but no checkpoint path is configured.")
        return state

    if load_checkpoint_setting is False:
        return state

    candidates = _checkpoint_candidates(checkpoint_path)
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
            if candidate != checkpoint_path:
                logger.info("Loaded checkpoint %s from %s", checkpoint_path, candidate)
            return loaded
        except FileNotFoundError as exc:
            last_error = exc
            logger.warning(
                "Checkpoint candidate %s could not be loaded (%s). Trying an older checkpoint.", candidate, exc
            )

    if load_checkpoint_setting is True:
        attempted = ", ".join(candidates)
        if last_error is None:
            raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
        raise FileNotFoundError(
            f"Could not load a checkpoint from {checkpoint_path}. Attempted: {attempted}"
        ) from last_error

    logger.info(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
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
            discover_latest=False,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=allow_partial,
        )
    except FileNotFoundError:
        # Backward compatibility: older grug runs saved {"train_state": state}.
        wrapped = load_fn(
            {"train_state": state},
            candidate,
            discover_latest=False,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=allow_partial,
        )
        logger.info("Loaded legacy wrapped grug checkpoint format from %s", candidate)
        return wrapped["train_state"]  # type: ignore[index]
