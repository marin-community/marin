# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import equinox as eqx
import fsspec
import jax
from fsspec import AbstractFileSystem
from haliax.nn import ArrayStacked
from levanter.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")

_LEGACY_EXPERT_PATH = ("stacked_blocks", "stacked", "mlp", "expert_mlp")


def load_june_checkpoint(
    tree: StateT,
    checkpoint_path: str,
    *,
    subpath: str | None = None,
    axis_mapping: object | None = None,
    mesh: jax.sharding.Mesh | None = None,
    allow_partial: bool = False,
    _load_fn: Callable[..., Any] = load_checkpoint,
) -> StateT:
    """Load June weights, adapting legacy array-stacked expert paths when needed.

    Pre-refactor untied checkpoints stored routed experts under each stacked block at
    ``params.stacked_blocks.stacked.mlp.expert_mlp``. The explicit-bank model stores the
    same stacked arrays at ``params.expert_banks.stacked``. This adapter loads the legacy
    expert subtree separately, loads every other requested leaf through the ordinary
    checkpoint loader, then installs the experts into the new single-owned bank container.

    The adapter applies only to an untied array-stacked model and does not migrate optimizer
    state. A tied target has fewer banks than the legacy per-layer checkpoint, so converting it
    requires an explicit merge policy.
    """
    params_location = _find_explicit_untied_params(tree)
    if params_location is None:
        return cast(
            StateT,
            _load_fn(
                tree,
                checkpoint_path,
                subpath=subpath,
                axis_mapping=axis_mapping,
                mesh=mesh,
                allow_partial=allow_partial,
            ),
        )

    params_path, params = params_location
    if _contains_optimizer_state(tree):
        raise ValueError(
            "legacy June optimizer-state migration is not supported; load params and pending_qb_betas "
            "with a weights-only exemplar, then initialize a fresh optimizer"
        )
    expert_banks = cast(ArrayStacked[Any], params.expert_banks)
    legacy_expert_path = params_path + _LEGACY_EXPERT_PATH
    legacy_expert_exemplar = _nested_mapping(legacy_expert_path, expert_banks.stacked)

    try:
        loaded_legacy_tree = _load_fn(
            legacy_expert_exemplar,
            checkpoint_path,
            subpath=subpath,
            axis_mapping=axis_mapping,
            mesh=mesh,
            allow_partial=False,
        )
    except FileNotFoundError:
        return cast(
            StateT,
            _load_fn(
                tree,
                checkpoint_path,
                subpath=subpath,
                axis_mapping=axis_mapping,
                mesh=mesh,
                allow_partial=allow_partial,
            ),
        )

    loaded_legacy_experts = _value_at_path(loaded_legacy_tree, legacy_expert_path)
    params_without_experts = _replace_field(params, "expert_banks", None)
    nonexpert_exemplar = _replace_at_path(tree, params_path, params_without_experts)
    loaded_nonexperts = _load_fn(
        nonexpert_exemplar,
        checkpoint_path,
        subpath=subpath,
        axis_mapping=axis_mapping,
        mesh=mesh,
        allow_partial=allow_partial,
    )
    loaded_params = _value_at_path(loaded_nonexperts, params_path)
    loaded_expert_banks = eqx.tree_at(lambda banks: banks.stacked, expert_banks, loaded_legacy_experts)
    loaded_params = _replace_field(loaded_params, "expert_banks", loaded_expert_banks)
    logger.info("Adapted legacy June array-stacked expert paths from %s", checkpoint_path)
    return cast(StateT, _replace_at_path(loaded_nonexperts, params_path, loaded_params))


def _find_explicit_untied_params(tree: Any, path: tuple[str, ...] = ()) -> tuple[tuple[str, ...], Any] | None:
    if _is_explicit_untied_params(tree):
        return path, tree
    if isinstance(tree, dict):
        for key in ("params", "train_state"):
            if key in tree:
                found = _find_explicit_untied_params(tree[key], (*path, key))
                if found is not None:
                    return found
        return None
    if hasattr(tree, "params"):
        return _find_explicit_untied_params(tree.params, (*path, "params"))
    return None


def _contains_optimizer_state(tree: Any) -> bool:
    if isinstance(tree, dict):
        if "opt_state" in tree:
            return True
        return any(_contains_optimizer_state(value) for key, value in tree.items() if key == "train_state")
    return hasattr(tree, "opt_state")


def _is_explicit_untied_params(value: Any) -> bool:
    expert_banks = getattr(value, "expert_banks", None)
    stacked_blocks = getattr(value, "stacked_blocks", None)
    config = getattr(value, "config", None)
    if not isinstance(expert_banks, ArrayStacked) or stacked_blocks is None or config is None:
        return False
    return config.resolved_expert_bank_for_layer == tuple(range(config.num_layers))


def _nested_mapping(path: tuple[str, ...], value: Any) -> Any:
    nested = value
    for key in reversed(path):
        nested = {key: nested}
    return nested


def _value_at_path(tree: Any, path: tuple[str, ...]) -> Any:
    value = tree
    for key in path:
        value = value[key] if isinstance(value, dict) else getattr(value, key)
    return value


def _replace_at_path(tree: Any, path: tuple[str, ...], value: Any) -> Any:
    if not path:
        return value
    key, *remaining = path
    child = tree[key] if isinstance(tree, dict) else getattr(tree, key)
    replaced_child = _replace_at_path(child, tuple(remaining), value)
    return _replace_field(tree, key, replaced_child)


def _replace_field(tree: Any, key: str, value: Any) -> Any:
    if isinstance(tree, dict):
        updated = dict(tree)
        updated[key] = value
        return updated
    return eqx.tree_at(
        lambda current: getattr(current, key),
        tree,
        value,
        is_leaf=lambda leaf: leaf is None,
    )


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
    initialize_from: str | None = None,
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

    # Nothing to auto-resume from. Fall back to initialize_from (first-time
    # init only): loads model + optimizer + step from an external source
    # checkpoint. Subsequent restarts will find this run's own checkpoints
    # in checkpoint_search_paths and take the auto-resume path above,
    # bypassing initialize_from.
    if initialize_from is not None:
        logger.info("No checkpoint found under %s; initializing from %s", checkpoint_search_paths, initialize_from)
        try:
            return _load_candidate_state(
                state=state,
                candidate=initialize_from,
                mesh=mesh,
                allow_partial=allow_partial,
                load_fn=_load_fn,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"initialize_from={initialize_from!r} but the checkpoint could not be loaded."
            ) from exc

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
