# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
import os
import urllib.parse
from collections.abc import Callable, Sequence
from typing import ClassVar, Protocol, TypeVar, cast

import fsspec
import jax
from fsspec import AbstractFileSystem
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.utils.jax_utils import barrier_sync_named

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")
RESTORE_COMPLETE_BARRIER = "grug_checkpoint_restore_complete"
# Older grug runs saved {"train_state": state}, so their leaves carry this prefix.
LEGACY_STATE_KEY = "train_state"
# Field name of the optional fp32 pinned-host master in a grug train state's checkpoint layout.
MASTER_PARAMS_KEY = "master_params"
# The barrier runs one clock, started by the first rank to arrive, so this bounds the spread
# between arrivals rather than the length of a restore. A gang whose object-store caches are only
# partly warm spreads widest, since re-reading a checkpoint is an order of magnitude faster than
# reading it cold, and killing a stalled gang is what leaves a fleet in that state. Expiring
# aborts the attempt for the scheduler to retry, rather than holding every rank behind one that
# never arrives.
RESTORE_BARRIER_TIMEOUT = 40 * 60


class _GrugState(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]

    @property
    def step(self) -> jax.Array: ...

    @property
    def params(self) -> object: ...

    @property
    def ema_params(self) -> object | None: ...


GrugStateT = TypeVar("GrugStateT", bound=_GrugState)


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
    template_for_candidate: Callable[[str], StateT] | None = None,
    _load_fn: Callable[..., StateT] = load_checkpoint,
) -> StateT:
    """Restore the newest loadable checkpoint under the search paths, else return ``state``.

    ``template_for_candidate`` maps a candidate path to the template it is read with. A template
    is the pytree whose leaves name what restore loads. Pass the hook when the state layout
    differs between checkpoint generations, so each candidate is read with the template matching
    its layout; ``None`` reads every candidate with ``state``. Raising ``FileNotFoundError`` from
    the hook skips to the next-older candidate; any other exception aborts the restore, so a hook
    can refuse a checkpoint outright rather than quietly fall past it.
    """
    if not checkpoint_search_paths:
        if load_checkpoint_setting:
            raise FileNotFoundError("load_checkpoint=True but no checkpoint search paths are configured.")
        return state

    if load_checkpoint_setting is False:
        return state

    candidates = _checkpoint_candidates(checkpoint_search_paths)
    # A bare search root is always a candidate, so this is what separates "nothing has been
    # written yet" from "checkpoints exist here".
    written = [candidate for candidate in candidates if candidate not in checkpoint_search_paths]
    last_error: FileNotFoundError | None = None

    for candidate in candidates:
        try:
            loaded = _load_candidate_state(
                state=state if template_for_candidate is None else template_for_candidate(candidate),
                candidate=candidate,
                mesh=mesh,
                allow_partial=allow_partial,
                load_fn=_load_fn,
            )
            barrier_sync_named(RESTORE_COMPLETE_BARRIER, timeout=RESTORE_BARRIER_TIMEOUT)
            if candidate not in checkpoint_search_paths:
                logger.info("Loaded checkpoint from %s while searching %s", candidate, checkpoint_search_paths)
            return loaded
        except FileNotFoundError as exc:
            last_error = exc
            logger.warning(
                "Checkpoint candidate %s could not be loaded (%s). Trying an older checkpoint.", candidate, exc
            )

    search_path_summary = ", ".join(checkpoint_search_paths)
    if load_checkpoint_setting is True:
        attempted = ", ".join(candidates)
        if last_error is None:
            raise FileNotFoundError(f"Could not find checkpoint under any of: {search_path_summary}")
        raise FileNotFoundError(
            f"Could not load a checkpoint from search paths {search_path_summary}. Attempted: {attempted}"
        ) from last_error

    if written:
        # An optional resume that finds checkpoints and reads none of them is a failed resume, not
        # a first launch. Restarting at step 0 would overwrite them and still report a plausible MFU.
        raise FileNotFoundError(
            f"{len(written)} checkpoint(s) exist under {search_path_summary} but none could be loaded: "
            f"{', '.join(written)}"
        ) from last_error

    logger.info("No checkpoint under %s. Starting from scratch.", checkpoint_search_paths)
    return state


def init_weights_only_from_checkpoint(
    state: GrugStateT,
    checkpoint_path: str,
    *,
    mesh: jax.sharding.Mesh | None,
    allow_partial: bool,
    additional_weight_fields: Sequence[str] = (),
) -> GrugStateT:
    """Initialize a fresh Grug state from external weights."""
    if int(state.step) != 0:
        return state

    concrete_checkpoint_path = latest_checkpoint_path(checkpoint_path)
    weight_fields = ("params", *additional_weight_fields)
    exemplar = {field_name: getattr(state, field_name) for field_name in weight_fields}
    logger.info("Initializing model weights from %s", concrete_checkpoint_path)
    loaded = cast(
        dict[str, object],
        load_checkpoint(
            exemplar,
            concrete_checkpoint_path,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=allow_partial,
        ),
    )
    updates = {field_name: loaded[field_name] for field_name in weight_fields}
    if state.ema_params is not None:
        updates["ema_params"] = loaded["params"]
    return cast(GrugStateT, dataclasses.replace(state, **updates))


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
        wrapped = load_fn(
            {LEGACY_STATE_KEY: state},
            candidate,
            axis_mapping=None,
            mesh=mesh,
            allow_partial=allow_partial,
        )
        logger.info("Loaded legacy wrapped grug checkpoint format from %s", candidate)
        return wrapped[LEGACY_STATE_KEY]  # type: ignore[index]
