# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout record: the deploy/rollback state a controller restart leaves in remote state.

A ``controller restart`` writes this record next to the controller checkpoints in
the cluster's remote state dir. It carries the image being deployed, the image to
roll back to, and the pre-deploy checkpoint to restore on rollback, plus a phase
that drives the rollback state machine:

- ``PENDING``: a forward deploy is in flight, not yet health-verified.
- ``COMMITTED``: the forward deploy passed its post-restart health check.
- ``ROLLBACK_REQUESTED``: the next controller boot restores ``rollback_checkpoint``
  over its local DB, then rewrites the record ``ROLLED_BACK``.
- ``ROLLED_BACK``: the controller has restored the pre-deploy checkpoint.

The controller consumes ``ROLLBACK_REQUESTED`` exactly once and self-clears to
``ROLLED_BACK``, so a later crash or reboot reuses the restored DB instead of
rewinding to the checkpoint again.
"""

import logging
from enum import StrEnum

import fsspec.core
from pydantic import BaseModel, ConfigDict
from rigging.filesystem import prefix_join

logger = logging.getLogger(__name__)

ROLLOUT_RECORD_FILENAME = "rollout-record.json"


class RolloutPhase(StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLBACK_REQUESTED = "rollback_requested"
    ROLLED_BACK = "rolled_back"


class RolloutRecord(BaseModel):
    """A controller deploy and how to roll it back.

    - ``phase``: where this deploy sits in the rollback state machine.
    - ``image``: the controller image this deploy runs.
    - ``previous_image``: the image to roll back to, or None when there is nothing
      to revert to (first deploy, or a consumed rollback).
    - ``rollback_checkpoint``: the pre-deploy checkpoint
      (``gs://…/controller-state/<epoch_ms>``) to restore on rollback, taken at the
      previous image's schema. None when nothing needs restoring.
    - ``updated_at_ms``: wall-clock time this record was written.
    """

    model_config = ConfigDict(extra="ignore")

    phase: RolloutPhase
    image: str
    previous_image: str | None = None
    rollback_checkpoint: str | None = None
    updated_at_ms: int = 0


def _record_url(remote_state_dir: str) -> str:
    return prefix_join(remote_state_dir, ROLLOUT_RECORD_FILENAME)


def read_rollout_record(remote_state_dir: str) -> RolloutRecord | None:
    """Return the rollout record from remote state, or None if absent/unreadable."""
    url = _record_url(remote_state_dir)
    try:
        with fsspec.core.open(url, "r") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Ignoring unreadable rollout record %s: %s", url, exc)
        return None
    try:
        return RolloutRecord.model_validate_json(data)
    except ValueError as exc:
        logger.warning("Ignoring malformed rollout record %s: %s", url, exc)
        return None


def write_rollout_record(remote_state_dir: str, record: RolloutRecord) -> None:
    """Write the rollout record to remote state, overwriting any existing one."""
    url = _record_url(remote_state_dir)
    with fsspec.core.open(url, "w") as f:
        f.write(record.model_dump_json(indent=2))
