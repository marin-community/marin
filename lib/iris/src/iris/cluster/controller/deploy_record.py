# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy record: the rollback pointer a controller restart leaves in remote state.

Each forward ``controller restart`` records the image it deployed, the image it
replaced, and the pre-deploy checkpoint. ``controller restart --rollback`` reads
this back to revert to the previous image and its pre-deploy state without the
operator tracking coordinates by hand. The record sits next to the controller
checkpoints in the cluster's remote state dir.
"""

import json
import logging
from dataclasses import asdict, dataclass

import fsspec.core

logger = logging.getLogger(__name__)

DEPLOY_RECORD_FILENAME = "deploy-record.json"


@dataclass(frozen=True)
class DeployRecord:
    """What a controller restart deployed, and how to roll it back.

    - ``current_image``: the controller image running after this restart.
    - ``previous_image``: the image it replaced — the rollback target. None when
      there is nothing to roll back to (first deploy, or a consumed rollback).
    - ``pre_deploy_checkpoint``: the checkpoint taken before this deploy, at the
      previous image's schema; restored alongside ``previous_image`` on rollback.
    - ``recorded_at_ms``: wall-clock time the record was written.
    """

    current_image: str
    previous_image: str | None
    pre_deploy_checkpoint: str | None
    recorded_at_ms: int


def _record_url(remote_state_dir: str) -> str:
    return f"{remote_state_dir.rstrip('/')}/{DEPLOY_RECORD_FILENAME}"


def read_deploy_record(remote_state_dir: str) -> DeployRecord | None:
    """Return the deploy record from remote state, or None if absent/unreadable."""
    url = _record_url(remote_state_dir)
    try:
        with fsspec.core.open(url, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable deploy record %s: %s", url, exc)
        return None
    return DeployRecord(
        current_image=data["current_image"],
        previous_image=data.get("previous_image"),
        pre_deploy_checkpoint=data.get("pre_deploy_checkpoint"),
        recorded_at_ms=int(data.get("recorded_at_ms", 0)),
    )


def write_deploy_record(remote_state_dir: str, record: DeployRecord) -> None:
    """Write the deploy record to remote state, overwriting any existing one."""
    url = _record_url(remote_state_dir)
    with fsspec.core.open(url, "w") as f:
        json.dump(asdict(record), f, indent=2)
