# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for TPU orchestration."""

import logging
import os

import requests
from ray.exceptions import (
    ActorDiedError,
    ActorUnavailableError,
    NodeDiedError,
    RayError,
    RaySystemError,
    RayTaskError,
    WorkerCrashedError,
)

from fray.cluster.ray.tpu.types import MultisliceInfo, SliceInfo, TpuPreempted, TpuRunError

logger = logging.getLogger(__name__)

# Timeouts (in seconds)
HEALTH_CHECK_TIMEOUT = 60
TEARDOWN_ACTOR_TIMEOUT = 5 * 60
CANCEL_TASK_TIMEOUT = 4 * 60
TERMINATE_ACTOR_TIMEOUT = 5 * 60
START_ACTOR_TIMEOUT = 7 * 24 * 60 * 60  # 1 week

# Intervals (in seconds)
SCALE_UP_MULTISLICE_CHECK_INTERVAL = 3 * 60 * 60  # 3 hours
SCALE_UP_MULTISLICE_INTERVAL = 12 * 60 * 60  # 12 hours


def _multislice_info_from_head(head: SliceInfo, slice_id: int, num_slices: int) -> MultisliceInfo:
    """
    Create a MultisliceInfo object from the head slice info and the slice ID and number of slices.
    """
    return MultisliceInfo(
        coordinator_ip=head.ip_address,
        slice_id=slice_id,
        num_slices=num_slices,
        port=8081,  # default port for megascale
    )


def _multislice_info_to_env_vars(multislice: MultisliceInfo) -> dict[str, str]:
    if multislice is not None:
        mxla_env = {
            "MEGASCALE_COORDINATOR_ADDRESS": f"{multislice.coordinator_ip}:{multislice.port}",
            "MEGASCALE_NUM_SLICES": str(multislice.num_slices),
            "MEGASCALE_PORT": f"{multislice.port}",
            "MEGASCALE_SLICE_ID": str(multislice.slice_id),
        }
    else:
        mxla_env = {}
    return mxla_env


def get_current_tpu_is_preempted() -> bool:
    """
    Returns True if the current TPU is being preempted.

    This function queries the GCE metadata service to determine if the current
    TPU instance is in the process of being preempted.
    """
    try:
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
            headers={"Metadata-Flavor": "Google"},
            timeout=5,
        )
        response.raise_for_status()
        return response.text.lower() == "true"
    except Exception:
        return False


def _handle_ray_error(e: RayError):
    """
    Handle a Ray error that occurred on a TPU pod. Tries to determine if the error was due to a
    node failure or preemption or just an application error.
    """
    # treat node failures as preemptions
    if isinstance(e, NodeDiedError):
        logger.exception("Node died", exc_info=e)
        return TpuPreempted(e)
    elif isinstance(e, ActorUnavailableError | ActorDiedError):
        logger.exception("Actor died", exc_info=e)
        return TpuPreempted(e)
    elif isinstance(e, WorkerCrashedError):
        logger.exception("Worker crashed", exc_info=e)
        return TpuPreempted(e)
    elif isinstance(e, RaySystemError):
        logger.exception("System error", exc_info=e)
        return TpuRunError(e)
    elif isinstance(e, RayTaskError):
        # node preemptions don't always show up as one of the above errors and can just be a RayTaskError. We have
        # to try to sniff out the TPU's status.
        if get_current_tpu_is_preempted():
            logger.exception("Preempted", exc_info=e)
            return TpuPreempted(e)

        logger.exception(f"Task error {e}", exc_info=e)
        if isinstance(e.cause, TimeoutError) or "timed out" in str(e):
            logger.exception("Timeout error. Assuming preempted", exc_info=e)
            return TpuPreempted(e)
        return TpuRunError(e)

    else:
        logger.exception("Unknown error", exc_info=e)
        return TpuRunError(e)


def _hacky_remove_tpu_lockfile():
    """
    This is a hack to remove the lockfile that TPU pods create on the host filesystem.

    libtpu only allows one process to access the TPU at a time, and it uses a lockfile to enforce this.
    Ordinarily a lockfile would be removed when the process exits, but in the case of Ray, the process is
    a long-running daemon that doesn't typically exit until the node is shut down. This means that the lockfile
    persists across Ray tasks. This doesn't apply to our docker-based workloads, but it does apply to other
    tasks that use JAX directly.
    """
    if os.path.exists("/tmp/libtpu_lockfile"):
        try:
            os.unlink("/tmp/libtpu_lockfile")
        except FileNotFoundError:
            pass
        except PermissionError:
            logger.warning("Failed to remove lockfile")
            try:
                os.system("sudo rm /tmp/libtpu_lockfile")
            except Exception:
                pass
