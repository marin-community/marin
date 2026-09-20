# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Resume the arXiv sweep with 4 GiB preparation workers while the central1 CPU pool is full.

Only the four data-preparation CPU tasks (raw pool, parent, matched subset, S2ORC cache) change
their host-memory request; their recipes, identities, the plan and every TPU request are unchanged.
"""

import logging
from dataclasses import replace

from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient, IrisJobHandle
from fray.types import CpuConfig, JobRequest
from iris.client.client import IrisClient, get_iris_ctx

from experiments.domain_phase_mix import launch_tpp10_arxiv_sweep as sweep
from experiments.domain_phase_mix import prepare_tpp10_arxiv_sweep as preparation

logger = logging.getLogger(__name__)
PREPARATION_TASKS = ("prepare_raw-", "prepare_parent-", "prepare_subset-")
PREPARATION_RAM = "4g"


def smaller_preparation_request(request: JobRequest) -> JobRequest:
    """Lower the RAM of a frozen preparation task; refuse any other CPU task."""
    if not isinstance(request.resources.device, CpuConfig):
        return request
    if not request.name.startswith(PREPARATION_TASKS) or request.resources != preparation.CPU:
        raise ValueError(f"arXiv sweep recovery expects only the frozen preparation CPU tasks, got {request.name}")
    logger.info("Dispatching %s with 2 CPU / %s; recipe unchanged", request.name, PREPARATION_RAM)
    return replace(request, resources=replace(request.resources, ram=PREPARATION_RAM))


class PreparationClient(FrayIrisClient):
    """Apply the smaller request after artifact identities are resolved."""

    def __init__(self, client: IrisClient):
        self._iris = client

    def submit(self, request: JobRequest, adopt_existing: bool = True) -> IrisJobHandle:
        return super().submit(smaller_preparation_request(request), adopt_existing)


def main() -> None:
    context = get_iris_ctx()
    if context is None or context.client is None:
        raise ValueError("arXiv sweep recovery requires an Iris coordinator")
    with set_current_client(PreparationClient(context.client)):
        sweep.main()


if __name__ == "__main__":
    main()
