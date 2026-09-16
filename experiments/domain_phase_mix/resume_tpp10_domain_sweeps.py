# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Resume the frozen survey with a smaller CPU subset-preparation request."""

import logging
from dataclasses import replace

from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient, IrisJobHandle
from fray.types import CpuConfig, JobRequest
from iris.client.client import IrisClient, get_iris_ctx

from experiments.domain_phase_mix import launch_tpp10_domain_sweeps as survey
from experiments.domain_phase_mix import prepare_tpp10_domain_sweeps as preparation

logger = logging.getLogger(__name__)


class SubsetPreparationClient(FrayIrisClient):
    """Change only subset-worker RAM, after artifact identities are resolved."""

    def __init__(self, client: IrisClient):
        self._iris = client

    def submit(self, request: JobRequest, adopt_existing: bool = True) -> IrisJobHandle:
        if isinstance(request.resources.device, CpuConfig):
            if not request.name.startswith("prepare_subset-") or request.resources != preparation.CPU:
                raise ValueError("Survey recovery expects only the unfinished frozen subset CPU task")
            logger.info("Dispatching %s with 2 CPU / 4 GiB; subset recipe unchanged", request.name)
            request = replace(request, resources=replace(request.resources, ram="4g"))
        return super().submit(request, adopt_existing)


def main() -> None:
    context = get_iris_ctx()
    if context is None or context.client is None:
        raise ValueError("Survey recovery requires an Iris coordinator")
    with set_current_client(SubsetPreparationClient(context.client)):
        survey.main()


if __name__ == "__main__":
    main()
