# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Keep the refinement coordinator interactive while its TPU jobs use batch."""

from dataclasses import replace

from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient, IrisJobHandle
from fray.types import JobRequest, TpuConfig
from iris.client.client import IrisClient, get_iris_ctx
from iris.rpc import job_pb2

from experiments.domain_phase_mix import launch_starcoder_tpp10_refinement as refinement


class BatchTrainingClient(FrayIrisClient):
    """Override scheduling priority without changing frozen training artifacts."""

    def __init__(self, client: IrisClient):
        self._iris = client

    def submit(self, request: JobRequest, adopt_existing: bool = True) -> IrisJobHandle:
        if not isinstance(request.resources.device, TpuConfig):
            raise ValueError("Refinement recovery should dispatch only frozen TPU training jobs")
        return super().submit(replace(request, priority=job_pb2.PRIORITY_BAND_BATCH), adopt_existing)


def main() -> None:
    context = get_iris_ctx()
    if context is None or context.client is None:
        raise ValueError("Batch refinement recovery requires an Iris coordinator")
    with set_current_client(BatchTrainingClient(context.client)):
        refinement.main()


if __name__ == "__main__":
    main()
