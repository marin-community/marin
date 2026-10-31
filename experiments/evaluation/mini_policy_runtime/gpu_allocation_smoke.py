# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check privileged-task CUDA and Ray visibility against NVIDIA's allocation."""

import json

import ray
import torch
from marinskyrl.gpu_allocation import pin_task_cuda_devices


def main() -> None:
    allocation = pin_task_cuda_devices()
    assert allocation is not None
    expected = {device.removeprefix("GPU-") for device in allocation.split(",")}
    assert len(expected) == 2, "This diagnostic requires exactly two allocated GPUs"
    assert torch.cuda.device_count() == 2
    ray.init(include_dashboard=False)
    try:
        assert ray.cluster_resources()["GPU"] == 2
        # Keep both actors alive together so Ray must use both allocated devices.
        workers = [DeviceActor.remote() for _ in range(2)]
        observed = set(ray.get([worker.uuid.remote() for worker in workers]))
        assert observed == expected, (observed, expected)
        print(
            json.dumps(
                {"allocated_gpu_count": 2, "torch_gpu_count": 2, "ray_gpu_count": 2, "assigned_uuids_match": True}
            )
        )
    finally:
        ray.shutdown()


@ray.remote(num_gpus=1)
class DeviceActor:
    def uuid(self) -> str:
        assert torch.cuda.device_count() == 1
        return str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")


if __name__ == "__main__":
    main()
