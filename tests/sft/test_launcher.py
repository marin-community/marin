# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.types import ANY_REGION, GpuConfig

from experiments.sft.launcher import resources_from_accelerator


def test_gpu_accelerator_sizes_host_resources_with_device_count() -> None:
    resources = resources_from_accelerator("4xGB200")

    assert resources.device == GpuConfig(variant="GB200", count=4)
    assert resources.cpu == 32
    assert resources.ram == "384g"
    assert resources.disk == "192g"
    assert resources.regions == [ANY_REGION]
