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

"""Ray TPU orchestration for gang scheduling and multislice coordination."""

from fray.cluster.ray.tpu.config import (
    TPU_CONFIGS,
    TPUConfig,
    get_tpu_config,
)
from fray.cluster.ray.tpu.execution import (
    run_on_pod,
    run_on_pod_multislice,
    run_on_pod_multislice_resumable,
    run_on_pod_ray,
    run_on_pod_resumable,
)
from fray.cluster.ray.tpu.orchestration import (
    ResourcePoolManager,
    SliceActor,
    SlicePoolManager,
    TPUHostActor,
)
from fray.cluster.ray.tpu.types import (
    MultisliceInfo,
    SliceInfo,
    TPUHostInfo,
    TpuCancelled,
    TpuFailed,
    TpuPreempted,
    TpuRunError,
    TpuSuccess,
)
from fray.cluster.ray.tpu.utils import (
    HEALTH_CHECK_TIMEOUT,
    START_ACTOR_TIMEOUT,
    get_current_tpu_is_preempted,
)

__all__ = [
    "HEALTH_CHECK_TIMEOUT",
    "START_ACTOR_TIMEOUT",
    "TPU_CONFIGS",
    "MultisliceInfo",
    "ResourcePoolManager",
    "SliceActor",
    "SliceInfo",
    "SlicePoolManager",
    "TPUConfig",
    "TPUHostActor",
    "TPUHostInfo",
    "TpuCancelled",
    "TpuFailed",
    "TpuPreempted",
    "TpuRunError",
    "TpuSuccess",
    "get_current_tpu_is_preempted",
    "get_tpu_config",
    "run_on_pod",
    "run_on_pod_multislice",
    "run_on_pod_multislice_resumable",
    "run_on_pod_ray",
    "run_on_pod_resumable",
]
