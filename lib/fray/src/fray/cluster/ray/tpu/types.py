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

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Type definitions for TPU orchestration results and actor pool management."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from ray.actor import ActorHandle


# My kingdom for ADTs
@dataclass
class _TpuRunResult:
    """Internal class to hold the result of a TPU job."""

    pass


@dataclass
class TpuSuccess(_TpuRunResult):
    result: object


@dataclass
class TpuPreempted(_TpuRunResult):
    error: Exception


@dataclass
class TpuFailed(_TpuRunResult):
    error: Exception


@dataclass
class TpuRunError(_TpuRunResult):
    error: Exception


@dataclass
class TpuCancelled(_TpuRunResult):
    error: Exception


@dataclass
class MultisliceInfo:
    """
    Information about a TPU multislice.

    This is used to pass information about a TPU multislice to the worker tasks.
    """

    coordinator_ip: str
    slice_id: int
    num_slices: int
    port: int = 8081


@dataclass
class SliceInfo:
    """
    Information about a TPU slice.

    This is used to pass information about a TPU slice to the worker tasks.
    """

    slice_name: str
    num_vms: int
    ip_address: str
    num_tpus_per_vm: int


@dataclass(frozen=True)
class TPUHostInfo:
    slice_name: str
    worker_index: int
    node_id: str
    num_tpus: int


ActorInfoT = TypeVar("ActorInfoT")


@dataclass(frozen=True)
class ActorPoolMember(Generic[ActorInfoT]):
    actor: ActorHandle
    actor_info: ActorInfoT


SliceResource = ActorPoolMember[SliceInfo]
