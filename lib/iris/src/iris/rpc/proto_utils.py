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

"""Protobuf enum utilities."""

from iris.rpc import vm_pb2, cluster_pb2

_VM_STATE = vm_pb2.DESCRIPTOR.enum_types_by_name["VmState"]
_SCALING_ACTION = vm_pb2.DESCRIPTOR.enum_types_by_name["ScalingAction"]
_TASK_STATE = cluster_pb2.DESCRIPTOR.enum_types_by_name["TaskState"]


def vm_state_name(state: int) -> str:
    """Return enum name like 'VM_STATE_READY'."""
    try:
        return _VM_STATE.values_by_number[state].name
    except KeyError:
        return f"UNKNOWN({state})"


def scaling_action_name(action: int) -> str:
    """Return enum name like 'SCALING_ACTION_SCALE_UP'."""
    try:
        return _SCALING_ACTION.values_by_number[action].name
    except KeyError:
        return f"UNKNOWN({action})"


def task_state_name(state: int) -> str:
    """Return enum name like 'TASK_STATE_RUNNING'."""
    try:
        return _TASK_STATE.values_by_number[state].name
    except KeyError:
        return f"UNKNOWN({state})"
