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

"""Platform-agnostic shared types for VM lifecycle.

These types define the contract for VM lifecycle helpers without leaking
provider-specific details into controller code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from collections.abc import Mapping


class VmState(Enum):
    """Platform-agnostic VM lifecycle states."""

    BOOTING = "booting"
    RUNNING = "running"
    TERMINATED = "terminated"
    FAILED = "failed"


@dataclass(frozen=True)
class VmInfo:
    """Minimal VM identity + connectivity info."""

    vm_id: str
    address: str
    zone: str | None
    labels: Mapping[str, str]
    state: VmState
    created_at_ms: int


@dataclass(frozen=True)
class ContainerSpec:
    """Container launch specification."""

    image: str
    entrypoint: list[str]
    env: Mapping[str, str]
    ports: Mapping[str, int]
    health_port: int | None = None


@dataclass(frozen=True)
class VmBootstrapSpec:
    """Bootstrap specification for a VM role."""

    role: Literal["controller", "worker"]
    container: ContainerSpec
    labels: Mapping[str, str]
    bootstrap_script: str | None = None
    provider_overrides: Mapping[str, object] = field(default_factory=dict)
