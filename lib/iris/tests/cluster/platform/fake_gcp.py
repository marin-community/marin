# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Fake gcloud CLI for testing GcpPlatform without patching subprocess.run per test.

Intercepts subprocess.run calls that invoke gcloud, simulating TPU and GCE instance
lifecycle with in-memory state. Supports failure injection for error-path testing.

Usage:
    fake = FakeGcloud()
    fake.set_failure("tpu_create", "RESOURCE_EXHAUSTED: no capacity")

    # Use as a pytest fixture via fake_gcloud, which patches subprocess.run automatically.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest


@dataclass
class FakeResult:
    """Drop-in for subprocess.CompletedProcess returned by FakeGcloud."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _parse_flag(cmd: list[str], flag: str) -> str | None:
    """Extract value from --flag=value style args in a command list.

    Returns None if the flag is not present.
    """
    prefix = f"--{flag}="
    for arg in cmd:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _parse_labels_string(s: str) -> dict[str, str]:
    """Parse 'k1=v1,k2=v2' into {'k1': 'v1', 'k2': 'v2'}."""
    if not s:
        return {}
    result: dict[str, str] = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def _parse_filter_labels(filter_str: str) -> dict[str, str]:
    """Parse gcloud filter expressions like 'labels.k=v AND labels.k2=v2' into a dict."""
    if not filter_str:
        return {}
    labels: dict[str, str] = {}
    for match in re.finditer(r"labels\.([^=\s]+)=([^\s]+)", filter_str):
        labels[match.group(1)] = match.group(2)
    return labels


@dataclass
class FakeGcloud:
    """In-memory gcloud CLI fake that intercepts subprocess.run calls.

    Maintains TPU and VM state dictionaries keyed by (name, zone). Each gcloud
    subcommand is dispatched to a handler method. Unrecognized commands raise
    ValueError so test bugs surface immediately.
    """

    _tpus: dict[tuple[str, str], dict] = field(default_factory=dict)
    _vms: dict[tuple[str, str], dict] = field(default_factory=dict)
    _failures: dict[str, tuple[str, int]] = field(default_factory=dict)

    def set_failure(self, operation: str, error: str, code: int = 1) -> None:
        """Make a specific operation type fail on the next call.

        Args:
            operation: One of "tpu_create", "tpu_list", "tpu_describe", "tpu_delete",
                       "vm_create", "vm_list", "vm_describe", "vm_delete".
            error: The stderr message to return.
            code: The returncode to return (default 1).
        """
        self._failures[operation] = (error, code)

    def clear_failure(self) -> None:
        """Remove all injected failures."""
        self._failures.clear()

    def _check_failure(self, operation: str) -> FakeResult | None:
        if operation in self._failures:
            error, code = self._failures.pop(operation)
            return FakeResult(returncode=code, stderr=error)
        return None

    def __call__(self, cmd: list[str], **kwargs) -> FakeResult:
        """Drop-in replacement for subprocess.run. Dispatches by gcloud subcommand."""
        if not cmd or cmd[0] != "gcloud":
            raise ValueError(f"FakeGcloud: unrecognized command: {cmd}")

        # Extract subcommand tokens, skipping flags and their arguments.
        # Handles both --flag=value and --flag value forms.
        tokens: list[str] = []
        skip_next = False
        for arg in cmd[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg.startswith("--"):
                # If it's --flag (no =), the next arg is the value
                if "=" not in arg:
                    skip_next = True
                continue
            tokens.append(arg)

        if _matches(tokens, ["compute", "tpus", "tpu-vm", "create", None]):
            return self._tpu_create(cmd, tokens[4])
        if _matches(tokens, ["compute", "tpus", "tpu-vm", "list"]):
            return self._tpu_list(cmd)
        if _matches(tokens, ["compute", "tpus", "tpu-vm", "describe", None]):
            return self._tpu_describe(cmd, tokens[4])
        if _matches(tokens, ["compute", "tpus", "tpu-vm", "delete", None]):
            return self._tpu_delete(cmd, tokens[4])
        if _matches(tokens, ["compute", "instances", "create", None]):
            return self._vm_create(cmd, tokens[3])
        if _matches(tokens, ["compute", "instances", "describe", None]):
            return self._vm_describe(cmd, tokens[3])
        if _matches(tokens, ["compute", "instances", "list"]):
            return self._vm_list(cmd)
        if _matches(tokens, ["compute", "instances", "delete", None]):
            return self._vm_delete(cmd, tokens[3])
        if _matches(tokens, ["compute", "instances", "update", None]):
            return self._vm_update(cmd, tokens[3])
        if _matches(tokens, ["compute", "instances", "add-metadata", None]):
            return self._vm_add_metadata(cmd, tokens[3])

        raise ValueError(f"FakeGcloud: unrecognized command: {cmd}")

    # -------------------------------------------------------------------------
    # TPU handlers
    # -------------------------------------------------------------------------

    def _tpu_create(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("tpu_create"):
            return failure

        accel_type = _parse_flag(cmd, "accelerator-type")
        if not accel_type:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.tpus.tpu-vm.create) argument --accelerator-type: expected one argument",
            )

        zone = _parse_flag(cmd, "zone")
        if not zone:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.tpus.tpu-vm.create) argument --zone: expected one argument",
            )

        labels: dict[str, str] = {}
        # Labels can appear as --labels=K=V,... or --labels K=V,...
        labels_str = _parse_flag(cmd, "labels")
        if labels_str is None:
            # Check for the two-token form: --labels K=V,...
            for i, arg in enumerate(cmd):
                if arg == "--labels" and i + 1 < len(cmd):
                    labels_str = cmd[i + 1]
                    break
        if labels_str:
            labels = _parse_labels_string(labels_str)

        idx = len(self._tpus)
        tpu_data = {
            "name": name,
            "state": "READY",
            "acceleratorType": accel_type,
            "labels": labels,
            "networkEndpoints": [{"ipAddress": f"10.0.0.{idx + 1}"}],
            "createTime": "2024-01-15T10:30:00.000Z",
        }
        self._tpus[(name, zone)] = tpu_data
        return FakeResult(returncode=0)

    def _tpu_list(self, cmd: list[str]) -> FakeResult:
        if failure := self._check_failure("tpu_list"):
            return failure

        zone = _parse_flag(cmd, "zone")
        filter_str = _parse_flag(cmd, "filter")
        required_labels = _parse_filter_labels(filter_str) if filter_str else {}

        matching = []
        for (_, tpu_zone), tpu in self._tpus.items():
            if zone and tpu_zone != zone:
                continue
            tpu_labels = tpu.get("labels", {})
            if all(tpu_labels.get(k) == v for k, v in required_labels.items()):
                matching.append(tpu)

        return FakeResult(returncode=0, stdout=json.dumps(matching))

    def _tpu_describe(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("tpu_describe"):
            return failure

        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._tpus:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        return FakeResult(returncode=0, stdout=json.dumps(self._tpus[key]))

    def _tpu_delete(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("tpu_delete"):
            return failure

        zone = _parse_flag(cmd, "zone")
        self._tpus.pop((name, zone), None)
        return FakeResult(returncode=0)

    # -------------------------------------------------------------------------
    # VM handlers
    # -------------------------------------------------------------------------

    def _vm_create(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("vm_create"):
            return failure

        machine_type = _parse_flag(cmd, "machine-type")
        if not machine_type:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.instances.create) argument --machine-type: expected one argument",
            )

        zone = _parse_flag(cmd, "zone")
        if not zone:
            return FakeResult(
                returncode=1,
                stderr="ERROR: (gcloud.compute.instances.create) argument --zone: expected one argument",
            )

        labels_str = _parse_flag(cmd, "labels")
        labels = _parse_labels_string(labels_str) if labels_str else {}

        metadata_str = _parse_flag(cmd, "metadata")
        metadata = _parse_labels_string(metadata_str) if metadata_str else {}

        idx = len(self._vms) + 1
        vm_data = {
            "name": name,
            "status": "RUNNING",
            "networkInterfaces": [
                {
                    "networkIP": f"10.128.0.{idx}",
                    "accessConfigs": [{"natIP": f"34.1.2.{idx}"}],
                }
            ],
            "labels": labels,
            "metadata": metadata,
        }
        self._vms[(name, zone)] = vm_data
        return FakeResult(returncode=0, stdout=json.dumps([vm_data]))

    def _vm_describe(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("vm_describe"):
            return failure

        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._vms:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        fmt = _parse_flag(cmd, "format")
        if fmt == "value(status)":
            return FakeResult(returncode=0, stdout=self._vms[key]["status"] + "\n")

        return FakeResult(returncode=0, stdout=json.dumps(self._vms[key]))

    def _vm_list(self, cmd: list[str]) -> FakeResult:
        if failure := self._check_failure("vm_list"):
            return failure

        # gcloud instances list uses --zones (plural)
        zones_str = _parse_flag(cmd, "zones")
        zones = set(zones_str.split(",")) if zones_str else set()
        filter_str = _parse_flag(cmd, "filter")
        required_labels = _parse_filter_labels(filter_str) if filter_str else {}

        matching = []
        for (_, vm_zone), vm in self._vms.items():
            if zones and vm_zone not in zones:
                continue
            vm_labels = vm.get("labels", {})
            if all(vm_labels.get(k) == v for k, v in required_labels.items()):
                matching.append(vm)

        return FakeResult(returncode=0, stdout=json.dumps(matching))

    def _vm_delete(self, cmd: list[str], name: str) -> FakeResult:
        if failure := self._check_failure("vm_delete"):
            return failure

        zone = _parse_flag(cmd, "zone")
        self._vms.pop((name, zone), None)
        return FakeResult(returncode=0)

    def _vm_update(self, cmd: list[str], name: str) -> FakeResult:
        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._vms:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        update_labels_str = _parse_flag(cmd, "update-labels")
        if update_labels_str:
            new_labels = _parse_labels_string(update_labels_str)
            self._vms[key].setdefault("labels", {}).update(new_labels)

        return FakeResult(returncode=0)

    def _vm_add_metadata(self, cmd: list[str], name: str) -> FakeResult:
        zone = _parse_flag(cmd, "zone")
        key = (name, zone)
        if key not in self._vms:
            return FakeResult(returncode=1, stderr="NOT_FOUND")

        metadata_str = _parse_flag(cmd, "metadata")
        if metadata_str:
            new_metadata = _parse_labels_string(metadata_str)
            self._vms[key].setdefault("metadata", {}).update(new_metadata)

        return FakeResult(returncode=0)


def _matches(tokens: list[str], pattern: list[str | None]) -> bool:
    """Check if tokens match a pattern where None means 'any single token'."""
    if len(tokens) != len(pattern):
        return False
    return all(p is None or t == p for t, p in zip(tokens, pattern, strict=True))


@pytest.fixture
def fake_gcloud() -> FakeGcloud:
    """Pytest fixture that patches iris.cluster.platform.gcp.subprocess.run with a FakeGcloud."""
    fake = FakeGcloud()
    with patch("iris.cluster.platform.gcp.subprocess.run", fake):
        yield fake
