# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Condition

import pytest
from iris.runtime.jax_compile_guard import check_distributed_compile_fingerprint


@dataclass
class _FakeCoordinationService:
    values: dict[str, str] = field(default_factory=dict)
    arrivals: dict[str, set[int]] = field(default_factory=dict)
    condition: Condition = field(default_factory=Condition)

    def client(self, process_id: int) -> "_FakeClient":
        return _FakeClient(self, process_id)


@dataclass
class _FakeClient:
    service: _FakeCoordinationService
    process_id: int

    def key_value_set(self, key: str, value: str, allow_overwrite: bool = False) -> None:
        with self.service.condition:
            assert allow_overwrite is False
            assert key not in self.service.values
            self.service.values[key] = value
            self.service.condition.notify_all()

    def key_value_dir_get(self, key: str) -> list[tuple[str, str]]:
        with self.service.condition:
            return [(name, value) for name, value in self.service.values.items() if name.startswith(key)]

    def key_value_delete(self, key: str) -> None:
        with self.service.condition:
            del self.service.values[key]

    def wait_at_barrier(self, barrier_id: str, timeout_in_ms: int, process_ids: tuple[int, ...]) -> None:
        with self.service.condition:
            self.service.arrivals.setdefault(barrier_id, set()).add(self.process_id)
            self.service.condition.notify_all()
            if not self.service.condition.wait_for(
                lambda: self.service.arrivals[barrier_id] == set(process_ids), timeout_in_ms / 1000
            ):
                raise TimeoutError(barrier_id)


def test_distributed_compile_divergent_fingerprints_report_every_rank() -> None:
    service = _FakeCoordinationService()

    def check(process_id: int) -> str:
        with pytest.raises(RuntimeError) as error:
            check_distributed_compile_fingerprint(
                service.client(process_id), (0, 1), process_id, 0, f"module-{process_id}"
            )
        return str(error.value)

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = list(executor.map(check, (0, 1)))

    assert all("fingerprint mismatch" in error for error in errors)
    assert all("0: 'module-0'" in error and "1: 'module-1'" in error for error in errors)


def test_distributed_compile_identical_fingerprints_continue_and_release_keys() -> None:
    service = _FakeCoordinationService()

    def check(process_id: int) -> None:
        check_distributed_compile_fingerprint(service.client(process_id), (0, 1), process_id, 0, "same-module")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(check, (0, 1)))

    assert service.values == {}


def test_distributed_compile_missing_peer_reports_observed_fingerprints() -> None:
    service = _FakeCoordinationService()

    with pytest.raises(RuntimeError, match="expected processes \\(0, 1\\); observed \\{0: 'module-0'\\}"):
        check_distributed_compile_fingerprint(service.client(0), (0, 1), 0, 0, "module-0", timeout_ms=10)
