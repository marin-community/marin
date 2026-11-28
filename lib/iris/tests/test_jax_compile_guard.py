# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition

import pytest
from iris.jax.compile_guard import (
    _autotune_cache_file_count,
    _optimized_hlo_text,
    check_distributed_compile_fingerprint,
    check_distributed_executable_text,
)


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


def test_distributed_executable_identical_text_continues_and_releases_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeCoordinationService()
    monkeypatch.setattr("iris.jax.compile_guard.os.scandir", lambda _: pytest.fail("autotune directory scanned"))

    def check(process_id: int) -> None:
        check_distributed_executable_text(
            service.client(process_id), (0, 1), process_id, 0, "same-cache-key", "same optimized HLO", "/cache"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(check, (0, 1)))

    assert service.values == {}


def test_distributed_executable_divergent_text_reports_ranks_and_autotune_inputs(tmp_path: Path) -> None:
    service = _FakeCoordinationService()
    hashes = [hashlib.sha256(f"HLO {process_id}".encode()).hexdigest() for process_id in (0, 1)]
    cache_dirs = [tmp_path / f"process-{process_id}" for process_id in (0, 1)]
    for process_id, cache_dir in enumerate(cache_dirs):
        cache_dir.mkdir()
        for file_index in range(process_id + 1):
            (cache_dir / f"{file_index}.textproto").write_text("HLO")

    def check(process_id: int) -> str:
        with pytest.raises(RuntimeError) as error:
            check_distributed_executable_text(
                service.client(process_id),
                (0, 1),
                process_id,
                0,
                "same-cache-key",
                f"HLO {process_id}",
                str(cache_dirs[process_id]),
            )
        return str(error.value)

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = list(executor.map(check, (0, 1)))

    for error in errors:
        assert "same-cache-key" in error
        assert all(f"{digest}: [{rank}]" in error for rank, digest in enumerate(hashes))
        assert str(cache_dirs[0]) in error and "file_count=1" in error
        assert str(cache_dirs[1]) in error and "file_count=2" in error


def test_distributed_executable_unsupported_text_continues_on_all_ranks(caplog: pytest.LogCaptureFixture) -> None:
    service = _FakeCoordinationService()

    def check(process_id: int) -> None:
        check_distributed_executable_text(
            service.client(process_id),
            (0, 1),
            process_id,
            0,
            "same-cache-key",
            None if process_id == 0 else "HLO",
            "",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(check, (0, 1)))

    assert service.values == {}
    warnings = [
        record
        for record in caplog.records
        if record.name == "iris.jax.compile_guard" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1


def test_distributed_executable_extraction_error_fails_all_ranks() -> None:
    service = _FakeCoordinationService()

    def check(process_id: int) -> RuntimeError:
        with pytest.raises(RuntimeError, match="extraction failed on processes \\[0\\]") as error:
            check_distributed_executable_text(
                service.client(process_id),
                (0, 1),
                process_id,
                0,
                "same-cache-key",
                None if process_id == 0 else "HLO",
                "",
                ValueError("invalid HLO") if process_id == 0 else None,
            )
        return error.value

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = list(executor.map(check, (0, 1)))

    assert isinstance(errors[0].__cause__, ValueError)
    assert errors[1].__cause__ is None


def test_optimized_hlo_text_falls_back_to_modules() -> None:
    class Loaded:
        def hlo_modules(self) -> list["Module"]:
            return [Module("first"), Module("second")]

    @dataclass
    class Module:
        text: str

        def to_string(self) -> str:
            return self.text

    assert _optimized_hlo_text(Loaded(), RuntimeError) == "first\n\nsecond"


def test_optimized_hlo_text_unsupported_backend_returns_none() -> None:
    class Loaded:
        def get_hlo_text(self) -> str:
            raise RuntimeError("UNIMPLEMENTED: optimized HLO is unavailable")

        def hlo_modules(self) -> list[object]:
            raise NotImplementedError

    assert _optimized_hlo_text(Loaded(), RuntimeError) is None


def test_autotune_cache_file_count_counts_files_in_active_directory(tmp_path: Path) -> None:
    (tmp_path / "first.textproto").write_text("first")
    (tmp_path / "second.textproto").write_text("second")
    (tmp_path / "nested").mkdir()
    assert _autotune_cache_file_count(str(tmp_path)) == 2


def test_autotune_cache_file_count_non_directory_path_has_unknown_count(tmp_path: Path) -> None:
    not_a_directory = tmp_path / "file.textproto"
    not_a_directory.write_text("file")
    assert _autotune_cache_file_count(str(not_a_directory)) is None
