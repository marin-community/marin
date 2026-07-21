# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

HERE = Path(__file__).parent
PROBE_SOURCE = HERE / "cuda_module_probe.cc"
FAKE_DRIVER_SOURCE = HERE / "cuda_module_probe_fake_driver.cc"
CLIENT_SOURCE = HERE / "cuda_module_probe_client.cc"


@dataclass(frozen=True)
class ProbeBuild:
    probe: Path
    fake_driver: Path
    client: Path
    log_dir: Path
    fake_log: Path

    def run(self, **variables: str) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "LD_PRELOAD": str(self.probe),
            "MARIN_CUDA_MODULE_PROBE_PROFILE": "trace",
            "MARIN_CUDA_MODULE_PROBE_LOG_DIR": str(self.log_dir),
            "MARIN_CUDA_MODULE_PROBE_REQUIRED": "1",
            "FAKE_CUDA_LOG": str(self.fake_log),
            **variables,
        }
        return subprocess.run(
            [str(self.client), str(self.fake_driver)], check=False, capture_output=True, text=True, env=env, timeout=10
        )

    def events(self) -> list[dict]:
        return [json.loads(line) for path in self.log_dir.glob("*.ndjson") for line in path.read_text().splitlines()]


def _compile(compiler: str, arguments: list[str]) -> None:
    result = subprocess.run([compiler, *arguments], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def probe_build(tmp_path: Path) -> ProbeBuild:
    compiler = os.environ.get("CXX", "c++")
    fake_driver = tmp_path / "libfakecuda.so"
    client = tmp_path / "cuda_module_probe_client"
    probe = tmp_path / "libcuda_module_probe.so"
    _compile(compiler, ["-std=c++20", "-O2", "-fPIC", "-shared", str(FAKE_DRIVER_SOURCE), "-o", str(fake_driver)])
    _compile(compiler, ["-std=c++20", "-O2", str(CLIENT_SOURCE), "-o", str(client), "-ldl", "-pthread"])
    _compile(
        compiler,
        ["-std=c++20", "-O2", "-fPIC", "-shared", str(PROBE_SOURCE), "-o", str(probe), "-ldl", "-pthread"],
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return ProbeBuild(probe, fake_driver, client, log_dir, tmp_path / "fake.log")


def test_private_handle_fatbinary_lookup_is_redirected(probe_build: ProbeBuild) -> None:
    result = probe_build.run()

    assert result.returncode == 0, result.stderr
    assert any(
        event["event"] == "symbol_redirect" and event["symbol"] == "cuModuleLoadFatBinary"
        for event in probe_build.events()
    )
    client_output = json.loads(result.stdout)
    assert client_output == {"module_result": 0, "unload_result": 0, "module": 0xF00}


def test_valid_elf_load_records_bounded_identity(probe_build: ProbeBuild) -> None:
    result = probe_build.run()

    assert result.returncode == 0, result.stderr
    enter = next(event for event in probe_build.events() if event["event"] == "load_enter")
    exit_event = next(event for event in probe_build.events() if event["event"] == "load_exit")
    image = bytearray(128)
    image[:6] = b"\x7fELF\x02\x01"
    image[40:48] = (64).to_bytes(8, "little")
    image[52:54] = (64).to_bytes(2, "little")
    image[58:60] = (64).to_bytes(2, "little")
    image[60:62] = (1).to_bytes(2, "little")
    assert enter["input_kind"] == "elf64"
    assert enter["size"] == len(image)
    assert enter["sha256"] == hashlib.sha256(image).hexdigest()
    assert 0 <= enter["address_mod_4096"] < 4096
    assert exit_event["sequence"] == enter["sequence"]


def test_malformed_elf_uses_original_pointer_without_hashing(probe_build: ProbeBuild) -> None:
    result = probe_build.run(CLIENT_INVALID_ELF="1")

    assert result.returncode == 0, result.stderr
    enter = next(event for event in probe_build.events() if event["event"] == "load_enter")
    exit_event = next(event for event in probe_build.events() if event["event"] == "load_exit")
    assert enter["input_kind"] == "invalid_elf64"
    assert "sha256" not in enter
    assert [attempt["name"] for attempt in exit_event["attempts"]] == ["original"]


@pytest.mark.parametrize(
    ("fat_results", "expected_attempts", "expected_module"),
    [
        ("1,0", ["original", "same_pointer"], 0xF00),
        ("1,1,0", ["original", "same_pointer", "owned_copy"], 0xF00),
        ("1,1,1", ["original", "same_pointer", "owned_copy", "owned_copy_data"], 0xD00),
    ],
)
def test_trace_profile_recovers_with_distinguishable_attempts(
    probe_build: ProbeBuild, fat_results: str, expected_attempts: list[str], expected_module: int
) -> None:
    result = probe_build.run(FAKE_CUDA_FAT_RESULTS=fat_results, FAKE_CUDA_DATA_RESULTS="0")

    assert result.returncode == 0, result.stderr
    exit_event = next(event for event in probe_build.events() if event["event"] == "load_exit")
    assert [attempt["name"] for attempt in exit_event["attempts"]] == expected_attempts
    assert json.loads(result.stdout)["module"] == expected_module


def test_sync_profile_returns_prior_async_error_without_loading(probe_build: ProbeBuild) -> None:
    result = probe_build.run(MARIN_CUDA_MODULE_PROBE_PROFILE="sync", FAKE_CUDA_SYNC_RESULT="700")

    assert result.returncode == 5
    exit_event = next(event for event in probe_build.events() if event["event"] == "load_exit")
    assert exit_event["pre_sync_result"] == 700
    assert exit_event["attempts"] == []
    assert probe_build.fake_log.read_text().splitlines() == ["sync:700"]


def test_split_profile_uses_canonical_iris_task_id(probe_build: ProbeBuild) -> None:
    result = probe_build.run(
        MARIN_CUDA_MODULE_PROBE_PROFILE="trace_sync_split",
        IRIS_TASK_ID="/user/job/1:0",
        FAKE_CUDA_SYNC_RESULT="700",
    )

    assert result.returncode == 5
    enter_event = next(event for event in probe_build.events() if event["event"] == "load_enter")
    exit_event = next(event for event in probe_build.events() if event["event"] == "load_exit")
    assert enter_event["effective_profile"] == "sync"
    assert exit_event["pre_sync_result"] == 700


def test_data_direct_uses_data_for_valid_elf(probe_build: ProbeBuild) -> None:
    result = probe_build.run(
        MARIN_CUDA_MODULE_PROBE_PROFILE="data_direct", FAKE_CUDA_FAT_RESULTS="1", FAKE_CUDA_DATA_RESULTS="0"
    )

    assert result.returncode == 0, result.stderr
    exit_event = next(event for event in probe_build.events() if event["event"] == "load_exit")
    assert [attempt["name"] for attempt in exit_event["attempts"]] == ["data_direct"]
    assert probe_build.fake_log.read_text().splitlines() == ["data:0"]


def test_pressure_profile_retries_data_only_after_releasing_reserve(probe_build: ProbeBuild) -> None:
    result = probe_build.run(
        MARIN_CUDA_MODULE_PROBE_PROFILE="pressure",
        MARIN_CUDA_MODULE_PROBE_RESERVE_BYTES=str(64 << 20),
        FAKE_CUDA_FAT_RESULTS="1,1,1",
        FAKE_CUDA_DATA_RESULTS="1,0",
    )

    assert result.returncode == 0, result.stderr
    exit_event = next(event for event in probe_build.events() if event["event"] == "load_exit")
    assert [attempt["name"] for attempt in exit_event["attempts"]] == [
        "original",
        "same_pointer",
        "owned_copy",
        "owned_copy_data",
        "post_release_data",
    ]
    assert probe_build.fake_log.read_text().splitlines() == [
        "alloc:0",
        "fat:1",
        "fat:1",
        "fat:1",
        "data:1",
        "free:0",
        "data:0",
    ]


def test_concurrent_loads_have_unique_sequences_and_overlap(probe_build: ProbeBuild) -> None:
    result = probe_build.run(CLIENT_THREADS="2", FAKE_CUDA_BARRIER="1")

    assert result.returncode == 0, result.stderr
    enters = [event for event in probe_build.events() if event["event"] == "load_enter"]
    assert len({event["sequence"] for event in enters}) == 2
    assert max(event["in_flight"] for event in enters) == 2


def test_task_zero_capture_is_named_by_content_hash(probe_build: ProbeBuild) -> None:
    result = probe_build.run(MARIN_CUDA_MODULE_PROBE_CAPTURE_CUBIN="1", IRIS_TASK_ID="/user/job/0:0")

    assert result.returncode == 0, result.stderr
    enter = next(event for event in probe_build.events() if event["event"] == "load_enter")
    captures = list(probe_build.log_dir.glob("*.cubin"))
    assert [path.name for path in captures] == [f"{enter['sha256']}.cubin"]
    assert hashlib.sha256(captures[0].read_bytes()).hexdigest() == enter["sha256"]
