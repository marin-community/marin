"""Regression: per-trial object-store (fsspec/s3fs) I/O must run OFF the shared
RolloutCoordinator asyncio event loop.

Live py-spy on the CoreWeave RL path proved every coordinator stall bottomed out
in ``fsspec/asyn.py sync`` — an async filesystem driven SYNCHRONOUSLY, blocking the
coordinator's ONE event-loop thread. That serializes all trial coroutines and caps
the ``generate()`` dispatch rate (vLLM engines starved). The RL jobs use a REMOTE
``s3://`` ``trials_dir`` (durable across preemption; all job-health / trace tooling
reads it directly), so a local-``trials_dir`` swap is NOT viable — the fix is to
offload each blocking sync fsspec call via ``asyncio.to_thread`` so the loop is free.

These tests lock the offload at the sites caught by py-spy:
  - env layer: ``DaytonaEnvironment._sdk_download_dir`` (agent-log + verifier-dir +
    artifact-dir downloads all route here) and ``._sdk_download_file``;
  - trial layer: ``Trial._finalize`` (result.json) and ``Trial._init_result``
    (config.json + the redundant remote mkdir skip);
  - verifier layer: ``Verifier._resolve_rewards`` (reward-file exists/read);
  - artifact layer: ``ArtifactHandler.download_artifacts`` (manifest.json write).

The trial/artifact single-file writes now route through the bounded background
artifact writer (``to_thread`` fills the shared default executor under
concurrent trials); the tests assert both that the write never runs on the
event-loop thread and that the bytes land.
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from upath import UPath

from harbor.environments.daytona import environment as daytona_env
from harbor.environments.daytona.environment import DaytonaEnvironment
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.trial import artifact_handler as artifact_handler_module
from harbor.trial import trial as trial_module
from harbor.trial.artifact_handler import ArtifactHandler
from harbor.trial.single_step import SingleStepTrial
from harbor.utils.artifact_writer import ArtifactWriter
from harbor.verifier.verifier import Verifier


class _ThreadRecordingWriter:
    """Stand-in for a ``UPath`` result/config path: records the thread the
    artifact-writer ``open`` ran on and persists the bytes to a real local file
    so behaviour is exercised."""

    def __init__(self, path: Path):
        self._path = path
        self.parent = path.parent
        self.thread_id: int | None = None

    def __str__(self) -> str:
        return str(self._path)

    def open(self, mode: str):
        self.thread_id = threading.get_ident()
        return self._path.open(mode)


def _configure_trial_for_init_result(trial: SingleStepTrial, tmp_path: Path) -> None:
    trial.config = TrialConfig(
        task=TaskConfig(path=tmp_path),
        trial_name="t0",
    )
    trial._agent_timeout_sec = None
    trial._verifier_timeout_sec = 600
    trial._agent_setup_timeout_sec = 360
    trial._environment_build_timeout_sec = 600


# ── env layer ────────────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_sdk_download_dir_copies_remote_tree_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    remote_target = tmp_path / "remote"  # local UPath -> real copy runs

    env = object.__new__(DaytonaEnvironment)
    env._sandbox = object()  # truthy: passes the "sandbox exists" guard

    async def _fake_local_dl(source_dir: str, staging: Path) -> None:
        Path(staging, "trajectory.json").write_text("{}")

    env._download_dir_to_local = _fake_local_dl  # type: ignore[attr-defined]

    # Force the REMOTE staging branch (target is not treated as a local target).
    monkeypatch.setattr(daytona_env, "_local_download_target", lambda _t: None)

    real_copy = daytona_env._copy_local_tree_to_remote
    seen: dict[str, int] = {}

    def _spy_copy(local_root: Path, remote_root: UPath) -> None:
        seen["thread"] = threading.get_ident()
        real_copy(local_root, remote_root)

    monkeypatch.setattr(daytona_env, "_copy_local_tree_to_remote", _spy_copy)

    await env._sdk_download_dir("/logs/agent", UPath(str(remote_target)))

    assert (remote_target / "trajectory.json").read_text() == "{}"
    assert seen["thread"] != loop_thread, "remote tree copy ran ON the event loop"


@pytest.mark.unit
async def test_sdk_download_file_writes_remote_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    remote_target = tmp_path / "out" / "result.json"

    async def _fake_download_file(source_path: str, dest: str) -> None:
        Path(dest).write_text("payload")

    env = object.__new__(DaytonaEnvironment)
    env._sandbox = SimpleNamespace(
        fs=SimpleNamespace(download_file=AsyncMock(side_effect=_fake_download_file))
    )

    monkeypatch.setattr(daytona_env, "_local_download_target", lambda _t: None)

    real_copy = daytona_env._copy_local_file_to_remote
    seen: dict[str, int] = {}

    def _spy_copy(local_file: Path, target: UPath) -> None:
        seen["thread"] = threading.get_ident()
        real_copy(local_file, target)

    monkeypatch.setattr(daytona_env, "_copy_local_file_to_remote", _spy_copy)

    await env._sdk_download_file("/logs/result.json", UPath(str(remote_target)))

    assert remote_target.read_text() == "payload"
    assert seen["thread"] != loop_thread, "remote file write ran ON the event loop"


# ── trial layer ──────────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_finalize_writes_result_json_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    artifact_writer = ArtifactWriter()
    monkeypatch.setattr(trial_module, "get_artifact_writer", lambda: artifact_writer)
    writer = _ThreadRecordingWriter(tmp_path / "result.json")

    trial = object.__new__(SingleStepTrial)
    trial._attempt_index = None
    trial._stop_agent_environment = AsyncMock()  # type: ignore[attr-defined]
    # Round-3: _enforce_verifier_result_invariant is now async (it offloads its
    # own exception.txt write off the loop), so _finalize awaits it.
    trial._enforce_verifier_result_invariant = AsyncMock()  # type: ignore[assignment]
    trial._emit = AsyncMock()  # type: ignore[attr-defined]
    trial._result = SimpleNamespace(
        finished_at=None,
        model_dump_json=lambda indent: '{"ok": true}',
    )
    trial.paths = SimpleNamespace(result_path=writer, trial_dir=tmp_path)

    # _finalize awaits the required result, so the write is durable on return.
    await trial._finalize()

    assert writer.thread_id is not None
    assert writer.thread_id != loop_thread, "result.json write ran ON the event loop"
    assert (tmp_path / "result.json").read_text() == '{"ok": true}'


@pytest.mark.unit
async def test_init_result_skips_remote_mkdir_and_offloads_config_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    artifact_writer = ArtifactWriter()
    monkeypatch.setattr(trial_module, "get_artifact_writer", lambda: artifact_writer)
    writer = _ThreadRecordingWriter(tmp_path / "config.json")
    trial_dir = mock.MagicMock()  # chainable: .expanduser().resolve().as_uri()

    trial = object.__new__(SingleStepTrial)
    trial.paths = SimpleNamespace(
        _is_local=lambda: False,  # REMOTE trials_dir
        trial_dir=trial_dir,
        config_path=writer,
    )
    _configure_trial_for_init_result(trial, tmp_path)
    trial.task = SimpleNamespace(name="task", checksum="abc")
    trial.agent = SimpleNamespace(to_agent_info=lambda: None)

    with mock.patch("harbor.trial.trial.TrialResult", return_value=object()):
        await trial._init_result()
    assert artifact_writer.flush(timeout=30) == 0

    trial_dir.mkdir.assert_not_called()  # redundant remote round-trip skipped
    assert writer.thread_id is not None
    assert writer.thread_id != loop_thread, "config.json write ran ON the event loop"
    persisted = json.loads((tmp_path / "config.json").read_text())
    assert persisted["resolved_timeouts"] == {
        "agent": None,
        "verifier": 600,
        "agent_setup": 360,
        "environment_build": 600,
        "trial_attempt": None,
        "trial_cleanup_grace": 30.0,
        "steps": [],
    }


@pytest.mark.unit
async def test_init_result_mkdirs_on_local_trials_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trial_module, "get_artifact_writer", lambda: ArtifactWriter())
    writer = _ThreadRecordingWriter(tmp_path / "config.json")
    trial_dir = mock.MagicMock()

    trial = object.__new__(SingleStepTrial)
    trial.paths = SimpleNamespace(
        _is_local=lambda: True,  # LOCAL trials_dir still needs the dir
        trial_dir=trial_dir,
        config_path=writer,
    )
    _configure_trial_for_init_result(trial, tmp_path)
    trial.task = SimpleNamespace(name="task", checksum="abc")
    trial.agent = SimpleNamespace(to_agent_info=lambda: None)

    with mock.patch("harbor.trial.trial.TrialResult", return_value=object()):
        await trial._init_result()

    trial_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


# ── verifier layer ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_verifier_resolve_rewards_reads_json(tmp_path: Path) -> None:
    (tmp_path / "reward.json").write_text('{"reward": 1.0}')
    verifier = object.__new__(Verifier)
    verifier.trial_paths = SimpleNamespace(
        reward_json_path=UPath(str(tmp_path / "reward.json")),
        reward_text_path=UPath(str(tmp_path / "reward.txt")),
    )
    assert verifier._resolve_rewards() == {"reward": 1.0}


@pytest.mark.unit
def test_verifier_resolve_rewards_reads_text_fallback(tmp_path: Path) -> None:
    (tmp_path / "reward.txt").write_text("0.5")
    verifier = object.__new__(Verifier)
    verifier.trial_paths = SimpleNamespace(
        reward_json_path=UPath(str(tmp_path / "reward.json")),
        reward_text_path=UPath(str(tmp_path / "reward.txt")),
    )
    assert verifier._resolve_rewards() == {"reward": 0.5}


# ── artifact layer ───────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_download_artifacts_writes_manifest_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_writer = ArtifactWriter()
    monkeypatch.setattr(
        artifact_handler_module, "get_artifact_writer", lambda: artifact_writer
    )

    handler = ArtifactHandler(artifacts=[], logger=mock.MagicMock())

    source_env = SimpleNamespace(
        capabilities=SimpleNamespace(mounted=False),
        is_dir=AsyncMock(return_value=True),
        download_dir=AsyncMock(),
    )

    await handler.download_artifacts(
        source_env,  # type: ignore[arg-type]
        tmp_path / "artifacts",
        source_artifacts_dir="/logs/artifacts",
    )
    # The optional manifest write runs on the background writer, never on the
    # event-loop thread. Drain it here to observe the persisted diagnostic.
    assert artifact_writer.flush(timeout=30) == 0

    manifest_path = tmp_path / "artifacts" / "manifest.json"
    assert manifest_path.exists(), "manifest.json was not written"
