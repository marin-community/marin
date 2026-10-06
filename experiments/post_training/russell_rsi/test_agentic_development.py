# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Harbor checks do not qualify the prepared images or the real QEMU runtime."""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("harbor")
pytest.importorskip("minisweagent")

from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle
from shellbox.backends.qemu.environment import QemuEnvironment

from experiments.post_training.russell_rsi.agentic_development import (
    HARBOR_COMMIT,
    NATIVE_MODEL_RETRY_ATTEMPTS,
    TASK_COMMIT,
    TASK_IDS,
    DevelopmentPlan,
    FrozenProducer,
    PairedReportConfig,
    binding,
    cohort,
    file_sha256,
    materialize_tasks,
    paired_report,
    run_trial_slot,
    scripted_endpoint,
)
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal
from lib.shellbox.tests.test_qemu_machine import local_guest


class LocalGuestEnvironment(QemuEnvironment):
    """Use the guest command loop and local files at the environment I/O boundary."""

    starts = 0

    async def start(self, force_build: bool) -> None:
        type(self).starts += 1
        root = Path(self.guest_bundle)
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        self.machine = await local_guest(root, env={"PATH": os.defpath})
        (root / "logs/verifier").mkdir(parents=True)

    def local_path(self, path: str) -> Path:
        return Path(self.guest_bundle) / path.lstrip("/")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: int | str | None = None,
    ):
        for path in ("/logs", "/tests", "/solution"):
            command = command.replace(path, str(self.local_path(path)))
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(Path(str(source_dir)), self.local_path(target_dir), dirs_exist_ok=True)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(self.local_path(source_dir), Path(str(target_dir)), dirs_exist_ok=True)

    async def download_file(self, source_path, target_path):
        shutil.copyfile(self.local_path(source_path), Path(str(target_path)))


class CwdFailureEnvironment(LocalGuestEnvironment):
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: int | str | None = None,
    ):
        if command == "pwd":
            raise FileNotFoundError("Fixture cwd is unavailable")
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)


@pytest.fixture
def local_trial(tmp_path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Write correct into answer and submit.")
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 30\n[verifier]\ntimeout_sec = 30\n'
        "[environment]\nallow_internet = false\n"
    )
    attempt = AttemptJournal(
        StoragePath(str(tmp_path / "journal")), {"producer": "producer-a", "task": "task-a", "bundle": "bundle-a"}
    )
    marker = tmp_path / "journal/pre-verifier/complete.json"
    (task / "tests/test.sh").write_text(
        f"#!/bin/sh\nset -eu\ntest -f {marker}\n"
        'if test "$(cat answer 2>/dev/null)" = correct; then reward=1; else reward=0; fi\n'
        f'echo "$reward" > {tmp_path / "guest/logs/verifier/reward.txt"}\n'
    )
    config = TrialConfig(
        task=TaskConfig(path=task),
        trial_name="fixture",
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(
            import_path="shellbox.mini_agent:NativeMiniAgent",
            model_name="producer-a",
            model_alias="hosted_vllm/scripted",
            upload_agent_logs=False,
            kwargs={"model_retry_attempts": NATIVE_MODEL_RETRY_ATTEMPTS},
        ),
        environment=EnvironmentConfig(
            import_path="experiments.post_training.russell_rsi.test_agentic_development:LocalGuestEnvironment",
            kwargs={"guest_bundle": str(tmp_path / "guest"), "network_policy": "deny"},
        ),
    )
    return attempt, config


@pytest.mark.parametrize(
    "limit,damage_git,missing_cwd", [(0, False, False), (1, False, False), (0, True, False), (0, False, True)]
)
def test_native_trial_persists_evidence_before_verifier_and_completed_replay_has_no_requests(
    local_trial, limit, damage_git, missing_cwd
):
    attempt, config = local_trial
    if missing_cwd:
        config = config.model_copy(
            update={
                "environment": config.environment.model_copy(
                    update={
                        "import_path": (
                            "experiments.post_training.russell_rsi.test_agentic_development:CwdFailureEnvironment"
                        )
                    }
                )
            }
        )
    command = "rm -rf .git; printf correct > answer" if damage_git else "printf correct > answer"
    with scripted_endpoint((command, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")) as (
        endpoint,
        requests,
    ):
        config = config.model_copy(
            update={
                "agent": config.agent.model_copy(
                    update={
                        "kwargs": {
                            **config.agent.kwargs,
                            "api_base": endpoint,
                            "config_specs": [
                                "mini.yaml",
                                "model.model_kwargs.num_retries=0",
                                f"agent.step_limit={limit}",
                            ],
                        }
                    }
                )
            }
        )
        first = asyncio.run(run_trial_slot(attempt, config, native=True))
        request_count = len(requests)
        starts = LocalGuestEnvironment.starts
        second = asyncio.run(run_trial_slot(attempt, config, native=True))
    assert first["status"] == "valid_grade" and first["grade"] == 1
    assert second == first
    assert request_count == len(requests) == (1 if limit else 2)
    assert first["error_category"] == ("agent_error" if limit else None)
    assert first["trial_result"]["agent_result"]["n_input_tokens"] == (1 if limit else 2)
    assert LocalGuestEnvironment.starts == starts
    assert (attempt.directory / "canonical-trial-result.json").exists()
    workspace = json.loads((attempt.directory / "pre-verifier/workspace.json").read_text())
    if damage_git:
        assert workspace["records"]["inventory"]["return_code"] != 0
    else:
        assert workspace["records"]["inventory"]["stdout"].find("answer") >= 0
    if missing_cwd:
        assert workspace["records"]["cwd"]["exception_type"] == "FileNotFoundError"
        assert workspace["cwd"] is None
    assert workspace["full_workspace_recovery"] is False
    assert "reward.txt" in first["verifier_inventory"]


@pytest.mark.parametrize(
    "changed,exception", [(None, RuntimeError), ("producer", ValueError), ("task", ValueError), ("bundle", ValueError)]
)
def test_reserved_trial_refuses_reuse_before_environment_start(local_trial, changed, exception):
    attempt, config = local_trial
    write_once(attempt.directory / "reservation.json", attempt.binding)
    if changed is not None:
        attempt = replace(attempt, binding=attempt.binding | {changed: "changed"})
    starts = LocalGuestEnvironment.starts
    with pytest.raises(exception):
        asyncio.run(run_trial_slot(attempt, config, native=True))
    assert LocalGuestEnvironment.starts == starts
    assert not Path(str(config.trials_dir)).exists()


def report_plan(tmp_path):
    pin = PinnedFile(str(tmp_path / "unused"), "a" * 64)
    producer = FrozenProducer("producer-a", "export-a", pin, pin, "tokenizer", "revision", pin)
    plan = DevelopmentPlan(
        pin,
        pin,
        RuntimeBundle("manifest", "b" * 64, "archive", "c" * 64),
        pin,
        100,
        (),
        {},
        (),
        (producer, replace(producer, identity="producer-b", export_uri="export-b")),
        "journal",
        "fixture-cluster",
    )
    tasks = [{"task_id": task_id, "source_files": [], "dockerfile_sha256": "d" * 64} for task_id in TASK_IDS]
    source = {
        "tasks": tasks,
        "task_source_commit": TASK_COMMIT,
        "harbor_commit": HARBOR_COMMIT,
    }
    images = [{"task_id": task_id, "dockerfile_sha256": "d" * 64} for task_id in TASK_IDS]
    for name, content in (("source", source), ("images", images)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content))
    return replace(
        plan,
        source_manifest=PinnedFile(str(tmp_path / "source.json"), file_sha256(tmp_path / "source.json")),
        image_manifest=PinnedFile(str(tmp_path / "images.json"), file_sha256(tmp_path / "images.json")),
    )


@pytest.mark.parametrize("missing", [False, True])
def test_paired_report_keeps_sixteen_slots_and_excludes_infrastructure_errors(tmp_path, missing):
    plan = report_plan(tmp_path)
    frozen = binding(plan, cohort(plan))
    paths = (str(tmp_path / "producer-0"), str(tmp_path / "producer-1"))
    for index, path in enumerate(paths):
        slots: dict[str, dict[str, object]] = {
            task_id: {"status": "valid_grade", "grade": index, "error_category": None} for task_id in TASK_IDS
        }
        if index == 1:
            slots[TASK_IDS[0]] = {
                "status": "infrastructure_error",
                "grade": None,
                "error_category": "infrastructure_error",
            }
        if missing and index == 1:
            del slots[TASK_IDS[-1]]
        write_once(
            StoragePath(path) / "checkpoint-results.json", {"binding": frozen, "producer_index": index, "slots": slots}
        )
    if missing:
        with pytest.raises(ValueError, match="exactly sixteen"):
            paired_report(PairedReportConfig(plan, paths, str(tmp_path / "report")))
        assert not (tmp_path / "report/development-comparison.json").exists()
        return
    paired_report(PairedReportConfig(plan, paths, str(tmp_path / "report")))
    report = json.loads((tmp_path / "report/development-comparison.json").read_text())
    assert len({(row["producer_identity"], row["task_id"]) for row in report["slots"]}) == 16
    assert report["valid_grade_count"] == 15 and report["infrastructure_error_count"] == 1
    assert report["paired_valid_count"] == report["paired_net_gain"] == 7
    assert TASK_IDS[0] not in report["wins"]


@pytest.mark.parametrize("altered", [False, True])
def test_task_archive_materializes_frozen_files_and_rejects_changed_bytes(tmp_path, altered):
    plan = report_plan(tmp_path)
    source = plan.source_manifest.read_json()
    archive_root = tmp_path / "archive-source/tasks"
    for task in source["tasks"]:
        content = f"Fixture instruction for {task['task_id']}".encode()
        task["source_files"] = [
            {
                "path": f"datasets/swebench-verified/{task['task_id']}/instruction.md",
                "size": len(content),
                "sha": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
            }
        ]
        path = archive_root / task["task_id"] / "instruction.md"
        path.parent.mkdir(parents=True)
        path.write_bytes(content[::-1] if altered else content)
    source_path = tmp_path / "source-archive-manifest.json"
    source_path.write_text(json.dumps(source))
    archive = tmp_path / "tasks.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(archive_root, arcname="tasks")
    plan = replace(
        plan,
        source_manifest=PinnedFile(str(source_path), file_sha256(source_path)),
        task_archive=PinnedFile(str(archive), file_sha256(archive)),
        task_archive_size=archive.stat().st_size,
    )
    destination = tmp_path / "materialized"
    destination.mkdir()
    if altered:
        with pytest.raises(ValueError, match="Canonical task source bytes changed"):
            materialize_tasks(plan, cohort(plan), destination)
        return
    root = materialize_tasks(plan, cohort(plan), destination)
    assert [(root / task_id / "instruction.md").read_text() for task_id in TASK_IDS] == [
        f"Fixture instruction for {task_id}" for task_id in TASK_IDS
    ]


def test_native_model_failure_retains_canonical_grade_and_model_error_category(local_trial):
    attempt, config = local_trial
    with scripted_endpoint(("unused",), response_status=401) as (endpoint, requests):
        config = config.model_copy(
            update={
                "agent": config.agent.model_copy(
                    update={
                        "kwargs": {
                            **config.agent.kwargs,
                            "api_base": endpoint,
                            "config_specs": ["mini.yaml", "model.model_kwargs.num_retries=0"],
                        }
                    }
                )
            }
        )
        result = asyncio.run(run_trial_slot(attempt, config, native=True))
    assert len(requests) == 1
    assert result["status"] == "valid_grade" and result["grade"] == 0
    assert result["error_category"] == "model_error"
    assert result["trial_result"]["exception_info"]["exception_type"] == "NonZeroAgentExitCodeError"
