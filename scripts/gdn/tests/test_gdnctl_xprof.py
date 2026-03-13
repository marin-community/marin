# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import marin.profiling

from scripts.gdn import gdnctl


def test_build_remote_xprof_compare_command() -> None:
    command = gdnctl._build_remote_xprof_compare_command(
        before_remote_path="marin/.agents/xprof_compare/before.xplane.pb",
        after_remote_path="marin/.agents/xprof_compare/after.xplane.pb",
        output_remote_path="marin/.agents/xprof_compare/result.json",
        top_k=12,
        normalize_positive_deltas_ms=47.97,
    )

    assert "uv run --with xprof python -m marin.profiling.cli xprof-compare" in command
    assert "--before-xplane marin/.agents/xprof_compare/before.xplane.pb" in command
    assert "--after-xplane marin/.agents/xprof_compare/after.xplane.pb" in command
    assert "--output marin/.agents/xprof_compare/result.json" in command
    assert "--top-k 12" in command
    assert "--normalize-positive-deltas-ms 47.97" in command
    assert "cat marin/.agents/xprof_compare/result.json" in command


def test_cmd_xprof_compare_runs_remote_analysis_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    before = tmp_path / "before.xplane.pb"
    after = tmp_path / "after.xplane.pb"
    output = tmp_path / "compare.json"
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    calls: list[list[str]] = []

    def fake_run(
        cmd,
        *,
        cwd=None,
        input_text=None,
        capture_output=False,
        check=True,
        extra_env=None,
    ):
        del cwd, input_text, check, extra_env
        call = list(cmd)
        calls.append(call)
        if call[:3] == ["ssh", "dev-tpu-demo", "mkdir"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:1] == ["scp"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:3] == ["ssh", "dev-tpu-demo", "bash"]:
            payload = {"framework_op_stats": {"positive_deltas": []}, "op_profile_category": {"positive_deltas": []}}
            return subprocess.CompletedProcess(call, 0, json.dumps(payload), "")
        if call[:3] == ["ssh", "dev-tpu-demo", "rm"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        raise AssertionError(call)

    monkeypatch.setattr(gdnctl, "_ensure_dev_tpu_ssh_config", lambda tpu_name: "dev-tpu-demo")
    monkeypatch.setattr(gdnctl, "_run", fake_run)

    args = argparse.Namespace(
        cluster="us-east5-a",
        tpu_name="demo",
        before_xplane=before,
        after_xplane=after,
        top_k=20,
        normalize_positive_deltas_ms=47.972424,
        remote_stage_dir=".agents/xprof_compare/iter92",
        sync_repo=False,
        keep_remote=False,
        output=output,
    )

    rc = gdnctl.cmd_xprof_compare(args)
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["framework_op_stats"]["positive_deltas"] == []
    assert any(call[:1] == ["scp"] for call in calls)
    assert any(call[:3] == ["ssh", "dev-tpu-demo", "rm"] for call in calls)


def test_cmd_xprof_compare_can_sync_repo_first(tmp_path: Path, monkeypatch) -> None:
    before = tmp_path / "before.xplane.pb"
    after = tmp_path / "after.xplane.pb"
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    calls: list[list[str]] = []

    def fake_run(
        cmd,
        *,
        cwd=None,
        input_text=None,
        capture_output=False,
        check=True,
        extra_env=None,
    ):
        del cwd, input_text, check, extra_env
        call = list(cmd)
        calls.append(call)
        if call[:3] == gdnctl.DEV_TPU:
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:3] == ["ssh", "dev-tpu-demo", "mkdir"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:1] == ["scp"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:3] == ["ssh", "dev-tpu-demo", "bash"]:
            payload = {"framework_op_stats": {"positive_deltas": []}, "op_profile_category": {"positive_deltas": []}}
            return subprocess.CompletedProcess(call, 0, json.dumps(payload), "")
        if call[:3] == ["ssh", "dev-tpu-demo", "rm"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        raise AssertionError(call)

    monkeypatch.setattr(gdnctl, "_ensure_dev_tpu_ssh_config", lambda tpu_name: "dev-tpu-demo")
    monkeypatch.setattr(gdnctl, "_run", fake_run)

    args = argparse.Namespace(
        cluster="us-east5-a",
        tpu_name="demo",
        before_xplane=before,
        after_xplane=after,
        top_k=20,
        normalize_positive_deltas_ms=None,
        remote_stage_dir=".agents/xprof_compare/iter92",
        sync_repo=True,
        keep_remote=True,
        output=None,
    )

    rc = gdnctl.cmd_xprof_compare(args)
    assert rc == 0
    assert calls[0][:3] == gdnctl.DEV_TPU
    assert not any(call[:3] == ["ssh", "dev-tpu-demo", "rm"] for call in calls)


def test_cmd_xprof_compare_runs_downloads_artifacts_and_compares(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "compare.json"
    download_roots: list[Path] = []

    def fake_download_latest_profile_artifact_for_run(*args, **kwargs):
        run_target = args[0]
        download_root = Path(kwargs["download_root"])
        download_roots.append(download_root)
        filename = "before.xplane.pb" if run_target.endswith("/before") else "after.xplane.pb"
        (download_root / filename).write_bytes(filename.encode("utf-8"))
        artifact_ref = "entity/project/before:v0" if filename.startswith("before") else "entity/project/after:v0"
        return SimpleNamespace(artifact_ref=artifact_ref, artifact_dir=download_root)

    monkeypatch.setattr(
        marin.profiling,
        "download_latest_profile_artifact_for_run",
        fake_download_latest_profile_artifact_for_run,
    )
    monkeypatch.setattr(
        gdnctl,
        "_run_xprof_compare_remote",
        lambda **kwargs: {
            "framework_op_stats": {"family_positive_deltas": []},
            "op_profile_category": {"positive_deltas": []},
            "before_xplane": str(kwargs["before_xplane"]),
            "after_xplane": str(kwargs["after_xplane"]),
        },
    )

    args = argparse.Namespace(
        cluster="us-east5-a",
        tpu_name="demo",
        before_run_target="entity/project/runs/before",
        after_run_target="entity/project/runs/after",
        entity=None,
        project=None,
        alias="latest",
        download_root=tmp_path / "downloads",
        top_k=20,
        normalize_positive_deltas_ms=47.972424,
        remote_stage_dir=".agents/xprof_compare/iter92",
        sync_repo=False,
        keep_remote=False,
        output=output,
    )

    rc = gdnctl.cmd_xprof_compare_runs(args)
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert download_roots == [tmp_path / "downloads" / "before", tmp_path / "downloads" / "after"]
    assert payload["before_artifact_ref"] == "entity/project/before:v0"
    assert payload["after_artifact_ref"] == "entity/project/after:v0"
    assert payload["before_xplane_local"].endswith("before.xplane.pb")
    assert payload["after_xplane_local"].endswith("after.xplane.pb")


def test_run_xprof_compare_remote_falls_back_to_downloaded_json(tmp_path: Path, monkeypatch) -> None:
    before = tmp_path / "before.xplane.pb"
    after = tmp_path / "after.xplane.pb"
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    calls: list[list[str]] = []

    def fake_run(
        cmd,
        *,
        cwd=None,
        input_text=None,
        capture_output=False,
        check=True,
        extra_env=None,
    ):
        del cwd, input_text, capture_output, check, extra_env
        call = list(cmd)
        calls.append(call)
        if call[:3] == ["ssh", "dev-tpu-demo", "mkdir"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:3] == ["ssh", "dev-tpu-demo", "bash"]:
            return subprocess.CompletedProcess(call, 0, "", "remote stderr noise")
        if call[:1] == ["scp"] and call[1].startswith("dev-tpu-demo:"):
            Path(call[2]).write_text('{"framework_op_stats": {"positive_deltas": []}}', encoding="utf-8")
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:1] == ["scp"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        if call[:3] == ["ssh", "dev-tpu-demo", "rm"]:
            return subprocess.CompletedProcess(call, 0, "", "")
        raise AssertionError(call)

    monkeypatch.setattr(gdnctl, "_ensure_dev_tpu_ssh_config", lambda tpu_name: "dev-tpu-demo")
    monkeypatch.setattr(gdnctl, "_run", fake_run)

    payload = gdnctl._run_xprof_compare_remote(
        cluster="us-east5-a",
        tpu_name="demo",
        before_xplane=before,
        after_xplane=after,
        top_k=20,
        normalize_positive_deltas_ms=None,
        remote_stage_dir=".agents/xprof_compare/iter93",
        sync_repo=False,
        keep_remote=False,
    )

    assert payload["framework_op_stats"]["positive_deltas"] == []
    assert any(call[:1] == ["scp"] and call[1].startswith("dev-tpu-demo:") for call in calls)
