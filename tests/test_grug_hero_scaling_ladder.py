# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run

HERO_TRIGGER = Path(__file__).resolve().parents[1] / "experiments/grug/moe_hero_ep/trigger_hero.sh"


def test_diagnostic_run_matches_the_d6144_rack_local_recipe():
    diagnostic = build_diagnostic_run(
        run_id="test-diagnostic",
        dp_racks=1,
        num_steps=1,
        schedule_steps=390_251,
        version="dev",
    )
    ladder = build_ladder_run(run_id="test-ladder", size="d6144", version="dev")
    diagnostic_config = diagnostic.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=diagnostic.runtime_args, deps=diagnostic.deps)
    )
    ladder_config = ladder.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=ladder.runtime_args, deps=ladder.deps)
    )

    assert diagnostic_config.model == ladder_config.model
    assert diagnostic_config.processes_per_task == ladder_config.processes_per_task
    assert diagnostic_config.tensorstore_cache_bytes == ladder_config.tensorstore_cache_bytes
    assert diagnostic_config.trainer == dataclasses.replace(
        ladder_config.trainer,
        trainer=diagnostic_config.trainer.trainer,
        replica_axis_size=1,
        save_checkpoints=False,
    )
    assert diagnostic_config.trainer.trainer == dataclasses.replace(
        ladder_config.trainer.trainer,
        id=diagnostic_config.trainer.trainer.id,
        train_batch_size=diagnostic_config.trainer.trainer.train_batch_size,
        profiler=diagnostic_config.trainer.trainer.profiler,
        tracker=diagnostic_config.trainer.trainer.tracker,
        progress_watchdog=diagnostic_config.trainer.trainer.progress_watchdog,
        checkpointer=diagnostic_config.trainer.trainer.checkpointer,
        load_checkpoint_path=diagnostic_config.trainer.trainer.load_checkpoint_path,
    )
    assert diagnostic_config.data.target_budget is ladder_config.data.target_budget is None
    assert diagnostic_config.data.experiment_budget is ladder_config.data.experiment_budget is None
    assert diagnostic_config.data.train_weights == [
        (step, {name: weight for name, weight in weights.items() if weight > 0})
        for step, weights in ladder_config.data.train_weights
    ]


@pytest.mark.parametrize(
    ("size", "num_steps", "expected_simulated_epoching"),
    [("d2048", None, True), ("d6144", 1, True), ("d6144", None, False)],
)
def test_scaling_ladder_disables_simulated_epoching_above_flop_limit(size, num_steps, expected_simulated_epoching):
    step = build_ladder_run(run_id=f"test-{size}", size=size, num_steps=num_steps, version="2026.08.18")
    ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)

    data = step.build_config(ctx).data

    assert (data.target_budget is not None) is expected_simulated_epoching
    assert (data.experiment_budget is not None) is expected_simulated_epoching


def test_scaling_ladder_searches_cluster_and_data_local_temp_roots(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", "s3://hero-checkpoints")
    step = build_ladder_run(run_id="test-d6144", size="d6144", num_steps=1, version="2026.08.18")
    output_path = "s3://marin-us-east-02a/marin/grug/test-d6144/v"
    ctx = dataclasses.replace(
        StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps),
        output_path=output_path,
    )

    trainer = step.build_config(ctx).trainer.trainer
    assert trainer.checkpoint_search_paths("test-d6144") == [
        f"{output_path}/checkpoints",
        "s3://hero-checkpoints/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/test-d6144/v/checkpoints",
        "s3://marin-us-east-02a/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/test-d6144/v/checkpoints",
    ]


@pytest.mark.parametrize(
    ("dirty", "comment_succeeds"),
    [(False, True), (True, True), (False, False)],
)
def test_hero_trigger_records_the_submitted_commit_and_tree_state(tmp_path, dirty, comment_succeeds):
    repo = tmp_path / "repo"
    repo.mkdir()
    trigger = repo / "trigger_hero.sh"
    shutil.copy2(HERO_TRIGGER, trigger)

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "add", trigger.name], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "test trigger"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        (repo / "untracked.txt").write_text("dirty\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['HERO_TRIGGER_CAPTURE']).write_text(json.dumps({\n"
        "    'argv': sys.argv[1:],\n"
        "}))\n"
    )
    fake_uv.chmod(0o755)
    fake_uuidgen = fake_bin / "uuidgen"
    fake_uuidgen.write_text("#!/bin/sh\necho 12345678-1234-1234-1234-123456789abc\n")
    fake_uuidgen.chmod(0o755)
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['HERO_GH_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(int(os.environ.get('HERO_GH_EXIT_CODE', '0')))\n"
    )
    fake_gh.chmod(0o755)

    capture = tmp_path / "capture.json"
    gh_capture = tmp_path / "gh.json"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "WANDB_API_KEY": "test-key",
        "HERO_TRIGGER_CAPTURE": str(capture),
        "HERO_GH_CAPTURE": str(gh_capture),
        "HERO_GH_EXIT_CODE": "0" if comment_succeeds else "1",
    }
    result = subprocess.run([str(trigger)], cwd=repo, env=env, check=False, capture_output=True, text=True)

    expected_dirty = str(dirty).lower()
    expected_state = "dirty" if dirty else "clean"
    expected_job_name = f"hero-12d8b6f0-dee637-coord-{commit[:8]}-{expected_state}-12345678"
    assert json.loads(gh_capture.read_text()) == [
        "issue",
        "comment",
        "https://github.com/marin-community/marin/issues/8506",
        "--body",
        (
            "Hero launch requested.\n\n"
            "- Run ID: `hero-12d8b6f0-dee637`\n"
            f"- Commit: `{commit}`\n"
            f"- Tree dirty: `{expected_dirty}`\n"
            f"- Coordinator job: `{expected_job_name}`\n"
            "- Target: `cw-us-east-08a` (11 x NVL72)"
        ),
    ]
    if not comment_succeeds:
        assert result.returncode == 1
        assert not capture.exists()
        return

    assert result.returncode == 0
    submitted = json.loads(capture.read_text())
    argv = submitted["argv"]
    assert argv[argv.index("--system-reason") + 1] == (f"hero run; commit={commit}; tree_dirty={expected_dirty}")
    assert argv[argv.index("--job-name") + 1] == expected_job_name
    assert argv[argv.index("GIT_COMMIT") + 1] == commit
    assert argv[argv.index("HERO_LAUNCH_TREE_DIRTY") + 1] == expected_dirty
