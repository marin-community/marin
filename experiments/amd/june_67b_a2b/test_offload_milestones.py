# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.amd.june_67b_a2b.vultr.offload_milestones import offload_once


def _checkpoint(run_dir, subdir, step, *, is_temporary, complete=True):
    path = run_dir / subdir / f"step-{step}"
    (path / "params").mkdir(parents=True)
    (path / "params" / "chunk").write_bytes(bytes([step % 256]) * 1000)
    if complete:
        (path / "metadata.json").write_text(json.dumps({"step": step, "is_temporary": is_temporary}))
    return path


def test_offloads_milestones_and_keeps_the_newest_checkpoint_local(tmp_path):
    run_dir, destination = tmp_path / "run", tmp_path / "bucket"
    first = _checkpoint(run_dir, "checkpoints", 100, is_temporary=False)
    second = _checkpoint(run_dir, "checkpoints", 200, is_temporary=False)
    temporary = _checkpoint(run_dir, "checkpoints-temp", 250, is_temporary=True)
    partial = _checkpoint(run_dir, "checkpoints", 300, is_temporary=False, complete=False)

    copied = offload_once(run_dir, str(destination), delete_after_offload=True, workers=2)

    assert copied == [100, 200]
    for step in (100, 200):
        assert (destination / f"step-{step}" / ".verified-copy-manifest.json").is_file()
        assert (destination / f"step-{step}" / "params" / "chunk").read_bytes() == bytes([step]) * 1000
    assert not (destination / "step-250").exists() and not (destination / "step-300").exists()
    # Both milestones are older than the temporary step-250, so neither is needed to resume locally.
    assert not first.exists() and not second.exists()
    assert temporary.exists() and partial.exists()

    assert offload_once(run_dir, str(destination), delete_after_offload=True, workers=2) == []


def test_newest_milestone_stays_local_until_a_newer_checkpoint_commits(tmp_path):
    run_dir, destination = tmp_path / "run", tmp_path / "bucket"
    milestone = _checkpoint(run_dir, "checkpoints", 100, is_temporary=False)

    assert offload_once(run_dir, str(destination), delete_after_offload=True, workers=2) == [100]
    assert milestone.exists()

    _checkpoint(run_dir, "checkpoints-temp", 150, is_temporary=True)
    assert offload_once(run_dir, str(destination), delete_after_offload=True, workers=2) == []
    assert not milestone.exists()


def test_without_delete_local_every_milestone_stays(tmp_path):
    run_dir, destination = tmp_path / "run", tmp_path / "bucket"
    milestone = _checkpoint(run_dir, "checkpoints", 100, is_temporary=False)
    _checkpoint(run_dir, "checkpoints-temp", 150, is_temporary=True)

    assert offload_once(run_dir, str(destination), delete_after_offload=False, workers=2) == [100]
    assert milestone.exists()
