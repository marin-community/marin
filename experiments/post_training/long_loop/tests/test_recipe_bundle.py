# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import subprocess
import tarfile

import pytest

from experiments.post_training.long_loop.archive import ArchiveInput, seal_archive
from experiments.post_training.long_loop.backfill_recipe import backfill_recipe
from experiments.post_training.long_loop.recipe_bundle import (
    require_clean_revision,
    runtime_source_manifest,
    verify_recipe_bundle,
    write_recipe_bundle,
)


def test_bundle_pins_runtime_and_consumed_configs_and_rejects_corruption(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "train.py"
    source.write_text("LR = 0.00005\n")
    snapshot = runtime_source_manifest(workspace, "a" * 40)
    config = tmp_path / "consumed.yaml"
    config.write_text("loss_mode: full_token\n")
    inputs = [ArchiveInput(str(config), "sft/consumed.yaml", "config")]
    root = str(tmp_path / "recipe")
    _, receipt = write_recipe_bundle(root, inputs, snapshot, workspace)
    assert verify_recipe_bundle(root, receipt["manifest_sha256"]) == receipt
    with tarfile.open(tmp_path / "recipe/recipe.tar.gz") as archive:
        assert archive.extractfile("source/train.py").read() == source.read_bytes()
        assert archive.extractfile("evidence/sft/consumed.yaml").read() == config.read_bytes()
    source.write_text("LR = 1\n")
    with pytest.raises(ValueError, match="source changed"):
        write_recipe_bundle(str(tmp_path / "changed"), inputs, snapshot, workspace)
    # A valid tar stream with a changed config must fail the original manifest.
    with tarfile.open(tmp_path / "recipe/recipe.tar.gz", "w:gz") as archive:
        for name, content in (("source/train.py", b"LR = 0.00005\n"), ("evidence/sft/consumed.yaml", b"changed\n")):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    with pytest.raises(ValueError, match="member failed verification"):
        verify_recipe_bundle(root, receipt["manifest_sha256"])
    manifest = tmp_path / "recipe/manifest.json"
    document = json.loads(manifest.read_text())
    document["marin_revision"] = "b" * 40
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        verify_recipe_bundle(root, receipt["manifest_sha256"])


def test_production_refuses_tracked_and_untracked_source_changes(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "--quiet")
    source = tmp_path / "recipe.py"
    source.write_text("LR = 0.00005\n")
    git("add", "recipe.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "Initial recipe")
    revision = git("rev-parse", "HEAD")
    require_clean_revision(tmp_path, revision)
    source.write_text("LR = 0.01\n")
    with pytest.raises(ValueError, match="clean source tree"):
        require_clean_revision(tmp_path, revision)
    git("restore", "recipe.py")
    (tmp_path / "uncommitted.py").write_text("LR = 1\n")
    with pytest.raises(ValueError, match="including untracked"):
        require_clean_revision(tmp_path, revision)


def test_historical_backfill_uses_committed_sources_and_preserves_sealed_archive(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()

    git("init", "--quiet")
    (repository / "recipe.py").write_text("LR = 0.00005\n")
    git("add", "recipe.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "Initial recipe")
    revision = git("rev-parse", "HEAD")
    config = tmp_path / "consumed.yaml"
    config.write_text("batch: 256\n")
    root = tmp_path / "sealed"
    seal_archive(str(root), [ArchiveInput(str(config), "sft/config.yaml", "config")], {"marin_commit": revision})
    raw = (root / "manifest.json").read_bytes()
    result = backfill_recipe(
        archive_root=str(root),
        archive_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        output=str(tmp_path / "backfill"),
        repository=repository,
        source_revisions=(revision,),
        backfill_revision=revision,
        limitations=("Historical dirty source bytes are not reconstructed by a commit.",),
    )
    assert (root / "manifest.json").read_bytes() == raw
    assert not result["producer_replay"] and not result["gpu_allocation"] and not result["historical_archive_modified"]
    assert verify_recipe_bundle(str(tmp_path / "backfill"), result["manifest_sha256"])["files"] == 3
    with tarfile.open(tmp_path / "backfill/recipe.tar.gz") as archive:
        assert archive.extractfile(f"source/revisions/{revision}/recipe.py").read() == b"LR = 0.00005\n"
