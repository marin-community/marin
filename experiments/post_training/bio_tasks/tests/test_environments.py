# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io

import pytest

from experiments.post_training.bio_tasks.build import identity, task_files
from experiments.post_training.bio_tasks.install_environment import download
from experiments.post_training.bio_tasks.recipes import RECIPES


def test_distinct_biological_inputs_share_the_locked_environment_context():
    recipe = next(recipe for recipe in RECIPES if recipe.id == "real-protein-alignment")
    first, second = (
        task_files(recipe, recipe.generate(seed), identity(recipe, seed, 0), "python@sha256:" + "0" * 64, "1" * 40)
        for seed in (0, 1)
    )
    assert first.under("environment/") == second.under("environment/")
    assert first.under("setup_files/inputs/") != second.under("setup_files/inputs/")


def test_package_download_checks_bytes_and_bounds_before_installation(tmp_path, monkeypatch):
    content = b"package artifact bytes"
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(content))
    target = tmp_path / "package.conda"
    url = "https://conda.anaconda.org/conda-forge/linux-64/example.conda"
    checksum = hashlib.sha256(content).hexdigest()
    assert download(url, target, "sha256", checksum, len(content)) == len(content)
    assert target.read_bytes() == content
    with pytest.raises(ValueError, match="checksum mismatch"):
        download(url, target, "sha256", "0" * 64, len(content))
    with pytest.raises(ValueError, match="byte limit"):
        download(url, target, "sha256", checksum, len(content) - 1)
