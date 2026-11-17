# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.post_training.long_loop.serving import check_serving_pin, serving_release


def test_endpoint_release_matches_rl_and_rejects_drift(tmp_path):
    project = Path(__file__).resolve().parents[4] / "vendor/skyrl/pyproject.toml"
    release = serving_release()
    check_serving_pin(release, project)
    changed = tmp_path / "pyproject.toml"
    changed.write_text(project.read_text().replace(release.version, "0.0.0.old"))
    with pytest.raises(ValueError, match="version differs"):
        check_serving_pin(release, changed)
    bad_wheel = replace(release.wheels[0], url="https://example.com/other.whl")
    with pytest.raises(ValueError, match="wheels differ"):
        check_serving_pin(replace(release, wheels=(bad_wheel, *release.wheels[1:])), project)
