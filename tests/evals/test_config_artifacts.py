# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from marin.evaluation.config_artifacts import HARBOR_CONFIG
from marin.evaluation.harbor_runner import _HARBOR_SPEC

from scripts.ci.update_evaluation_config_artifacts import ArtifactPin, apply_pins


def test_harbor_trial_runtime_matches_the_resolver_revision() -> None:
    assert _HARBOR_SPEC == HARBOR_CONFIG.source_requirement


def test_artifact_updater_rewrites_only_marked_pins_and_is_idempotent(tmp_path: Path) -> None:
    config_module = tmp_path / "lib/marin/src/marin/evaluation/config_artifacts.py"
    config_module.parent.mkdir(parents=True)
    config_module.write_text("# BEGIN GENERATED CONFIG ARTIFACT PINS\n# END GENERATED CONFIG ARTIFACT PINS\n")
    marin_project = tmp_path / "lib/marin/pyproject.toml"
    marin_project.parent.mkdir(parents=True, exist_ok=True)
    marin_project.write_text(
        "dependencies = [\n"
        "    # BEGIN GENERATED CONFIG ARTIFACT DEPENDENCIES\n"
        "    # END GENERATED CONFIG ARTIFACT DEPENDENCIES\n"
        "]\n"
    )
    root_project = tmp_path / "pyproject.toml"
    root_project.write_text("# BEGIN GENERATED HARBOR RUNTIME PIN\n# END GENERATED HARBOR RUNTIME PIN\n")
    harbor = ArtifactPin(
        package="harbor-config",
        repository="marin-community/harbor",
        revision="harbor-revision",
        release_tag="harbor-tag",
        wheel_url="https://example.test/harbor.whl",
        wheel_sha256="harbor-sha",
        schema_fingerprint="harbor-schema",
        resolver_fingerprint="harbor-resolver",
    )
    evalchemy = ArtifactPin(
        package="evalchemy-config",
        repository="marin-community/evalchemy",
        revision="evalchemy-revision",
        release_tag="evalchemy-tag",
        wheel_url="https://example.test/evalchemy.whl",
        wheel_sha256="evalchemy-sha",
        schema_fingerprint="evalchemy-schema",
        resolver_fingerprint=None,
    )

    assert apply_pins(tmp_path, harbor, evalchemy)
    assert not apply_pins(tmp_path, harbor, evalchemy)
    assert "harbor-revision" in root_project.read_text()
    assert "https://example.test/harbor.whl#sha256=harbor-sha" in marin_project.read_text()
    assert "evalchemy-resolver" not in config_module.read_text()
