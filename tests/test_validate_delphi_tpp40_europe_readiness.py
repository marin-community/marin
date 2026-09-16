# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.processing.tokenize import step_to_lm_mixture_component

from experiments.domain_phase_mix.audit_delphi_tpp40_evaluation_caches import evaluation_paths_sha256
from experiments.domain_phase_mix.delphi_tpp40_europe_runtime_caches import EUROPE_HISTORICAL_STACK_INPUT_PREFIX
from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_evaluation_caches import evaluation_steps
from experiments.domain_phase_mix.validate_delphi_tpp40_europe_readiness import (
    EXPECTED_STACK_ELEMENTS,
    EXPECTED_STACK_INPUTS,
    EXPECTED_STACK_TOKENS,
    validate_stack_artifact,
)


def _artifact() -> dict[str, object]:
    return {
        "input_configs": {
            name: {"cache_dir": f"{EUROPE_HISTORICAL_STACK_INPUT_PREFIX}{name.replace('/', '_')}-hash"}
            for name in EXPECTED_STACK_INPUTS
        }
    }


def test_validate_stack_artifact_accepts_complete_europe_cache() -> None:
    cache_dirs = validate_stack_artifact(
        artifact=_artifact(),
        stats={"total_tokens": EXPECTED_STACK_TOKENS, "total_elements": EXPECTED_STACK_ELEMENTS},
    )

    assert len(cache_dirs) == 15


def test_validate_stack_artifact_rejects_cross_region_input() -> None:
    artifact = _artifact()
    input_configs = artifact["input_configs"]
    assert isinstance(input_configs, dict)
    input_configs["stack_edu/C"] = {"cache_dir": "gs://marin-us-east5/tokenized/stack_edu_C-hash"}

    with pytest.raises(ValueError, match="historical Europe namespace"):
        validate_stack_artifact(
            artifact=artifact,
            stats={"total_tokens": EXPECTED_STACK_TOKENS, "total_elements": EXPECTED_STACK_ELEMENTS},
        )


def test_validate_stack_artifact_rejects_missing_language() -> None:
    artifact = _artifact()
    input_configs = artifact["input_configs"]
    assert isinstance(input_configs, dict)
    input_configs.pop("stack_edu/SQL")

    with pytest.raises(ValueError, match="15-language"):
        validate_stack_artifact(
            artifact=artifact,
            stats={"total_tokens": EXPECTED_STACK_TOKENS, "total_elements": EXPECTED_STACK_ELEMENTS},
        )


def test_europe_evaluation_cache_outputs_are_region_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")

    steps = evaluation_steps()
    components = [step_to_lm_mixture_component(step, include_raw_paths=False) for step in steps.values()]

    assert len(steps) == 23
    assert all(component.cache_dir.startswith("gs://marin-eu-west4/") for component in components)
    paths = {name: step.path() for name, step in steps.items()}
    assert evaluation_paths_sha256(paths) == evaluation_paths_sha256(dict(reversed(tuple(paths.items()))))
    assert evaluation_paths_sha256(paths) == "4d57689b68ccf4c2e280364aff6b66beafbf3dc9854f82fb6af7a80ca9e025b3"


def test_evaluation_cache_outputs_reject_cross_region_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")

    with pytest.raises(ValueError, match="MARIN_PREFIX"):
        evaluation_steps(region="europe-west4")
