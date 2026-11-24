# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from experiments.post_training.curriculum_rl import launch
from experiments.post_training.skyrl_test_support import native_json
from tests.rl.launch_document_census import runtime_launch as _runtime_launch

runtime_launch = pytest.fixture(_runtime_launch)


@pytest.mark.timeout(180)  # A cold cache installs the independently locked CPU launcher.
@pytest.mark.parametrize(
    ("policy", "preset"),
    [(launch.QWEN_POLICY, launch.SMOKE), (launch.QWEN_POLICY, launch.FULL)]
    + [(launch.SNOWBALL_POLICY, preset) for preset in launch.SCALES.values() if preset.label.startswith("snowball-")],
    ids=lambda value: value.label,
)
def test_curriculum_arms_compose_in_nested_runtime(policy, preset, tmp_path, runtime_launch) -> None:
    # Validate every sampler arm at the executable boundary without allocating GPUs.
    arms = launch.build_arms(
        specs=tuple(launch.ARMS.values()), scale=preset.label, policy=policy, version="2026.10.19.1"
    )
    paths = []
    for name, arm in arms.items():
        config = runtime_launch(arm.rl)
        path = tmp_path / f"{name}.yaml"
        path.write_text(config.launch_config_yaml)
        paths.append(str(path))
    run_native = native_json()
    rows = run_native(Path(__file__).parents[1] / "rl_tests/native_curriculum.py", paths)
    assert rows
    assert [{key: row[key] for key in ("profile", "strategy")} for row in rows] == [
        {"profile": "megatron", "strategy": "megatron"}
    ] * len(arms)
    if preset.label == "snowball-smoke":
        assert [row["allocation"] for row in rows] == [
            {
                "policy_gpus": 32,
                "reference_gpus": 16,
                "allocated_gpus": 56,
                "colocate_policy_ref": False,
            }
        ] * len(arms)
