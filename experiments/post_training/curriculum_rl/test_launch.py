# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training import rl_gate as gate
from experiments.post_training.curriculum_rl import launch
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
    rows = gate.compose_configs(paths)
    assert len(rows) == len(arms)
    for row in rows:
        trainer = row["skyrl"]["trainer"]
        assert (row["runtime"]["profile"], trainer["strategy"]) == ("megatron", "megatron")
        if preset.label == "snowball-smoke":
            placement = trainer["placement"]
            allocation = row["iris"]["allocation"]
            assert placement["policy_num_nodes"] * placement["policy_num_gpus_per_node"] == 32
            assert placement["ref_num_nodes"] * placement["ref_num_gpus_per_node"] == 16
            assert allocation["num_nodes"] * allocation["gpus_per_node"] == 56
            assert not placement["colocate_policy_ref"]
