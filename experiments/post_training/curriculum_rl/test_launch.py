# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
from marin.execution.lazy import StepContext

from experiments.post_training.curriculum_rl import launch
from experiments.post_training.snowball_tests.fixtures import native_json

COMPOSE_LAUNCHES = """
import json
import sys
from pathlib import Path
from cloud.iris.launch_config import load_launch_config

rows = []
for path in sys.argv[1:]:
    config = load_launch_config(Path(path))
    placement = config.skyrl.trainer.placement
    allocation = config.iris.allocation
    rows.append({
        "profile": config.runtime.profile,
        "strategy": config.skyrl.trainer.strategy,
        "allocation": {
            "policy_gpus": placement.policy_num_nodes * placement.policy_num_gpus_per_node,
            "reference_gpus": placement.ref_num_nodes * placement.ref_num_gpus_per_node,
            "allocated_gpus": allocation.num_nodes * allocation.gpus_per_node,
            "colocate_policy_ref": placement.colocate_policy_ref,
        },
    })
print(json.dumps(rows))
"""


@pytest.mark.timeout(180)  # A cold cache installs the independently locked CPU launcher.
@pytest.mark.parametrize(
    ("policy", "preset"),
    [(launch.QWEN_POLICY, launch.SMOKE), (launch.QWEN_POLICY, launch.FULL)]
    + [(launch.SNOWBALL_POLICY, preset) for preset in launch.SCALES.values() if preset.label.startswith("snowball-")],
    ids=lambda value: value.label,
)
def test_curriculum_arms_compose_in_nested_runtime(policy, preset, tmp_path) -> None:
    # Validate every sampler arm at the executable boundary without allocating GPUs.
    arms = launch.build_arms(
        specs=tuple(launch.ARMS.values()), scale=preset.label, policy=policy, version="2026.10.19.1"
    )
    paths = []
    for name, arm in arms.items():
        config = arm.rl.build_config(StepContext.for_fingerprint(arm.rl.runtime_args, arm.rl.deps))
        path = tmp_path / f"{name}.yaml"
        path.write_text(config.launch_config_yaml)
        paths.append(str(path))
    run_native = native_json()
    rows = run_native(COMPOSE_LAUNCHES, paths)
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
