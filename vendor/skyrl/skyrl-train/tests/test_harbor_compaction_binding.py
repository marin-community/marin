"""Behavior tests for task-scoped Pi compaction reserve bindings."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from harbor.agents.factory import AgentFactory
from harbor_config.models.trial.config import TrialConfig
from marinskyrl.harbor_environments import HarborTaskBinding, apply_task_binding
from pydantic import ValidationError


def _pi_trial_config(tmp_path: Path) -> TrialConfig:
    return TrialConfig.model_validate(
        {
            "task": {"path": str(tmp_path / "task")},
            "agent": {
                "name": "pi",
                "model_name": "open-athena/Snowball-1005",
                "kwargs": {
                    "version": "0.87.0",
                    "thinking": "high",
                    "model_info": {
                        "max_input_tokens": 102400,
                        "max_output_tokens": 8192,
                    },
                    "compaction_reserve_tokens": 8192,
                },
            },
        }
    )


@pytest.mark.parametrize(
    ("reserve_tokens", "expected_keep_recent"),
    [
        (8192, 47104),
        (77824, 12288),
    ],
)
def test_task_binding_overlays_pi_compaction_settings_preserving_window(
    tmp_path: Path,
    reserve_tokens: int,
    expected_keep_recent: int,
) -> None:
    base = _pi_trial_config(tmp_path)
    bound = apply_task_binding(
        base,
        HarborTaskBinding(compaction_reserve_tokens=reserve_tokens),
    )

    assert base.agent.kwargs["compaction_reserve_tokens"] == 8192
    assert bound.agent.kwargs["compaction_reserve_tokens"] == reserve_tokens
    assert bound.agent.kwargs["model_info"] == {
        "max_input_tokens": 102400,
        "max_output_tokens": 8192,
    }

    agent = AgentFactory.create_agent_from_config(bound.agent, tmp_path / "logs")
    cmd = agent._build_register_compaction_command()
    assert cmd is not None
    home = tmp_path / f"home-{reserve_tokens}"
    home.mkdir()
    subprocess.run(["bash", "-c", cmd], check=True, env={"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/local/bin:/home/karan/.nvm/versions/node/v22.23.4/bin"})
    settings = json.loads((home / ".pi/agent/settings.json").read_text())
    assert settings["compaction"] == {
        "enabled": True,
        "reserveTokens": reserve_tokens,
        "keepRecentTokens": expected_keep_recent,
    }


def test_task_binding_none_leaves_compaction_reserve_unchanged(tmp_path: Path) -> None:
    base = _pi_trial_config(tmp_path)
    bound = apply_task_binding(base, HarborTaskBinding())
    assert bound.agent.kwargs["compaction_reserve_tokens"] == 8192


@pytest.mark.parametrize("bad_value", [True, False, 0, -1, "77824"])
def test_task_binding_rejects_non_positive_or_non_strict_int(bad_value: object) -> None:
    with pytest.raises(ValidationError):
        HarborTaskBinding(compaction_reserve_tokens=bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("out_of_window", [8191, 102400, 120000])
def test_pi_rejects_bound_compaction_reserve_outside_output_and_window_bounds(
    tmp_path: Path,
    out_of_window: int,
) -> None:
    bound = apply_task_binding(
        _pi_trial_config(tmp_path),
        HarborTaskBinding(compaction_reserve_tokens=out_of_window),
    )
    with pytest.raises(ValueError, match="compaction_reserve_tokens"):
        AgentFactory.create_agent_from_config(bound.agent, tmp_path / "logs")
