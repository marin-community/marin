# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math

from experiments.domain_phase_mix.agentic_coding_eval_dataset import (
    coderforge_outcome,
    openhands_outcome,
    parse_messages,
    render_assistant_action_messages,
    truncate_utf8,
)
from experiments.domain_phase_mix.launch_300m_agentic_coding_bpb_evals import _metrics_from_summary_path


def test_coderforge_outcome_from_reward() -> None:
    assert coderforge_outcome(1.0) == "success"
    assert coderforge_outcome("1.5") == "success"
    assert coderforge_outcome(0.0) == "fail"


def test_openhands_outcome_from_resolved() -> None:
    assert openhands_outcome(1) == "success"
    assert openhands_outcome("2") == "success"
    assert openhands_outcome(0) == "fail"


def test_render_assistant_action_excludes_non_assistant_text() -> None:
    messages = [
        {"role": "system", "content": "hidden policy"},
        {"role": "user", "content": "fix issue text"},
        {
            "role": "assistant",
            "content": "I will inspect the repo.",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"cmd": "rg failing_test"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "very long observation"},
    ]

    rendered = render_assistant_action_messages(messages)

    assert "I will inspect the repo." in rendered
    assert "bash" in rendered
    assert "rg failing_test" in rendered
    assert "hidden policy" not in rendered
    assert "fix issue text" not in rendered
    assert "very long observation" not in rendered


def test_parse_messages_accepts_json_strings() -> None:
    messages = parse_messages('[{"role": "assistant", "content": "hello"}]')

    assert messages == [{"role": "assistant", "content": "hello"}]


def test_truncate_utf8_preserves_valid_encoding() -> None:
    text = "abc🙂def"
    truncated = truncate_utf8(text, 6)

    assert len(truncated.encode("utf-8")) <= 6
    assert truncated.encode("utf-8").decode("utf-8") == truncated


def test_metrics_from_summary_computes_success_and_failed_macros(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    rows = (
        ("agentic_coding/coderforge_swe_rebench_success", 1.0),
        ("agentic_coding/coderforge_swe_smith_success", 2.0),
        ("agentic_coding/coderforge_r2e_gym_success", 3.0),
        ("agentic_coding/openhands_swe_rebench_success", 4.0),
        ("agentic_coding/coderforge_swe_rebench_fail", 5.0),
        ("agentic_coding/coderforge_swe_smith_fail", 6.0),
        ("agentic_coding/coderforge_r2e_gym_fail", 7.0),
        ("agentic_coding/openhands_swe_rebench_fail", 8.0),
    )
    summary_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": name,
                        "documents": 512,
                        "bytes": 10,
                        "bits": bpb * 10,
                        "bpb": bpb,
                    }
                    for name, bpb in rows
                ]
            }
        ),
        encoding="utf-8",
    )

    metrics, error = _metrics_from_summary_path(str(summary_path))

    assert error == ""
    assert math.isclose(metrics["eval/agentic_coding/success_macro_bpb"], 2.5)
    assert math.isclose(metrics["eval/agentic_coding/coderforge_success_macro_bpb"], 2.0)
    assert math.isclose(metrics["eval/agentic_coding/failed_macro_bpb"], 6.5)
    assert math.isclose(metrics["eval/agentic_coding/success_minus_failed_bpb"], -4.0)
