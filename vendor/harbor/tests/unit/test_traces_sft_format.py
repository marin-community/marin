"""Tests for the ``sft_format`` export path in ``traces_utils``.

When ``sft_format=True``, the exporter emits serve-shaped rows:
- ``role: system`` preserved (not mapped to ``user``)
- structured ``tool_calls`` on assistant turns (OpenAI format)
- ``role: tool`` for tool observations (not ``role: user``)
- column key ``messages`` (not ``conversations``)

The default (``sft_format=False``) path must remain byte-identical.
"""

import json
from pathlib import Path

from harbor.utils import traces_utils as _tu
from harbor.utils.traces_utils import (
    _extract_complete_subagent_conversation,
    _extract_single_episode_conversation,
    _step_tool_calls_to_openai,
    collect_conversations_from_trial,
    export_traces,
    extract_conversations_from_trajectory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RUN_META = {
    "agent_name": "test-agent",
    "model_name": "test-model",
    "model_provider": "test-provider",
    "start_time": "2024-01-01T00:00:00Z",
    "task_name": "test-task",
    "trial_name": "trial-1",
    "run_id": "run-1",
}

_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]


def _write_trajectory(
    trial_dir: Path, steps: list[dict], agent: dict | None = None
) -> None:
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    traj = {
        "agent": agent or {"name": "test-agent", "model_name": "test-model"},
        "steps": steps,
    }
    (agent_dir / "trajectory.json").write_text(json.dumps(traj))


def _run_meta_with_tools():
    meta = dict(_RUN_META)
    meta["tool_definitions"] = _TOOL_DEFS
    return meta


# ---------------------------------------------------------------------------
# _step_tool_calls_to_openai helper
# ---------------------------------------------------------------------------


def test_step_tool_calls_to_openai_basic():
    step_calls = [{"function_name": "bash", "arguments": {"command": "ls -la"}}]
    result = _step_tool_calls_to_openai(step_calls)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "bash"
    assert json.loads(result[0]["function"]["arguments"]) == {"command": "ls -la"}


def test_step_tool_calls_to_openai_multiple():
    step_calls = [
        {"function_name": "bash", "arguments": {"command": "echo hi"}},
        {"function_name": "read", "arguments": {"path": "/tmp/foo"}},
    ]
    result = _step_tool_calls_to_openai(step_calls)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "bash"
    assert result[1]["function"]["name"] == "read"


def test_step_tool_calls_to_openai_empty_args():
    step_calls = [{"function_name": "status", "arguments": {}}]
    result = _step_tool_calls_to_openai(step_calls)
    assert result[0]["function"]["name"] == "status"
    assert json.loads(result[0]["function"]["arguments"]) == {}


def test_step_tool_calls_to_openai_skips_missing_name():
    step_calls = [{"arguments": {"x": 1}}, {"function_name": "ok", "arguments": {}}]
    result = _step_tool_calls_to_openai(step_calls)
    assert len(result) == 1
    assert result[0]["function"]["name"] == "ok"


def test_step_tool_calls_to_openai_preserves_string_args():
    """Arguments that are already strings should pass through."""
    step_calls = [{"function_name": "bash", "arguments": '{"command": "ls"}'}]
    result = _step_tool_calls_to_openai(step_calls)
    assert result[0]["function"]["arguments"] == '{"command": "ls"}'


def test_step_tool_calls_to_openai_empty_list():
    assert _step_tool_calls_to_openai([]) == []


# ---------------------------------------------------------------------------
# _extract_single_episode_conversation — sft_format=True
# ---------------------------------------------------------------------------


def test_sft_preserves_system_role():
    """system steps must produce role: system, not role: user."""
    steps = [
        {"source": "system", "message": "You are a helpful agent."},
        {"source": "agent", "message": "Hello!"},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    msgs = conv["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are a helpful agent."


def test_sft_column_key_is_messages():
    steps = [
        {"source": "system", "message": "Task."},
        {"source": "agent", "message": "Done."},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    assert "messages" in conv
    assert "conversations" not in conv


def test_sft_structured_tool_calls():
    """Assistant tool_calls must be OpenAI-structured, not inline text."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [
                {"function_name": "bash", "arguments": {"command": "ls"}},
            ],
        },
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    asst = conv["messages"][-1]
    assert asst["role"] == "assistant"
    assert len(asst["tool_calls"]) == 1
    assert asst["tool_calls"][0]["type"] == "function"
    assert asst["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls"
    }
    # No inline <tool_call> tags in content
    assert "<tool_call>" not in asst["content"]


def test_sft_multiple_tool_calls_one_step():
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [
                {"function_name": "bash", "arguments": {"command": "echo a"}},
                {"function_name": "read", "arguments": {"path": "/x"}},
                {"function_name": "write", "arguments": {"path": "/y", "content": "z"}},
            ],
        },
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    asst = conv["messages"][-1]
    assert len(asst["tool_calls"]) == 3
    names = [tc["function"]["name"] for tc in asst["tool_calls"]]
    assert names == ["bash", "read", "write"]


def test_sft_no_tool_calls_emits_empty_list():
    steps = [
        {"source": "system", "message": "Task."},
        {"source": "agent", "message": "I'm done."},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    asst = conv["messages"][-1]
    assert asst["tool_calls"] == []


def test_sft_observation_is_tool_role():
    """Tool observations must be role: tool, not role: user."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "ls"}}],
            "observation": {"results": [{"content": "file1\nfile2"}]},
        },
        {"source": "agent", "message": "Done."},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    msgs = conv["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "file1\nfile2"


def test_sft_does_not_embed_tools_in_system():
    """No <tools> block should appear in the system message."""
    steps = [
        {"source": "system", "message": "You are an agent."},
        {"source": "agent", "message": "Hi"},
    ]
    conv = _extract_single_episode_conversation(
        steps, 0, _run_meta_with_tools(), sft_format=True
    )
    sys_msg = conv["messages"][0]
    assert "<tools>" not in sys_msg["content"]
    assert sys_msg["content"] == "You are an agent."


def test_sft_preserves_reasoning_content():
    """<think> tags from reasoning_content must be preserved in assistant content."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "The answer is 42.",
            "reasoning_content": "Let me think...",
            "tool_calls": [],
        },
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    asst = conv["messages"][-1]
    assert "<think>Let me think...</think>" in asst["content"]
    assert "The answer is 42." in asst["content"]


def test_sft_fixes_orphaned_think_tag():
    """An orphaned </think> (missing opener) must get <think> prepended."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "reasoning here</think>\nThe answer.",
        },
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    asst = conv["messages"][-1]
    assert asst["content"].count("<think>") == 1
    assert asst["content"].startswith("<think>")


def test_sft_multi_turn_conversation():
    """A multi-turn trajectory produces system → assistant → tool → assistant."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "pwd"}}],
            "observation": {"results": [{"content": "/home/user"}]},
        },
        {"source": "agent", "message": "The directory is /home/user."},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    msgs = conv["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "bash"
    assert msgs[2]["content"] == "/home/user"
    assert msgs[3]["content"] == "The directory is /home/user."


def test_sft_user_step_preserved():
    """A 'user' source step should still be role: user in sft_format."""
    steps = [
        {"source": "system", "message": "Task."},
        {"source": "user", "message": "Please help me."},
        {"source": "agent", "message": "Sure!"},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_sft_last_agent_observation_excluded():
    """The last agent step's observation belongs to the next episode — it must be excluded."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "Done.",
            "observation": {"results": [{"content": "obs"}]},
        },
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    msgs = conv["messages"]
    # Only system + assistant — no tool message (obs belongs to next episode)
    assert len(msgs) == 2
    assert [m["role"] for m in msgs] == ["system", "assistant"]


# ---------------------------------------------------------------------------
# _extract_single_episode_conversation — default path unchanged
# ---------------------------------------------------------------------------


def test_default_system_maps_to_user():
    steps = [
        {"source": "system", "message": "Task instructions."},
        {"source": "agent", "message": "Done."},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META)
    assert conv["conversations"][0]["role"] == "user"
    assert conv["conversations"][0]["content"] == "Task instructions."


def test_default_inline_tool_calls():
    """Default path must still inline <tool_call> tags, not structured tool_calls."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "ls"}}],
        },
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META)
    asst = conv["conversations"][-1]
    assert "tool_calls" not in asst  # no structured field
    assert "<tool_call>" in asst["content"]
    assert '"name": "bash"' in asst["content"]


def test_default_observation_is_user_role():
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"function_name": "bash", "arguments": {}}],
            "observation": {"results": [{"content": "output"}]},
        },
        {"source": "agent", "message": "Done."},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META)
    obs_msgs = [
        m
        for m in conv["conversations"]
        if m["role"] == "user" and m["content"] == "output"
    ]
    assert len(obs_msgs) == 1


def test_default_embeds_tools_in_system():
    conv = _extract_single_episode_conversation(
        [
            {"source": "system", "message": "You are an agent."},
            {"source": "agent", "message": "hi"},
        ],
        0,
        _run_meta_with_tools(),
    )
    sys_msg = conv["conversations"][0]
    assert "<tools>" in sys_msg["content"]


def test_default_column_key_is_conversations():
    conv = _extract_single_episode_conversation(
        [{"source": "system", "message": "T."}, {"source": "agent", "message": "a"}],
        0,
        _RUN_META,
    )
    assert "conversations" in conv
    assert "messages" not in conv


# ---------------------------------------------------------------------------
# extract_conversations_from_trajectory — sft_format threading
# ---------------------------------------------------------------------------


def test_extract_from_trajectory_sft(tmp_path):
    traj_file = tmp_path / "trajectory.json"
    traj = {
        "agent": {"name": "test-agent", "model_name": "test-model"},
        "steps": [
            {"source": "system", "message": "Task."},
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"function_name": "bash", "arguments": {"command": "ls"}}
                ],
                "observation": {"results": [{"content": "a\nb"}]},
            },
            {"source": "agent", "message": "Done."},
        ],
    }
    traj_file.write_text(json.dumps(traj))

    convs = extract_conversations_from_trajectory(traj_file, _RUN_META, sft_format=True)
    assert len(convs) == 2  # 2 agent steps → 2 episodes
    # episode 0: system + assistant (no tool obs — belongs to next episode)
    assert all(m["role"] != "tool" for m in convs[0]["messages"])
    # episode 1: system + assistant + tool obs + assistant
    assert any(m["role"] == "tool" for m in convs[1]["messages"])
    for c in convs:
        assert c["messages"][0]["role"] == "system"


def test_extract_from_trajectory_sft_multi_episode(tmp_path):
    """Each agent step marks the end of an episode; multi-agent → multi-episode."""
    traj_file = tmp_path / "trajectory.json"
    traj = {
        "agent": {"name": "test-agent", "model_name": "test-model"},
        "steps": [
            {"source": "system", "message": "Task."},
            {"source": "agent", "message": "Turn 1."},
            {"source": "agent", "message": "Turn 2."},
            {"source": "agent", "message": "Turn 3."},
        ],
    }
    traj_file.write_text(json.dumps(traj))

    convs = extract_conversations_from_trajectory(traj_file, _RUN_META, sft_format=True)
    assert len(convs) == 3  # 3 agent steps → 3 episodes
    for c in convs:
        assert "messages" in c
        assert "conversations" not in c
        assert c["messages"][0]["role"] == "system"


def test_extract_from_trajectory_default_unchanged(tmp_path):
    """Default path still produces conversations, not messages."""
    traj_file = tmp_path / "trajectory.json"
    traj = {
        "agent": {"name": "test-agent", "model_name": "test-model"},
        "steps": [
            {"source": "system", "message": "Task."},
            {"source": "agent", "message": "Done."},
        ],
    }
    traj_file.write_text(json.dumps(traj))

    convs = extract_conversations_from_trajectory(traj_file, _RUN_META)
    assert len(convs) == 1
    assert "conversations" in convs[0]
    assert "messages" not in convs[0]


# ---------------------------------------------------------------------------
# collect_conversations_from_trial — sft_format
# ---------------------------------------------------------------------------


def test_collect_from_trial_sft(tmp_path):
    trial_dir = tmp_path / "trial-1"
    trial_dir.mkdir()
    _write_trajectory(
        trial_dir,
        [
            {"source": "system", "message": "Task."},
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"function_name": "bash", "arguments": {"command": "ls"}}
                ],
                "observation": {"results": [{"content": "file.txt"}]},
            },
            {"source": "agent", "message": "Done."},
        ],
    )
    convs = collect_conversations_from_trial(trial_dir, _RUN_META, sft_format=True)
    assert len(convs) >= 1
    for c in convs:
        assert "messages" in c
        assert "conversations" not in c
        assert c["messages"][0]["role"] == "system"


# ---------------------------------------------------------------------------
# Subagent traces — _extract_complete_subagent_conversation
# ---------------------------------------------------------------------------


def test_subagent_sft_format(tmp_path):
    traj_file = tmp_path / "trajectory.subagent-1-summary.json"
    traj = {
        "agent": {"name": "subagent", "model_name": "test-model"},
        "steps": [
            {"source": "system", "message": "Summarize the context."},
            {
                "source": "agent",
                "message": "",
                "tool_calls": [{"function_name": "read", "arguments": {"path": "/x"}}],
                "observation": {"results": [{"content": "data"}]},
            },
            {"source": "agent", "message": "Here is the summary."},
        ],
    }
    traj_file.write_text(json.dumps(traj))

    conv = _extract_complete_subagent_conversation(
        traj_file, _RUN_META, sft_format=True
    )
    assert conv is not None
    assert "messages" in conv
    assert "conversations" not in conv
    msgs = conv["messages"]
    assert msgs[0]["role"] == "system"
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert len(asst_msgs) == 2
    assert len(asst_msgs[0]["tool_calls"]) == 1
    assert asst_msgs[0]["tool_calls"][0]["function"]["name"] == "read"
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1


def test_subagent_default_unchanged(tmp_path):
    traj_file = tmp_path / "trajectory.subagent-1-summary.json"
    traj = {
        "agent": {"name": "subagent", "model_name": "test-model"},
        "steps": [
            {"source": "system", "message": "Task."},
            {"source": "agent", "message": "Done."},
        ],
    }
    traj_file.write_text(json.dumps(traj))

    conv = _extract_complete_subagent_conversation(traj_file, _RUN_META)
    assert conv is not None
    assert "conversations" in conv
    assert "messages" not in conv


# ---------------------------------------------------------------------------
# End-to-end: export_traces with sft_format=True
# ---------------------------------------------------------------------------


def _make_job(tmp_path, steps=None):
    """Create a job dir with one trial + result.json for export_traces."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = job_dir / "trial-x"
    trial_dir.mkdir()
    _write_trajectory(
        trial_dir,
        steps
        or [
            {"source": "system", "message": "Task."},
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"function_name": "bash", "arguments": {"command": "ls"}}
                ],
                "observation": {"results": [{"content": "out"}]},
            },
            {"source": "agent", "message": "Done."},
        ],
        agent={"name": "terminus-2", "model_name": "test-model"},
    )
    result_payload = {
        "config": {"agent": {"name": "terminus-2", "model_name": "test-model"}},
        "agent_info": {"name": "terminus-2", "model_info": {"name": "test-model"}},
        "started_at": "2024-01-01T00:00:00Z",
        "task_name": "test-task",
        "trial_name": "trial-x",
        "run_id": "test-run",
    }
    (job_dir / "result.json").write_text(json.dumps(result_payload))
    _tu._RESULT_JSON_CACHE.clear()
    return job_dir


def test_export_traces_sft_format(tmp_path):
    job_dir = _make_job(tmp_path)
    ds = export_traces(
        job_dir,
        recursive=True,
        use_rich_progress=False,
        sft_format=True,
        verbose=False,
    )
    assert len(ds) == 2  # 2 agent steps → 2 episodes
    assert "messages" in ds.column_names
    assert "conversations" not in ds.column_names
    for row in ds:
        msgs = row["messages"]
        assert msgs[0]["role"] == "system"
        assert any(m["role"] == "assistant" for m in msgs)
    # episode-1 should have a tool observation
    ep1 = [r for r in ds if r["episode"] == "episode-1"]
    assert ep1 and any(m["role"] == "tool" for m in ep1[0]["messages"])


def test_export_traces_default_unchanged(tmp_path):
    job_dir = _make_job(tmp_path)
    ds = export_traces(
        job_dir,
        recursive=True,
        use_rich_progress=False,
        verbose=False,
    )
    assert "conversations" in ds.column_names
    assert "messages" not in ds.column_names


def test_export_traces_sft_with_literal_tokens(tmp_path):
    """sft_format and include_literal_tokens must work together."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "Done.",
            "metrics": {
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4, 5],
                "logprobs": [-0.1, -0.2],
            },
        },
    ]
    job_dir = _make_job(tmp_path, steps=steps)
    ds = export_traces(
        job_dir,
        recursive=True,
        use_rich_progress=False,
        sft_format=True,
        include_literal_tokens=True,
        verbose=False,
    )
    assert len(ds) == 1  # 1 agent step → 1 episode
    assert "messages" in ds.column_names
    assert "prompt_token_ids" in ds.column_names
    assert ds[0]["messages"][0]["role"] == "system"


def test_export_traces_sft_no_subagents(tmp_path):
    """sft_format with export_subagents=False."""
    job_dir = _make_job(tmp_path)
    ds = export_traces(
        job_dir,
        recursive=True,
        use_rich_progress=False,
        sft_format=True,
        export_subagents=False,
        verbose=False,
    )
    assert "messages" in ds.column_names


# ---------------------------------------------------------------------------
# Round-trip: SFT format messages are renderable by a tools-aware chat template
# ---------------------------------------------------------------------------


def test_sft_messages_have_consistent_schema():
    """Every assistant message must have a tool_calls key (even if empty)."""
    steps = [
        {"source": "system", "message": "Task."},
        {"source": "agent", "message": "First response."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "ls"}}],
            "observation": {"results": [{"content": "out"}]},
        },
        {"source": "agent", "message": "Final."},
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    asst_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
    assert len(asst_msgs) == 3
    for m in asst_msgs:
        assert "tool_calls" in m
        assert isinstance(m["tool_calls"], list)
    assert len(asst_msgs[0]["tool_calls"]) == 0
    assert len(asst_msgs[1]["tool_calls"]) == 1
    assert len(asst_msgs[2]["tool_calls"]) == 0


def test_sft_tool_call_arguments_is_json_string():
    """All tool_call.function.arguments must be a JSON string (not a dict)."""
    steps = [
        {"source": "system", "message": "Task."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [
                {
                    "function_name": "bash",
                    "arguments": {"command": "ls", "timeout": 30},
                },
                {"function_name": "read", "arguments": {"path": "/tmp/x"}},
            ],
        },
    ]
    conv = _extract_single_episode_conversation(steps, 0, _RUN_META, sft_format=True)
    asst = conv["messages"][-1]
    for tc in asst["tool_calls"]:
        assert isinstance(tc["function"]["arguments"], str)
        parsed = json.loads(tc["function"]["arguments"])
        assert isinstance(parsed, dict)
