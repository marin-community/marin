"""Pre-action DAgger plans and actor attribution are fail-closed."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.pi import Pi
from harbor.models.trial.config import AgentConfig
from harbor_config.models.trial.pi import PiAgentConfig, PiDaggerPlan
from harbor.models.agent.context import AgentContext
from tests.unit.agents.installed.pi_trajectory_fixture import fake_pi_trajectory


def plan(protocol="turn_mixture", executors=None):
    return {
        "protocol": protocol,
        "teacher_model": "teacher",
        "teacher_api_base": "http://127.0.0.1:8001/v1",
        "executors": executors if executors is not None else ["student", "teacher"],
        "trial_id": "trial",
        "policy_step": 4,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(executors=[]),
        lambda p: p.update(executors=["student", "unknown"]),
        lambda p: p.update(protocol="other"),
        lambda p: p.update(policy_step=-1),
        lambda p: p.update(teacher_api_base="file:///tmp/teacher"),
        lambda p: p.update(
            protocol="prefix_takeover", executors=["teacher", "student"]
        ),
    ],
)
def test_rejects_invalid_plan(mutate):
    value = plan()
    mutate(value)
    with pytest.raises(ValueError):
        PiDaggerPlan.parse(value)


def test_pi_capability_is_typed_and_checked_before_agent_creation(tmp_path):
    config = AgentConfig.model_validate(
        {
            "name": "pi",
            "model_name": "hosted_vllm/student",
            "kwargs": {
                "api_base": "http://localhost:8000/v1",
                "thinking_format": "qwen-chat-template",
            },
            "pi": {"mode": "dagger", "plan": plan()},
        }
    )
    assert config.pi is not None
    assert config.pi.plan is not None
    assert config.pi.plan.executors == ("student", "teacher")
    assert AgentConfig.model_validate_json(config.model_dump_json()).pi == config.pi
    with pytest.raises(FrozenInstanceError):
        config.pi.plan.policy_step = 5
    with pytest.raises(ValueError, match="unsupported Pi DAgger protocol"):
        PiDaggerPlan(
            "unknown", "teacher", "http://teacher:8001/v1", ("student",), "trial", 2
        )
    with pytest.raises(ValidationError):
        config.pi.mode = "literal"
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(agent, Pi)
    assert agent._dagger_plan == config.pi.plan
    assert agent._collect_rollout_details

    for bad in (
        {"name": "opencode", "pi": {"mode": "literal"}},
        {"name": "pi", "pi": {"mode": "dagger"}},
        {"name": "pi", "pi": {"mode": "literal", "plan": plan()}},
        {
            "name": "pi",
            "pi": {"mode": "dagger", "plan": plan(), "extension_path": "/tmp/code.js"},
        },
        {"name": "pi", "kwargs": {"dagger_plan": plan()}},
        {
            "name": "pi",
            "pi": {"mode": "literal"},
            "kwargs": {"collect_rollout_details": True},
        },
    ):
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(bad)


def test_dagger_requires_model_endpoint(tmp_path):
    with pytest.raises(ValueError, match="requires api_base"):
        Pi(
            logs_dir=tmp_path,
            model_name="served/student",
            pi_config=PiAgentConfig(mode="dagger", plan=PiDaggerPlan.parse(plan())),
        )


def test_teacher_label_is_private_and_attributed(tmp_path):
    agent = Pi(
        logs_dir=tmp_path,
        model_name="served/student",
        api_base="http://localhost:8000/v1",
        thinking_format="qwen-chat-template",
        pi_config=PiAgentConfig(mode="dagger", plan=PiDaggerPlan.parse(plan())),
    )
    initial = [{"role": "user", "content": "solve"}]
    student = {"role": "assistant", "content": "student"}
    entries = [
        {
            "request_id": "student-action",
            "status_code": 200,
            "actor": "student",
            "request": {"model": "student", "messages": initial},
            "literal": {
                "prompt_token_ids": [1],
                "completion_token_ids": [2],
                "logprobs": [-0.1],
                "assistant_message": student,
            },
        },
        {
            "request_id": "teacher-action",
            "status_code": 200,
            "actor": "teacher",
            "request": {"model": "teacher", "messages": [*initial, student]},
            "literal": {
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3],
                "logprobs": None,
                "assistant_message": {"role": "assistant", "content": "teacher second"},
            },
        },
    ]
    private_label = {
        "request_id": "teacher-private",
        "status_code": 200,
        "request": {"model": "teacher", "messages": initial},
        "literal": {
            "prompt_token_ids": [1],
            "completion_token_ids": [4],
            "logprobs": None,
            "assistant_message": {"role": "assistant", "content": "teacher first"},
        },
    }
    turns = [
        {
            "turn": 0,
            "actor": "student",
            "teacher_label": private_label,
            "executed_request_id": "student-action",
        },
        {
            "turn": 1,
            "actor": "teacher",
            "teacher_label": {
                key: value for key, value in entries[1].items() if key != "actor"
            },
            "executed_request_id": "teacher-action",
        },
    ]
    agent._in_memory_entries = deepcopy(entries)
    agent._dagger_turns = deepcopy(turns)
    agent._in_memory_trajectory = fake_pi_trajectory(
        entries, dagger_turns=turns, trial_id="trial", policy_step=4
    )
    agent._in_memory_pi_output = "\n".join(
        [
            '{"type":"message_end","message":{"role":"assistant"}}',
            '{"type":"message_end","message":{"role":"assistant"}}',
        ]
    )
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert (
        context.metadata["pi_dagger_turns"][0]["teacher_label"]["literal"][
            "assistant_message"
        ]
        != student
    )
    assert [turn["actor"] for turn in context.metadata["pi_dagger_turns"]] == [
        "student",
        "teacher",
    ]
    assert [item["content"] for item in context.metadata["all_messages"]] == [
        "solve",
        "student",
        "teacher second",
    ]
    assert context.rollout_details[0]["logprobs"] == [[-0.1], []]
    assert (
        context.metadata["pi_hydrated_trajectory"]["turns"][0]["teacher_label"][
            "logprobs"
        ]
        is None
    )
    assert not list(tmp_path.iterdir())

    for damaged in (
        lambda: agent._dagger_turns[0].update(actor="teacher"),
        lambda: agent._dagger_turns[0]["teacher_label"]["request"].update(messages=[]),
        lambda: agent._in_memory_entries[1].update(actor="student"),
    ):
        agent._in_memory_entries = deepcopy(entries)
        agent._dagger_turns = deepcopy(turns)
        damaged()
        with pytest.raises(ValueError):
            agent._validate_dagger_turns(agent._in_memory_entries)
