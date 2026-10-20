"""Teacher-only Pi trials keep execution authorship separate from student behavior."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.pi import Pi
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig
from harbor_config.models.trial.pi import PiAgentConfig
from tests.unit.agents.installed.pi_trajectory_fixture import fake_pi_trajectory


def teacher_call() -> dict:
    return {
        "request_id": "teacher-1",
        "status_code": 200,
        "actor": "teacher",
        "request": {
            "model": "teacher-revision",
            "messages": [{"role": "user", "content": "solve"}],
        },
        "literal": {
            "prompt_token_ids": [1],
            "completion_token_ids": [2, 3],
            "logprobs": None,
            "assistant_message": {"role": "assistant", "content": "answer"},
        },
    }


def test_teacher_trace_capability_and_evidence(tmp_path):
    config = AgentConfig.model_validate(
        {
            "name": "pi",
            "model_name": "hosted_vllm/teacher-revision",
            "kwargs": {
                "api_base": "http://localhost:8001/v1",
                "thinking_format": "qwen-chat-template",
            },
            "pi": {"mode": "teacher_trace"},
        }
    )
    assert AgentConfig.model_validate_json(config.model_dump_json()).pi == config.pi
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(agent, Pi)
    assert agent._capture_actor == "teacher"
    assert agent._dagger_plan is None
    assert agent._collect_rollout_details

    agent._in_memory_entries = [teacher_call()]
    agent._in_memory_trajectory = fake_pi_trajectory(
        agent._in_memory_entries, trial_id=agent._trial_identity
    )
    agent._in_memory_pi_output = '{"type":"message_end","message":{"role":"assistant"}}'
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.metadata["pi_executed_actors"] == ["teacher"]
    assert context.metadata["all_messages"] == [
        {"role": "user", "content": "solve"},
        {"role": "assistant", "content": "answer"},
    ]
    assert context.rollout_details[0]["completion_token_ids"] == [[2, 3]]
    assert context.rollout_details[0]["logprobs"] == [[]]
    assert (
        context.metadata["pi_hydrated_trajectory"]["turns"][0]["executed"]["logprobs"]
        is None
    )
    assert not list(tmp_path.iterdir())

    for damaged in ("student", None):
        agent._in_memory_entries = [deepcopy(teacher_call())]
        if damaged is None:
            del agent._in_memory_entries[0]["actor"]
        else:
            agent._in_memory_entries[0]["actor"] = damaged
        with pytest.raises(ValueError, match="incomplete literal evidence"):
            agent.populate_context_post_run(AgentContext())


def test_teacher_trace_rejects_dagger_plan(tmp_path):
    with pytest.raises(ValidationError, match="other modes forbid one"):
        PiAgentConfig(
            mode="teacher_trace",
            plan={
                "protocol": "turn_mixture",
                "teacher_model": "t",
                "teacher_api_base": "http://teacher:8001/v1",
                "executors": ["teacher"],
                "trial_id": "trial",
                "policy_step": 0,
            },
        )
    with pytest.raises(ValueError, match="requires api_base"):
        Pi(
            logs_dir=tmp_path,
            model_name="served/teacher",
            pi_config=PiAgentConfig(mode="teacher_trace"),
        )
