"""A private Pi teacher critique is not an executed teacher action or student token."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.pi import Pi
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig
from harbor_config.models.trial.pi import PiAgentConfig, PiCritiquePlan
from tests.unit.agents.installed.pi_trajectory_fixture import fake_pi_trajectory


def sample() -> tuple[list[dict], dict]:
    prompt = [{"role": "user", "content": [{"type": "text", "text": "Solve"}]}]
    action = {"role": "assistant", "content": "Try X"}
    feedback = "Teacher feedback on your previous attempt (not an executed action):\nTry Y\nRevise."
    entries = [
        {
            "request_id": "student-1",
            "status_code": 200,
            "actor": "student",
            "request": {"model": "student", "messages": prompt},
            "literal": {
                "prompt_token_ids": [1],
                "completion_token_ids": [2],
                "logprobs": [-0.1],
                "assistant_message": action,
            },
        },
        {
            "request_id": "revision-1",
            "status_code": 200,
            "actor": "revision",
            "request": {
                "model": "student",
                "messages": [
                    *prompt,
                    action,
                    {"role": "user", "content": [{"type": "text", "text": feedback}]},
                ],
            },
            "literal": {
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4],
                "logprobs": [-0.2],
                "assistant_message": {"role": "assistant", "content": "Y"},
            },
        },
    ]
    critique = {
        "trial_id": "trial-1",
        "teacher_model": "teacher",
        "student_request_id": "student-1",
        "teacher_request_id": "teacher-1",
        "teacher_request": {
            "model": "teacher",
            "stream": False,
            "messages": [
                *prompt,
                action,
                {"role": "user", "content": "Critique this attempt"},
            ],
        },
        "teacher_response": {"role": "assistant", "content": "Try Y"},
        "feedback": feedback,
    }
    return entries, critique


def agent(tmp_path) -> Pi:
    config = AgentConfig.model_validate(
        {
            "name": "pi",
            "model_name": "hosted_vllm/student",
            "kwargs": {
                "api_base": "http://student:8080/v1",
                "thinking_format": "qwen-chat-template",
            },
            "pi": {
                "mode": "critique_revision",
                "critique": {
                    "teacher_model": "teacher",
                    "teacher_api_base": "http://teacher:8080/v1",
                    "principle": "Be precise",
                    "trial_id": "trial-1",
                },
            },
        }
    )
    assert AgentConfig.model_validate_json(config.model_dump_json()).pi == config.pi
    instance = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(instance, Pi)
    return instance


def test_critique_actors_and_private_feedback(tmp_path):
    instance = agent(tmp_path)
    entries, critique = sample()
    instance._in_memory_entries = entries
    instance._in_memory_trajectory = fake_pi_trajectory(entries, critique=critique)
    instance._critique_evidence = critique
    instance._in_memory_pi_output = (
        '{"type":"message_end","message":{"role":"assistant"}}\n' * 2
    )
    context = AgentContext()
    instance.populate_context_post_run(context)
    assert context.metadata["pi_executed_actors"] == ["student", "revision"]
    assert context.metadata["pi_critique"]["teacher_request_id"] == "teacher-1"
    assert "Critique this attempt" not in str(context.metadata["all_messages"])
    assert len(context.rollout_details[0]["completion_token_ids"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_critique",
        "wrong_actor",
        "absent_feedback",
        "private_instruction_leaked",
        "revision_tool",
        "extra_turn",
        "wrong_teacher",
        "wrong_trial",
        "missing_response",
    ],
)
def test_critique_rejects_unattributed_or_incomplete_evidence(tmp_path, mutation):
    instance = agent(tmp_path)
    entries, critique = sample()
    entries, critique = deepcopy(entries), deepcopy(critique)
    if mutation == "missing_critique":
        critique = None
    elif mutation == "wrong_actor":
        entries[1]["actor"] = "teacher"
    elif mutation == "absent_feedback":
        entries[1]["request"]["messages"][-1]["content"][0]["text"] = "anything else"
    elif mutation == "private_instruction_leaked":
        entries[1]["request"]["messages"].insert(
            -1, critique["teacher_request"]["messages"][-1]
        )
    elif mutation == "revision_tool":
        entries[1]["literal"]["assistant_message"]["tool_calls"] = [
            {
                "id": "bad",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ]
    elif mutation == "extra_turn":
        entries.append(deepcopy(entries[1]))
    elif mutation == "wrong_teacher":
        critique["teacher_model"] = "unknown"
    elif mutation == "wrong_trial":
        critique["trial_id"] = "other"
    elif mutation == "missing_response":
        critique["teacher_response"]["content"] = ""
    instance._in_memory_entries = entries
    instance._critique_evidence = critique
    with pytest.raises(ValueError, match="Pi critique"):
        instance._validate_critique(entries)


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "critique_revision"},
        {
            "mode": "literal",
            "critique": {
                "teacher_model": "t",
                "teacher_api_base": "http://host/v1",
                "principle": "p",
                "trial_id": "id",
            },
        },
        {
            "mode": "critique_revision",
            "critique": {
                "teacher_model": "t",
                "teacher_api_base": "http://host/v1",
                "principle": "p",
                "trial_id": "id",
                "extension": "x.js",
            },
        },
        {
            "mode": "critique_revision",
            "critique": {
                "teacher_model": "t",
                "teacher_api_base": "file:///v1",
                "principle": "p",
                "trial_id": "id",
            },
        },
    ],
)
def test_critique_plan_fails_closed(config):
    with pytest.raises((ValidationError, ValueError)):
        PiAgentConfig.model_validate(config)


@pytest.mark.parametrize("key", ["critique_plan", "critique"])
def test_critique_not_accepted_in_untyped_kwargs(key):
    with pytest.raises(ValidationError, match="typed AgentConfig.pi"):
        AgentConfig.model_validate(
            {
                "name": "pi",
                "model_name": "hosted_vllm/student",
                "kwargs": {key: {"model": "teacher"}},
            }
        )


def test_critique_frozen_plan():
    plan = PiCritiquePlan("teacher", "http://teacher:8080/v1", "Review", "id")
    with pytest.raises(AttributeError):
        plan.teacher_model = "other"
