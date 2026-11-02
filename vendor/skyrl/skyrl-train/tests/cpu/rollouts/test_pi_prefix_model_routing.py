"""Pi student and private teacher requests share a leased proxy, not a policy source."""

import pytest

from rolloutengine.models import ModelTurn
from skyrl_train.rollouts.agent_backend import (
    AgentInferenceBackend,
    EXECUTED_TEACHER_HEADER,
    PI_EXECUTION_MODEL_OPTION,
    TRIAL_HEADER,
)


class Decoder:
    def decode(self, ids):
        return str(ids[0])


async def tokenize(_):
    return {"tokens": [1]}


@pytest.mark.asyncio
async def test_prefix_routes_independent_models_to_same_frozen_pre_action_context():
    seen = []
    backend = AgentInferenceBackend(
        "student", 256, tokenize, Decoder(), teacher_model_name="teacher", teacher_decoder=Decoder()
    )

    def handler(actor):
        async def complete(prompt):
            seen.append((actor, prompt.messages))
            return ModelTurn(
                message={"role": "assistant", "content": actor},
                prompt_token_ids=(1, 2),
                response_token_ids=(3 if actor == "student" else 4,),
                logprobs=(-0.2,),
                stop_reason="stop",
                text=actor,
            )

        return complete

    payload = {
        "headers": {TRIAL_HEADER: "physical_0"},
        "json": {"model": "student", "messages": [{"role": "user", "content": "Solve the task"}], "logprobs": True},
    }
    with backend.session("physical_0", handler("student"), teacher_complete=handler("teacher")):
        student = await backend.chat_completion(payload)
        teacher = await backend.chat_completion({**payload, "json": {**payload["json"], "model": "teacher"}})
        assert seen == [
            ("student", ({"role": "user", "content": "Solve the task"},)),
            ("teacher", ({"role": "user", "content": "Solve the task"},)),
        ]
        assert student["model"] == "student" and student["choices"][0]["token_ids"] == [3]
        assert teacher["model"] == "teacher" and teacher["choices"][0]["token_ids"] == [4]
        with pytest.raises(ValueError, match="unregistered"):
            await backend.chat_completion({**payload, "json": {**payload["json"], "model": "other"}})
    with pytest.raises(ValueError, match="No active agent session"):
        await backend.chat_completion(payload)


@pytest.mark.asyncio
async def test_executed_teacher_uses_the_model_session_but_private_label_does_not():
    calls = []
    backend = AgentInferenceBackend(
        "student", 256, tokenize, Decoder(), teacher_model_name="teacher", teacher_decoder=Decoder()
    )

    def handler(source):
        async def complete(prompt):
            calls.append((source, prompt.options.get(PI_EXECUTION_MODEL_OPTION)))
            return ModelTurn(
                message={"role": "assistant", "content": source},
                prompt_token_ids=(1,),
                response_token_ids=(4,),
                logprobs=None,
                stop_reason="stop",
                text=source,
            )

        return complete

    payload = {
        "headers": {TRIAL_HEADER: "physical_0"},
        "json": {"model": "teacher", "messages": [{"role": "user", "content": "Solve"}]},
    }
    with backend.session("physical_0", handler("model_session"), teacher_complete=handler("private_label")):
        await backend.chat_completion({**payload, "headers": {**payload["headers"], EXECUTED_TEACHER_HEADER: "0"}})
        await backend.chat_completion({**payload, "headers": {**payload["headers"], EXECUTED_TEACHER_HEADER: "1"}})
        assert calls == [("private_label", None), ("model_session", "teacher")]
        with pytest.raises(ValueError, match="Only independently routed"):
            await backend.chat_completion(
                {
                    **payload,
                    "headers": {
                        **payload["headers"],
                        EXECUTED_TEACHER_HEADER: "invalid",
                    },
                }
            )
        with pytest.raises(ValueError, match="reserved"):
            await backend.chat_completion(
                {
                    **payload,
                    "json": {
                        **payload["json"],
                        PI_EXECUTION_MODEL_OPTION: "teacher",
                    },
                }
            )


@pytest.mark.asyncio
async def test_prefix_refuses_a_student_session_without_its_private_teacher():
    backend = AgentInferenceBackend(
        "student", 256, tokenize, Decoder(), teacher_model_name="teacher", teacher_decoder=Decoder()
    )
    with pytest.raises(ValueError, match="teacher session"):
        with backend.session("physical_0", lambda _: None):
            pass
