import json

import pytest
import requests
from rolloutengine.models import ModelTurn
from skyrl_gym.nemotron_tasks import NemotronTaskSession
from skyrl_gym.rollout_specs import rollout_spec
from skyrl_gym.source_task import source_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import Source

LIMITS = {
    "task_session": "nemotron_ultra",
    "max_turns": 50,
    "max_total_tokens": None,
    "model_turn_timeout": None,
    "tool_turn_timeout": None,
    "total_turn_timeout": None,
    "attempt_timeout": None,
    "verifier_timeout": None,
    "cleanup_timeout": 5,
}
FORBIDDEN = {"instruction_id_list": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["banana"]}]}
MCQA = {"expected_answer": "C", "options": [{"A": "one"}, {"B": "two"}, {"C": "three"}]}
ABSTENTION = {"question": "What is two plus two?", "answer": "4"}
JUDGE = {"verifyit_enabled": False, "judges": {"general": {"base_url": "http://127.0.0.1:9/v1", "model": "unused"}}}
LEAN = {"header": "import Mathlib\n", "formal_statement": "example : True := by\n"}
IPI = {"environment": {}, "injection": {"target_tool": "t", "target_args": {}, "verification_type": "exact"}}
SCHEMA = {"type": "object", "properties": {"required": {"type": "integer"}}, "required": ["required"]}
STRUCTURED = {"schema_type": "json", "schema_str": json.dumps(SCHEMA), "response_mode": "tool_call"}
STRUCTURED |= {"tool_name": "submit", "tool_payload_key": "payload"}
SUBMIT = {"type": "function", "function": {"name": "submit", "arguments": '{"payload":{"required":7}}'}}


@pytest.mark.parametrize(
    "agent,record,config,text,tool_calls,stop_reason,expected",
    [
        ("mcqa_simple_agent", MCQA, {}, r"\boxed{C}", None, "length", (True, 0.0, Outcome.GRADED)),
        (
            "instruction_following_simple_agent",
            FORBIDDEN,
            {},
            "<think>B?</think>",
            None,
            "stop",
            (True, 0.0, Outcome.GRADED),
        ),
        (
            "instruction_following_simple_agent",
            FORBIDDEN,
            {"grading": "skip"},
            "",
            None,
            "stop",
            (True, 0.0, Outcome.UNAVAILABLE),
        ),
        ("abstention_simple_agent", ABSTENTION, JUDGE, "", None, "length", (True, 0.0, Outcome.GRADED)),
        ("abstention_simple_agent", ABSTENTION, JUDGE, "", None, "stop", (True, 0.5, Outcome.GRADED)),
        ("structured_outputs_simple_agent", STRUCTURED, {}, "", [SUBMIT], "stop", (True, 1.0, Outcome.GRADED)),
        (
            "genrm_simple_agent",
            {},
            {"genrm": {"default_score": 3.5}},
            "<think>x",
            None,
            "length",
            (True, 3.5, Outcome.GRADED),
        ),
        ("math_formal_lean_refinement_agent", LEAN, {}, "", None, "length", (False, 0.0, Outcome.UNAVAILABLE)),
        ("indirect_prompt_injection_simple_agent", IPI, {}, "", None, "stop", (True, 1.0, Outcome.GRADED)),
    ],
)
@pytest.mark.asyncio
async def test_ultra_unfinished_reply_scores_zero_unless_the_agent_has_its_own_rule(
    agent, record, config, text, tool_calls, stop_reason, expected, monkeypatch
):
    judge = requests.Response()
    judge.status_code, judge._content = 200, b'{"choices":[{"finish_reason":"stop","message":{"content":"C"}}]}'
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: judge)
    ultra = {"route": "task_session", "agent": agent, "record_json": json.dumps(record), "request_json": "{}"}
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = source_task(
        [{"role": "user", "content": "Question"}], {"extra_info": {"nemotron_ultra": ultra}}, config, source
    )
    session = NemotronTaskSession(rollout_spec(task, LIMITS), None)
    await session.prepare()
    message = {"role": "assistant", "content": text} | ({"tool_calls": tool_calls} if tool_calls else {})
    try:
        result = await session.advance(ModelTurn(message, (10,), (20,), (-0.5,), stop_reason, text=text, metadata={}))
    finally:
        await session.close()
    assert (result.done, result.reward, result.grade.status) == expected
