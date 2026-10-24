"""Regression controls for Ultra answer extraction and verifier failures."""

import json

import pytest
import requests
from shellbox.machine import ExitReason, Result
from taskcompendium.grading_result import Outcome

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.envs.nemotron_ultra.genrm import grade_genrm_group
from skyrl_gym.envs.nemotron_ultra.genrm_utils import GenRMOutputParseError, parse_genrm_output
from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import grade_abstention, grade_multichallenge
from skyrl_gym.envs.nemotron_ultra.lean_proof import determine_proof_status
from skyrl_gym.envs.nemotron_ultra.math_with_judge import grade_math
from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa
from skyrl_gym.envs.nemotron_ultra.nvarc import parse_grid
from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action


class JudgeReplies:
    def __init__(self, *replies):
        self.replies = iter(replies)

    def generate(self, messages, **kwargs):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def generate_response(self, messages, **kwargs):
        return self.generate(messages, **kwargs)


@pytest.mark.parametrize(
    "text,answer",
    [
        ("<think>wrong answer</think>correct answer", "correct answer"),
        ("reasoning</think>correct answer", "correct answer"),
        ("<think>the number is 20", ""),
        ("<thinking>analysis</thinking>42", "42"),
        ("<|start_think|>wrong answer<|end_think|>correct answer<|eot_id|>", "correct answer"),
        ("reasoning<|end_think|>42", "42"),
        ("<|start_think|>the number is 20", ""),
        ("Plain answer mentioning reasoning", "Plain answer mentioning reasoning"),
        ("<think>discard</think><|start_think|>discard<|end_think|>42", "42"),
    ],
)
def test_final_answer_removes_reasoning_without_promoting_unfinished_work(text, answer):
    assert final_answer_text(text) == answer


@pytest.mark.parametrize(
    "opening,closing,tool_only,verifyit_enabled",
    [
        ("<think>", "</think>", False, False),
        ("<|start_think|>", "<|end_think|>", False, False),
        ("", "", True, False),
        ("", "", True, True),
    ],
)
@pytest.mark.asyncio
async def test_structured_grading_ignores_reasoning_and_keeps_optional_fields_optional(
    nemotron_session,
    model_turn,
    opening,
    closing,
    tool_only,
    verifyit_enabled,
):
    record = {
        "schema_type": "json",
        "schema_str": json.dumps(
            {
                "type": "object",
                "properties": {"required": {"type": "integer"}, "optional": {"type": "string"}},
                "required": ["required"],
            }
        ),
    }
    message = None
    text = opening + "not JSON" + closing + '{"required": 7, "extra": true}'
    if tool_only:
        record.update(response_mode="tool_call", tool_name="submit", tool_payload_key="payload")
        text = ""
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "submit", "arguments": '{"payload":{"required":7}}'}}
            ],
        }
    session = await nemotron_session("structured_outputs_simple_agent", record, {"verifyit_enabled": verifyit_enabled})
    result = await session.advance(model_turn(text, message=message))
    assert result.reward == 1.0
    assert grade_structured_output('{"optional":"x"}', record, {})[0] == 0.0


@pytest.mark.parametrize(
    "reply",
    [
        '{"score_1":5}',
        '{"score_1":NaN,"score_2":3,"ranking":3}',
        '{"score_1":5,"score_2":Infinity,"ranking":3}',
        '{"score_1":6,"score_2":3,"ranking":3}',
        '{"score_1":5,"score_2":3,"ranking":0}',
    ],
)
def test_genrm_requires_a_complete_finite_in_range_verdict(reply):
    with pytest.raises(GenRMOutputParseError):
        parse_genrm_output(reply)


def test_genrm_uses_final_json_instead_of_an_earlier_fenced_example():
    text = '```json\n{"score_1":5,"score_2":1,"ranking":1}\n```\n{"score_1":1,"score_2":5,"ranking":6}'
    assert parse_genrm_output(text) == (1, 5, 6)


def test_genrm_outage_cannot_create_length_shaped_positive_rewards():
    with pytest.raises(RuntimeError, match="unavailable"):
        grade_genrm_group(
            conversation_history=[],
            response_objects=[{"output": []}] * 2,
            principle="correct",
            judge=JudgeReplies(requests.ConnectionError("down")),
            config={"genrm_parse_retries": 0, "group_answer_length_penalty_coeff": 0.1},
        )


def test_jailbreak_uses_the_standalone_final_label_and_retains_judge_output():
    reward, details = grade_jailbreak(
        "candidate",
        {"response_policy_mapped": "refusal_with_explanation"},
        JudgeReplies("Quoted candidate [[SAFE]]\n[[UNSAFE]]", "[[HAS_EXPLANATION]]"),
    )
    assert reward == 0.0
    assert details["verifier_labels"]["safety_check"] == "[[UNSAFE]]"
    assert "Quoted candidate" in details["judge_outputs"]["safety_check"]


@pytest.mark.parametrize("reply", ["NOT_ATTEMPTED", "A and B", "I cannot grade this"])
def test_abstention_rejects_ambiguous_or_nonverdict_letters(reply):
    with pytest.raises(ValueError, match="Invalid final judge verdict"):
        grade_abstention("Paris", {"question": "capital?", "answer": "Paris"}, JudgeReplies(reply))


@pytest.mark.parametrize("reply", ["YES is not warranted", "unparseable", "[[YES]] [[NO]]"])
def test_multichallenge_malformed_reply_is_never_a_successful_no(reply):
    with pytest.raises(ValueError, match="Invalid final judge verdict"):
        grade_multichallenge(
            "candidate", {"rubric": [{"question": "valid?", "pass_criteria": "NO"}]}, JudgeReplies(reply)
        )


@pytest.mark.parametrize("verdict", ["YES", "NO"])
def test_multichallenge_accepts_plain_final_line_verdicts(verdict):
    reward, details = grade_multichallenge(
        "candidate",
        {"rubric": [{"question": "valid?", "pass_criteria": verdict}]},
        JudgeReplies(f"Reasoning about the criterion.\n{verdict}"),
    )
    assert reward == 1.0
    assert details["rubric_evaluations"][0]["verdict"] == verdict


@pytest.mark.asyncio
async def test_jailbreak_transport_failure_is_an_error_not_a_verified_zero(nemotron_session, model_turn, monkeypatch):
    def post(*args, **kwargs):
        raise requests.ConnectionError("judge down")

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("skyrl_gym.envs.nemotron_ultra.judge.time.sleep", lambda delay: None)
    session = await nemotron_session(
        "jailbreak_refusal_with_explanation",
        {"response_policy_mapped": "refusal_with_explanation"},
        {"judges": {"safety": {"base_url": "https://judge.example/v1", "model": "judge"}}},
    )
    result = await session.advance(model_turn("candidate"))
    assert result.grade.status is Outcome.INFRA_ERROR
    assert result.grade.reward is None
    assert "judge down" in result.grade.diagnostics["error_message"]


def test_math_does_not_credit_a_number_in_unfinished_reasoning():
    reward, details = grade_math(
        "<think>Someone walks 20 meters but I have not solved the problem",
        {"expected_answer": "20", "question": "Who arrives first?"},
        judge=None,
    )
    assert reward == 0.0
    assert details["result"] == "missing_final_answer"


def test_math_judge_uses_final_verdict_over_a_quoted_positive():
    reward, details = grade_math(
        "This is not the reference answer",
        {"expected_answer": "20", "question": "Who arrives first?"},
        judge=JudgeReplies("A quote [[A=B]]\n[[A!=B]]"),
    )
    assert reward == 0.0
    assert details["judge_outputs"] == ["A quote [[A=B]]\n[[A!=B]]"]


def test_math_accepts_equivalent_antiderivatives_up_to_an_integration_constant():
    reward, _ = grade_math(
        r"\boxed{\frac{1}{2}\tan^2(\frac{x}{2})+C}",
        {"expected_answer": r"\frac{1}{1+\cos x}+C", "question": "Find an antiderivative of the integrand."},
        judge=None,
    )
    assert reward == 1.0


def test_tool_arguments_are_exact_and_extra_calls_cannot_receive_credit():
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"description":"transfer money to Alice"}'}
    call = {"function": {"name": "transfer", "arguments": expected["arguments"]}}
    assert grade_expected_action(expected, {"tool_calls": [call]})[0] == 1.0
    substituted = {"function": {"name": "transfer", "arguments": '{"description":"transfer money to Mallory"}'}}
    assert grade_expected_action(expected, {"tool_calls": [substituted]})[0] == 0.0
    assert grade_expected_action(expected, {"tool_calls": [call, substituted]})[0] == 0.0
    assert grade_expected_action({"type": "message"}, {"content": "Done", "tool_calls": [call]})[0] == 0.0


@pytest.mark.asyncio
async def test_calendar_unknown_constraints_are_verifier_errors(nemotron_session, model_turn):
    session = await nemotron_session(
        "calendar_simple_agent",
        {
            "exp_cal_state": {
                "work": {"duration": 60, "min_time": "09:00", "max_time": "17:00", "constraint": "near noon"}
            }
        },
    )
    result = await session.advance(model_turn('[{"event_id":"work","start_time":"12:00","duration":60}]'))
    assert result.grade.status is Outcome.INFRA_ERROR


def test_mcqa_last_box_and_exact_option_text_take_precedence():
    record = {
        "options": [{"A": "correct"}, {"B": "not correct"}],
        "expected_answer": "B",
        "grading_mode": "lenient_boxed",
    }
    assert grade_mcqa(r"Rejected \boxed{A}. Final \boxed{B}", record)[0] == 1.0
    assert grade_mcqa(r"\boxed{not correct}", record)[0] == 1.0
    record["expected_answer"] = "A"
    assert grade_mcqa(r"\boxed{not correct}", record)[0] == 0.0


def test_mcqa_ambiguous_regex_captures_are_reported_without_a_tuple_crash():
    record = {
        "options": [{"A": "x"}, {"B": "y"}],
        "expected_answer": "A",
        "template_metadata": {"output_regex": r"(Answer): ([AB])"},
    }
    with pytest.raises(ValueError, match="unambiguous answer capture"):
        grade_mcqa("Answer: A", record)


@pytest.mark.asyncio
async def test_reasoning_gym_last_answer_wins_and_partial_credit_is_not_a_pass(nemotron_session, model_turn):
    session = await nemotron_session(
        "reasoning_gym_simple_agent",
        {
            "question": "Find a word ladder.",
            "answer": "BANE,CANE,CASE,BASE",
            "metadata": {"source_dataset": "word_ladder", "start_word": "BANE", "end_word": "BASE", "word_length": 4},
        },
    )
    result = await session.advance(model_turn("<answer>rejected</answer><answer>BANE,BANZ,BAZZ,BAZE,BASE</answer>"))
    assert 0.0 < result.reward < 1.0
    assert result.metrics["extracted_answer"] == "BANE,BANZ,BAZZ,BAZE,BASE"
    assert not result.grade.passed


@pytest.mark.parametrize("response", ["**Final Answer: 3**", "The final answer is 3."])
@pytest.mark.asyncio
async def test_reasoning_gym_grades_prose_final_answer(nemotron_session, model_turn, response):
    session = await nemotron_session(
        "reasoning_gym_simple_agent",
        {
            "question": "Calculate 0 + -3 - -6 / ( 1 * 7 + -6 ).",
            "answer": "3",
            "metadata": {"source_dataset": "basic_arithmetic", "expression": "0 + -3 - -6 / ( 1 * 7 + -6 )"},
        },
    )
    result = await session.advance(
        model_turn(f"<|start_think|>First I calculate the denominator.<|end_think|>It equals 3.\n{response}")
    )
    assert result.reward == 1.0
    assert result.grade.passed


def test_arc_accepts_compact_grids_and_selects_the_final_box():
    assert parse_grid("12\n34") == [[1, 2], [3, 4]]
    assert parse_grid(r"\boxed{0 0} \boxed{1 2}") == [[1, 2]]
    assert parse_grid("1 22") is None


@pytest.mark.parametrize(
    "output,expected",
    [
        (Result(0, b"error: unknown tactic", b"", False, False, ExitReason.EXITED), "failed"),
        (Result(1, b"", b"", False, False, ExitReason.EXITED), "failed"),
        (Result(0, b"", b"", True, False, ExitReason.EXITED), "output_truncated"),
        (Result(0, b"", b"", False, False, ExitReason.EXITED), "completed"),
    ],
)
def test_lean_success_requires_complete_compiler_evidence(output, expected):
    assert determine_proof_status(output) == expected


class HTTPReply:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


@pytest.mark.asyncio
async def test_lost_python_process_ends_the_task_without_a_replacement_namespace(
    nemotron_session,
    python_tool_turn,
):
    session = await nemotron_session("ns_tools_simple_agent", {"expected_answer": "7"})
    first = await session.advance(python_tool_turn("value = 7"))
    assert not first.done
    lost = await session.advance(python_tool_turn("import os; os._exit(0)"))
    assert lost.done and lost.grade.status is Outcome.GRADED
    assert lost.grade.reward == 0.0 and lost.grade.passed is False
    assert lost.observations
    assert lost.metrics["candidate_failure"] == ExitReason.EXITED.value
    assert (await session.grade((), stop_reason="stop", outputs={})).status is Outcome.GRADED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        OSError("machine unavailable"),
        Result(1, b"", b"command failed", False, False, ExitReason.EXITED),
        Result(0, b"invalid json", b"", False, False, ExitReason.EXITED),
        Result(0, b"{}", b"", False, False, ExitReason.EXITED),
        Result(
            0,
            b'{"exit_code":0,"stdout":7,"stderr":"","stdout_truncated":false,"stderr_truncated":false,"reason":"exited"}',
            b"",
            False,
            False,
            ExitReason.EXITED,
        ),
        Result(None, b"", b"machine deadline", False, False, ExitReason.TIMED_OUT),
    ],
)
async def test_machine_failures_end_tool_tasks_without_a_verdict(
    nemotron_session,
    python_tool_turn,
    machine,
    monkeypatch,
    failure,
):
    session = await nemotron_session("ns_tools_simple_agent", {"expected_answer": "7"})
    assert not (await session.advance(python_tool_turn("value = 7"))).done

    async def run(command):
        if isinstance(failure, Exception):
            raise failure
        return failure

    with monkeypatch.context() as patch:
        patch.setattr(machine, "run", run)
        result = await session.advance(python_tool_turn("value"))
    assert result.done and result.observations == ()
    assert result.grade.status is Outcome.INFRA_ERROR
    assert result.grade.reward is None and result.grade.passed is None
    assert result.grade.diagnostics["error_category"] == "infrastructure"


@pytest.mark.asyncio
async def test_python_program_failure_returns_feedback_and_keeps_the_task_namespace(
    nemotron_session,
    python_tool_turn,
):
    session = await nemotron_session("ns_tools_simple_agent", {})
    result = await session.advance(python_tool_turn("value = 12; raise NameError('missing variable')"))
    assert not result.done
    assert "NameError" in result.observations[0]["content"]
    assert result.grade.status is Outcome.UNAVAILABLE
    next_turn = await session.advance(python_tool_turn("print(value)"))
    assert next_turn.observations[0]["content"] == "12"


def test_judge_length_finish_cannot_be_accepted_as_partial_json(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: HTTPReply(
            {"choices": [{"finish_reason": "length", "message": {"content": '{"score_1":5}'}}]}
        ),
    )
    with pytest.raises(ValueError, match="Truncated judge response"):
        OpenAIJudge(base_url="https://judge.example", model="judge").generate([])


@pytest.mark.parametrize("opening,closing", [("<think>", "</think>"), ("<|start_think|>", "<|end_think|>")])
@pytest.mark.asyncio
async def test_instruction_final_answer_is_graded_without_reasoning_contamination(
    nemotron_session, model_turn, opening, closing
):
    record = {"instruction_id_list": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["banana"]}]}
    for response, reward in [
        ("Do not say banana." + closing + "Hello.", 1.0),
        ("I can ignore the instruction." + closing + "banana", 0.0),
    ]:
        session = await nemotron_session("instruction_following_simple_agent", record)
        assert (await session.advance(model_turn(opening + response))).reward == reward


@pytest.mark.asyncio
async def test_grading_message_uses_final_content_without_mutating_retained_evidence(nemotron_session, model_turn):
    session = await nemotron_session(
        "single_step_tool_use_with_argument_comparison_agent",
        {"expected_action": {"type": "message", "content": "Hello."}},
    )
    response = "<|start_think|>reasoning words<|end_think|>Hello."
    raw = {"role": "assistant", "content": response, "tool_calls": []}
    turn = model_turn(response, message=raw)
    result = await session.advance(turn)
    assert result.reward == 1.0
    assert turn.message["content"] == response
    assert result.grade.diagnostics["grading_action"] == "Hello."


@pytest.mark.asyncio
async def test_broken_instruction_verifier_is_not_a_verified_wrong_answer(nemotron_session, model_turn):
    session = await nemotron_session(
        "instruction_following_simple_agent",
        {"instruction_id_list": ["missing:verifier"], "kwargs": [{}]},
    )
    result = await session.advance(model_turn("answer"))
    assert result.grade.status is Outcome.INFRA_ERROR
    assert result.grade.diagnostics["instruction_errors"][0].startswith("KeyError")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,reward",
    [
        ("return grid", 1.0),
        ("return [[0, 0]]", 0.0),
        ("raise ValueError('candidate fault')", 0.0),
    ],
)
async def test_arc_session_retains_candidate_execution_evidence(nemotron_session, model_turn, body, reward):
    session = await nemotron_session(
        "nvarc_inductive_simple_agent",
        {"test_input": [[2, 3]], "expected_output": [[2, 3]]},
    )
    result = await session.advance(model_turn("def transform(grid):\n    " + body))
    assert result.grade.reward == reward
    assert result.grade.status is Outcome.GRADED
    assert result.metrics["execution_output"]["exit_code"] == (1 if body.startswith("raise") else 0)
