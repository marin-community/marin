"""Actual source judge compositions with a real local model protocol boundary."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_profiles_verifyit import (
    grade_judge_profile_verifyit,
)
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import (
    _MULTICHALLENGE_PROMPT,
    grade_abstention,
    grade_multichallenge,
)
from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak, _configuration


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            owner.requests.append(request)
            text = owner.by_prompt.get(request["messages"][-1]["content"], owner.reply)
            budget = request.get("max_completion_tokens", request.get("max_tokens"))
            complete = owner.complete_at is None or budget >= owner.complete_at
            response = {
                "id": "source-judge",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop" if complete else "length",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
            }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    owner = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    owner.requests, owner.by_prompt, owner.reply = [], {}, "A"
    owner.complete_at = None
    thread = threading.Thread(target=owner.serve_forever, daemon=True)
    thread.start()
    judge = OpenAIJudge(
        base_url=f"http://127.0.0.1:{owner.server_port}/v1",
        model="source-judge",
        api_key="test",
        timeout_seconds=10.0,
    )
    yield owner, judge
    owner.shutdown()
    owner.server_close()
    thread.join()


@pytest.mark.parametrize("verifyit_enabled", [False, True])
@pytest.mark.asyncio
async def test_profile_judge_recovers_from_8192_token_truncation(
    server, nemotron_session, model_turn, verifyit_enabled
):
    owner, judge = server
    owner.complete_at = 16384
    owner.reply = "C"
    session = await nemotron_session(
        "abstention_simple_agent",
        {"question": "What is two plus two?", "answer": "4"},
        {
            "verifyit_enabled": verifyit_enabled,
            "judges": {"general": {"base_url": judge.base_url, "model": judge.model, "api_key": "test"}},
        },
    )
    result = await session.advance(model_turn("An uncertain answer."))
    assert result.done
    assert result.reward == result.grade.reward == 0.5
    assert {request.get("max_completion_tokens", request.get("max_tokens")) for request in owner.requests} == {
        8192,
        16384,
    }


@pytest.mark.parametrize("reply,score", [("Assistant assessed Answer.\nA", 1.0), ("B", 0.0), ("C", 0.5)])
def test_abstention_native_and_existing_label_composition(server, reply, score):
    owner, judge = server
    owner.reply = reply
    record = {"question": "What is two plus two?", "answer": "4"}
    native = grade_abstention(r"\boxed{4}", record, judge)
    requests = list(owner.requests)
    owner.requests.clear()
    cutover = grade_judge_profile_verifyit(r"\boxed{4}", record, judge, kind="abstention")
    assert native == cutover
    assert cutover[0] == score
    assert owner.requests == requests


def test_idk_normalization_retains_half_without_a_model(server):
    owner, judge = server
    record = {"question": "What is two plus two?", "answer": "4"}
    native = grade_abstention(r"\boxed{[IDK]}", record, judge)
    cutover = grade_judge_profile_verifyit(r"\boxed{[IDK]}", record, judge, kind="abstention")
    assert native == cutover == (0.5, {"verdict": "abstain", "extracted_answer": "[IDK]"})
    assert owner.requests == []


@pytest.mark.parametrize("response", ["", " \n ", "<think>Working through the question.</think>"])
@pytest.mark.asyncio
async def test_empty_abstention_retains_native_not_attempted_reward(server, response, nemotron_session, model_turn):
    owner, judge = server
    owner.reply = "C"
    record = {"question": "What is two plus two?", "answer": "4"}
    native_reward, native_details = grade_abstention(response, record, judge)
    owner.requests.clear()

    reward, details = grade_judge_profile_verifyit(response, record, judge, kind="abstention")

    assert reward == native_reward == 0.5
    assert details["verdict"] == native_details["verdict"] == "abstain"
    assert details["extracted_answer"].strip() == ""
    assert owner.requests == []
    for verifyit_enabled in [False, True]:
        for stop_reason, expected in [("stop", 0.5), ("length", 0.0)]:
            session = await nemotron_session(
                "abstention_simple_agent",
                record,
                {
                    "verifyit_enabled": verifyit_enabled,
                    "judges": {"general": {"base_url": judge.base_url, "model": judge.model, "api_key": "test"}},
                },
            )
            result = await session.advance(model_turn(response, stop_reason=stop_reason))
            assert result.done
            assert result.reward == result.grade.reward == expected


@pytest.mark.parametrize(
    "expected,reply,score",
    [("YES", "[[YES]]", 1.0), ("NO", "[[NO]]", 1.0), ("YES", "[[NO]]", 0.0)],
)
def test_multichallenge_expected_direction_and_source_prompts(server, expected, reply, score):
    owner, judge = server
    owner.reply = reply
    record = {
        "context": "trusted {context} {response} {pass_criteria} {candidate} {reference}",
        "rubric": [
            {"question": "Criterion one?", "pass_criteria": expected},
            {
                "question": "Criterion two?",
                "pass_criteria": "NO" if expected == "YES" else "YES",
            },
        ],
    }
    native = grade_multichallenge("candidate {reference}", record, judge)
    requests = sorted(owner.requests, key=lambda item: item["messages"][-1]["content"])
    owner.requests.clear()
    cutover = grade_judge_profile_verifyit("candidate {reference}", record, judge, kind="multichallenge")
    assert native == cutover
    assert cutover[0] == 0.5
    assert cutover[1]["rubric_evaluations"][0]["score"] == score
    assert sorted(owner.requests, key=lambda item: item["messages"][-1]["content"]) == requests


@pytest.mark.parametrize(
    "policy",
    [
        "engagement_with_disclaimer",
        "hard_refusal_no_redirection",
        "hard_refusal_with_helplines",
        "refusal_with_explanation",
    ],
)
@pytest.mark.parametrize("kind", ["positive", "negative"])
def test_jailbreak_all_source_policies_use_existing_label_grading(server, policy, kind):
    owner, judge = server
    config = _configuration()
    names = config["policy_verifier_map"][policy]["verifiers"]
    record = {
        "response_policy_mapped": policy,
        "adversarial_prompt": "trusted harmless fixture question",
    }
    for name in names:
        verifier = config[name]
        prompt = verifier["prompt_template"].format(
            adversarial_prompt=record["adversarial_prompt"],
            model_response="candidate response",
        )
        owner.by_prompt[prompt] = verifier["labels"][kind]
    native = grade_jailbreak("candidate response", record, judge)
    requests = sorted(owner.requests, key=lambda item: item["messages"][-1]["content"])
    owner.requests.clear()
    cutover = grade_judge_profile_verifyit("candidate response", record, judge, kind="jailbreak")
    assert native == cutover
    assert sorted(owner.requests, key=lambda item: item["messages"][-1]["content"]) == requests


@pytest.mark.parametrize(
    "kind,record",
    [
        ("abstention", {"question": "Question", "answer": ""}),
        ("multichallenge", {"rubric": []}),
        (
            "jailbreak",
            {"response_policy_mapped": "unknown", "adversarial_prompt": "Question"},
        ),
    ],
)
def test_malformed_trusted_judge_contract_cannot_gain_positive_credit(server, kind, record):
    owner, judge = server
    score, detail = grade_judge_profile_verifyit("candidate", record, judge, kind=kind)
    assert score == 0.0
    assert detail["error_type"] == "schema_error"
    assert owner.requests == []


def test_later_multichallenge_failure_discards_prior_credit(server):
    owner, judge = server
    owner.reply = "[[YES]]"
    record = {"rubric": [{"question": "First?"}, {"question": "Second?"}]}

    owner.by_prompt[
        _MULTICHALLENGE_PROMPT.format(context="", response="candidate", question="Second?", pass_criteria="YES")
    ] = "ambiguous [[YES]] and [[NO]]"
    score, detail = grade_judge_profile_verifyit("candidate", record, judge, kind="multichallenge")
    assert score == 0.0
    assert detail["error_type"] == "verification_error"
    assert len(owner.requests) == 2
