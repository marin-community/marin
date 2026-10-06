# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from verifyit.grade import InvalidTask, Status, grade, run, write_reward
from verifyit.modes import grade_judge
from verifyit.spec import Constraint, EmptyOutputPolicy, JudgeSpec, SampleResolution, parse_spec, render_spec

# The dataset's own reference answer, apostrophe included: the gate must fold case, spacing and
# punctuation without mangling non-ASCII text.
REFERENCE = (
    "Yes, if the non-state actor’s actions amount to an armed attack and the host state is "  # noqa: RUF001
    "unwilling or unable to suppress the threat."
)
BOXED_RESPONSE = f"The victim state may act only in the narrow case described.\n\\boxed{{{REFERENCE}}}"
SPACED_RESPONSE = REFERENCE.lower().replace("attack and", "attack   and").rstrip(".")
QUESTION = "Under the narrow interpretation of Article 51 of the UN Charter, when may force be used?"
ENV_VARS = ("VERIFYIT_JUDGE_BASE_URL", "VERIFYIT_JUDGE_API_KEY", "VERIFYIT_JUDGE_MODEL")


class FakeJudgeServer(ThreadingHTTPServer):
    """A one-endpoint stand-in for an OpenAI-compatible chat server."""

    replies: list[str]
    prompts: list[str]
    finish_reason: str
    requests: list[dict]
    finish_reasons: list[str]
    http_status: int
    message_fields: dict
    response_fields: dict
    raw_body: str | None
    usage_tokens: list[int]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        assert self.path.endswith(("/chat/completions", "/responses"))
        server: FakeJudgeServer = self.server  # pyrefly: ignore[bad-assignment]
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(request)
        inputs = request.get("messages", request.get("input"))
        server.prompts.append(inputs if isinstance(inputs, str) else inputs[-1]["content"])
        reply = server.replies[min(len(server.prompts) - 1, len(server.replies) - 1)]
        usage = {}
        if server.usage_tokens:
            index = min(len(server.prompts) - 1, len(server.usage_tokens) - 1)
            usage = {"usage": {"completion_tokens": server.usage_tokens[index]}}
        body = json.dumps(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply, **server.message_fields},
                        "finish_reason": (
                            server.finish_reasons[min(len(server.prompts) - 1, len(server.finish_reasons) - 1)]
                            if server.finish_reasons
                            else server.finish_reason
                        ),
                    }
                ],
                **usage,
            }
        ).encode()
        if self.path.endswith("/responses"):
            body = json.dumps(
                {
                    "id": "resp-fake",
                    "object": "response",
                    "status": "completed",
                    "error": None,
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": reply}],
                        }
                    ],
                    **server.response_fields,
                }
            ).encode()
        if server.raw_body is not None:
            body = server.raw_body.encode()
        self.send_response(server.http_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def fake_judge(monkeypatch):
    server = FakeJudgeServer(("127.0.0.1", 0), _Handler)
    server.replies = ["SCORE: 1"]
    server.prompts = []
    server.requests = []
    server.finish_reasons = []
    server.http_status = 200
    server.message_fields = {}
    server.response_fields = {}
    server.raw_body = None
    server.usage_tokens = []
    server.finish_reason = "stop"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("VERIFYIT_JUDGE_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("VERIFYIT_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("VERIFYIT_JUDGE_MODEL", "fake/judge-9b")
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def unconfigured_judge(monkeypatch):
    """No judge endpoint at all, so any model call fails loudly instead of reaching the network."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _workspace(tmp_path: Path, response: str) -> Path:
    (tmp_path / "answer.txt").write_text(response)
    return tmp_path


@pytest.mark.parametrize(
    "response", [pytest.param(BOXED_RESPONSE, id="boxed"), pytest.param(SPACED_RESPONSE, id="unboxed_case_and_spacing")]
)
def test_exact_gate_scores_one_without_calling_a_model(tmp_path, unconfigured_judge, response):
    spec = JudgeSpec(references=("Paris", REFERENCE), question=QUESTION)
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, response))
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)
    assert reward.detail == {"gate": "exact"}


def test_gate_ignores_articles_and_trailing_punctuation(tmp_path, unconfigured_judge):
    reward = grade_judge.grade(JudgeSpec(references=("Paris",)), tmp_path, _workspace(tmp_path, "\\boxed{The Paris.}"))
    assert reward.reward == 1.0


def test_paraphrase_falls_through_to_the_model(tmp_path, fake_judge):
    fake_judge.replies = ["The candidate omits the unwilling-or-unable condition.\nSCORE: 0.5"]
    response = "\\boxed{Only when the armed group's attack is attributable to the host state.}"
    spec = JudgeSpec(references=(REFERENCE,), question=QUESTION)
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, response))
    assert (reward.reward, reward.status) == (0.5, Status.SCORED)
    assert reward.detail["model"] == "fake/judge-9b"
    assert "unwilling-or-unable" in reward.detail["reasoning"]
    prompt = fake_judge.prompts[0]
    assert REFERENCE in prompt
    assert QUESTION in prompt
    assert response in prompt


def test_disabled_gate_sends_even_an_exact_match_to_the_model(tmp_path, fake_judge):
    fake_judge.replies = ["Same answer.\nSCORE: 1"]
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, REFERENCE))
    assert reward.reward == 1.0
    assert len(fake_judge.prompts) == 1


def test_spec_model_overrides_the_environment(tmp_path, fake_judge):
    fake_judge.replies = ["Wrong answer.\nSCORE: 0"]
    spec = JudgeSpec(references=(REFERENCE,), model="override/judge-70b", exact_gate=False)
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "The moon is made of cheese."))
    assert (reward.reward, reward.status, reward.detail["model"]) == (0.0, Status.SCORED, "override/judge-70b")


@pytest.mark.parametrize("rubric", ["reference", "checklist"])
def test_unparseable_reply_is_retried_once_then_masks_candidate(tmp_path, fake_judge, rubric):
    fake_judge.replies = ["I cannot grade this."]
    spec = JudgeSpec(rubric=rubric, references=(REFERENCE,), criteria=("Be correct",), exact_gate=False)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    reward = run(spec_path, _workspace(tmp_path, "something else"))
    assert reward.status == Status.INFRA_ERROR
    assert len(fake_judge.prompts) == 2


def test_second_attempt_is_accepted(tmp_path, fake_judge):
    fake_judge.replies = ["I cannot grade this.", "Matches the reference.\nSCORE: 1"]
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "a paraphrase"))
    assert reward.reward == 1.0
    assert reward.detail["attempt_count"] == 2
    assert reward.detail["attempts"] == [
        {"finish_reason": "stop", "completion_tokens": None},
        {"finish_reason": "stop", "completion_tokens": None},
    ]


def test_reference_retries_truncation_with_budget_and_records_attempts(tmp_path, fake_judge):
    fake_judge.replies = ["SCORE: 1", "SCORE: 0.5"]
    fake_judge.finish_reasons = ["length", "stop"]
    fake_judge.usage_tokens = [1024, 1600]
    spec = JudgeSpec(
        references=(REFERENCE,),
        exact_gate=False,
        max_completion_tokens=1024,
        incomplete_retry_tokens=2048,
        reasoning_effort="low",
    )

    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "a paraphrase"))

    assert (reward.reward, reward.status) == (0.5, Status.SCORED)
    assert [request["max_completion_tokens"] for request in fake_judge.requests] == [1024, 2048]
    assert [request["reasoning_effort"] for request in fake_judge.requests] == ["low", "low"]
    assert reward.detail["attempt_count"] == 2
    assert reward.detail["attempts"] == [
        {"finish_reason": "length", "completion_tokens": 1024},
        {"finish_reason": "stop", "completion_tokens": 1600},
    ]


def test_reference_exhausted_truncation_is_unscored_with_attempts(tmp_path, fake_judge):
    fake_judge.replies = ["SCORE: 1"]
    fake_judge.finish_reasons = ["length", "length"]
    fake_judge.usage_tokens = [1024, 2048]
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False, max_completion_tokens=1024, incomplete_retry_tokens=2048)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))

    reward = run(spec_path, _workspace(tmp_path, "a paraphrase"))

    assert (reward.reward, reward.status) == (0.0, Status.INFRA_ERROR)
    assert reward.detail["attempt_count"] == 2
    assert reward.detail["attempts"] == [
        {"finish_reason": "length", "completion_tokens": 1024},
        {"finish_reason": "length", "completion_tokens": 2048},
    ]


def test_missing_endpoint_configuration_is_an_infra_error(tmp_path, unconfigured_judge):
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    with pytest.raises(RuntimeError, match="VERIFYIT_JUDGE_BASE_URL"):
        grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "a paraphrase"))


def test_missing_model_configuration_is_an_infra_error(tmp_path, fake_judge, monkeypatch):
    monkeypatch.delenv("VERIFYIT_JUDGE_MODEL")
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    with pytest.raises(RuntimeError, match="VERIFYIT_JUDGE_MODEL"):
        grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "a paraphrase"))


def test_no_output_scores_zero(tmp_path, unconfigured_judge):
    (tmp_path / "answer.txt").write_text("   \n")
    reward = grade_judge.grade(JudgeSpec(references=(REFERENCE,)), tmp_path, tmp_path)
    assert (reward.reward, reward.detail) == (0.0, {"reason": "no_output"})


CRITERIA = ("Does the response give exactly three steps?", "Is the tone formal?", "Does it mention Ada Lovelace?")


def _checklist(**overrides) -> JudgeSpec:
    return JudgeSpec(rubric="checklist", criteria=CRITERIA, **overrides)


def test_checklist_scores_the_fraction_of_criteria_the_judge_passes(tmp_path, fake_judge):
    fake_judge.replies = ["Three steps.\nSCORE: 1", "Casual.\nSCORE: 0", "Names her.\nSCORE: 1"]
    reward = grade_judge.grade(_checklist(), tmp_path, _workspace(tmp_path, "1. Ask Ada Lovelace. 2. Wait. 3. Done."))
    assert (reward.reward, reward.status) == (pytest.approx(2 / 3), Status.SCORED)
    assert [c["passed"] for c in reward.detail["criteria"]] == [True, False, True]
    assert len(fake_judge.prompts) == 3
    assert CRITERIA[1] in fake_judge.prompts[1] and CRITERIA[0] not in fake_judge.prompts[1]


def test_checklist_records_attempts_per_criterion(tmp_path, fake_judge):
    fake_judge.replies = ["SCORE: 1", "SCORE: 1", "SCORE: 0"]
    fake_judge.finish_reasons = ["length", "stop", "stop"]
    fake_judge.usage_tokens = [1024, 1500, 100]
    spec = JudgeSpec(
        rubric="checklist",
        criteria=CRITERIA[:2],
        max_completion_tokens=1024,
        incomplete_retry_tokens=2048,
    )

    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "candidate"))

    assert (reward.reward, reward.status) == (0.5, Status.SCORED)
    assert [request["max_completion_tokens"] for request in fake_judge.requests] == [1024, 2048, 1024]
    assert [criterion["attempt_count"] for criterion in reward.detail["criteria"]] == [2, 1]
    assert [criterion["attempts"] for criterion in reward.detail["criteria"]] == [
        [
            {"finish_reason": "length", "completion_tokens": 1024},
            {"finish_reason": "stop", "completion_tokens": 1500},
        ],
        [{"finish_reason": "stop", "completion_tokens": 100}],
    ]


def test_checklist_truncated_criterion_keeps_partial_diagnostics_without_score(tmp_path, fake_judge):
    fake_judge.replies = ["SCORE: 1"]
    fake_judge.finish_reasons = ["stop", "length", "length"]
    fake_judge.usage_tokens = [10, 1024, 2048]
    spec = JudgeSpec(rubric="checklist", criteria=CRITERIA[:2], max_completion_tokens=1024, incomplete_retry_tokens=2048)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))

    reward = run(spec_path, _workspace(tmp_path, "candidate"))

    assert (reward.reward, reward.status) == (0.0, Status.INFRA_ERROR)
    assert [criterion["attempt_count"] for criterion in reward.detail["criteria"]] == [1, 2]
    assert [attempt["finish_reason"] for attempt in reward.detail["criteria"][1]["attempts"]] == ["length", "length"]
    assert [attempt["completion_tokens"] for attempt in reward.detail["criteria"][1]["attempts"]] == [1024, 2048]


def test_majority_samples_resolve_each_criterion_and_keep_every_verdict(tmp_path, fake_judge):
    fake_judge.replies = [
        "Clear.\nSCORE: 1",
        "Unsure.\nSCORE: 0",
        "Clear again.\nSCORE: 1",
        "SCORE: 0",
        "SCORE: 0",
        "SCORE: 1",
    ]
    spec = JudgeSpec(rubric="checklist", criteria=CRITERIA[:2], samples=3, sample_temperature=0.7)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    reward = run(spec_path, _workspace(tmp_path, "1. Ask Ada Lovelace. 2. Wait. 3. Done."))
    assert (reward.status, reward.reward) == (Status.SCORED, 0.5)
    assert [request["temperature"] for request in fake_judge.requests] == [0.7] * 6
    criteria = reward.detail["criteria"]
    assert [[sample["passed"] for sample in c["samples"]] for c in criteria] == [
        [True, False, True],
        [False, False, True],
    ]
    assert [(c["passed"], c["attempt_count"], c["reasoning"]) for c in criteria] == [(True, 3, "Clear."), (False, 3, "")]
    assert [sample["reasoning"] for sample in criteria[0]["samples"]] == ["Clear.", "Unsure.", "Clear again."]


def test_two_then_third_asks_a_third_time_only_on_disagreement(tmp_path, fake_judge):
    # Criterion one: the first two agree. Criterion two: they disagree and the third decides.
    fake_judge.replies = ["SCORE: 0", "SCORE: 0", "SCORE: 1", "SCORE: 0", "SCORE: 1"]
    spec = JudgeSpec(
        rubric="checklist", criteria=CRITERIA[:2], samples=3, sample_resolution=SampleResolution.TWO_THEN_THIRD
    )
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    reward = run(spec_path, _workspace(tmp_path, "1. Ask Ada Lovelace. 2. Wait. 3. Done."))
    assert (reward.status, reward.reward) == (Status.SCORED, 0.5)
    assert [request["temperature"] for request in fake_judge.requests] == [0.0] * 5
    criteria = reward.detail["criteria"]
    assert [[sample["passed"] for sample in c["samples"]] for c in criteria] == [[False, False], [True, False, True]]
    assert [c["attempt_count"] for c in criteria] == [2, 3]


def test_a_failed_sample_leaves_the_whole_checklist_unscored_with_its_completed_samples(tmp_path, fake_judge):
    fake_judge.replies = ["SCORE: 1"]
    fake_judge.finish_reasons = ["stop", "stop", "length"]
    fake_judge.usage_tokens = [10, 11, 8192]
    spec = JudgeSpec(rubric="checklist", criteria=CRITERIA, samples=3)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    reward = run(spec_path, _workspace(tmp_path, "1. Ask Ada Lovelace. 2. Wait. 3. Done."))
    assert (reward.status, reward.reward) == (Status.INFRA_ERROR, 0.0)
    assert len(fake_judge.requests) == 3
    [partial] = reward.detail["criteria"]
    assert (partial["criterion"], len(partial["samples"]), partial["attempt_count"]) == (CRITERIA[0], 2, 3)
    assert [attempt["finish_reason"] for attempt in partial["attempts"]] == ["stop", "stop", "length"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples": 0},
        {"samples": 2},
        {"samples": 2, "sample_resolution": "two_then_third"},
        {"samples": 3, "sample_temperature": -0.5},
        {"samples": 3, "sample_temperature": float("inf")},
        {"sample_temperature": 0.7},
    ],
)
def test_invalid_sample_settings_never_call_the_model(tmp_path, fake_judge, overrides):
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_checklist(**overrides)))
    reward = run(spec_path, _workspace(tmp_path, "a paraphrase"))
    assert reward.status is Status.INVALID_TASK
    assert fake_judge.requests == []


def test_repeated_judgments_on_a_reference_rubric_are_an_invalid_task(tmp_path, fake_judge):
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False, samples=3)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    reward = run(spec_path, _workspace(tmp_path, "a paraphrase"))
    assert (reward.status, reward.reward) == (Status.INVALID_TASK, 0.0)
    assert fake_judge.requests == []


def test_checklist_shows_the_context_file_to_the_judge(tmp_path, fake_judge):
    context = "é" * (grade_judge.CONTEXT_LIMIT - 4) + "TAIL" + "excluded"
    (tmp_path / "conversation.txt").write_text(context)
    fake_judge.replies = ["SCORE: 1"]
    spec = JudgeSpec(rubric="checklist", criteria=(CRITERIA[0],), context="conversation.txt")
    grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "1. 2. 3."))
    assert context[: grade_judge.CONTEXT_LIMIT] in fake_judge.prompts[0]
    assert "excluded" not in fake_judge.prompts[0]


def test_missing_context_file_is_an_invalid_task(tmp_path, fake_judge):
    spec = JudgeSpec(rubric="checklist", criteria=CRITERIA, context="missing.txt")
    with pytest.raises(grade_judge.InvalidTask):
        grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "text"))


def test_constraints_gate_scores_zero_without_calling_the_judge(tmp_path, unconfigured_judge):
    spec = _checklist(constraints=(Constraint("startend:end_checker", {"end_phrase": "Sincerely."}),))
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "1. Ask Ada Lovelace."))
    assert (reward.reward, reward.detail["gate"], reward.detail["failed"]) == (
        0.0,
        "constraints",
        ["startend:end_checker"],
    )


def test_constraints_that_pass_hand_over_to_the_judge(tmp_path, fake_judge):
    fake_judge.replies = ["SCORE: 1"]
    spec = JudgeSpec(
        rubric="checklist",
        criteria=(CRITERIA[2],),
        constraints=(Constraint("startend:end_checker", {"end_phrase": "Sincerely."}),),
    )
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "Ada Lovelace was first. Sincerely."))
    assert reward.reward == 1.0 and len(fake_judge.prompts) == 1


def test_checklist_without_criteria_is_an_invalid_task(tmp_path, unconfigured_judge):
    with pytest.raises(grade_judge.InvalidTask):
        grade_judge.grade(JudgeSpec(rubric="checklist"), tmp_path, _workspace(tmp_path, "text"))


@pytest.mark.parametrize("rubric", ["reference", "checklist"])
@pytest.mark.parametrize(
    "budgets",
    [
        {"max_completion_tokens": 0},
        {"max_completion_tokens": 1024, "incomplete_retry_tokens": 1024},
    ],
)
def test_score_rubrics_reject_invalid_budgets_before_judge_call(tmp_path, fake_judge, rubric, budgets):
    spec = JudgeSpec(rubric=rubric, references=(REFERENCE,), criteria=(CRITERIA[0],), exact_gate=False, **budgets)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))

    reward = run(spec_path, _workspace(tmp_path, "candidate"))

    assert reward.status is Status.INVALID_TASK
    assert fake_judge.requests == []


@pytest.mark.parametrize("rubric", ["reference", "checklist"])
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls", "function_call", "unknown", None])
def test_incomplete_judge_score_is_unscored_and_removes_stale_rewards(tmp_path, fake_judge, rubric, finish_reason):
    fake_judge.replies = ["SCORE: 1"]
    fake_judge.finish_reason = finish_reason
    spec = JudgeSpec(rubric=rubric, references=(REFERENCE,), criteria=("Be correct",), exact_gate=False)
    workspace = _workspace(tmp_path, "a candidate paraphrase")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text('{"reward": 1.0}')
    (logs / "reward.txt").write_text("1.0")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    verdict = run(spec_path, workspace)
    write_reward(logs, verdict)
    assert verdict.status == Status.INFRA_ERROR
    assert json.loads((logs / "verdict.json").read_text())["status"] == "infra_error"
    assert not (logs / "reward.json").exists()
    assert not (logs / "reward.txt").exists()


@pytest.mark.parametrize(
    ("rubric", "reply"),
    [
        ("reference", "SCORE: 1e-9"),
        ("reference", "SCORE: 1garbage"),
        ("reference", "SCORE: 0.7"),
        ("checklist", "SCORE: 0.5"),
        ("reference", "SCORE: 1\nActually unable to grade"),
    ],
)
def test_malformed_final_judge_labels_never_award_reward(tmp_path, fake_judge, rubric, reply):
    fake_judge.replies = [reply]
    spec = JudgeSpec(rubric=rubric, references=(REFERENCE,), criteria=("Be correct",), exact_gate=False)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text('{"reward":1.0}')
    verdict = run(spec_path, _workspace(tmp_path, "a candidate paraphrase"))
    write_reward(logs, verdict)
    assert (verdict.status, verdict.reward) == (Status.INFRA_ERROR, 0.0)
    assert not (logs / "reward.json").exists()


def _label_spec(**overrides) -> JudgeSpec:
    values = dict(
        rubric="labels",
        references=("reference answer",),
        question="trusted question",
        system_prompt="Compare {reference} against {candidate} for {question}.",
        prompt_template="A: {reference}\nB: {candidate}",
        label_scores={"[[A=B]]": 1.0, "[[A!=B]]": 0.0},
        strip_reasoning_blocks=True,
    )
    values.update(overrides)
    return JudgeSpec(**values)


@pytest.mark.parametrize(
    "reply,reward,status",
    [
        ("[[A=B]]", 1.0, Status.SCORED),
        ("[[A!=B]]", 0.0, Status.SCORED),
        ("reasoning\n[[A=B]]\n", 1.0, Status.SCORED),
        ("<think>[[A!=B]]</think>\n[[A=B]]", 1.0, Status.SCORED),
        ("earlier [[A!=B]]\n[[A=B]]", 0.0, Status.INFRA_ERROR),
        ("[[A=B]] but unclear", 0.0, Status.INFRA_ERROR),
        ("<think>unfinished [[A=B]]", 0.0, Status.INFRA_ERROR),
        ("", 0.0, Status.INFRA_ERROR),
    ],
)
def test_configured_judge_labels_on_real_http_boundary(tmp_path, fake_judge, reply, reward, status):
    fake_judge.replies = [reply]
    workspace = _workspace(tmp_path, "candidate {reference}")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec()))
    result = run(spec_path, workspace)
    assert result.status is status
    assert result.reward == reward
    if status is Status.SCORED:
        assert result.detail["completion"] == reply
    assert fake_judge.requests[0]["messages"] == [
        {"role": "system", "content": "Compare reference answer against candidate {reference} for trusted question."},
        {"role": "user", "content": "A: reference answer\nB: candidate {reference}"},
    ]


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls", None])
def test_configured_judge_incomplete_score_cannot_pass(tmp_path, fake_judge, reason):
    fake_judge.replies = ["[[A=B]]"]
    fake_judge.finish_reason = reason
    _workspace(tmp_path, "candidate")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec()))
    result = run(spec_path, tmp_path)
    assert result.status is Status.INFRA_ERROR
    assert result.reward == 0.0


def test_configured_judge_only_retries_incomplete_completion_budget(tmp_path, fake_judge):
    fake_judge.replies = ["[[A=B]]", "[[A!=B]]"]
    fake_judge.finish_reasons = ["length", "stop"]
    _workspace(tmp_path, "candidate")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec(incomplete_retry_tokens=16384)))
    result = run(spec_path, tmp_path)
    assert result.status is Status.SCORED
    assert result.reward == 0.0
    assert [request["max_completion_tokens"] for request in fake_judge.requests] == [8192, 16384]


@pytest.mark.parametrize(
    "overrides",
    [
        {"references": ()},
        {"references": ("",)},
        {"references": ("", "reference")},
        {"prompt_template": "{unknown}"},
        {"system_prompt": "", "prompt_template": "{candidate}"},
        {"label_scores": {}},
        {"label_scores": {"[[A=B]]": True}},
        {"label_scores": {"[[A=B]]": 1.5}},
        {"label_scores": {"": 1.0}},
        {"request_timeout": -1.0},
        {"request_timeout": float("inf")},
        {"incomplete_retry_tokens": 8192},
    ],
)
def test_configured_judge_invalid_contract_never_calls_model(tmp_path, fake_judge, overrides):
    _workspace(tmp_path, "candidate")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec(**overrides)))
    result = run(spec_path, tmp_path)
    assert result.status is Status.INVALID_TASK
    assert result.reward == 0.0
    assert fake_judge.requests == []


def test_configured_judge_http_failure_cannot_use_positive_body(tmp_path, fake_judge):
    fake_judge.http_status = 500
    fake_judge.replies = ["[[A=B]]"]
    _workspace(tmp_path, "candidate")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec()))
    result = run(spec_path, tmp_path)
    assert result.status is Status.INFRA_ERROR
    assert result.reward == 0.0
    assert len(fake_judge.requests) == 1


@pytest.mark.parametrize(
    "message_fields",
    [
        {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"function_call": {"name": "tool", "arguments": "{}"}},
        {"refusal": "Cannot judge this answer."},
    ],
)
def test_configured_judge_rejects_positive_text_with_structured_refusal(tmp_path, fake_judge, message_fields):
    fake_judge.replies = ["[[A=B]]"]
    fake_judge.message_fields = message_fields
    workspace = _workspace(tmp_path, "candidate")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec()))
    result = run(spec_path, workspace)
    assert (result.status, result.reward) == (Status.INFRA_ERROR, 0.0)


@pytest.mark.parametrize(
    "overrides", [{"system_prompt": None}, {"prompt_template": 1}, {"question": []}, {"references": (1,)}]
)
def test_direct_configured_judge_rejects_nontext_contract(tmp_path, fake_judge, overrides):
    workspace = _workspace(tmp_path, "candidate")
    with pytest.raises(grade_judge.InvalidTask):
        grade_judge.grade(_label_spec(**overrides), tmp_path, workspace)
    assert fake_judge.requests == []


def test_judge_positional_output_argument_selects_the_candidate_file(tmp_path, unconfigured_judge):
    spec = JudgeSpec(("expected",), (), "", "", (), "reference", "", True, 120.0, "/app/custom-answer.txt")
    (tmp_path / "answer.txt").write_text("wrong candidate")
    (tmp_path / "custom-answer.txt").write_text("expected")
    result = grade_judge.grade(spec, tmp_path, tmp_path)
    assert (result.status, result.reward) == (Status.SCORED, 1.0)


@pytest.mark.parametrize(
    "reply,reward,status",
    [
        ("Assistant gives an Answer.\nA", 1.0, Status.SCORED),
        ("A\nB", 0.0, Status.INFRA_ERROR),
        ("B", 0.0, Status.SCORED),
        ("C", 0.5, Status.SCORED),
        ("", 0.0, Status.INFRA_ERROR),
        ("Answer: A", 0.0, Status.INFRA_ERROR),
    ],
)
def test_completed_line_labels_do_not_scan_letters_inside_prose(tmp_path, fake_judge, reply, reward, status):
    fake_judge.replies = [reply]
    workspace = _workspace(tmp_path, "candidate")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec(label_scores={"A": 1.0, "B": 0.0, "C": 0.5}, label_scan="lines")))
    result = run(spec_path, workspace)
    assert (result.status, result.reward) == (status, reward)


@pytest.mark.parametrize("label_scan", ["tokens", "", None, 1])
def test_unsupported_label_scan_is_invalid_before_judge_call(tmp_path, fake_judge, label_scan):
    workspace = _workspace(tmp_path, "candidate")
    with pytest.raises(grade_judge.InvalidTask):
        grade_judge.grade(_label_spec(label_scan=label_scan), tmp_path, workspace)
    assert fake_judge.requests == []


@pytest.mark.parametrize("candidate", ["", " \n\t"])
def test_explicit_empty_judge_policy_delegates_present_text_to_task_labels(tmp_path, fake_judge, candidate):
    fake_judge.replies = ["C"]
    spec = _label_spec(empty_output=EmptyOutputPolicy.GRADE, label_scores={"A": 1.0, "B": 0.0, "C": 0.5})
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    _workspace(tmp_path, candidate)
    result = run(spec_path, tmp_path)
    assert (result.status, result.reward, result.detail["verdict"]) == (Status.SCORED, 0.5, "C")
    assert len(fake_judge.requests) == 1
    assert fake_judge.requests[0]["messages"][-1]["content"] == spec.prompt_template.format(
        question=spec.question, reference=spec.references[0], candidate=candidate
    )
    (tmp_path / "answer.txt").unlink()
    missing = run(spec_path, tmp_path)
    assert (missing.status, missing.reward) == (Status.SCORED, 0.0)
    assert len(fake_judge.requests) == 1


@pytest.mark.parametrize("fault", ["default_policy", "invalid_reference", "incomplete_provider", "empty_provider"])
def test_empty_judge_policy_never_turns_task_or_transport_failure_into_abstention(tmp_path, fake_judge, fault):
    fake_judge.replies = ["C"]
    overrides = {"empty_output": EmptyOutputPolicy.GRADE, "label_scores": {"A": 1.0, "B": 0.0, "C": 0.5}}
    status = Status.INFRA_ERROR
    if fault == "default_policy":
        overrides["empty_output"] = EmptyOutputPolicy.ZERO
        status = Status.SCORED
    elif fault == "invalid_reference":
        overrides["references"] = ("",)
        status = Status.INVALID_TASK
    elif fault == "incomplete_provider":
        fake_judge.finish_reason = "length"
    else:
        fake_judge.replies = [""]
    _workspace(tmp_path, "")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(_label_spec(**overrides)))
    result = run(spec_path, tmp_path)
    assert (result.status, result.reward) == (status, 0.0)
    if fault in {"default_policy", "invalid_reference"}:
        assert fake_judge.requests == []


def test_empty_judged_answer_cannot_pass_a_vacuous_normalized_exact_gate(tmp_path, fake_judge):
    fake_judge.replies = ["SCORE: 0"]
    spec = JudgeSpec(references=("!!!",), empty_output=EmptyOutputPolicy.GRADE)
    _workspace(tmp_path, "")
    result = grade_judge.grade(spec, tmp_path, tmp_path)
    assert result.reward == 0
    assert len(fake_judge.requests) == 1


@pytest.mark.parametrize("rubric", ["reference", "checklist"])
@pytest.mark.parametrize(
    "message_fields",
    [
        {"refusal": "Cannot judge."},
        {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"function_call": {"name": "tool", "arguments": "{}"}},
        {"role": "user"},
    ],
)
def test_score_rubrics_reject_positive_text_in_invalid_envelope(tmp_path, fake_judge, rubric, message_fields):
    fake_judge.replies = ["SCORE: 1"]
    fake_judge.message_fields = message_fields
    spec = (
        JudgeSpec(references=("reference",), exact_gate=False)
        if rubric == "reference"
        else JudgeSpec(rubric="checklist", criteria=("Correct answer",))
    )
    workspace = _workspace(tmp_path, "candidate")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    result = run(spec_path, workspace)
    assert (result.status, result.reward) == (Status.INFRA_ERROR, 0.0)
    assert len(fake_judge.requests) == 1


def test_direct_candidate_uses_explicit_connection_and_preserves_text(fake_judge, monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    fake_judge.replies = ["CORRECT", "INCORRECT"]
    connection = grade_judge.JudgeConnection(f"http://127.0.0.1:{fake_judge.server_port}/v1", "private-test-key")
    spec = JudgeSpec(
        references=("answer",),
        rubric="labels",
        model="explicit-model",
        prompt_template="Reference: {reference}\nCandidate: {candidate}",
        label_scores={"CORRECT": 1.0, "INCORRECT": 0.0},
        label_scan="lines",
        empty_output=EmptyOutputPolicy.GRADE,
    )
    assert grade_judge.grade_judge_candidate(spec, "  candidate\n", connection=connection).reward == 1
    assert grade_judge.grade_judge_candidate(spec, "", connection=connection).reward == 0
    assert fake_judge.prompts == ["Reference: answer\nCandidate:   candidate\n", "Reference: answer\nCandidate: "]
    assert all(request["model"] == "explicit-model" for request in fake_judge.requests)
    assert "private-test-key" not in repr(connection)


def test_direct_candidate_validates_trusted_constraints_before_empty_gate():
    spec = JudgeSpec(references=("answer",), constraints=(Constraint("unknown_instruction", {}),))
    with pytest.raises(InvalidTask):
        grade_judge.grade_judge_candidate(spec, "")


def test_direct_candidate_provider_failure_never_returns_a_score(fake_judge):
    fake_judge.message_fields = {"refusal": "Cannot judge"}
    fake_judge.replies = ["SCORE: 1"]
    spec = JudgeSpec(references=("answer",), exact_gate=False)
    with pytest.raises(RuntimeError, match="refusal"):
        grade_judge.grade_judge_candidate(spec, "wrong")


def test_paired_ordinal_matches_source_ties_and_circular_mean():
    edges = [(0, 1), (1, 2), (2, 0)]
    ratings = [
        dict(left=0, right=1, score_left=1, score_right=1, ranking=1),
        dict(left=1, right=2, score_left=5, score_right=5, ranking=6),
        dict(left=2, right=0, score_left=4, score_right=2, ranking=3.5),
    ]
    # Independent source arithmetic: equal ratings receive +/- (3.5-ranking).
    expected_raw = [(3.5 + 2) / 2, (-1.5 + 2.5) / 2, (7.5 + 4) / 2]
    verdicts = grade_judge.grade_paired_ordinal(3, edges, ratings)
    assert [v.detail["raw_score"] for v in verdicts] == pytest.approx(expected_raw)
    assert [v.reward for v in verdicts] == pytest.approx([(raw + 1.5) / 9 for raw in expected_raw])
    assert all(v.status is Status.SCORED for v in verdicts)
    # Opposite directed edges are distinct, including the source's size-two circle.
    pair = grade_judge.grade_paired_ordinal(
        2, [(0, 1), (1, 0)], [ratings[0], dict(left=1, right=0, score_left=1, score_right=1, ranking=3.5)]
    )
    assert [v.detail["raw_score"] for v in pair] == pytest.approx([2.25, -0.25])
    assert [v.reward for v in pair] == pytest.approx([3.75 / 9, 1.25 / 9])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "self", "bool", "nonfinite", "range"])
def test_paired_ordinal_invalidates_whole_provider_cohort(fault):
    ratings = [
        dict(left=0, right=1, score_left=5, score_right=4, ranking=1),
        dict(left=1, right=0, score_left=5, score_right=4, ranking=1),
    ]
    if fault == "missing":
        ratings.pop()
    elif fault == "duplicate":
        ratings[1] = dict(ratings[0])
    elif fault == "self":
        ratings[1]["right"] = 1
    elif fault == "bool":
        ratings[1]["left"] = True
    elif fault == "nonfinite":
        ratings[1]["score_left"] = float("nan")
    else:
        ratings[1]["ranking"] = 7
    with pytest.raises(RuntimeError, match="protocol"):
        grade_judge.grade_paired_ordinal(2, [(0, 1), (1, 0)], ratings)


@pytest.mark.parametrize("size,edges", [(1, [(0, 0)]), (2, [(0, 0)]), (2, [(0, 1), (0, 1)]), (3, [(0, 1)])])
def test_paired_ordinal_rejects_invalid_trusted_graph_before_provider(size, edges):
    with pytest.raises(InvalidTask):
        grade_judge.grade_paired_ordinal(size, edges, None)


def test_paired_ordinal_rejects_overflowing_derived_bounds():
    with pytest.raises(InvalidTask, match="derived"):
        grade_judge.grade_paired_ordinal(2, [(0, 1)], None, ranking_bounds=(-1e308, 1e308))


def test_paired_ordinal_json_integer_indices_and_huge_trusted_bounds():
    ratings = [dict(left=0.0, right=1.0, score_left=3, score_right=3, ranking=3.5)]
    verdicts = grade_judge.grade_paired_ordinal(2, [(0, 1)], ratings)
    assert [v.detail["raw_score"] for v in verdicts] == [3, 3]
    with pytest.raises(InvalidTask, match="bounds"):
        grade_judge.grade_paired_ordinal(2, [(0, 1)], ratings, rating_bounds=(1, 10**400))


@pytest.mark.parametrize(
    "reply,reward,status",
    [
        (" correct ", 1, Status.SCORED),
        ("INCORRECT", 0, Status.SCORED),
        ("Explanation\nCORRECT", 0, Status.INFRA_ERROR),
        ("CORRECT INCORRECT", 0, Status.INFRA_ERROR),
    ],
)
def test_responses_whole_labels_preserve_source_request_and_reject_prose(tmp_path, fake_judge, reply, reward, status):
    fake_judge.replies = [reply]
    fake_judge.response_fields = {
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "INCORRECT"}]},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": reply}],
            },
        ]
    }
    spec = _label_spec(
        api="responses",
        label_scan="whole",
        label_case="upper",
        strip_reasoning_blocks=False,
        system_prompt="",
        label_scores={"CORRECT": 1, "INCORRECT": 0},
        max_completion_tokens=1024,
        reasoning_effort="low",
    )
    path = tmp_path / "verifier.toml"
    path.write_text(render_spec(spec))
    result = run(path, _workspace(tmp_path, "candidate text"))
    assert (result.reward, result.status) == (reward, status)
    assert fake_judge.requests == [
        {
            "model": "fake/judge-9b",
            "input": "A: reference answer\nB: candidate text",
            "max_output_tokens": 1024,
            "reasoning": {"effort": "low"},
        }
    ]


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        {"error": {"message": "failed"}},
        {"output": [{"type": "reasoning", "summary": [{"text": "CORRECT"}]}]},
        {"output": [{"type": "function_call", "name": "CORRECT"}]},
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "CORRECT"}],
                }
            ]
        },
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [{"type": "output_text", "text": "CORRECT"}],
                }
            ]
        },
    ],
)
def test_responses_nonanswers_cannot_preserve_stale_credit(tmp_path, fake_judge, fields):
    fake_judge.replies = ["CORRECT"]
    fake_judge.response_fields = fields
    path = tmp_path / "verifier.toml"
    path.write_text(render_spec(_label_spec(api="responses", label_scan="whole", label_scores={"CORRECT": 1})))
    logs = tmp_path / "logs"
    write_reward(logs, grade_judge.scored(1))
    result = run(path, _workspace(tmp_path, "candidate"))
    write_reward(logs, result)
    assert (result.reward, result.status) == (0, Status.INFRA_ERROR)
    assert not (logs / "reward.txt").exists()
    assert not (logs / "reward.json").exists()


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"failed","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"[[A=B]]"}]}]}',
        '{"status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"[[A=B]]"}]}],"usage":NaN}',
    ],
)
def test_responses_ambiguous_raw_payload_fails_closed(tmp_path, fake_judge, raw):
    fake_judge.raw_body = raw
    path = tmp_path / "verifier.toml"
    path.write_text(render_spec(_label_spec(api="responses")))
    result = run(path, _workspace(tmp_path, "candidate"))
    assert (result.reward, result.status) == (0, Status.INFRA_ERROR)


def test_case_folded_label_collision_is_invalid_before_blank_candidate(tmp_path, fake_judge):
    path = tmp_path / "verifier.toml"
    path.write_text(render_spec(_label_spec(label_case="upper", label_scores={"correct": 0, "CORRECT": 1})))
    result = run(path, _workspace(tmp_path, ""))
    assert (result.reward, result.status) == (0, Status.INVALID_TASK)
    assert not fake_judge.requests


@pytest.mark.parametrize("reasoning_status", ["in_progress", "incomplete"])
def test_responses_completed_envelope_cannot_hide_incomplete_reasoning(tmp_path, fake_judge, reasoning_status):
    fake_judge.response_fields = {
        "output": [
            {"type": "reasoning", "status": reasoning_status},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "[[A=B]]"}],
            },
        ]
    }
    path = tmp_path / "verifier.toml"
    path.write_text(render_spec(_label_spec(api="responses")))
    result = run(path, _workspace(tmp_path, "candidate"))
    assert (result.reward, result.status) == (0, Status.INFRA_ERROR)


def test_label_exact_gate_roundtrip_uses_declared_reward_without_provider(fake_judge):
    spec = _label_spec(label_scores={"A": 1.0, "B": 0.0, "C": 0.5}, exact_gate_answers=("idk",), exact_gate_label="C")
    spec = parse_spec(render_spec(spec))
    result = grade_judge.grade_judge_candidate(spec, "[IDK]", gate_candidate="idk")
    assert result.status == Status.SCORED
    assert result.reward == 0.5
    assert result.detail["verdict"] == "C"
    assert not fake_judge.requests
    fake_judge.replies = ["A"]
    result = grade_judge.grade_judge_candidate(spec, "Original Answer", gate_candidate="original answer")
    assert result.reward == 1.0
    assert "Original Answer" in fake_judge.prompts[-1]


def test_label_exact_gate_validates_reference_and_table_before_provider_avoidance(fake_judge):
    for changes in ({"references": ("",)}, {"label_scores": {"C": 0.5, "B": float("nan")}}):
        spec = _label_spec(exact_gate_answers=("idk",), exact_gate_label="C", **changes)
        with pytest.raises(InvalidTask):
            grade_judge.grade_judge_candidate(spec, "idk")
    assert not fake_judge.requests


@pytest.mark.parametrize(
    "spec, candidate, reply, expected",
    [
        (JudgeSpec(references=("Mars",)), "Mars", "SCORE: 0", 1.0),
        (JudgeSpec(references=("Mars",)), "the red planet", "SCORE: 0.5", 0.5),
        (JudgeSpec(rubric="checklist", criteria=("Names Mars",)), "Mars", "SCORE: 1", 1.0),
        (JudgeSpec(references=("Mars",)), " \n", "SCORE: 1", 0.0),
        (
            JudgeSpec(references=("Mars",), constraints=(Constraint("startend:end_checker", {"end_phrase": "."}),)),
            "Mars",
            "SCORE: 1",
            0.0,
        ),
    ],
)
def test_in_memory_and_file_grading_match(tmp_path, fake_judge, spec, candidate, reply, expected):
    fake_judge.replies = [reply]
    memory_reward = grade_judge.grade_judge_candidate(spec, candidate, context="Private setting")
    (tmp_path / "context.txt").write_text("Private setting")
    file_reward = grade_judge.grade(replace(spec, context="context.txt"), tmp_path, _workspace(tmp_path, candidate))
    assert memory_reward == file_reward
    assert (memory_reward.reward, memory_reward.status) == (expected, Status.SCORED)
    if spec.rubric == "checklist":
        assert all("Private setting" in prompt for prompt in fake_judge.prompts)


def test_invalid_rubric_precedes_candidate_file_failure(tmp_path):
    spec = JudgeSpec(rubric="unsupported", output="x" * 5000)
    with pytest.raises(grade_judge.InvalidTask):
        grade_judge.grade(spec, tmp_path, tmp_path)


def test_missing_context_remains_invalid_when_candidate_is_unreadable(tmp_path, monkeypatch):
    (tmp_path / "answer.txt").write_text("candidate")

    def unreadable_file(_path, *args, **kwargs):
        raise PermissionError("candidate cannot be read")

    monkeypatch.setattr(Path, "read_text", unreadable_file)
    spec = JudgeSpec(references=("Mars",), context="missing-context.txt")
    reward = grade(spec, tests_dir=tmp_path, workspace=tmp_path)

    assert (reward.reward, reward.status) == (0.0, Status.INVALID_TASK)


@pytest.mark.parametrize("candidate", ["Mars", "the red planet"])
def test_missing_runtime_keeps_exact_gate_and_blocks_environment_endpoint(fake_judge, candidate):
    spec = JudgeSpec(references=("Mars",))
    if candidate == "Mars":
        reward = grade_judge.grade_judge_candidate(spec, candidate, runtime=None)
        assert (reward.reward, reward.status) == (1.0, Status.SCORED)
    else:
        with pytest.raises(RuntimeError, match="explicit endpoint selection"):
            grade_judge.grade_judge_candidate(spec, candidate, runtime=None)
    assert fake_judge.prompts == []
