# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from tasktrove_verify.grade import Status
from tasktrove_verify.modes import grade_judge
from tasktrove_verify.spec import Constraint, JudgeSpec

# The dataset's own reference answer, apostrophe included: the gate must fold case, spacing and
# punctuation without mangling non-ASCII text.
REFERENCE = (
    "Yes, if the non-state actor’s actions amount to an armed attack and the host state is "  # noqa: RUF001
    "unwilling or unable to suppress the threat."
)
BOXED_RESPONSE = f"The victim state may act only in the narrow case described.\n\\boxed{{{REFERENCE}}}"
SPACED_RESPONSE = REFERENCE.lower().replace("attack and", "attack   and").rstrip(".")
QUESTION = "Under the narrow interpretation of Article 51 of the UN Charter, when may force be used?"
ENV_VARS = ("TASKTROVE_JUDGE_BASE_URL", "TASKTROVE_JUDGE_API_KEY", "TASKTROVE_JUDGE_MODEL")


class FakeJudgeServer(ThreadingHTTPServer):
    """A one-endpoint stand-in for an OpenAI-compatible chat server."""

    replies: list[str]
    prompts: list[str]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        assert self.path.endswith("/chat/completions")
        server: FakeJudgeServer = self.server  # pyrefly: ignore[bad-assignment]
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.prompts.append(request["messages"][-1]["content"])
        reply = server.replies[min(len(server.prompts) - 1, len(server.replies) - 1)]
        body = json.dumps(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            }
        ).encode()
        self.send_response(200)
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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TASKTROVE_JUDGE_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("TASKTROVE_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("TASKTROVE_JUDGE_MODEL", "fake/judge-9b")
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
    "response",
    [
        pytest.param(BOXED_RESPONSE, id="boxed"),
        pytest.param(SPACED_RESPONSE, id="unboxed_case_and_spacing"),
    ],
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
    assert (reward.reward, reward.detail["model"]) == (0.0, "override/judge-70b")


def test_unparseable_reply_is_retried_once_then_raises_infrastructure_error(tmp_path, fake_judge):
    fake_judge.replies = ["I cannot grade this."]
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    with pytest.raises(grade_judge.JudgeInfrastructureError, match="no parseable score") as error:
        grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "something else"))

    assert [attempt["response"] for attempt in error.value.evidence["attempts"]] == [
        "I cannot grade this.",
        "I cannot grade this.",
    ]
    assert len(fake_judge.prompts) == 2


@pytest.mark.parametrize("reply, expected", [("Result: SCORE: 1.0", 1.0), ("SCORE = **0.75**", 0.75)])
def test_reference_accepts_existing_score_syntax(tmp_path, fake_judge, reply, expected):
    fake_judge.replies = [reply]
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "something else"))
    assert (reward.reward, reward.status) == (expected, Status.SCORED)
    assert len(fake_judge.prompts) == 1


def test_second_attempt_is_accepted(tmp_path, fake_judge):
    fake_judge.replies = ["I cannot grade this.", "Matches the reference.\nSCORE: 1"]
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "a paraphrase"))
    assert reward.reward == 1.0


def test_missing_endpoint_configuration_is_an_infra_error(tmp_path, unconfigured_judge):
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    with pytest.raises(RuntimeError, match="TASKTROVE_JUDGE_BASE_URL"):
        grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "a paraphrase"))


def test_missing_model_configuration_is_an_infra_error(tmp_path, fake_judge, monkeypatch):
    monkeypatch.delenv("TASKTROVE_JUDGE_MODEL")
    spec = JudgeSpec(references=(REFERENCE,), exact_gate=False)
    with pytest.raises(RuntimeError, match="TASKTROVE_JUDGE_MODEL"):
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


def test_checklist_partial_score_counts_as_failed_criterion(tmp_path, fake_judge):
    fake_judge.replies = ["Partly meets it.\nSCORE: 0.5"]
    spec = JudgeSpec(rubric="checklist", criteria=(CRITERIA[0],))
    reward = grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "two steps"))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["criteria"][0]["passed"] is False
    assert len(fake_judge.prompts) == 1


def test_checklist_shows_the_context_file_to_the_judge(tmp_path, fake_judge):
    (tmp_path / "conversation.txt").write_text("[USER]: plan my week\n[ASSISTANT]: sure")
    fake_judge.replies = ["SCORE: 1"]
    spec = JudgeSpec(rubric="checklist", criteria=(CRITERIA[0],), context="conversation.txt")
    grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "1. 2. 3."))
    assert "plan my week" in fake_judge.prompts[0]


def test_reference_rubric_shows_private_context_to_the_judge(tmp_path, fake_judge):
    (tmp_path / "context.txt").write_text("The setting is the year 2042.")
    fake_judge.replies = ["The answer matches.\nSCORE: 1"]
    spec = JudgeSpec(references=("Mars",), context="context.txt", exact_gate=False)
    grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "The red planet"))
    assert "The setting is the year 2042." in fake_judge.prompts[0]


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


def test_constraint_exception_is_infrastructure_error_not_zero_reward(tmp_path, fake_judge, monkeypatch):
    def broken_check(_candidate: str, _params: dict):
        raise OSError("checker storage unavailable")

    spec = _checklist(constraints=(Constraint("broken", {}),))
    monkeypatch.setattr(grade_judge, "resolve_checks", lambda constraints: [(constraints[0], broken_check)])
    with pytest.raises(grade_judge.JudgeInfrastructureError, match="checker storage unavailable") as error:
        grade_judge.grade(spec, tmp_path, _workspace(tmp_path, "candidate"))
    assert error.value.evidence == {
        "constraint": "broken_check",
        "diagnostic": "OSError: checker storage unavailable",
    }
    assert fake_judge.prompts == []


def test_checklist_without_criteria_is_an_invalid_task(tmp_path, unconfigured_judge):
    with pytest.raises(grade_judge.InvalidTask):
        grade_judge.grade(JudgeSpec(rubric="checklist"), tmp_path, _workspace(tmp_path, "text"))


@pytest.mark.parametrize(
    "spec, candidate, reply, expected",
    [
        (JudgeSpec(references=("Mars",)), "Mars", "SCORE: 0", 1.0),
        (JudgeSpec(references=("Mars",)), "the red planet", "SCORE: 0.75", 0.75),
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
    memory_reward = grade_judge.grade_candidate(spec, candidate, context="Private setting")
    (tmp_path / "context.txt").write_text("Private setting")
    file_reward = grade_judge.grade(replace(spec, context="context.txt"), tmp_path, _workspace(tmp_path, candidate))
    assert memory_reward == file_reward
    assert (memory_reward.reward, memory_reward.status) == (expected, Status.SCORED)
    assert all("Private setting" in prompt for prompt in fake_judge.prompts)
