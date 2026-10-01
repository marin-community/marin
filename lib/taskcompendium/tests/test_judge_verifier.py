# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove judge rubrics through TaskCompendium's Harbor verifier lifecycle."""

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from tasktrove_verify.modes import grade_judge
from tasktrove_verify.spec import RUBRIC_CHECKLIST, Constraint, JudgeRuntimeConfig, JudgeSpec

from taskcompendium.grading import Outcome
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.submission import PlainText
from taskcompendium.verifiers.judge import judge_answer

from .harbor_replay import run_replay_trial


@dataclass
class JudgeEndpoint:
    url: str
    requests: list[dict]
    replies: list[str]


@pytest.fixture
def judge_endpoint():
    endpoint = JudgeEndpoint("", [], [])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            endpoint.requests.append(body)
            reply = endpoint.replies.pop(0)
            response = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):  # noqa: A002 - match BaseHTTPRequestHandler
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _task(verifier, prompt: str = "Name the red planet.") -> TaskSpec:
    return TaskSpec(
        id="judge-fixture",
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=Source(dataset="hand-authored", revision="fixture-1", row="judge-fixture", importer_revision="1"),
    )


def _runtime(endpoint: JudgeEndpoint, *, max_requests: int = 8) -> JudgeRuntimeConfig:
    return JudgeRuntimeConfig(
        model="fixture-judge", base_url=endpoint.url, api_key_env=None, request_timeout=10.0, max_requests=max_requests
    )


async def _run(
    tmp_path, endpoint, task, answer: str, trial_name: str, *, max_requests: int = 8, provide_runtime: bool = True
):
    environment_config = HarborEnvironmentConfig()
    task_dir = lower_to_harbor(task, PlainText(id="plain"), environment_config, tmp_path / f"task-{trial_name}")
    result = await run_replay_trial(
        task_dir,
        {"role": "assistant", "content": answer},
        tmp_path / "trials",
        trial_name,
        _runtime(endpoint, max_requests=max_requests) if provide_runtime else None,
    )
    return result, task_dir


async def test_judge_exact_gate_skips_endpoint_and_keeps_reference_private(tmp_path, judge_endpoint):
    verifier = judge_answer(
        JudgeSpec(references=("Mars",), question="What is the red planet?", model="other-judge"),
        context="Scoring note that must stay private.",
    )
    result, task_dir = await _run(tmp_path, judge_endpoint, _task(verifier), "Mars", "exact-gate", provide_runtime=False)

    outcome = json.loads((tmp_path / "trials/exact-gate/verifier/taskcompendium-result.json").read_text())
    assert result.exception_info is None, result.exception_info
    assert outcome["status"] == Outcome.GRADED
    assert outcome["reward"] == 1.0
    assert outcome["evidence"] == {"gate": "exact"}
    assert judge_endpoint.requests == []
    instruction = (task_dir / "instruction.md").read_text()
    assert "Mars" not in instruction
    assert "Scoring note" not in instruction


async def test_judge_reference_reuses_shared_rubric_and_records_evidence(tmp_path, judge_endpoint):
    judge_endpoint.replies.append("The answer is substantially correct.\nSCORE: 1")
    verifier = judge_answer(JudgeSpec(references=("Mars",), model="fixture-judge"))
    result, _ = await _run(tmp_path, judge_endpoint, _task(verifier), "The red planet", "reference")

    outcome = json.loads((tmp_path / "trials/reference/verifier/taskcompendium-result.json").read_text())
    assert result.exception_info is None, result.exception_info
    assert outcome["status"] == Outcome.GRADED
    assert outcome["reward"] == 1.0
    assert outcome["evidence"]["model"] == "fixture-judge"
    assert outcome["evidence"]["response"].endswith("SCORE: 1")
    assert len(judge_endpoint.requests) == 1
    assert "The red planet" in judge_endpoint.requests[0]["messages"][0]["content"]


async def test_judge_checklist_averages_valid_scores(tmp_path, judge_endpoint):
    judge_endpoint.replies.extend(("SCORE: 1", "SCORE: 0"))
    verifier = judge_answer(
        JudgeSpec(criteria=("Names Mars.", "Explains its color."), rubric=RUBRIC_CHECKLIST, model="fixture-judge"),
        context="Private rubric context.",
    )
    result, _ = await _run(tmp_path, judge_endpoint, _task(verifier), "Mars is red.", "checklist")

    outcome = json.loads((tmp_path / "trials/checklist/verifier/taskcompendium-result.json").read_text())
    assert result.exception_info is None, result.exception_info
    assert outcome["status"] == Outcome.GRADED
    assert outcome["reward"] == 0.5
    assert outcome["evidence"]["passed"] == 1
    assert outcome["evidence"]["total"] == 2
    assert len(judge_endpoint.requests) == 2


async def test_malformed_judge_response_is_infra_error_without_reward(tmp_path, judge_endpoint):
    judge_endpoint.replies.extend(("I cannot determine the score.", "Still no score."))
    verifier = judge_answer(
        JudgeSpec(references=("Mars",), exact_gate=False, model="fixture-judge", request_timeout=2.0)
    )
    result, _ = await _run(tmp_path, judge_endpoint, _task(verifier), "The red planet", "malformed")

    outcome = json.loads((tmp_path / "trials/malformed/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == Outcome.INFRA_ERROR
    assert outcome["reward"] is None
    assert "no parseable score" in outcome["error"]
    assert [item["response"] for item in outcome["evidence"]["attempts"]] == [
        "I cannot determine the score.",
        "Still no score.",
    ]
    assert result.verifier_result is None
    assert len(judge_endpoint.requests) == 2


async def test_fractional_checklist_score_counts_as_unmet_criterion(tmp_path, judge_endpoint):
    judge_endpoint.replies.append("Partly.\nSCORE: 0.5")
    verifier = judge_answer(
        JudgeSpec(criteria=("Names Mars.",), rubric=RUBRIC_CHECKLIST, model="fixture-judge", exact_gate=False)
    )
    result, _ = await _run(tmp_path, judge_endpoint, _task(verifier), "Mars", "fractional")

    outcome = json.loads((tmp_path / "trials/fractional/verifier/taskcompendium-result.json").read_text())
    assert result.exception_info is None
    assert outcome["status"] == Outcome.GRADED
    assert outcome["reward"] == 0.0
    assert outcome["evidence"]["criteria"][0]["passed"] is False
    assert len(judge_endpoint.requests) == 1


async def test_malformed_reply_retry_counts_against_request_budget(tmp_path, judge_endpoint):
    judge_endpoint.replies.append("No score line.")
    verifier = judge_answer(JudgeSpec(references=("Mars",), model="fixture-judge", exact_gate=False))
    result, _ = await _run(
        tmp_path, judge_endpoint, _task(verifier), "The red planet", "malformed-budget", max_requests=1
    )

    outcome = json.loads((tmp_path / "trials/malformed-budget/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == Outcome.INFRA_ERROR
    assert outcome["reward"] is None
    assert "request budget exhausted after 1" in outcome["error"]
    assert outcome["evidence"]["attempts"] == [{"attempt": 1, "response": "No score line."}]
    assert result.verifier_result is None
    assert len(judge_endpoint.requests) == 1


async def test_constraint_checker_exception_is_infra_error_without_reward(tmp_path, judge_endpoint, monkeypatch):
    def broken_check(_candidate: str, _params: dict):
        raise OSError("checker unavailable")

    monkeypatch.setattr(grade_judge, "resolve_checks", lambda constraints: [(constraints[0], broken_check)])
    verifier = judge_answer(
        JudgeSpec(
            criteria=("Names Mars.",),
            rubric=RUBRIC_CHECKLIST,
            constraints=(Constraint("broken", {}),),
            model="fixture-judge",
        )
    )
    result, _ = await _run(tmp_path, judge_endpoint, _task(verifier), "Mars", "constraint-infra")

    outcome = json.loads((tmp_path / "trials/constraint-infra/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == Outcome.INFRA_ERROR
    assert outcome["reward"] is None
    assert result.verifier_result is None
    assert outcome["evidence"]["constraint"] == "broken_check"
    assert "checker unavailable" in outcome["evidence"]["diagnostic"]
    assert judge_endpoint.requests == []


async def test_judge_request_budget_failure_keeps_partial_evidence_and_no_reward(tmp_path, judge_endpoint):
    judge_endpoint.replies.append("SCORE: 1")
    verifier = judge_answer(
        JudgeSpec(criteria=("Names Mars.", "Explains its color."), rubric=RUBRIC_CHECKLIST, model="fixture-judge")
    )
    result, _ = await _run(
        tmp_path,
        judge_endpoint,
        _task(verifier),
        "Mars is red.",
        "budget",
        max_requests=1,
    )

    outcome = json.loads((tmp_path / "trials/budget/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == Outcome.INFRA_ERROR
    assert outcome["reward"] is None
    assert "request budget exhausted" in outcome["error"]
    assert len(outcome["evidence"]["criteria"]) == 1
    assert result.verifier_result is None
    assert len(judge_endpoint.requests) == 1
