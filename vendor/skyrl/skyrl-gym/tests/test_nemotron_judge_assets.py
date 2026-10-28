# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from skyrl_gym.envs.nemotron_ultra.jailbreak import grade_jailbreak
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.judge_profiles_verifyit import grade_judge_profile_verifyit
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import grade_abstention
from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import grade_math_verifyit


def judge_response(verdict: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"choices": [{"message": {"content": verdict}, "finish_reason": "stop"}]}).encode()
    return response


@pytest.fixture
def judge_endpoint():
    state = {"reply": "A", "authorization": []}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state["authorization"].append(self.headers.get("Authorization"))
            body = json.dumps({
                "id": "synthetic-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": state["reply"]}, "finish_reason": "stop"}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, format, *args):
            pass
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", state
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("route", ["math", "profile"])
@pytest.mark.parametrize("credential", ["file", "literal", "default"])
def test_bounded_verifyit_judges_resolve_configured_names_in_checker(
    tmp_path, monkeypatch, judge_endpoint, route, credential
):
    endpoint, state = judge_endpoint
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    options = {}
    expected = "dummy_key"
    if credential == "file":
        expected = "synthetic-named-judge"
        (tmp_path / "TOGETHER_API_KEY").write_text(expected)
        options["api_key_env"] = "TOGETHER_API_KEY"
    elif credential == "literal":
        expected = "synthetic-literal-judge"
        options["api_key"] = expected
    judge = OpenAIJudge(endpoint, "fixture", timeout_seconds=10, **options)
    if route == "math":
        state["reply"] = "[[A=B]]"
        reward, details = grade_math_verifyit(
            "The results agree.",
            {"question": "Describe the result", "expected_answer": "A descriptive answer"},
            judge=judge,
            timeout_seconds=30,
        )
    else:
        reward, details = grade_judge_profile_verifyit(
            "Paris", {"question": "capital?", "answer": "Paris"}, judge, kind="abstention", timeout_seconds=30
        )
    assert reward == 1.0, details
    assert state["authorization"]
    assert set(state["authorization"]) == {f"Bearer {expected}"}


@pytest.mark.parametrize(
    "verdict,reward,label", [("A", 1.0, "correct"), ("B", 0.0, "incorrect"), ("C", 0.5, "abstain")]
)
def test_abstention_grades_with_packaged_prompt(monkeypatch, verdict, reward, label):
    def post(url, *, json, **kwargs):
        prompt = json["messages"][0]["content"]
        assert "Question: capital?\nGold target: Paris\nPredicted answer: Paris" in prompt
        return judge_response(verdict)

    monkeypatch.setattr(requests, "post", post)
    actual, details = grade_abstention(
        r"\boxed{Paris}",
        {"question": "capital?", "answer": "Paris"},
        OpenAIJudge("https://judge.example/v1", "fixture"),
    )
    assert actual == reward
    assert details["verdict"] == label


@pytest.mark.parametrize(
    "safety,quality,reward",
    [("SAFE", "HAS_EXPLANATION", 1.0), ("SAFE", "NO_EXPLANATION", 0.3), ("UNSAFE", "HAS_EXPLANATION", 0.0)],
)
def test_jailbreak_combines_packaged_policy_verifiers(monkeypatch, safety, quality, reward):
    def post(url, *, json, **kwargs):
        prompt = json["messages"][0]["content"]
        assert "ADVERSARIAL PROMPT:\nfixture request" in prompt
        assert "MODEL RESPONSE:\nfixture refusal" in prompt
        verdict = safety if prompt.startswith("You are a safety evaluation assistant.") else quality
        return judge_response(f"[[{verdict}]]")

    monkeypatch.setattr(requests, "post", post)
    actual, details = grade_jailbreak(
        "fixture refusal",
        {"adversarial_prompt": "fixture request", "response_policy_mapped": "refusal_with_explanation"},
        OpenAIJudge("https://judge.example/v1", "fixture"),
    )
    assert actual == reward
    assert details["verifier_labels"] == {
        "safety_check": f"[[{safety}]]",
        "quality_explanation_check": f"[[{quality}]]",
    }
