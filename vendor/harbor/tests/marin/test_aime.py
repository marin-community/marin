# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AIME request routing and persisted answers in Harbor's isolated runtime."""

import io
import json
import subprocess
import urllib.error
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from marin.evaluation.harbor import single_turn_aime_agent
from marin.evaluation.harbor.single_turn_aime_agent import SingleTurnAimeAgent


@dataclass
class CompletionEndpoint:
    content: str | None = "The answer is 42"
    finish_reason: str | None = None
    reasoning_content: str | None = None
    transient_failures: int = 0
    models: list[str] = field(default_factory=list)

    def __call__(self, request, timeout):
        self.models.append(json.loads(request.data)["model"])
        if len(self.models) <= self.transient_failures:
            raise urllib.error.HTTPError(
                request.full_url, 503, "Service Unavailable", None, None
            )
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": self.content,
                                "reasoning_content": self.reasoning_content,
                            },
                            "finish_reason": self.finish_reason,
                        }
                    ]
                }
            ).encode()
        )


class LocalShell:
    async def exec(self, command):
        completed = subprocess.run(
            ["/bin/sh", "-c", command], check=False, capture_output=True, text=True
        )
        return SimpleNamespace(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@pytest.fixture
def endpoint(monkeypatch):
    endpoint = CompletionEndpoint()
    monkeypatch.setattr(single_turn_aime_agent.urllib.request, "urlopen", endpoint)
    # The endpoint fake completes immediately; retry backoff is outside this I/O contract.
    monkeypatch.setattr(single_turn_aime_agent.time, "sleep", lambda _delay: None)
    return endpoint


@pytest.fixture
def agent(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    return SingleTurnAimeAgent(
        logs_dir=logs,
        model_name="hosted_vllm/iceball-micro",
        api_base="https://inference.test/v1",
        answer_path=str(tmp_path / "answer.txt"),
        max_tokens=256,
    )


@pytest.mark.parametrize("alias", [None, "hosted_vllm/resumed-endpoint"])
async def test_aime_uses_served_alias_and_persists_answer(
    tmp_path, endpoint, agent, alias
):
    if alias is not None:
        agent.model_alias = alias
    await agent.run("Solve the problem", LocalShell(), SimpleNamespace())
    assert endpoint.models == ["resumed-endpoint" if alias else "iceball-micro"]
    assert (tmp_path / "answer.txt").read_text() == "42\n"


@pytest.mark.parametrize(
    ("content", "reasoning", "answer"),
    [
        (
            r"An incomplete derivation mentions 12. The final result is \boxed{137}.",
            None,
            "137\n",
        ),
        (None, None, "-1\n"),
        (None, r"An incomplete derivation suggests \boxed{137}.", "-1\n"),
    ],
)
async def test_aime_length_truncation_grades_final_text_only(
    tmp_path, endpoint, agent, content, reasoning, answer
):
    endpoint.content, endpoint.reasoning_content, endpoint.finish_reason = (
        content,
        reasoning,
        "length",
    )
    await agent.run("Solve the problem", LocalShell(), SimpleNamespace())
    assert (tmp_path / "answer.txt").read_text() == answer
    assert (tmp_path / "logs/response.txt").read_text() == (content or "")


@pytest.mark.parametrize("finish_reason", [None, "stop"])
async def test_aime_missing_final_text_without_truncation_does_not_write_answer(
    tmp_path, endpoint, agent, finish_reason
):
    endpoint.content, endpoint.finish_reason = None, finish_reason
    with pytest.raises(ValueError, match="text content"):
        await agent.run("Solve the problem", LocalShell(), SimpleNamespace())
    assert not (tmp_path / "answer.txt").exists()


async def test_aime_retries_transient_endpoint_failure_and_writes_answer(
    tmp_path, endpoint, agent
):
    endpoint.transient_failures = 1
    await agent.run("Solve the problem", LocalShell(), SimpleNamespace())
    assert (tmp_path / "answer.txt").read_text() == "42\n"
    assert endpoint.models == ["iceball-micro", "iceball-micro"]
