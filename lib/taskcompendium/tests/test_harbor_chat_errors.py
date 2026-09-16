# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain model endpoint failures through Harbor trial archiving."""

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

from taskcompendium.execution import HarborExecutionConfig, HarborLaunchConfig
from taskcompendium.harbor.generation import GenerationRequest, generate_attempts
from taskcompendium.importers.nemo_workplace import build_sample
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import Outcome


async def test_provider_http_failure_retains_response_detail(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise HTTPError(
            "http://localhost/v1/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":{"message":"model name mismatch"}}'),
        )

    monkeypatch.setattr("taskcompendium.harbor.agents.urllib.request.urlopen", fail)
    sample = build_sample(Path(__file__).parent / "fixtures/nemo")
    task = lower_to_harbor(
        sample.specification,
        (sample.rendering,),
        sample.binding,
        tmp_path / "task",
        reference_execution=HarborExecutionConfig(sample.binding, HarborLaunchConfig("provider_chat")),
        model_name="fixture",
        agent_kwargs={"api_base": "http://localhost/v1", "max_turns": 3},
    )
    execution = json.loads((task / "reference-execution.json").read_text())
    attempt = (
        await generate_attempts([GenerationRequest(task, execution, "provider", 0)], tmp_path / "trials", concurrency=1)
    )[0]

    assert attempt.status == Outcome.INFRA_ERROR
    assert attempt.reward is None
    assert attempt.exception is not None
    assert attempt.exception["exception_type"] == "RuntimeError"
    assert "model name mismatch" in attempt.exception["exception_message"]
