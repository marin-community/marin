# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native single-completion contracts cross Harbor without a shell/tool loop."""

import hashlib
import json

import httpx
import pytest
from rolloutengine.models import RolloutLimits
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.models import ConversationInput, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.models import RawRow
from taskcompendium.request_budget import PromptBudget, RequestBudget, TokenizationIdentity, tool_free_prompt_sha256
from taskcompendium.runtime.oracle import oracle_answer_attempt
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from harbor.agents import taskcompendium as agents


def source_task():
    task = exact_answer_task(
        RawRow("native", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), {}),
        prompt="coordinates", answers=("C:T670:N",), ignore_case=False,
    )
    return TaskSpec.model_validate({**task.model_dump(), "completion_protocol": "tool_free_single", "context": ConversationInput(events=(
        TextMessage(role="system", content="Use coordinate text only."), TextMessage(role="user", content="Find an atom."),
    ))})


@pytest.mark.asyncio
async def test_native_candidate_requires_budget_and_does_not_execute_a_returned_tool(tmp_path, monkeypatch):
    task = source_task()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}assistant"
    messages = [{"role": event.role, "content": event.content} for event in task.context.events]
    identity = TokenizationIdentity(
        tokenizer="fixture/native", revision="a" * 40,
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(), template_arguments={},
    )
    budget = PromptBudget(
        prompt_sha256=tool_free_prompt_sha256(messages), model="candidate", endpoint="https://candidate.fixture/v1", backend_identity_sha256="b" * 64,
        tokenization=identity, endpoint_limit_source="fixture native endpoint limits",
        limits=RequestBudget(100, 99, 70), prompt_tokens=len(tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True,
        )),
    )
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps({budget.prompt_sha256: budget.model_dump(mode="json")}))
    requests = []

    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "reply", "object": "chat.completion", "created": 0, "model": "candidate",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "C:T670:N", "tool_calls": [{
                    "id": "call", "type": "function", "function": {"name": "shell", "arguments": "not JSON"},
                }],
            }}],
            "usage": {"prompt_tokens": budget.prompt_tokens, "completion_tokens": 1, "total_tokens": budget.prompt_tokens + 1},
        })

    real_client = agents.AsyncOpenAI
    monkeypatch.setattr(agents, "AsyncOpenAI", lambda **kwargs: real_client(
        **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(response)),
    ))
    loaded = []
    monkeypatch.setattr(agents.AutoTokenizer, "from_pretrained", lambda name, *, revision: loaded.append((name, revision)) or tokenizer)
    common = dict(
        logs_dir=tmp_path / "agent", model_name="openai/candidate", api_base="https://candidate.fixture/v1",
        max_turns=16, max_tokens=100, request_parameters={"temperature": .7},
    )
    limits = RolloutLimits(
        max_turns=16, max_total_tokens=None, model_turn_timeout=10, tool_turn_timeout=None,
        total_turn_timeout=10, attempt_timeout=20, verifier_timeout=5, cleanup_timeout=1,
    )
    unconfigured = agents.TaskSpecChatAgent(**common)
    with pytest.raises(ValueError, match="native budget manifest"):
        await unconfigured.run_task(task, None, limits)
    assert unconfigured.attempt.reply_status is None
    assert requests == [] and loaded == []
    agent = agents.TaskSpecChatAgent(
        **common, budget_manifest=str(path), budget_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        backend_identity_sha256="b" * 64, tokenization_identity=identity.model_dump(mode="json"),
    )
    attempt = await agent.run_task(task, None, limits)
    assert len(requests) == 1 and requests[0]["messages"] == messages
    assert loaded == [(identity.tokenizer, identity.revision)]
    assert attempt.conversation.events[-1].content == "C:T670:N"
    assert attempt.reply_status.origin == "provider" and attempt.reply_status.tool_events
    assert (tmp_path / "agent/tool-free/raw_response.json").is_file()


@pytest.mark.asyncio
async def test_native_original_oracle_and_empty_controls_have_explicit_control_provenance(tmp_path):
    task = source_task()
    limits = RolloutLimits(
        max_turns=1, max_total_tokens=None, model_turn_timeout=10, tool_turn_timeout=None,
        total_turn_timeout=10, attempt_timeout=20, verifier_timeout=5, cleanup_timeout=1,
    )
    nop = await agents.TaskSpecEmptyAgent(logs_dir=tmp_path).run_task(task, None, limits)
    for attempt in (oracle_answer_attempt(task, b"C:T670:N"), nop, agents.control_attempt(task, [{"role": "assistant", "content": "C:T670:N"}])):
        assert attempt.reply_status.origin == "control"
        assert not attempt.reply_status.truncated and not attempt.reply_status.provider_refusal and not attempt.reply_status.tool_events
