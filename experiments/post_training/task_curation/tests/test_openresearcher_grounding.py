# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.task_curation.datasets.openresearcher.grounding import (
    GroundingCompletionIncomplete,
    GroundingJudge,
    screen_population,
    screen_question,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import GroundedRequirement
from experiments.post_training.task_curation.tests.review_server import JudgeReply


def grounding_tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{m['role']}}: {{m['content']}}{% endfor %}"
    return tokenizer


@pytest.mark.asyncio
@pytest.mark.parametrize("fabricated_quote", [False, True])
async def test_grounding_verifies_corpus_evidence_before_independent_answer(review_server, tmp_path, fabricated_quote):
    text = "Nina Calder founded Cedar Press in 1974."
    quote = "A different person founded Cedar Press in 1974." if fabricated_quote else text
    review_server.judge_replies = [
        JudgeReply(
            json.dumps(
                {
                    "decomposable": True,
                    "requirements": [{"id": "h1", "claim": "CLAIM_ONLY", "queries": ["Cedar founder", "Cedar 1974"]}],
                }
            )
        ),
        JudgeReply(
            json.dumps({"supported": True, "reason": "Supported.", "evidence": [{"docid": "d1", "quote": quote}]})
        ),
        JudgeReply(
            json.dumps({"answerable": True, "answer": "Nina Calder", "explanation": "The document identifies her."})
        ),
        JudgeReply("extracted_final_answer: Nina Calder\nreasoning: Same name.\ncorrect: yes\nconfidence: 100"),
        JudgeReply(
            json.dumps(
                {
                    "all_clues_covered": True,
                    "all_hops_supported": True,
                    "answer_supported": True,
                    "uniquely_determined": True,
                    "reason": "All required relations are supported.",
                }
            )
        ),
    ]
    tokenizer = grounding_tokenizer()

    async def retrieve(queries):
        return {"d1": text}

    async with httpx.AsyncClient() as client:
        judge = GroundingJudge(
            JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture", default_model="fixture"),
            client,
            tokenizer,
            StoragePath(str(tmp_path / "observations")),
        )
        result = await screen_question(
            {"id": "fixture", "question": "Who founded Cedar Press in 1974?", "answer": "Nina Calder"}, judge, retrieve
        )
    assert result["keep"] is not fabricated_quote
    if fabricated_quote:
        assert result["reason"] == "required_hop_not_grounded_or_citation_not_literal"
    else:
        blind_request = review_server.requests[2]["messages"][-1]["content"]
        assert '"proposed_answer"' not in blind_request and "CLAIM_ONLY" not in blind_request
        assert result["answer_match"]["correct"]
        observation = json.loads((tmp_path / "observations/audit-0.json").read_text())
        assert observation["response"]["model"] == "fixture"


@pytest.mark.asyncio
async def test_truncated_grounding_retries_with_the_same_sealed_budget(review_server, tmp_path):
    verdict = {"supported": False, "reason": "No evidence.", "evidence": []}
    review_server.judge_replies = [
        JudgeReply(json.dumps({**verdict, "supported": True}), finish_reason="length"),
        JudgeReply(json.dumps(verdict)),
    ]
    async with httpx.AsyncClient() as client:
        judge = GroundingJudge(
            JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture", default_model="fixture"),
            client,
            grounding_tokenizer(),
            StoragePath(str(tmp_path)),
        )
        result = await judge.structured("ground-h1", "Audit the evidence.", GroundedRequirement)
    assert result.model_dump(mode="json") == verdict
    assert len(review_server.requests) == 2
    assert review_server.requests[0] == review_server.requests[1]
    assert review_server.requests[0]["max_tokens"] == 4096
    assert review_server.requests[0]["temperature"] == 0.7
    first = json.loads((tmp_path / "ground-h1-0.json").read_text())
    second = json.loads((tmp_path / "ground-h1-1.json").read_text())
    assert first["response"]["choices"][0]["finish_reason"] == "length"
    assert second["response"]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_exhausted_grounding_truncations_remain_execution_failures(review_server, tmp_path):
    review_server.judge_replies = [JudgeReply("{", finish_reason="length") for _ in range(3)]

    async def retrieve(queries):
        raise AssertionError("Incomplete decomposition must not reach retrieval")

    with pytest.raises(ExceptionGroup) as failure:
        await screen_population(
            [{"id": "fixture", "question": "Who founded Cedar Press?", "answer": "Nina Calder"}],
            JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture", default_model="fixture"),
            grounding_tokenizer(),
            retrieve,
            StoragePath(str(tmp_path)),
            1,
        )
    assert isinstance(failure.value.exceptions[0], GroundingCompletionIncomplete)
    assert len(review_server.requests) == 3
    decision = json.loads((tmp_path / "decisions/fixture.json").read_text())
    assert decision["status"] == "execution_failure"
    assert "keep" not in decision
    for attempt in range(3):
        observation = json.loads((tmp_path / f"observations/fixture/decompose-{attempt}.json").read_text())
        assert observation["response"]["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_grounding_preserves_http_failure_response_before_aborting(tmp_path):
    error = {"error": {"message": "Invalid fixture request", "type": "invalid_request_error"}}
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json=error))
    async with httpx.AsyncClient(transport=transport) as client:
        judge = GroundingJudge(
            JudgeConnection("http://fixture/v1", "private-test-api-key", default_model="fixture"),
            client,
            grounding_tokenizer(),
            StoragePath(str(tmp_path)),
        )
        with pytest.raises(httpx.HTTPStatusError) as failure:
            await judge.structured("ground-h1", "Audit the evidence.", GroundedRequirement)
    assert failure.value.response.status_code == 400
    saved = (tmp_path / "ground-h1-0-http-error.json").read_text()
    observation = json.loads(saved)
    assert observation["status_code"] == 400
    assert json.loads(observation["response_body"]) == error
    assert observation["request"]["messages"][-1]["content"] == "Audit the evidence."
    assert "private-test-api-key" not in saved
    assert len(list(tmp_path.glob("*-http-error.json"))) == 1
