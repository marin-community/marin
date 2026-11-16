# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.task_curation.datasets.openresearcher.grounding import (
    GroundingJudge,
    screen_population,
    screen_question,
    validate_technical_exclusions,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import GroundedRequirement
from experiments.post_training.task_curation.datasets.openresearcher.grounding_pipeline import retrieval_response
from experiments.post_training.task_curation.tests.review_server import JudgeReply


def grounding_tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{m['role']}}: {{m['content']}}{% endfor %}"
    return tokenizer


@pytest.mark.asyncio
async def test_grounding_retrieval_preserves_large_documents_and_response_boundaries():
    first = {"documents": {"large": "corpus evidence " * (2 * 1024 * 1024)}, "searches": []}
    second = {"documents": {"small": "The next response."}, "searches": []}
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    reader.feed_data((json.dumps(first) + "\n" + json.dumps(second) + "\n").encode())
    reader.feed_eof()
    assert await retrieval_response(reader) == first
    assert await retrieval_response(reader) == second


@pytest.mark.asyncio
async def test_grounding_retrieval_rejects_truncated_response():
    reader = asyncio.StreamReader(limit=16)
    reader.feed_data(json.dumps({"documents": {"large": "incomplete evidence " * 20}}).encode())
    reader.feed_eof()
    with pytest.raises(asyncio.IncompleteReadError):
        await retrieval_response(reader)


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
@pytest.mark.parametrize("incomplete_questions", [1, 2])
async def test_exhausted_grounding_truncations_drop_questions_with_a_population_cap(
    review_server, tmp_path, incomplete_questions
):
    rows = [{"id": str(index), "question": "Who founded Cedar Press?", "answer": "Nina Calder"} for index in range(100)]
    review_server.judge_replies = [JudgeReply("{", finish_reason="length") for _ in range(3 * incomplete_questions)] + [
        JudgeReply(json.dumps({"decomposable": False, "requirements": []})) for _ in rows[incomplete_questions:]
    ]

    async def retrieve(queries):
        raise AssertionError("Incomplete decomposition must not reach retrieval")

    args = (
        rows,
        JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture", default_model="fixture"),
        grounding_tokenizer(),
        retrieve,
        StoragePath(str(tmp_path)),
        1,
    )
    if incomplete_questions == 1:
        decisions = await screen_population(*args)
        assert len(decisions) == 100
        assert sum(d["reason"] == "grounding_incomplete_exhausted" for d in decisions) == 1
    else:
        with pytest.raises(ValueError, match="exclusions exceed one percent"):
            await screen_population(*args)
    for index in range(incomplete_questions):
        decision = json.loads((tmp_path / f"decisions/{index}.json").read_text())
        assert decision["status"] == "technical_exclusion"
        assert decision["keep"] is False
        assert decision["reason"] == "grounding_incomplete_exhausted"
        requests = []
        for uri in decision["observations"]:
            observation = json.loads(StoragePath(uri).read_text())
            assert observation["response"]["choices"][0]["finish_reason"] == "length"
            assert observation["response"]["usage"]["total_tokens"] == 16
            requests.append(observation["request"])
        assert len(requests) == 3
        assert requests[0] == requests[1] == requests[2]


def test_grounding_exclusion_cap_uses_the_complete_population_and_forbids_retention():
    decisions = [
        (
            {"status": "technical_exclusion", "reason": "grounding_incomplete_exhausted", "keep": False}
            if index < 61
            else {"status": "screened", "reason": "pass", "keep": True}
        )
        for index in range(6102)
    ]
    validate_technical_exclusions(decisions)
    decisions[0]["keep"] = True
    with pytest.raises(ValueError, match="explicit technical exclusion"):
        validate_technical_exclusions(decisions)
    decisions[0]["keep"] = False
    decisions[61] = dict(decisions[0])
    with pytest.raises(ValueError, match="exclusions exceed one percent"):
        validate_technical_exclusions(decisions)


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


@pytest.mark.asyncio
async def test_grounding_counts_prompt_tokens_before_sending_oversized_packet(review_server, tmp_path):
    async def retrieve(queries):
        raise AssertionError("Oversized packet must not reach retrieval")

    result = await screen_population(
        [{"id": "oversized", "question": "evidence " * 28000, "answer": "Nina Calder"}],
        JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture", default_model="fixture"),
        grounding_tokenizer(),
        retrieve,
        StoragePath(str(tmp_path)),
        1,
    )
    assert result[0]["status"] == "screened"
    assert result[0]["keep"] is False
    assert result[0]["reason"] == "complete_evidence_exceeds_sealed_context_budget"
    assert review_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("requirements_count", [1, 11])
async def test_grounding_checks_requirement_budget_before_nested_query_validation(
    review_server, tmp_path, requirements_count
):
    response = json.dumps(
        {
            "decomposable": True,
            "requirements": [
                {"id": f"h{index + 1}", "claim": "A required fact.", "queries": ["one", "two", "three", "four", "five"]}
                for index in range(requirements_count)
            ],
        }
    )
    review_server.judge_replies = [JudgeReply(response) for _ in range(3)]

    async def retrieve(queries):
        raise AssertionError("Invalid or over-budget decomposition must not reach retrieval")

    args = (
        [{"id": "fixture", "question": "Who founded Cedar Press?", "answer": "Nina Calder"}],
        JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture", default_model="fixture"),
        grounding_tokenizer(),
        retrieve,
        StoragePath(str(tmp_path)),
        1,
    )
    if requirements_count == 11:
        result = await screen_population(*args)
        assert result[0]["keep"] is False
        assert result[0]["status"] == "screened"
        assert result[0]["reason"] == "complete_requirements_exceed_sealed_budget"
        assert len(review_server.requests) == 1
    else:
        with pytest.raises(ExceptionGroup) as failure:
            await screen_population(*args)
        assert isinstance(failure.value.exceptions[0], ValidationError)
        assert len(review_server.requests) == 3
        decision = json.loads((tmp_path / "decisions/fixture.json").read_text())
        assert decision["status"] == "execution_failure"
        assert "keep" not in decision
