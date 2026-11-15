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

from experiments.post_training.task_curation.datasets.openresearcher.grounding import GroundingJudge, screen_question
from experiments.post_training.task_curation.tests.review_server import JudgeReply


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
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{m['role']}}: {{m['content']}}{% endfor %}"

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
