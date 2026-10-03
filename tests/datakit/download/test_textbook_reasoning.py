# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

from marin.datakit.download.textbook_reasoning import HF_DATASET_ID, row_to_chat_doc


def test_row_to_chat_doc_preserves_turns_and_source_identity():
    question = "Why is the sky blue?"

    documents = row_to_chat_doc({"question": question, "answer": "Rayleigh scattering."})

    assert len(documents) == 1
    document = documents[0]
    assert document["source"] == HF_DATASET_ID
    assert document["source_id"] == hashlib.sha256(question.encode()).hexdigest()
    assert document["messages"][0]["role"] == "user"
    assert document["messages"][0]["content"] == [{"type": "text", "text": question}]
    assert document["messages"][1]["role"] == "assistant"
    assert document["messages"][1]["content"] == [{"type": "text", "text": "Rayleigh scattering."}]


def test_row_to_chat_doc_drops_incomplete_rows():
    assert row_to_chat_doc({"question": "", "answer": "An answer"}) == []
    assert row_to_chat_doc({"question": "A question", "answer": "  "}) == []
