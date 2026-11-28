# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.abstention_sft.rows import chat_rows

ANSWER_ID, ABSTAIN_ID, DROP_ID, HELD_ID = "law-00001", "law-00002", "law-00003", "law-00004"
PROBLEMS = [
    {"request_id": request_id, "problem": f"Question {request_id}?", "answer": "Key", "domain": "Law"}
    for request_id in (ANSWER_ID, ABSTAIN_ID, DROP_ID, HELD_ID)
]
SAMPLES = [
    {"request_id": ANSWER_ID, "sample": 0, "output": "<|start_think|>Wrong path.<|end_think|>Other", "correct": False},
    {"request_id": ANSWER_ID, "sample": 1, "output": "<|start_think|>I know this.<|end_think|>Key", "correct": True},
    {"request_id": HELD_ID, "sample": 0, "output": "<|start_think|>Sure.<|end_think|>Key", "correct": True},
]
LABELS = [
    {"request_id": ANSWER_ID, "label": "answer", "held_out": False},
    {"request_id": ABSTAIN_ID, "label": "abstain", "held_out": False},
    {"request_id": DROP_ID, "label": "drop", "held_out": False},
    {"request_id": HELD_ID, "label": "answer", "held_out": True},
]
REPLIES = {ABSTAIN_ID: "I don't know; a legal database would say."}


def _assistant(row):
    return row["messages"][-1]


def test_answer_row_uses_the_students_own_correct_sample():
    train, _ = chat_rows(PROBLEMS, SAMPLES, LABELS, REPLIES)
    [row] = [row for row in train if row["id"] == ANSWER_ID]
    assert row["messages"][0]["content"] == f"Question {ANSWER_ID}?"
    assert (_assistant(row)["content"], _assistant(row)["reasoning_content"]) == ("Key", "I know this.")


def test_abstain_row_uses_the_written_reply():
    train, _ = chat_rows(PROBLEMS, SAMPLES, LABELS, REPLIES)
    [row] = [row for row in train if row["id"] == ABSTAIN_ID]
    assert _assistant(row)["content"] == REPLIES[ABSTAIN_ID]


def test_dropped_question_has_no_row():
    train, held = chat_rows(PROBLEMS, SAMPLES, LABELS, REPLIES)
    assert DROP_ID not in {row["id"] for row in train + held}


def test_held_out_questions_never_reach_train():
    train, held = chat_rows(PROBLEMS, SAMPLES, LABELS, REPLIES)
    assert HELD_ID not in {row["id"] for row in train}
    assert [row["id"] for row in held] == [HELD_ID]
