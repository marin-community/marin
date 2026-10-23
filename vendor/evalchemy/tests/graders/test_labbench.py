"""LAB-Bench dataset adaptation."""

import pytest

from eval.sources.labbench import prepare_question


@pytest.mark.parametrize("subset", ["ProtocolQA", "SeqQA", "CloningScenarios"])
def test_question_preserves_choices_and_protocol_after_shuffling(subset):
    row = {
        "id": "example", "question": "Which option?", "protocol": "Protocol context.\n",
        "ideal": "correct answer", "distractors": ["wrong one", "wrong two", "wrong three"],
        "subtask": "example subtask", "canary": "hidden-canary", "source": "hidden-source",
    }
    question = prepare_question(row, subset, seed=42)
    prompt = question["prompt"]
    choices = [line for line in prompt.splitlines() if line.startswith("(")]
    assert len(choices) == 4
    assert set(line[4:] for line in choices) == {row["ideal"], *row["distractors"]}
    assert f"({question['answer']}) correct answer" in choices
    assert ("Protocol context.\nWhich option?" in prompt) == (subset == "ProtocolQA")
    assert "hidden-canary" not in prompt and "hidden-source" not in prompt
    prepare_question({**row, "id": "another-example"}, subset, seed=42)
    assert prepare_question(row, subset, seed=42) == question
