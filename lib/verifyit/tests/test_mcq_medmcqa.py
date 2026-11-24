# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status
from verifyit.grade import grade as dispatch
from verifyit.modes import grade_mcq
from verifyit.spec import McqExtraction, McqSpec

SPEC = McqSpec("B", options=4, extraction=McqExtraction.MEDMCQA_COT_REFERENCE)


# Expected letters follow lm-eval's medmcqa_cot regex filter with take_first, applied after any inline
# reasoning and requiring a standalone letter.
@pytest.mark.parametrize(
    "response,reward,extracted",
    [
        ("Scurvy follows vitamin C deficiency.\nFinal Answer: B", 1, "B"),
        ("Final Answer: **B**", 1, "B"),
        ("Final Answer:\nB", 1, "B"),
        ("Final Answer: A", 0, "A"),
        ("Final Answer: A\nOn reflection, Final Answer: B", 0, "A"),
        ("<think>Final Answer: A</think>\nFinal Answer: B", 1, "B"),
        ("final answer: b", 0, None),
        ("The answer is B.", 0, None),
        ("Answer: B", 0, None),
        ("Final Answer: (B)", 0, None),
        ("Final Answer: Bone marrow", 0, None),
        ("Final Answer: ABCD", 0, None),
        ("Final Answer: B.", 1, "B"),
        ("", 0, None),
    ],
)
def test_medmcqa_cot_reference_extraction(response, reward, extracted):
    result = grade_mcq.grade_mcq_candidate(SPEC, response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["extracted"] == extracted


def test_medmcqa_cot_reference_grades_the_answer_file(tmp_path):
    (tmp_path / "answer.txt").write_text("B is the vitamin C option.\n\nFinal Answer: B")
    assert dispatch(SPEC, tmp_path, tmp_path).reward == 1
