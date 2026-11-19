# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IMO-Bench autograder verdicts through the judge transport."""

import pytest
from verifyit.grade import GradingInfraError, Status
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate

from taskcompendium.importers.imobench import answer_judge_spec, convert_answerbench, proof_judge_spec
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow


def _answerbench_row(**data: str) -> RawRow:
    source = Source(
        dataset="imobench/answerbench_v2", revision="a" * 40, row="answerbench_v2.csv:1", importer_revision="1"
    )
    return RawRow("imo-bench-algebra-001", source, data)


def test_answerbench_rejects_a_row_with_shifted_columns():
    # answerbench_v2's algebra-036, abridged: Category, Subcategory and Source are shifted one column left.
    shifted = _answerbench_row(
        **{
            "Problem ID": "imo-bench-algebra-036",
            "Problem": "Find all functions $Y$ ... $Y(x) = A + B/x$",
            "Short Answer": "Algebra",
            "Category": "Functional Equation",
            "Subcategory": "Iran 2002",
            "Source": "",
        }
    )
    rejection = convert_answerbench(shifted, ConversionContext({}, None))
    assert isinstance(rejection, ImportRejection)
    assert rejection.kind is ImportFailureKind.SOURCE_DEFECT

    valid = _answerbench_row(
        **{"Problem ID": "imo-bench-algebra-001", "Problem": "Find $N$.", "Short Answer": "3", "Category": "Algebra"}
    )
    assert isinstance(convert_answerbench(valid, ConversionContext({}, None)), TaskSpec)


@pytest.fixture
def connection(judge_server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    return JudgeConnection(
        f"http://127.0.0.1:{judge_server.server_port}/v1", "fixture-key", default_model="fixture-judge"
    )


@pytest.mark.parametrize(
    "reply,reward",
    [
        ("<thinking>\n1. **Golden Answer:** 3\n4. **Conclusion:** Correct\n</thinking>\n\\boxed{Correct}", 1.0),
        ("<thinking>Conclusion: Incorrect</thinking> \\boxed{incorrect}", 0.0),
        ("\\boxed{Correct}", 1.0),
    ],
)
def test_answer_grader_reads_the_box_after_thinking(judge_server, connection, reply, reward):
    judge_server.reply = reply
    verdict = grade_judge_candidate(answer_judge_spec("Find all $N$.", "$N=3$"), "So $N = 3$.", connection=connection)
    assert verdict.status is Status.SCORED
    assert verdict.reward == reward
    prompt = judge_server.requests[-1]["messages"][-1]["content"]
    assert prompt.endswith("Problem: Find all $N$.\nModel Solution: So $N = 3$.\nGolden Answer: $N=3$\n")
    assert "{1, 2} is equivalent to {2, 1}" in prompt


@pytest.mark.parametrize("points", [7, 6, 1, 0])
def test_proof_grader_scores_final_points_line_out_of_seven(judge_server, connection, points):
    judge_server.reply = f"The key lemma is proved.\n\n<points>{points} out of 7</points>"
    spec = proof_judge_spec("Prove $f(x)=0$.", "Set $x=0$.", "(Partial)\n 1. Proved $f(\\{0\\})=\\{0\\}$.")
    verdict = grade_judge_candidate(spec, "Proof.", connection=connection)
    assert verdict.status is Status.SCORED
    assert verdict.reward == pytest.approx(points / 7)
    prompt = judge_server.requests[-1]["messages"][-1]["content"]
    assert "**SPECIFIC GRADING GUIDELINES**\n(Partial)\n 1. Proved $f(\\{0\\})=\\{0\\}$.\n" in prompt


def test_proof_grader_without_a_points_line_is_not_scored(judge_server, connection):
    judge_server.reply = "The proof is complete, so it earns 7 points."
    with pytest.raises(GradingInfraError):
        grade_judge_candidate(proof_judge_spec("P", "S", "G"), "Proof.", connection=connection)
