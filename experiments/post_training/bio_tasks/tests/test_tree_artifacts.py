# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import pytest

from experiments.post_training.bio_tasks.build import verifier_archive
from experiments.post_training.bio_tasks.contract import Column, Contract, TreeContract, grade_files


@pytest.fixture
def tree_submission(tmp_path):
    contract = Contract(
        columns={"tips": Column(kind="integer", description="leaf count", unit="taxa")},
        expected={"summary": {"tips": 4}},
        trees={"tree.nwk": TreeContract(reference="((A:1,B:2):3,(C:4,D:5):6);", atol=1e-6, rtol=1e-5, max_bytes=1024)},
    )
    reference = tmp_path / "reference.json"
    reference.write_text(contract.model_dump_json())
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(contract.answer()))
    return reference, answer, tmp_path / "tree.nwk"


@pytest.mark.parametrize(
    "tree",
    [
        "((B:2,A:1):3,(D:5,C:4):6);",
        "(A:1,B:2,(C:4,D:5):9);",
        "(C:4,D:5,(A:1,B:2):9);",
        "('A':1e0,'B':2.0,('C':4,D:5)99[branch support]:9):0;",
        "(A:1.000001,B:2,(C:4,D:5):9);",
    ],
)
def test_equivalent_unrooted_tree_passes(tree_submission, tree):
    reference, answer, artifact = tree_submission
    artifact.write_text(tree)
    assert grade_files(reference, answer).reward == 1


@pytest.mark.parametrize(
    "tree",
    [
        "(A:1,C:4,(B:2,D:5):9);",  # Same taxa and pendant lengths; wrong split.
        "(A:1,B:2,(C:4,D:5):8);",  # Correct summary and topology; changed interior edge.
        "(A:1.01,B:2,(C:4,D:5):9);",
        "(A:1,B:2,C:13);",  # Missing taxon.
        "(A:1,B:2,(C:4,A:5):9);",  # Duplicate leaf.
        "(A:1,B:2,(C:4,E:5):9);",  # Replaced identity.
        "(A:1,B:2,(C:4,D:5));",  # Missing length must not become zero.
        "(A:1,B:2,(C:4,D:5):nan);",
        "(A:1,B:2,(C:4,D:5):-9);",
        "(A:1,B:2,(C:4,D:5):9):7;",  # Extraneous rooted stem.
        "(A:1,B:2,((C:4):0,D:5):9);",
        "(A:1,B:2,(C:4,D:5):9);(A:1,B:2,C:3);",
        "(A:1,B:2,(C:4,D:5):9)",
        "[unclosed(A:1,B:2,(C:4,D:5):9);",
        "(" * 2000,
    ],
)
def test_tree_errors_fail_despite_correct_summary(tree_submission, tree):
    reference, answer, artifact = tree_submission
    artifact.write_text(tree)
    assert grade_files(reference, answer).reward == 0


def test_packaged_verifier_checks_tree_in_isolated_python(tree_submission, tmp_path):
    reference, answer, artifact = tree_submission
    program = tmp_path / "verifier.pyz"
    program.write_bytes(verifier_archive())
    command = [
        sys.executable,
        "-I",
        str(program),
        "--reference",
        str(reference),
        "--answer",
        str(answer),
        "--logs",
        str(tmp_path / "logs"),
    ]
    for tree, expected in [("(A:1,B:2,(C:4,D:5):9);", 1), ("(A:1,C:4,(B:2,D:5):9);", 0)]:
        artifact.write_text(tree)
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=30)
        assert json.loads((tmp_path / "logs/reward.json").read_text()) == {"reward": expected}
