# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from itertools import combinations

from experiments.post_training.bio_tasks.contract import (
    AlignmentContract,
    Column,
    Contract,
    alignment_score,
    grade_files,
)
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.solvers.protein_alignment import pair_alignment


def test_alignment_accepts_alternative_gap_placements_and_rejects_damaged_or_poor_alignments(tmp_path):
    scoring = {a + b: 2 if a == b else -1 for a in "ACDEFGHIKLMNPQRSTVWY" for b in "ACDEFGHIKLMNPQRSTVWY"}
    sequences = {"a": "AAAC", "b": "AAC", "c": "AC"}
    target = AlignmentContract(
        sequences=sequences, scoring=scoring, gap_open=3, gap_extend=1, minimum_score=4, max_columns=8, max_bytes=256
    )
    contract = Contract(
        columns={"residues": Column(kind="integer", description="length", unit="residues")},
        expected={key: {"residues": len(sequence)} for key, sequence in sequences.items()},
        alignments={"alignment.fa": target},
    )
    reference, answer, artifact = [tmp_path / name for name in ("reference.json", "answer.json", "alignment.fa")]
    reference.write_text(contract.model_dump_json())
    answer.write_text(json.dumps(contract.answer()))
    assert grade_files(reference, answer).reward == 0
    for valid in (">a\nAAAC\n>b\n-AAC\n>c\n--AC\n", ">c comment\na--c\n>b\nA-\nAC\n>a\nAAAC\n"):
        artifact.write_text(valid)
        verdict = grade_files(reference, answer)
        assert verdict.reward == 1
        assert verdict.detail["artifact_checks"]["alignment.fa"]["score"] == 4
    for invalid in (
        ">a\nAAAC\n>b\nAAC-\n>c\nAC--\n",  # Correct sequences, poor objective.
        ">a\nAAAC\n>b\n-AAC\n>c\n--AA\n",  # Altered residue.
        ">a\nAAAC\n>b\nAAC\n>c\nAC\n",  # Ragged rows.
        ">a\n-AAAC\n>b\n--AAC\n>c\n---AC\n",  # All-gap column.
        ">a\nAAAC\n>a\n-AAC\n>c\n--AC\n",  # Duplicate identity.
        ">a\nAAAC\n>b\n-AAC\n>c\n--AC\n" + " " * 256,
    ):
        artifact.write_text(invalid)
        assert grade_files(reference, answer).reward == 0
    assert alignment_score(["AA-AA", "-----"], scoring, 3, 1) == -6
    artifact.unlink()
    outside = tmp_path / "outside.fa"
    outside.write_text(">a\nAAAC\n>b\n-AAC\n>c\n--AC\n")
    artifact.symlink_to(outside)
    assert grade_files(reference, answer).reward == 0


def test_global_affine_alignments_reproduce_independent_biopython_scores():
    data = json.loads(source_text("UniProt:globins-20260923", "uniprot-globins.json.gz"))
    for left, right in combinations(sorted(data["proteins"]), 2):
        a, b = (data["proteins"][key]["sequence"] for key in (left, right))
        aligned = pair_alignment(a, b, data["scoring"], data["gap_open"], data["gap_extend"])
        assert aligned[0].replace("-", "") == a
        assert aligned[1].replace("-", "") == b
        assert (
            alignment_score(list(aligned), data["scoring"], data["gap_open"], data["gap_extend"])
            == data["optimal_pair_scores"][f"{left}:{right}"]
        )
