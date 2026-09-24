# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from itertools import combinations

from experiments.post_training.bio_tasks.contract import (
    AlignmentContract,
    Column,
    Contract,
    FastaContract,
    alignment_score,
    grade_files,
)
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.solvers.protein_alignment import pair_alignment


def test_unaligned_fasta_checks_all_residues_and_identities_even_with_correct_summary(tmp_path):
    contract = Contract(
        columns={"proteins": Column(kind="integer", description="representative count", unit="proteins")},
        expected={"proteome": {"proteins": 2}},
        fasta={"representatives.fa": FastaContract(sequences={"P1": "MACDEX", "P2": "MKU*"}, max_bytes=256)},
    )
    reference, answer, artifact = [tmp_path / name for name in ("reference.json", "answer.json", "representatives.fa")]
    reference.write_text(contract.model_dump_json())
    answer.write_text(json.dumps(contract.answer()))
    assert grade_files(reference, answer).reward == 0
    for valid in (">P1\nMACDEX\n>P2\nMKU*\n", "\n>P2 comment\nmku*\n\n>P1\nmac\ndex\n"):
        artifact.write_text(valid)
        verdict = grade_files(reference, answer)
        assert verdict.reward == 1
        assert verdict.detail["artifact_checks"]["representatives.fa"]["residues"] == 10
    for invalid in (
        ">P1\nMACDEY\n>P2\nMKU*\n",  # Changed residue, unchanged length.
        ">P1\nMACDEX\n>P2\nMKU\n",  # Truncated final sequence.
        ">P1\nMACDE\n>P2\nMKU*\n",  # Truncated nonfinal sequence.
        ">P1\nMACDEX\n>P1\nMACDEX\n>P2\nMKU*\n",  # Duplicate expected record.
        ">P1\nMACDEX\n",  # Missing record.
        ">P1\nMACDEX\n>P2\nMKU*\n>P3\nM\n",  # Extra record.
        ">P1\nMACDEX\n>P2\n\n",  # Empty sequence.
        "MACDEX\n>P2\nMKU*\n",  # Missing header.
        ">\nMACDEX\n>P2\nMKU*\n",  # Empty identity.
        ">P1\nMAC-DEX\n>P2\nMKU*\n",  # Alignment gap in unaligned output.
        ">P1\nMACDEX\n>P2\nMKU*\n" + " " * 256,
    ):
        artifact.write_text(invalid)
        assert grade_files(reference, answer).reward == 0
    artifact.unlink()
    outside = tmp_path / "outside.fa"
    outside.write_text(">P1\nMACDEX\n>P2\nMKU*\n")
    artifact.symlink_to(outside)
    assert grade_files(reference, answer).reward == 0


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
