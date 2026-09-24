# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.post_training.bio_tasks.solvers.real_domains import domain_tables


def test_domains_use_alignment_boundaries_and_union_overlaps_without_dropping_unmatched_proteins(tmp_path):
    (tmp_path / "query.json").write_text(json.dumps({"model_accessions": ["PF1.1", "PF2.1", "PF3.1"]}))
    (tmp_path / "proteins.fa").write_text(">sp|A|first\nACDEFGHIKLMNPQRSTVWY\n>sp|B|second\nACDE\n>sp|C|third\nGHIK\n")
    (tmp_path / "proteins.tsv").write_text("Entry\tLength\tSequence version\nA\t20\t2\nB\t4\t1\nC\t4\t1\n")
    native = tmp_path / "domains.domtblout"
    native.write_text(
        "# target accession tlen query accession qlen E-value score bias # of c-Evalue i-Evalue "
        "score bias hmmfrom hmmto alifrom alito envfrom envto acc description\n"
        "sp|A|first - 20 First PF1.1 9 1e-10 30 0 1 2 1e-12 1e-10 25 0 1 5 4 8 1 9 0.9 first protein\n"
        "sp|A|first - 20 First PF1.1 9 1e-10 30 0 2 2 2e-12 2e-10 24 0 4 9 7 12 6 14 0.9 first protein\n"
        "sp|B|second - 4 First PF1.1 9 3e-10 29 0 1 1 3e-12 3e-10 23 0 1 2 1 2 1 3 0.9 second protein\n"
        "sp|A|first - 20 Second PF2.1 4 4e-10 28 0 1 1 4e-12 4e-10 22 0 1 4 12 15 10 18 0.9 first protein\n"
    )
    summaries, domains, proteins = domain_tables(tmp_path, native)
    assert summaries == [
        {"id": "PF1.1", "searched_proteins": 3, "matched_proteins": 2, "domains": 3, "covered_residues": 11},
        {"id": "PF2.1", "searched_proteins": 3, "matched_proteins": 1, "domains": 1, "covered_residues": 4},
        {"id": "PF3.1", "searched_proteins": 3, "matched_proteins": 0, "domains": 0, "covered_residues": 0},
    ]
    assert [(row["id"], row["sequence"]) for row in domains] == [
        ("PF1.1:A:1", "EFGHI"),
        ("PF1.1:A:2", "HIKLMN"),
        ("PF1.1:B:1", "AC"),
        ("PF2.1:A:1", "NPQR"),
    ]
    assert proteins == [
        {"id": "A", "length": 20, "sequence_version": 2, "domains": 3, "distinct_models": 2, "covered_residues": 12},
        {"id": "B", "length": 4, "sequence_version": 1, "domains": 1, "distinct_models": 1, "covered_residues": 2},
        {"id": "C", "length": 4, "sequence_version": 1, "domains": 0, "distinct_models": 0, "covered_residues": 0},
    ]
    assert (domains[0]["model_start"], domains[0]["alignment_start"], domains[0]["envelope_start"]) == (1, 4, 1)
    assert domains[0]["independent_evalue"] == 1e-10 and domains[0]["conditional_evalue"] == 1e-12
