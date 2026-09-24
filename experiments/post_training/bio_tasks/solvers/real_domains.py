# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read native HMMER domain tables and reconcile observed protein coverage."""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta


def domain_tables(inputs: Path, domtblout: Path) -> tuple[list[dict], list[dict], list[dict]]:
    query = json.loads((inputs / "query.json").read_text())
    sequences = {name.split("|")[1]: sequence for name, sequence in fasta(inputs / "proteins.fa").items()}
    with (inputs / "proteins.tsv").open() as handle:
        metadata = {row["Entry"]: row for row in csv.DictReader(handle, delimiter="\t")}
    if sequences.keys() != metadata.keys():
        raise ValueError("Protein metadata and FASTA identities differ")
    domains = []
    with domtblout.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(maxsplit=22)
            protein, model = fields[0].split("|")[1], fields[4]
            start, end = int(fields[17]), int(fields[18])
            if model not in query["model_accessions"] or len(sequences[protein]) != int(fields[2]):
                raise ValueError("Native search returned an inconsistent protein or model")
            if not 1 <= start <= end <= len(sequences[protein]):
                raise ValueError("Domain alignment lies outside its input sequence")
            domains.append(
                {
                    "id": f"{model}:{protein}:{fields[9]}",
                    "protein": protein,
                    "model": model,
                    "domain_index": int(fields[9]),
                    "alignment_start": start,
                    "alignment_end": end,
                    "envelope_start": int(fields[19]),
                    "envelope_end": int(fields[20]),
                    "model_start": int(fields[15]),
                    "model_end": int(fields[16]),
                    "protein_score": float(fields[7]),
                    "domain_score": float(fields[13]),
                    "independent_evalue": float(fields[12]),
                    "conditional_evalue": float(fields[11]),
                    "sequence": sequences[protein][start - 1 : end],
                }
            )
    intervals = defaultdict(list)
    for row in domains:
        intervals[(row["protein"], row["model"])].append((row["alignment_start"], row["alignment_end"]))
    summaries = []
    for model in query["model_accessions"]:
        members = {row["protein"] for row in domains if row["model"] == model}
        summaries.append(
            {
                "id": model,
                "searched_proteins": len(sequences),
                "matched_proteins": len(members),
                "domains": sum(row["model"] == model for row in domains),
                "covered_residues": sum(interval_union_length(intervals[(protein, model)]) for protein in members),
            }
        )
    proteins = []
    for protein, sequence in sorted(sequences.items()):
        hits = [row for row in domains if row["protein"] == protein]
        proteins.append(
            {
                "id": protein,
                "length": len(sequence),
                "sequence_version": int(metadata[protein]["Sequence version"]),
                "domains": len(hits),
                "distinct_models": len({row["model"] for row in hits}),
                "covered_residues": interval_union_length(
                    [(row["alignment_start"], row["alignment_end"]) for row in hits]
                ),
            }
        )
    return summaries, domains, proteins


def interval_union_length(intervals: list[tuple[int, int]]) -> int:
    """Measure closed interval unions without counting overlap more than once."""
    total, last = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start - 1, last))
        last = max(last, end)
    return total


def solve_domains(inputs: Path, output: Path) -> list[dict]:
    query = json.loads((inputs / "query.json").read_text())
    domtblout = output / "domains.domtblout"
    with (output / "hmmsearch.stdout").open("w") as stdout, (output / "hmmsearch.stderr").open("w") as stderr:
        subprocess.run(
            [
                "hmmsearch",
                "--cut_ga",
                "--cpu",
                "1",
                "--seed",
                str(query["seed"]),
                "--noali",
                "--domtblout",
                str(domtblout),
                str(inputs / "models.hmm"),
                str(inputs / "proteins.fa"),
            ],
            stdout=stdout,
            stderr=stderr,
            check=True,
            timeout=900,
        )
    summaries, domains, proteins = domain_tables(inputs, domtblout)
    for name, rows in [("domains.tsv", domains), ("proteins.tsv", proteins)]:
        with (output / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["id"]))
    return summaries
