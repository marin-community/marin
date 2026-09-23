# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent sequence references for additional repository-derived operations."""

from functools import partial
from itertools import combinations
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta, reverse_complement, tab_rows, table


def solve_repo_sequences(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation == "dna-unique-mapping":
        references = fasta(inputs / "reference.fa")
        for name, sequence in fasta(inputs / "reads.fa").items():
            hits = []
            if set(sequence) <= set("ACGT"):
                for reference, text in references.items():
                    for strand, query in [("+", sequence), ("-", reverse_complement(sequence))]:
                        start = text.find(query)
                        if start >= 0:
                            hits.append(
                                {
                                    "id": name,
                                    "reference": reference,
                                    "start": start,
                                    "end": start + len(query),
                                    "strand": strand,
                                }
                            )
            assert len(hits) <= 1
            answer.append(
                hits[0] if hits else {"id": name, "reference": None, "start": None, "end": None, "strand": "."}
            )
    elif operation == "protein-local-search":
        query = next(iter(fasta(inputs / "query.fa").values()))
        for name, sequence in fasta(inputs / "proteins.fa").items():
            start = sequence.find(query)
            if start >= 0:
                answer.append(
                    {"id": name, "start": start, "end": start + len(query), "identity": 1.0, "query_coverage": 1.0}
                )
    elif operation == "alignment-sum-of-pairs":
        sequences = fasta(inputs / "alignment.fa")
        for index, column in enumerate(zip(*sequences.values(), strict=True)):
            matches = mismatches = gaps = 0
            for a, b in combinations(column, 2):
                if "N" in (a, b) or a == b == "-":
                    continue
                if "-" in (a, b):
                    gaps += 1
                elif a == b:
                    matches += 1
                else:
                    mismatches += 1
            answer.append(
                {
                    "id": str(index),
                    "matches": matches,
                    "mismatches": mismatches,
                    "gap_pairs": gaps,
                    "score": 2 * matches - mismatches - 2 * gaps,
                }
            )
    elif operation == "hmmer-domain-extraction":
        proteins = fasta(inputs / "proteins.fa")
        for line in (inputs / "domains.domtblout").read_text().splitlines():
            if line.startswith("#"):
                continue
            fields = line.split(maxsplit=22)
            start, end = int(fields[17]) - 1, int(fields[18])
            answer.append(
                {
                    "id": fields[0] + ":" + fields[3] + ":" + fields[9],
                    "sequence": proteins[fields[0]][start:end],
                    "start": start,
                    "end": end,
                }
            )
    elif operation == "sequence-identity-clusters":
        proteins = fasta(inputs / "proteins.fa")
        cutoff = int((inputs / "max_mismatches.txt").read_text())
        adjacency = {name: set() for name in proteins}
        for a, b in combinations(proteins, 2):
            if sum(x != y for x, y in zip(proteins[a], proteins[b], strict=True)) <= cutoff:
                adjacency[a].add(b)
                adjacency[b].add(a)
        remaining = set(proteins)
        while remaining:
            members = {min(remaining)}
            while True:
                expanded = members | set().union(*(adjacency[name] for name in members))
                if expanded == members:
                    break
                members = expanded
            remaining -= members
            answer.extend({"id": name, "representative": min(members), "size": len(members)} for name in members)
    else:
        index = {row[0]: list(map(int, row[1:])) for row in tab_rows(inputs / "reference.fa.fai")}
        with (inputs / "reference.fa").open("rb") as handle:
            for region in table(inputs / "regions.csv"):
                _, offset, line_bases, line_bytes = index[region["contig"]]
                sequence = []
                for position in range(int(region["start"]) - 1, int(region["end"])):
                    handle.seek(offset + (position // line_bases) * line_bytes + position % line_bases)
                    sequence.append(handle.read(1).decode("ascii"))
                answer.append({"id": region["region"], "sequence": "".join(sequence)})
    return answer


NAMES = (
    "dna-unique-mapping",
    "protein-local-search",
    "alignment-sum-of-pairs",
    "hmmer-domain-extraction",
    "sequence-identity-clusters",
    "fasta-indexed-regions",
)
SOLVERS = {name: partial(solve_repo_sequences, operation=name) for name in NAMES}
