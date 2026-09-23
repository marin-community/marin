# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent FASTA/GFF3 solutions for the declared genome-analysis contracts."""

import json
from collections import Counter, defaultdict
from functools import partial
from itertools import pairwise, product
from pathlib import Path
from urllib.parse import unquote

from experiments.post_training.bio_tasks.solvers.formats import fasta, reverse_complement, tab_rows, translate


def coding_records(inputs: Path) -> list[dict]:
    """Join explicit GFF3 parts without deleting bases that continue a split codon."""
    sequences = fasta(inputs / "genome.fa")
    groups = defaultdict(list)
    for fields in tab_rows(inputs / "annotations.gff3"):
        if fields[2] != "CDS":
            continue
        attributes = dict(item.split("=", 1) for item in fields[8].split(";"))
        groups[unquote(attributes["ID"])].append((int(attributes["part"].split("/")[0]), fields))
    result = []
    for identifier, parts in groups.items():
        intervals = []
        coding = []
        ordered = [fields for _, fields in sorted(parts)]
        for fields in ordered:
            start, end = int(fields[3]) - 1, int(fields[4])
            intervals.append((start, end))
            segment = sequences[fields[0]][start:end]
            coding.append(reverse_complement(segment) if fields[6] == "-" else segment)
        result.append({"id": identifier, "parts": intervals, "strand": ordered[0][6], "coding": "".join(coding)})
    return result


def solve_real_genome(inputs: Path, operation: str) -> list[dict]:
    sequence = next(iter(fasta(inputs / "genome.fa").values()))
    query = json.loads((inputs / "query.json").read_text())
    genes = coding_records(inputs)
    if operation == "real-genome-cds-extraction":
        return [{"id": gene["id"], "sequence": gene["coding"], "length": len(gene["coding"])} for gene in genes]
    if operation == "real-genome-translation":
        return [
            {"id": gene["id"], "protein": "M" + translate(gene["coding"])[1:-1], "length": len(gene["coding"]) // 3 - 1}
            for gene in genes
        ]
    if operation == "real-genome-gc3":
        answer = []
        for gene in genes:
            codons = len(gene["coding"]) // 3 - 1
            gc = sum(gene["coding"][3 * index + 2] in {"G", "C"} for index in range(codons))
            answer.append({"id": gene["id"], "gc3": gc / codons, "gc_codons": gc, "codons": codons})
        return answer
    if operation == "real-genome-codon-counts":
        counts = Counter()
        for gene in genes:
            counts.update(map("".join, zip(*[iter(gene["coding"][:-3])] * 3, strict=True)))
        return [{"id": codon, "count": counts[codon]} for codon in map("".join, product("ACGT", repeat=3))]
    if operation == "real-genome-overlap":
        positions = {gene["id"]: set().union(*(range(start, end) for start, end in gene["parts"])) for gene in genes}
        answer = []
        for identifier, own in positions.items():
            others = set().union(*(sites for other, sites in positions.items() if other != identifier))
            answer.append({"id": identifier, "bases": len(own), "overlap": len(own & others)})
        return answer
    if operation == "real-genome-promoters":
        answer = []
        width = query["upstream_bases"]
        length = len(sequence)
        for gene in genes:
            start, end = gene["parts"][0]
            lower, upper = (start - width, start) if gene["strand"] == "+" else (end, end + width)
            if query["topology"] == "linear":
                bases = sequence[max(0, lower) : min(length, upper)]
            else:
                doubled = sequence * 3
                bases = doubled[lower + length : upper + length]
            if gene["strand"] == "-":
                bases = reverse_complement(bases)
            answer.append({"id": gene["id"], "sequence": bases, "length": len(bases)})
        return answer
    if operation == "real-genome-restriction-digest":
        answer = []
        length = len(sequence)
        for enzyme, profile in query["enzymes"].items():
            motif, offset = profile["motif"], profile["cut_after"]
            extended = sequence + sequence[: len(motif) - 1] if query["topology"] == "circular" else sequence
            cuts = sorted(
                (index + offset) % length for index in range(length) if extended[index : index + len(motif)] == motif
            )
            if query["topology"] == "circular" and cuts:
                fragments = [(cuts[(index + 1) % len(cuts)] - cut) % length or length for index, cut in enumerate(cuts)]
            else:
                boundaries = [0, *cuts, length]
                fragments = [end - start for start, end in pairwise(boundaries)]
            answer.append(
                {
                    "id": enzyme,
                    "cut_offsets": ",".join(map(str, cuts)),
                    "fragment_lengths": ",".join(map(str, sorted(fragments))),
                }
            )
        return answer
    raise ValueError(operation)


SOLVERS = {
    name: partial(solve_real_genome, operation=name)
    for name in (
        "real-genome-cds-extraction",
        "real-genome-translation",
        "real-genome-gc3",
        "real-genome-codon-counts",
        "real-genome-overlap",
        "real-genome-promoters",
        "real-genome-restriction-digest",
    )
}
