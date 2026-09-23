# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-reading annotation and interval solutions."""

import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from experiments.post_training.bio_tasks.solvers.formats import fasta, reverse_complement, tab_rows, translate


def solve_bed12(inputs: Path) -> list[dict]:
    answer = []
    for row in tab_rows(inputs / "transcripts.bed"):
        sizes = [int(x) for x in row[10].strip(",").split(",")]
        starts = [int(x) + int(row[1]) for x in row[11].strip(",").split(",")]
        blocks = list(zip(starts, sizes, strict=True))
        if row[5] == "-":
            blocks.reverse()
        answer.extend(
            {"id": f"{row[3]}/{rank}", "start": start, "end": start + size, "strand": row[5]}
            for rank, (start, size) in enumerate(blocks, 1)
        )
    return answer


def solve_gtf(inputs: Path) -> list[dict]:
    genome = fasta(inputs / "genome.fa")
    transcripts = defaultdict(list)
    for row in tab_rows(inputs / "annotations.gtf"):
        if row[2] == "exon":
            attrs = dict(re.findall(r'(\w+)\s+"([^"]*)"', row[8]))
            transcripts[attrs["transcript_id"]].append(row)
    answer = []
    for name, rows in transcripts.items():
        rows.sort(key=lambda row: int(row[3]))
        sequence = "".join(genome[row[0]][int(row[3]) - 1 : int(row[4])] for row in rows)
        if rows[0][6] == "-":
            sequence = reverse_complement(sequence)
        answer.append({"id": name, "sequence": sequence, "length": len(sequence)})
    return answer


def solve_cds(inputs: Path) -> list[dict]:
    genome = fasta(inputs / "genome.fa")
    transcripts = defaultdict(list)
    for row in tab_rows(inputs / "annotations.gff3"):
        if row[2] == "CDS":
            attrs = {k: unquote(v) for k, v in (a.split("=", 1) for a in row[8].split(";"))}
            transcripts[attrs["Parent"]].append(row)
    answer = []
    for name, rows in transcripts.items():
        minus = rows[0][6] == "-"
        rows.sort(key=lambda row: int(row[3]), reverse=minus)
        parts = [genome[row[0]][int(row[3]) - 1 : int(row[4])] for row in rows]
        if minus:
            parts = [reverse_complement(s) for s in parts]
        sequence = "".join(parts)[int(rows[0][7]) :]
        answer.append({"id": name, "protein": translate(sequence)})
    return answer


def merged_intervals(inputs: Path) -> dict[str, list[tuple[int, int]]]:
    grouped = defaultdict(list)
    for row in tab_rows(inputs / "features.bed"):
        grouped[row[0]].append((int(row[1]), int(row[2])))
    result = {}
    for chrom, spans in grouped.items():
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result[chrom] = merged
    return result


def solve_union(inputs: Path) -> list[dict]:
    merged = merged_intervals(inputs)
    answer = []
    for chrom, length in tab_rows(inputs / "chrom.sizes"):
        count = sum(b - a for a, b in merged.get(chrom, []))
        answer.append({"id": chrom, "covered": count, "fraction": count / int(length)})
    return answer


def solve_complement(inputs: Path) -> list[dict]:
    merged = merged_intervals(inputs)
    answer = []
    for chrom, length in tab_rows(inputs / "chrom.sizes"):
        cursor, rank = 0, 1
        for start, end in [*merged.get(chrom, []), (int(length), int(length))]:
            if start > cursor:
                answer.append({"id": f"{chrom}/{rank}", "start": cursor, "end": start})
                rank += 1
            cursor = max(cursor, end)
    return answer


def solve_nearest(inputs: Path) -> list[dict]:
    features = tab_rows(inputs / "features.bed")
    answer = []
    for chrom, start, _, name in tab_rows(inputs / "queries.bed"):
        pos = int(start)
        distance, feature = min((max(int(f[1]) - pos, pos - int(f[2]) + 1, 0), f[3]) for f in features if f[0] == chrom)
        answer.append({"id": name, "feature": feature, "distance": distance})
    return answer


def solve_promoters(inputs: Path) -> list[dict]:
    sizes = {c: int(n) for c, n in tab_rows(inputs / "chrom.sizes")}
    upstream, downstream = map(int, (inputs / "window.txt").read_text().split())
    answer = []
    for chrom, start, end, name, _, strand in tab_rows(inputs / "genes.bed"):
        if strand == "+":
            left, right = int(start) - upstream, int(start) + downstream
        else:
            left, right = int(end) - downstream, int(end) + upstream
        answer.append({"id": name, "start": max(0, left), "end": min(sizes[chrom], right)})
    return answer


def solve_signal(inputs: Path) -> list[dict]:
    track = tab_rows(inputs / "signal.bedgraph")
    answer = []
    for chrom, start, end, name in tab_rows(inputs / "windows.bed"):
        total = sum(
            max(0, min(int(end), int(b)) - max(int(start), int(a))) * int(v) for c, a, b, v in track if c == chrom
        )
        answer.append({"id": name, "integral": total, "mean": total / (int(end) - int(start))})
    return answer


SOLVERS = {
    "bed12-exons": solve_bed12,
    "gtf-splicing": solve_gtf,
    "gff-cds-translation": solve_cds,
    "bed-union-coverage": solve_union,
    "bed-complement": solve_complement,
    "bed-nearest-features": solve_nearest,
    "bed-stranded-promoters": solve_promoters,
    "bedgraph-weighted-signal": solve_signal,
}
