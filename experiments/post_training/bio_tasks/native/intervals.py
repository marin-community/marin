# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native interval expansion, binary track conversion, and threshold peak calls."""

from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import tab_rows


def solve_bedtools(inputs: Path, work: Path) -> list[dict]:
    output = execute(["bedtools", "bed12tobed6", "-i", str(inputs / "transcripts.bed"), "-n"], work, "exons.bed")
    return [
        {"id": name + "/" + rank, "start": int(start), "end": int(end), "strand": strand}
        for _, start, end, name, rank, strand in tab_rows(output)
    ]


def solve_kent(inputs: Path, work: Path) -> list[dict]:
    binary = work / "signal.bw"
    restored = work / "restored.bedgraph"
    execute(
        ["bedGraphToBigWig", str(inputs / "signal.bedgraph"), str(inputs / "chrom.sizes"), str(binary)],
        work,
        "encode.log",
    )
    execute(["bigWigToBedGraph", str(binary), str(restored)], work, "decode.log")
    track = tab_rows(restored)
    answer = []
    for chrom, start, end, name in tab_rows(inputs / "windows.bed"):
        left, right = int(start), int(end)
        signal = sum(
            max(0, min(right, int(b)) - max(left, int(a))) * float(value)
            for contig, a, b, value in track
            if contig == chrom
        )
        answer.append({"id": name, "integral": int(signal), "mean": signal / (right - left)})
    return answer


def solve_macs(inputs: Path, work: Path) -> list[dict]:
    peaks = work / "peaks.bed"
    execute(
        [
            "macs3",
            "bdgpeakcall",
            "-i",
            str(inputs / "signal.bedgraph"),
            "-c",
            (inputs / "threshold.txt").read_text().strip(),
            "-l",
            "1",
            "-g",
            "0",
            "-o",
            str(peaks),
        ],
        work,
        "macs.log",
    )
    track = tab_rows(inputs / "signal.bedgraph")
    answer = []
    counts = {}
    for row in tab_rows(peaks):
        if row[0].startswith("track"):
            continue
        chrom, start, end = row[:3]
        left, right = int(start), int(end)
        counts[chrom] = counts.get(chrom, 0) + 1
        maximum = max(
            float(value) for contig, a, b, value in track if contig == chrom and max(left, int(a)) < min(right, int(b))
        )
        answer.append({"id": f"{chrom}:{counts[chrom]}", "start": left, "end": right, "maximum": maximum})
    return answer
