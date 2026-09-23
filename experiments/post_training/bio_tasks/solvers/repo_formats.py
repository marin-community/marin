# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent references for repository-specific artifact bookkeeping."""

import math
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta, tab_rows, table
from experiments.post_training.bio_tasks.solvers.phylogeny import clades, leaves, newick
from experiments.post_training.bio_tasks.solvers.variants import vcf_records


def solve_repo_formats(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation == "paf-query-coverage":
        positions = defaultdict(set)
        counts = defaultdict(int)
        for row in tab_rows(inputs / "alignments.paf"):
            if "tp:A:P" in row[12:]:
                positions[row[0]].update(range(int(row[2]), int(row[3])))
                counts[row[0]] += 1
        for row in table(inputs / "queries.csv"):
            name = row["query"]
            answer.append(
                {
                    "id": name,
                    "covered": len(positions[name]),
                    "fraction": len(positions[name]) / int(row["length"]),
                    "alignments": counts[name],
                }
            )
    elif operation == "sam-pair-concordance":
        pairs = defaultdict(list)
        for row in tab_rows(inputs / "pairs.sam"):
            if row[0].startswith("@"):
                continue
            pairs[row[0]].append(row)
        for name, rows in pairs.items():
            primary = [row for row in rows if not int(row[1]) & (256 | 2048)]
            first = [row for row in primary if int(row[1]) & 64 and not int(row[1]) & 4]
            second = [row for row in primary if int(row[1]) & 128 and not int(row[1]) & 4]
            keep = False
            if len(first) == len(second) == 1:
                a, b = first[0], second[0]
                keep = (
                    a[2] == b[2]
                    and not int(a[1]) & 16
                    and bool(int(b[1]) & 16)
                    and int(a[3]) < int(b[3])
                    and int(b[3]) + 10 - int(a[3]) <= 100
                )
            answer.append({"id": name, "concordant": int(keep)})
    elif operation == "vcf-sample-qc":
        samples, records = vcf_records(inputs / "variants.vcf")
        counts = dict.fromkeys(samples, 0)
        for row in records:
            gt_index = row[8].split(":").index("GT")
            for name, call in zip(samples, row[9:], strict=True):
                counts[name] += "." in call.split(":")[gt_index]
        for name, count in counts.items():
            rate = count / len(records)
            answer.append({"id": name, "missing": count, "rate": rate, "keep": int(rate <= 0.25)})
    elif operation == "fastqc-report-reconciliation":
        for row in table(inputs / "samples.csv"):
            if row["include"] != "1":
                continue
            actual = len((inputs / row["fastq"]).read_text().splitlines()) // 4
            reported = int(
                next(
                    line.split("\t")[1]
                    for line in (inputs / row["report"]).read_text().splitlines()
                    if line.startswith("Total Sequences\t")
                )
            )
            answer.append(
                {
                    "id": row["sample"],
                    "actual_reads": actual,
                    "reported_reads": reported,
                    "difference": actual - reported,
                    "matches": int(actual == reported),
                }
            )
    elif operation == "alignment-partitions":
        length = len(next(iter(fasta(inputs / "alignment.fa").values())))
        membership = defaultdict(list)
        for name, start, end, stride in re.findall(
            r"charset\s+(\w+)\s*=\s*(\d+)-(\d+)(?:\\(\d+))?\s*;", (inputs / "partitions.nex").read_text(), re.IGNORECASE
        ):
            for position in range(int(start) - 1, int(end), int(stride or 1)):
                membership[position].append(name)
        for position in range(length):
            names = sorted(membership[position])
            answer.append(
                {
                    "id": str(position),
                    "partitions": ",".join(names),
                    "memberships": len(names),
                    "valid": int(len(names) == 1),
                }
            )
    elif operation == "newick-split-support":
        trees = [newick(line) for line in (inputs / "replicates.nwk").read_text().splitlines() if line]
        for row in table(inputs / "splits.csv"):
            target = set(row["side"].split(","))
            count = 0
            for tree in trees:
                all_tips = leaves(tree)
                count += any(leaves(node) == target or all_tips - leaves(node) == target for node in clades(tree))
            answer.append({"id": row["split"], "support": count / len(trees), "trees": count})
    elif operation == "gtf-coverage-counts":
        transcripts = {}
        lengths = defaultdict(int)
        for _, _, feature, start, end, _, _, _, text in tab_rows(inputs / "transcripts.gtf"):
            attrs = dict(re.findall(r'(\w+)\s+"([^"]*)"', text))
            if feature == "transcript":
                transcripts[attrs["transcript_id"]] = attrs
            elif feature == "exon":
                lengths[attrs["transcript_id"]] += int(end) - int(start) + 1
        read_length = int((inputs / "read_length.txt").read_text())
        for name, attrs in transcripts.items():
            answer.append(
                {
                    "id": name,
                    "gene": attrs["gene_id"],
                    "exon_length": lengths[name],
                    "estimated_count": math.ceil(float(attrs["cov"]) * lengths[name] / read_length),
                }
            )
    elif operation == "bedgraph-threshold-peaks":
        threshold = float((inputs / "threshold.txt").read_text())
        groups = defaultdict(list)
        for chrom, start, end, signal in tab_rows(inputs / "signal.bedgraph"):
            if float(signal) >= threshold:
                groups[chrom].append((int(start), int(end), float(signal)))
        for chrom, intervals in groups.items():
            merged = []
            for start, end, value in sorted(intervals):
                if merged and start == merged[-1][1]:
                    previous = merged.pop()
                    merged.append((previous[0], end, max(previous[2], value)))
                else:
                    merged.append((start, end, value))
            answer.extend(
                {"id": f"{chrom}:{i}", "start": start, "end": end, "maximum": value}
                for i, (start, end, value) in enumerate(merged, 1)
            )
    else:
        spots = defaultdict(list)
        for row in table(inputs / "spots.csv"):
            if row["read_type"] == "biological":
                spots[row["spot"]].append(row)
        for name, reads in spots.items():
            for i, row in enumerate(sorted(reads, key=lambda row: int(row["read_index"])), 1):
                answer.append(
                    {
                        "id": name + ":" + row["read_index"],
                        "destination": "single" if len(reads) == 1 else f"R{i}",
                        "sequence": row["sequence"],
                        "quality": row["quality"],
                    }
                )
    return answer


NAMES = (
    "paf-query-coverage",
    "sam-pair-concordance",
    "vcf-sample-qc",
    "fastqc-report-reconciliation",
    "alignment-partitions",
    "newick-split-support",
    "gtf-coverage-counts",
    "bedgraph-threshold-peaks",
    "sra-spot-export",
)
SOLVERS = {name: partial(solve_repo_formats, operation=name) for name in NAMES}
