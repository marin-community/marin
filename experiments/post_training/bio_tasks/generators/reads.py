# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequencing-read tasks constructed from per-read and per-base ledgers."""

import random
from collections import Counter
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def generate_reads(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    expected, inputs = {}, {}
    if operation.startswith("sam-"):
        lines = ["@HD\tVN:1.6\tSO:unsorted", "@SQ\tSN:chr1\tLN:200", "@SQ\tSN:chr2\tLN:200"]
        if operation == "sam-inclusion":
            cases = [
                (0, 30, 1),
                (16, 40, 1),
                (0, 29, 0),
                (256, 50, 0),
                (2048, 60, 0),
                (512, 50, 0),
                (1024, 60, 0),
                (4, 0, 0),
                (0, 255, 0),
            ]
            rng.shuffle(cases)
            unknown_id = f"r{next(i for i,case in enumerate(cases) if case[1]==255)}"
            for i, (flag, mapq, keep) in enumerate(cases):
                sequence = "".join(rng.choices("ACGT", k=12))
                fields = [
                    f"r{i}",
                    flag,
                    "*" if flag == 4 else "chr1",
                    0 if flag == 4 else 10 + i * 15,
                    mapq,
                    "*" if flag == 4 else "12M",
                    "*",
                    0,
                    0,
                    sequence,
                    "I" * 12,
                ]
                lines.append("\t".join(map(str, fields)))
                expected[f"r{i}"] = {"keep": keep}
            columns = {
                "keep": Column(kind="integer", unit="decision", description="1 for an eligible alignment, else 0")
            }
            prompt = (
                "Filter reads.sam: keep mapped primary alignments with MAPQ >= 30, excluding QC failures, "
                "duplicates, and supplementary alignments. MAPQ 255 means unavailable and must fail. Report"
                " every QNAME as id, including rejected records."
            )
            mutation = {
                "treated_mapq_255_as_high": [
                    {"id": k, "keep": 1 if k == unknown_id else v["keep"]} for k, v in expected.items()
                ]
            }
        elif operation == "sam-fragment-counts":
            categories = rng.sample(range(5), 5)
            for fragment, category in enumerate(categories):
                good = [category not in (2, 3, 4), category not in (1, 3, 4)]
                for mate in range(2):
                    flag = (99 if mate == 0 else 147) + (1024 if category == 4 else 0)
                    mapq = 40 if good[mate] or category == 4 else 20
                    position = 10 + fragment * 30 + mate * 10
                    lines.append(
                        "\t".join(
                            map(
                                str,
                                [
                                    f"p{fragment}",
                                    flag,
                                    "chr1",
                                    position,
                                    mapq,
                                    "8M",
                                    "=",
                                    position + (10 if mate == 0 else -10),
                                    18 if mate == 0 else -18,
                                    "ACGTACGT",
                                    "I" * 8,
                                ],
                            )
                        )
                    )
                lines.append(f"p{fragment}\t355\tchr1\t1\t60\t8M\t=\t10\t18\tACGTACGT\tIIIIIIII")
                expected[f"p{fragment}"] = {"keep": int(all(good)), "passing_mates": sum(good)}
            columns = {
                "keep": Column(kind="integer", unit="decision", description="1 only when both primary mates pass"),
                "passing_mates": Column(kind="integer", unit="mates", description="eligible primary mates"),
            }
            prompt = (
                "For each paired fragment QNAME in reads.sam, count primary mapped mates with MAPQ >= 30, "
                "excluding duplicate, QC-failed, secondary and supplementary records. Keep a fragment only "
                "if both its first and second primary mates pass. Count each mate once, and report all "
                "fragment IDs."
            )
            mutation = {
                "accepted_one_passing_mate": [
                    {"id": k, **v, "keep": int(v["passing_mates"] > 0)} for k, v in expected.items()
                ]
            }
        elif operation == "sam-allele-pileup":
            base_ledger = []
            for i in range(6):
                bases = list("".join(rng.choices("ACGT", k=8)))
                sequence = "TT" + "".join(bases[:4]) + "G" + "".join(bases[4:])
                scores = [35] * len(sequence)
                if i == 1:
                    scores[4] = 10
                flag = 1024 if i == 5 else (16 if i % 2 else 0)
                lines.append(
                    f"r{i}\t{flag}\tchr1\t11\t40\t2S4M1I4M\t*\t0\t0\t{sequence}\t" + "".join(chr(q + 33) for q in scores)
                )
                if i != 5:
                    for j, base in enumerate(bases):
                        query_index = 2 + j + (j >= 4)
                        if scores[query_index] >= 25:
                            base_ledger.append((10 + j, base))
            loci = [12, 14, 17, 50]
            inputs["loci.bed"] = "".join(f"chr1\t{p}\t{p+1}\tsite{i}\n" for i, p in enumerate(loci))
            for i, p in enumerate(loci):
                counts = Counter(base for position, base in base_ledger if position == p)
                expected[f"site{i}"] = {base: counts[base] for base in "ACGT"}
            columns = {
                base: Column(kind="integer", unit="base calls", description=f"eligible {base} calls") for base in "ACGT"
            }
            prompt = (
                "Count A/C/G/T at each one-base locus in loci.bed using reads.sam. Include mapped primary, "
                "nonduplicate, non-QC-failed, nonsupplementary alignments with MAPQ >= 30 and base quality "
                ">= 25 (Phred+33). Replay CIGAR: insertions and soft clips consume query only. SAM SEQ is "
                "already stored in alignment orientation, including reverse-strand records. Return zero "
                "counts at uncovered loci; use locus name as id."
            )
            mutation = {
                "counted_low_quality_base": [
                    {"id": k, **v, "A": v["A"] + int(k == "site0")} for k, v in expected.items()
                ]
            }
        else:
            aligned, junctions = [], []
            offset = rng.randint(1, 8)
            for i in range(5):
                start = 10 + offset + (i // 2) * 20
                flag = 256 if i == 4 else 0
                sequence = "".join(rng.choices("ACGT", k=14))
                lines.append(
                    f"r{i}\t{flag}\tchr1\t{start+1}\t40\t2S4M1I3M2D2M3N2M1H\t*\t0\t0\t{sequence}\t" + ("I" * 14)
                )
                if flag == 0:
                    aligned.extend(
                        [*range(start, start + 7), *range(start + 9, start + 11), *range(start + 14, start + 16)]
                    )
                    junctions.append((start + 11, start + 14))
            if operation == "sam-cigar-coverage":
                windows = [(0, 90), (12, 26), (90, 100)]
                inputs["windows.bed"] = "".join(f"chr1\t{a}\t{b}\tw{i}\n" for i, (a, b) in enumerate(windows))
                for i, (a, b) in enumerate(windows):
                    eligible = [p for p in aligned if a <= p < b]
                    expected[f"w{i}"] = {"depth_sum": len(eligible), "covered_bases": len(set(eligible))}
                columns = {
                    "depth_sum": Column(kind="integer", unit="aligned base calls", description="sum of per-base depths"),
                    "covered_bases": Column(kind="integer", unit="bases", description="positions with positive depth"),
                }
                prompt = (
                    "Compute base coverage over windows.bed from primary mapped reads in reads.sam. M, =, and X"
                    " contribute depth; D and N advance reference without contributing depth, I/S consume query"
                    " only, and H/P contribute neither. Exclude secondary and supplementary alignments. Report "
                    "depth_sum and covered_bases per window name."
                )
                mutation = {
                    "treated_deletions_as_coverage": [
                        {"id": k, **v, "depth_sum": v["depth_sum"] + 8 * int(k == "w0")} for k, v in expected.items()
                    ]
                }
            else:
                assert operation == "sam-junction-support"
                counts = Counter(junctions)
                expected = {f"chr1/{a}/{b}": {"reads": n} for (a, b), n in counts.items()}
                columns = {
                    "reads": Column(
                        kind="integer",
                        unit="distinct read names",
                        description="primary reads supporting this exact intron",
                    )
                }
                prompt = (
                    "Extract splice junctions from primary mapped reads in reads.sam. Only CIGAR N operations "
                    "define introns; D does not. Count distinct QNAMEs per exact junction, excluding "
                    "secondary/supplementary alignments. Use chr/start/end IDs with 0-based half-open intron "
                    "coordinates, and report all observed junctions."
                )
                mutation = {
                    "one_based_intron_coordinates": [
                        {"id": f"chr1/{a+1}/{b}", "reads": n} for (a, b), n in counts.items()
                    ]
                }
        header, body = lines[:3], lines[3:]
        rng.shuffle(body)
        inputs["reads.sam"] = "\n".join(header + body) + "\n"
    elif operation in {"fastq-adapter-trimming", "fastq-quality-trimming"}:
        reads = []
        for i in range(5):
            payload = "".join(rng.choices("ACGT", k=12 + i))
            if operation == "fastq-adapter-trimming":
                adapter = "AGATCGGAAGAGC"
                tail = ["", adapter, adapter[:8], adapter[:5], adapter + "TT"][i]
                sequence = payload + tail
                kept = payload if i in (1, 2, 4) else sequence
                quality = "I" * len(sequence)
            else:
                sequence = payload
                tail = 0 if i == 0 else 3 if i < 4 else len(sequence)
                quality = "I" * (len(sequence) - tail) + "!" * tail
                kept = sequence[: len(sequence) - tail]
            reads.append(f"@read{i}\n{sequence}\n+\n{quality}\n")
            expected[f"read{i}"] = {"sequence": kept, "quality": quality[: len(kept)], "length": len(kept)}
        rng.shuffle(reads)
        inputs = {"reads.fastq": "".join(reads)}
        columns = {
            "sequence": Column(kind="text", unit="DNA", description="trimmed sequence"),
            "quality": Column(kind="text", unit="Phred+33 characters", description="corresponding trimmed qualities"),
            "length": Column(kind="integer", unit="bases", description="trimmed read length"),
        }
        if operation == "fastq-adapter-trimming":
            prompt = (
                "Trim exact adapter AGATCGGAAGAGC from reads.fastq. If the full adapter occurs, trim from "
                "its first occurrence to the end. Otherwise trim the longest suffix matching an adapter "
                "prefix of at least 6 bases. Preserve reads with shorter matches. Trim qualities "
                "identically; retain every read ID."
            )
            mutation = {
                "trimmed_subthreshold_adapter": [
                    (
                        {
                            "id": k,
                            **v,
                            "sequence": v["sequence"][:-5],
                            "quality": v["quality"][:-5],
                            "length": v["length"] - 5,
                        }
                        if k == "read3"
                        else {"id": k, **v}
                    )
                    for k, v in expected.items()
                ]
            }
        else:
            prompt = (
                "Trim consecutive 3-prime bases with Phred+33 quality < 20 from reads.fastq; stop at the "
                "first base with quality >= 20. Preserve internal and 5-prime bases. Trim sequence and "
                "qualities together. Return all read IDs, including empty results."
            )
            mutation = {"dropped_fully_trimmed_read": [{"id": k, **v} for k, v in expected.items() if k != "read4"]}
    else:
        assert operation == "umi-deduplication"
        molecules, rows = {}, []
        for cell in ["c0", "c1", "c2"]:
            for gene in ["g0", "g1"]:
                number = rng.randint(1, 4) if cell != "c2" else 0
                molecules[(cell, gene)] = number
                for i in range(number):
                    umi = ["ACAT", "ACCT", "ACGT", "ACTT"][i]
                    rows.extend([{"cell": cell, "gene": gene, "umi": umi} for _ in range(rng.randint(2, 5))])
        rows.extend(
            [{"cell": "offlist", "gene": "g0", "umi": "ACAT"}, {"cell": "c0", "gene": "ambiguous", "umi": "ACAT"}]
        )
        rng.shuffle(rows)
        inputs = {"alignments.csv": csv_text(rows), "cells.txt": "c0\nc1\nc2\n", "genes.txt": "g0\ng1\n"}
        expected = {f"{cell}/{gene}": {"molecules": n} for (cell, gene), n in molecules.items()}
        columns = {
            "molecules": Column(
                kind="integer", unit="UMIs", description="distinct exact UMIs per eligible cell and gene"
            )
        }
        prompt = (
            "Deduplicate alignments.csv into molecules by exact (cell,gene,umi). Use only the cell and "
            "gene whitelists in cells.txt and genes.txt. The ambiguous gene label is excluded. No UMI "
            "correction or fuzzy matching. Report every cell/gene combination, including zero "
            "molecules, with cell/gene as id."
        )
        mutation = {
            "counted_reads_instead_of_molecules": [
                {"id": k, "molecules": v["molecules"] * 2} for k, v in expected.items()
            ]
        }
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), mutation
    )


PROFILES = {
    "sam-inclusion": (("sam1.6",), ("sam-flags", "mapping-quality", "unknown-quality")),
    "sam-fragment-counts": (("sam1.6",), ("mate-identity", "fragment-filtering", "primary-alignments")),
    "sam-allele-pileup": (("sam1.6", "bed4"), ("cigar-replay", "base-quality", "allele-counts")),
    "sam-cigar-coverage": (("sam1.6", "bed4"), ("cigar-consumption", "depth", "covered-bases")),
    "sam-junction-support": (("sam1.6",), ("splicing", "junction-coordinates", "read-deduplication")),
    "fastq-adapter-trimming": (("fastq-phred33",), ("adapter-matching", "partial-adapters", "quality-synchronization")),
    "fastq-quality-trimming": (("fastq-phred33",), ("three-prime-trimming", "quality-thresholds", "empty-reads")),
    "umi-deduplication": (("csv-header",), ("umi-identity", "whitelists", "molecule-counting")),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        formats,
        ("https://samtools.github.io/hts-specs/SAMv1.pdf", "https://github.com/marcelm/cutadapt"),
        partial(generate_reads, operation=name),
    )
    for name, (formats, skills) in PROFILES.items()
)
