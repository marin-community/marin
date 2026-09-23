# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Annotation and interval tasks with construction-based references."""

import random
from fractions import Fraction
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe


def generate_intervals(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    expected, inputs = {}, {}
    if operation == "bed12-exons":
        lines = []
        for t in range(4):
            origin = 20 + 80 * t
            lengths = [rng.randint(4, 9) for _ in range(3)]
            starts = [origin, origin + 20, origin + 45]
            strand = "+" if t % 2 == 0 else "-"
            lines.append(
                "\t".join(
                    map(
                        str,
                        [
                            "chr1",
                            origin,
                            starts[-1] + lengths[-1],
                            f"tx{t}",
                            0,
                            strand,
                            origin,
                            origin,
                            0,
                            3,
                            ",".join(map(str, lengths)) + ",",
                            ",".join(str(s - origin) for s in starts) + ",",
                        ],
                    )
                )
            )
            order = range(3) if strand == "+" else range(2, -1, -1)
            for rank, i in enumerate(order, 1):
                expected[f"tx{t}/{rank}"] = {"start": starts[i], "end": starts[i] + lengths[i], "strand": strand}
        inputs = {"transcripts.bed": "\n".join(lines) + "\n"}
        columns = {
            "start": Column(kind="integer", unit="bases", description="0-based exon start"),
            "end": Column(kind="integer", unit="bases", description="exclusive exon end"),
            "strand": Column(kind="text", unit="orientation", description="+ or -"),
        }
        prompt = (
            "Expand transcripts.bed (BED12) into exons. blockStarts are relative to chromStart. "
            "Number exons from the transcript 5-prime end, reversing genomic block order on minus strand. "
            "Use transcript_name/exon_rank as id, with ranks starting at 1. Preserve 0-based half-open coordinates."
        )
        mutation = {"used_one_based_start": [{"id": k, **v, "start": v["start"] + 1} for k, v in expected.items()]}
    elif operation in {"gtf-splicing", "gff-cds-translation"}:
        genome = list("".join(rng.choices("ACGT", k=600)))
        annotations = [] if operation == "gtf-splicing" else ["##gff-version 3", "##sequence-region chr1 1 600"]
        for t in range(4):
            strand = "+" if t % 2 == 0 else "-"
            origin = 10 + t * 140
            phase = 1 + t % 2
            if operation == "gtf-splicing":
                chunks = ["".join(rng.choices("ACGT", k=rng.randint(6, 12))) for _ in range(3)]
                expected[f"tx{t}"] = {"sequence": "".join(chunks), "length": sum(map(len, chunks))}
            else:
                amino_acid, codon = rng.choice([("A", "GCT"), ("G", "GGT"), ("V", "GTT")])
                coding = "ATG" + codon + "AAATAG"
                chunks = ["A" * phase + coding[:4], coding[4:]]
                expected[f"tx{t}"] = {"protein": f"M{amino_acid}K*"}
            # Plant transcript-oriented pieces directly; the solver reads genomic slices and strand.
            positions = [origin + i * 35 for i in range(len(chunks))]
            genomic_chunks = (
                chunks if strand == "+" else [s.translate(str.maketrans("ACGT", "TGCA"))[::-1] for s in reversed(chunks)]
            )
            if operation == "gff-cds-translation":
                annotations.append(
                    f"chr1\tsynthetic\tmRNA\t{positions[0]+1}\t{positions[-1]+len(genomic_chunks[-1])}\t.\t{strand}\t.\tID=tx{t}"
                )
            records = []
            for i, (start, chunk) in enumerate(zip(positions, genomic_chunks, strict=True)):
                genome[start : start + len(chunk)] = chunk
                transcript_index = i if strand == "+" else len(chunks) - i - 1
                if operation == "gtf-splicing":
                    attrs = f'gene_name "shared name"; transcript_id "tx{t}"; gene_id "g{t}";'
                    records.append(f"chr1\tsynthetic\texon\t{start+1}\t{start+len(chunk)}\t.\t{strand}\t.\t{attrs}")
                else:
                    cds_phase = phase if transcript_index == 0 else 2
                    records.append(
                        f"chr1\tsynthetic\tCDS\t{start+1}\t{start+len(chunk)}\t.\t{strand}\t{cds_phase}\tID=cds{t}_{i};Parent=tx{t}"
                    )
            rng.shuffle(records)
            annotations.extend(records)
        extension = "gtf" if operation == "gtf-splicing" else "gff3"
        inputs = {
            "genome.fa": ">chr1\n" + "\n".join("".join(genome[i : i + 60]) for i in range(0, 600, 60)) + "\n",
            f"annotations.{extension}": "\n".join(annotations) + "\n",
        }
        if operation == "gtf-splicing":
            columns = {
                "sequence": Column(kind="text", unit="DNA", description="spliced transcript sequence"),
                "length": Column(kind="integer", unit="bases", description="spliced length"),
            }
            prompt = (
                "Reconstruct each transcript in annotations.gtf using exon features and genome.fa. Group by "
                "quoted transcript_id, sort exons, splice out introns, and orient 5-prime to 3-prime. GTF coordinates "
                "are 1-based closed. Use transcript_id as id; shared gene_name values are not identifiers."
            )
            mutation = {
                "reversed_without_complementing": [
                    {"id": k, **v, "sequence": v["sequence"][::-1]} for k, v in expected.items()
                ]
            }
        else:
            columns = {
                "protein": Column(
                    kind="text", unit="amino acids", description="standard-code translation, including * stops"
                )
            }
            prompt = (
                "Translate each mRNA's CDS in annotations.gff3 against genome.fa using the standard genetic code. "
                "Order and orient CDS segments in transcript direction. The 5-prime end is partial: omit the first "
                "CDS segment's phase bases, then concatenate all remaining CDS bases. Internal phase describes "
                "codon continuation: do not discard those bases. Keep terminal stop as *. Use mRNA ID as id."
            )
            mutation = {"lost_terminal_stop": [{"id": k, "protein": v["protein"][:-1]} for k, v in expected.items()]}
    else:
        # Per-base membership is the construction reference; oracle algorithms use interval endpoints.
        length = rng.randint(80, 100)
        spans = [(5, 15), (12, 24), (35, 43), (43, rng.randint(48, 55))]
        shift = rng.randint(0, 8)
        spans = [(a + shift, b + shift) for a, b in spans]
        occupied = {p for a, b in spans for p in range(a, b)}
        inputs = {
            "features.bed": "".join(f"chr1\t{a}\t{b}\tf{i}\n" for i, (a, b) in enumerate(spans)),
            "chrom.sizes": f"chr1\t{length}\n",
        }
        if operation == "bed-union-coverage":
            expected = {"chr1": {"covered": len(occupied), "fraction": len(occupied) / length}}
            columns = {
                "covered": Column(kind="integer", unit="bases", description="distinct covered bases"),
                "fraction": Column(
                    kind="number",
                    unit="proportion",
                    description="union coverage / chromosome length",
                    atol=1e-10,
                    rtol=1e-8,
                ),
            }
            prompt = (
                "Compute the union coverage of features.bed for every chromosome in chrom.sizes. "
                "Overlapping or touching intervals must not double-count bases. Use chromosome as id."
            )
            mutation = {
                "summed_overlapping_lengths": [
                    {
                        "id": "chr1",
                        "covered": sum(b - a for a, b in spans),
                        "fraction": sum(b - a for a, b in spans) / length,
                    }
                ]
            }
        elif operation == "bed-complement":
            runs = []
            for pos in range(length):
                if pos in occupied:
                    continue
                if not runs or runs[-1][-1] != pos - 1:
                    runs.append([])
                runs[-1].append(pos)
            expected = {f"chr1/{i}": {"start": run[0], "end": run[-1] + 1} for i, run in enumerate(runs, 1)}
            columns = {
                "start": Column(kind="integer", unit="bases", description="0-based uncovered interval start"),
                "end": Column(kind="integer", unit="bases", description="exclusive uncovered interval end"),
            }
            prompt = (
                "Find maximal uncovered intervals in features.bed within chrom.sizes, including chromosome "
                "edges. Use 0-based half-open intervals and chromosome/rank IDs, ranked by increasing start"
                " from 1."
            )
            mutation = {"closed_end": [{"id": k, **v, "end": v["end"] - 1} for k, v in expected.items()]}
        elif operation == "bed-nearest-features":
            points = [2, 30 + shift, 60]
            inputs["queries.bed"] = "".join(f"chr1\t{p}\t{p+1}\tq{i}\n" for i, p in enumerate(points))
            for i, p in enumerate(points):
                distances = {f"f{j}": min(abs(p - x) for x in range(a, b)) for j, (a, b) in enumerate(spans)}
                winner = min(distances, key=lambda key: (distances[key], key))
                expected[f"q{i}"] = {"feature": winner, "distance": distances[winner]}
            columns = {
                "feature": Column(
                    kind="text", unit="feature ID", description="nearest feature name; lexicographic tie-break"
                ),
                "distance": Column(
                    kind="integer", unit="bases", description="minimum absolute difference of occupied base indices"
                ),
            }
            prompt = (
                "For each one-base query in queries.bed, find the nearest feature in features.bed on the "
                "same chromosome. Distance is the minimum absolute difference between occupied 0-based base"
                " indices; overlap is zero and adjacent bases differ by 1. Break ties by lexicographically "
                "smallest feature name. Use query name as id."
            )
            mutation = {
                "off_by_one_distance": [{"id": k, **v, "distance": v["distance"] + 1} for k, v in expected.items()]
            }
        elif operation == "bed-stranded-promoters":
            upstream, downstream = rng.randint(12, 18), 4
            genes = [(3, 12, "+"), (25, 40, "-"), (65, 78, "-")]
            inputs["genes.bed"] = "".join(f"chr1\t{a}\t{b}\tg{i}\t0\t{s}\n" for i, (a, b, s) in enumerate(genes))
            for i, (a, b, strand) in enumerate(genes):
                tss = a if strand == "+" else b - 1
                selected = [
                    p for p in range(length) if -upstream <= (p - tss) * (1 if strand == "+" else -1) < downstream
                ]
                expected[f"g{i}"] = {"start": min(selected), "end": max(selected) + 1}
            columns = {
                "start": Column(kind="integer", unit="bases", description="clipped promoter start"),
                "end": Column(kind="integer", unit="bases", description="exclusive promoter end"),
            }
            prompt = (
                "Construct promoters from BED6 genes.bed, clipped to chrom.sizes. "
                "TSS is start on + and end-1 on -. Include bases whose signed distance in "
                f"transcription direction from the TSS is >= -{upstream} and < {downstream}. "
                "Return half-open coordinates, gene name as id."
            )
            inputs["window.txt"] = f"{upstream}\t{downstream}\n"
            mutation = {
                "unclipped_chromosome_start": [
                    {"id": k, **v, "start": -1 if k == "g0" else v["start"]} for k, v in expected.items()
                ]
            }
        else:
            assert operation == "bedgraph-weighted-signal"
            values = [rng.randint(2, 6), rng.randint(7, 12)]
            track = [(10, 20, values[0]), (25, 45, values[1])]
            inputs["signal.bedgraph"] = "".join(f"chr1\t{a}\t{b}\t{v}\n" for a, b, v in track)
            windows = [(0, 50), (15, 35), (60, 70)]
            inputs["windows.bed"] = "".join(f"chr1\t{a}\t{b}\tw{i}\n" for i, (a, b) in enumerate(windows))
            signal = {p: v for a, b, v in track for p in range(a, b)}
            for i, (a, b) in enumerate(windows):
                total = sum(signal.get(p, 0) for p in range(a, b))
                expected[f"w{i}"] = {"integral": total, "mean": float(Fraction(total, b - a))}
            columns = {
                "integral": Column(kind="integer", unit="signal times bases", description="per-base signal sum"),
                "mean": Column(
                    kind="number", unit="signal", description="sum divided by full window length", atol=1e-10, rtol=1e-8
                ),
            }
            prompt = (
                "Summarize nonoverlapping signal.bedgraph over windows.bed. Weight signal by overlapping "
                "bases and treat uncovered bases as zero. Mean uses the entire window length. Use window "
                "name as id."
            )
            mutation = {
                "ignored_uncovered_bases": [
                    {"id": k, **v, "mean": v["integral"] / 30 if k == "w0" else v["mean"]} for k, v in expected.items()
                ]
            }
    contract = Contract(columns=columns, expected=expected)
    return Instance(prompt + " Inputs are in /app/inputs.", inputs, contract, mutation)


PROFILES = {
    "bed12-exons": (("bed12",), ("block-offsets", "strand-aware-exon-order")),
    "gtf-splicing": (("gtf-exons", "fasta-dna"), ("splicing", "transcript-joins", "strand")),
    "gff-cds-translation": (("gff3-cds", "fasta-dna"), ("cds-phase", "genetic-code", "strand")),
    "bed-union-coverage": (("bed4", "chrom-sizes"), ("interval-union", "coverage-denominators")),
    "bed-nearest-features": (("bed4",), ("nearest-feature", "coordinate-distance", "ties")),
    "bed-stranded-promoters": (("bed6", "chrom-sizes"), ("tss", "strand", "boundary-clipping")),
    "bed-complement": (("bed4", "chrom-sizes"), ("interval-complement", "chromosome-boundaries")),
    "bedgraph-weighted-signal": (("bedgraph", "bed4"), ("weighted-signal", "uncovered-bases")),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        formats,
        (
            "https://genome.ucsc.edu/FAQ/FAQformat.html",
            "https://jun2026.archive.ensembl.org/info/website/upload/gff.html",
        ),
        partial(generate_intervals, operation=name),
    )
    for name, (formats, skills) in PROFILES.items()
)
