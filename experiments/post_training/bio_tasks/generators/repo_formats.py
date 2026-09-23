# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repository-specific formats, reconciliation, and workflow seeds."""

import math
import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.generators.variants import vcf_text
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def generate_repo_formats(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    expected, inputs = {}, {}
    if operation == "paf-query-coverage":
        size = rng.randint(90, 130)
        offset = rng.randint(2, 8)
        records = [
            ("q0", offset, offset + 20, "+", 100, "P"),
            ("q0", offset + 15, offset + 40, "-", 200, "P"),
            ("q0", 0, size, "+", 400, "S"),
            ("q1", 0, 10, "-", 50, "P"),
            ("q1", 20, 30, "+", 90, "P"),
        ]
        inputs = {
            "alignments.paf": "".join(
                f"{name}\t{size}\t{start}\t{end}\t{strand}\tchr1\t1000\t{target}\t{target+end-start}\t{end-start}\t{end-start}\t60\ttp:A:{kind}\n"
                for name, start, end, strand, target, kind in records
            ),
            "queries.csv": csv_text([{"query": f"q{i}", "length": size} for i in range(3)]),
        }
        expected = {
            f"q{i}": {"covered": covered, "fraction": covered / size, "alignments": count}
            for i, covered, count in [(0, 40, 2), (1, 20, 2), (2, 0, 0)]
        }
        columns = {
            "covered": Column(kind="integer", unit="query bases", description="union of primary alignment query spans"),
            "fraction": Column(
                kind="number", unit="fraction", description="covered / full query length", atol=1e-10, rtol=1e-8
            ),
            "alignments": Column(kind="integer", unit="records", description="records with tp:A:P"),
        }
        prompt = (
            "Summarize query coverage from minimap2-style alignments.paf, including every queries.csv "
            "query. Include only tp:A:P alignments, union their 0-based half-open query spans, and "
            "divide by full query length. PAF query coordinates remain in forward query coordinates on "
            "either strand. Do not sum overlapping spans or count tp:A:S secondary alignments. Use "
            "query id."
        )
        wrong = [{"id": k, **v, "covered": 45 if k == "q0" else v["covered"]} for k, v in expected.items()]
        reason = "double_counted_split_alignment_overlap"
    elif operation == "sam-pair-concordance":
        lines = ["@HD\tVN:1.6\tSO:unsorted", "@SQ\tSN:chr1\tLN:1000", "@SQ\tSN:chr2\tLN:1000"]
        categories = rng.sample(["valid", "same_strand", "too_far", "different_contig", "orphan"], 5)
        for i, category in enumerate(categories):
            pos = rng.randint(20, 80)
            first = 99 if category == "valid" else 65
            second = 147 if category == "valid" else 145 if category != "same_strand" else 129
            mate = pos + 40 if category != "too_far" else pos + 200
            reference = "chr2" if category == "different_contig" else "chr1"
            for number, flag, chrom, start, other in [(1, first, "chr1", pos, mate), (2, second, reference, mate, pos)]:
                if category == "orphan" and number == 2:
                    continue
                lines.append(f"q{i}\t{flag}\t{chrom}\t{start}\t60\t10M\t=\t{other}\t0\tACGTACGTAC\tIIIIIIIIII")
            expected[f"q{i}"] = {"concordant": int(category == "valid")}
        inputs = {"pairs.sam": "\n".join(lines) + "\n"}
        columns = {
            "concordant": Column(
                kind="integer", unit="decision", description="1 if both primary mates satisfy declared fragment policy"
            )
        }
        prompt = (
            "Classify each QNAME in pairs.sam as concordant under this explicit policy: exactly one "
            "mapped primary first mate and one mapped primary second mate, same reference, "
            "inward-facing FR orientation (first forward and left, second reverse and right), and outer"
            " reference span <=100 bases. Ignore proper-pair flag 0x2 as a precomputed assertion; "
            "derive the decision from records. Exclude secondary/supplementary records. CIGARs here are"
            " all 10M. Missing mates fail. Return every QNAME."
        )
        wrong = [{"id": k, "concordant": 1} for k in expected]
        reason = "accepted_discordant_or_orphan_pairs"
    elif operation == "vcf-sample-qc":
        n = rng.randint(4, 7)
        records = []
        for i in range(n):
            calls = ["0/0", "0/1", "./." if i < 2 else "1/1", "./1" if i == 0 else "0/0"]
            records.append(["chr1", 10 + i, f"v{i}", "A", "G", 60, "PASS", ".", "GT", *calls])
        inputs = {"variants.vcf": vcf_text(records, ("s0", "s1", "s2", "s3"))}
        expected = {
            f"s{i}": {"missing": missing, "rate": missing / n, "keep": int(missing / n <= 0.25)}
            for i, missing in enumerate([0, 0, 2, 1])
        }
        columns = {
            "missing": Column(
                kind="integer", unit="genotypes", description="genotypes with at least one missing allele"
            ),
            "rate": Column(
                kind="number",
                unit="fraction",
                description="missing genotype count / all variant records",
                atol=1e-10,
                rtol=1e-8,
            ),
            "keep": Column(kind="integer", unit="decision", description="missing rate <=0.25"),
        }
        prompt = (
            "Compute sample missingness in variants.vcf and keep samples with missing-genotype rate "
            "<=0.25. A genotype with any missing allele counts as missing for this sample-QC task, "
            "including ./1. Denominator is all variant records, before sample filtering. Return each "
            "sample missing count, rate, and keep=0/1. This is genotype missingness, not allele-call "
            "missingness."
        )
        wrong = [{"id": k, **v, "missing": 0 if k == "s3" else v["missing"]} for k, v in expected.items()]
        reason = "partially_missing_genotype_counted_as_complete"
    elif operation == "fastqc-report-reconciliation":
        manifest = []
        for i in range(3):
            count = rng.randint(4, 9)
            length = rng.randint(10, 20)
            reported = count + 1 if i == 1 else count
            manifest.append(
                {"sample": f"s{i}", "fastq": f"s{i}.fastq", "report": f"s{i}_fastqc_data.txt", "include": int(i < 2)}
            )
            inputs[f"s{i}.fastq"] = "".join(
                f"@r{j}\n" + ("ACGT" * 10)[:length] + "\n+\n" + "I" * length + "\n" for j in range(count)
            )
            inputs[f"s{i}_fastqc_data.txt"] = (
                "##FastQC\t0.12.1\n>>Basic Statistics\tpass\n#Measure\tValue\n"
                f"Filename\ts{i}.fastq\nFile type\tConventional base calls\nEncoding\tSanger / Illumina 1.9\n"
                f"Total Sequences\t{reported}\nSequences flagged as poor quality\t0\n"
                f"Sequence length\t{length}\n%GC\t50\n>>END_MODULE\n"
            )
            if i < 2:
                expected[f"s{i}"] = {
                    "actual_reads": count,
                    "reported_reads": reported,
                    "difference": count - reported,
                    "matches": int(count == reported),
                }
        inputs["samples.csv"] = csv_text(manifest)
        columns = {
            name: Column(kind="integer", unit="reads" if name != "matches" else "decision", description=description)
            for name, description in [
                ("actual_reads", "FASTQ record count"),
                ("reported_reads", "FastQC Basic Statistics Total Sequences"),
                ("difference", "actual minus reported"),
                ("matches", "1 if counts agree"),
            ]
        }
        prompt = (
            "Audit each included samples.csv sample by reconciling its raw FASTQ record count with the "
            "Total Sequences field in its native FastQC data report. Resolve filenames through the "
            "manifest rather than file ordering; exclude include=0 samples. Preserve mismatches as "
            "matches=0 with signed actual-minus-reported difference. Report sample IDs; do not equate a"
            " FastQC pass label with correct sample accounting."
        )
        wrong = [{"id": k, **v, "matches": 1} for k, v in expected.items()]
        reason = "trusted_report_pass_without_reconciliation"
    elif operation == "alignment-partitions":
        codons = rng.randint(4, 8)
        end = 3 * codons
        alignment = "ATG" * codons + "A"
        inputs = {
            "alignment.fa": f">a\n{alignment}\n>b\n{alignment}\n",
            "partitions.nex": (
                f"#NEXUS\nbegin sets;\n charset codon1 = 1-{end}\\3;\n"
                f" charset codon2 = 2-{end}\\3;\n charset codon3 = 3-{end}\\3;\n"
                " charset extra = 1-3;\nend;\n"
            ),
        }
        for index in range(end + 1):
            names = [] if index == end else [f"codon{index%3+1}"] + (["extra"] if index < 3 else [])
            expected[str(index)] = {
                "partitions": ",".join(names),
                "memberships": len(names),
                "valid": int(len(names) == 1),
            }
        columns = {
            "partitions": Column(
                kind="text", unit="partition names", description="lexically sorted memberships, comma-separated"
            ),
            "memberships": Column(kind="integer", unit="partitions", description="number of sets containing site"),
            "valid": Column(kind="integer", unit="decision", description="1 if exactly one partition contains site"),
        }
        prompt = (
            "Audit partitions.nex NEXUS charsets against alignment.fa. Charset coordinates are 1-based "
            "inclusive; backslash denotes stride (e.g. 1-12\\3 selects 1,4,7,10). Report every alignment"
            " column using 0-based string id, all sorted partition memberships, their count, and "
            "valid=1 only for exactly one membership. Detect both overlapping assignments and "
            "unassigned sites rather than silently repairing them."
        )
        wrong = [{"id": k, **v, "valid": 1} for k, v in expected.items()]
        reason = "ignored_overlaps_and_unassigned_sites"
    elif operation == "newick-split-support":
        n = rng.randint(3, 6)
        length = rng.randint(1, 9) / 10
        trees = [f"((a:{length},b:0.2):0.1,(c:0.3,d:0.4):0.1);" for _ in range(n)]
        trees.extend(["(a:0.1,(b:0.2,(c:0.3,d:0.4):0.1):0.2);", "((a:0.1,c:0.2):0.1,(b:0.3,d:0.4):0.1);"])
        rng.shuffle(trees)
        inputs = {
            "replicates.nwk": "\n".join(trees) + "\n",
            "splits.csv": csv_text(
                [{"split": "ab_cd", "side": "a,b"}, {"split": "ac_bd", "side": "a,c"}, {"split": "ad_bc", "side": "a,d"}]
            ),
        }
        expected = {
            "ab_cd": {"support": (n + 1) / (n + 2), "trees": n + 1},
            "ac_bd": {"support": 1 / (n + 2), "trees": 1},
            "ad_bc": {"support": 0.0, "trees": 0},
        }
        columns = {
            "support": Column(
                kind="number",
                unit="fraction of replicate trees",
                description="unrooted bipartition occurrence frequency",
                atol=1e-10,
                rtol=1e-8,
            ),
            "trees": Column(kind="integer", unit="trees", description="trees displaying split, counted once per tree"),
        }
        prompt = (
            "Calculate support for splits.csv bipartitions across the Newick trees in replicates.nwk. "
            "Each side lists one subset of the common tip set; its complement defines the other. Treat "
            "trees as unrooted for split matching, ignore branch lengths, and count a split at most "
            "once per tree even if two root-child edges represent it. Different root placements must "
            "agree. Use split id."
        )
        wrong = [{"id": k, **v, "trees": 2 * v["trees"]} for k, v in expected.items()]
        reason = "double_counted_root_complement_edges"
    elif operation == "gtf-coverage-counts":
        lines = []
        read_length = rng.choice([50, 75, 100])
        for i in range(3):
            exon_a, exon_b = rng.randint(20, 40), rng.randint(20, 40)
            coverage = rng.randint(2, 8) + 0.25
            start = 100 * i + 1
            end = start + exon_a + 10 + exon_b - 1
            attributes = f'gene_id "g{i//2}"; transcript_id "t{i}";'
            lines.append(f'chr1\tStringTie\ttranscript\t{start}\t{end}\t.\t+\t.\t{attributes} cov "{coverage}";')
            for left, right in [(start, start + exon_a - 1), (start + exon_a + 10, end)]:
                lines.append(f"chr1\tStringTie\texon\t{left}\t{right}\t.\t+\t.\t{attributes}")
            expected[f"t{i}"] = {
                "gene": f"g{i//2}",
                "exon_length": exon_a + exon_b,
                "estimated_count": math.ceil(coverage * (exon_a + exon_b) / read_length),
            }
        inputs = {"transcripts.gtf": "\n".join(lines) + "\n", "read_length.txt": str(read_length) + "\n"}
        columns = {
            "gene": Column(kind="text", unit="gene ID", description="gene_id associated with transcript"),
            "exon_length": Column(kind="integer", unit="bases", description="sum of inclusive exon lengths"),
            "estimated_count": Column(
                kind="integer",
                unit="estimated reads",
                description="ceil(transcript coverage * exon length / read length)",
            ),
        }
        prompt = (
            "Reconstruct StringTie-style transcript read-count estimates from transcripts.gtf and "
            "read_length.txt. Sum 1-based inclusive exon lengths per transcript, read cov from its "
            "transcript feature, and compute ceil(cov*exon_length/read_length). Preserve "
            "transcript-to-gene mapping. Do not use genomic transcript span or call these observed "
            "molecule counts. Return transcript IDs."
        )
        wrong = [{"id": k, **v, "exon_length": v["exon_length"] + 10} for k, v in expected.items()]
        reason = "included_intron_in_transcript_length"
    elif operation == "bedgraph-threshold-peaks":
        start = rng.randint(5, 20)
        inputs = {
            "signal.bedgraph": (
                f"chr1\t{start}\t{start+5}\t4\nchr1\t{start+5}\t{start+10}\t5\nchr1\t{start+10}\t{start+15}\t7\nchr1\t{start+15}\t{start+20}\t0\nchr1\t{start+20}\t{start+25}\t6\n"
            ),
            "threshold.txt": "5\n",
        }
        expected = {
            "chr1:1": {"start": start + 5, "end": start + 15, "maximum": 7.0},
            "chr1:2": {"start": start + 20, "end": start + 25, "maximum": 6.0},
        }
        columns = {
            "start": Column(kind="integer", unit="0-based bases", description="inclusive peak start"),
            "end": Column(kind="integer", unit="0-based bases", description="exclusive peak end"),
            "maximum": Column(
                kind="number", unit="signal units", description="maximum signal across merged peak", atol=1e-8, rtol=1e-8
            ),
        }
        prompt = (
            "Call contiguous intervals where signal.bedgraph >= threshold.txt. Coordinates are 0-based "
            "half-open; merge directly touching qualifying intervals on the same chromosome, but do not"
            " bridge below-threshold or missing regions. Report each merged peak start/end and maximum "
            "signal with chromosome:1-based_peak_index in coordinate order. This prescribed threshold "
            "does not imply a statistical peak-calling p-value."
        )
        wrong = [{"id": k, **v, "start": v["start"] + 5 if k == "chr1:1" else v["start"]} for k, v in expected.items()]
        reason = "excluded_signal_equal_to_threshold"
    else:
        assert operation == "sra-spot-export"
        rows = []
        for i in range(3):
            for read, kind in [(1, "biological"), (2, "technical"), (3, "biological")]:
                if i == 2 and read == 3:
                    continue
                sequence = "".join(rng.choice("ACGT") for _ in range(12))
                quality = "".join(chr(33 + rng.randint(20, 40)) for _ in sequence)
                rows.append(
                    {"spot": f"spot{i}", "read_index": read, "read_type": kind, "sequence": sequence, "quality": quality}
                )
                if kind == "biological":
                    destination = "single" if i == 2 else "R1" if read == 1 else "R2"
                    expected[f"spot{i}:{read}"] = {"destination": destination, "sequence": sequence, "quality": quality}
        inputs = {"spots.csv": csv_text(rows)}
        columns = {
            name: Column(
                kind="text",
                unit=(
                    "read routing" if name == "destination" else "bases" if name == "sequence" else "Phred+33 characters"
                ),
                description=description,
            )
            for name, description in [
                ("destination", "R1, R2, or single after biological-read selection"),
                ("sequence", "unaltered biological read sequence"),
                ("quality", "unaltered quality string"),
            ]
        }
        prompt = (
            "Apply split-3 export semantics to the supplied spot/read ledger spots.csv: exclude "
            "technical reads, then send two biological reads of a spot to R1 and R2 in read_index "
            "order; a spot with one biological read goes to single. Preserve sequence and quality. "
            "Output spot:original_read_index IDs. The ledger is an independently authored source of "
            "truth, not a native SRA archive; no live accession download is required."
        )
        wrong = [
            {"id": k, **v, "destination": "R1" if v["destination"] == "single" else v["destination"]}
            for k, v in expected.items()
        ]
        reason = "routed_orphan_to_paired_output"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "paf-query-coverage": ("split-alignments", "query-span-union", "paf-strand"),
    "sam-pair-concordance": ("mate-identity", "fragment-orientation", "discordance"),
    "vcf-sample-qc": ("sample-missingness", "partial-genotypes", "inclusive-cutoffs"),
    "fastqc-report-reconciliation": ("qc-aggregation", "sample-identity", "raw-report-reconciliation"),
    "alignment-partitions": ("nexus-charsets", "codon-stride", "partition-validation"),
    "newick-split-support": ("unrooted-splits", "replicate-support", "root-invariance"),
    "gtf-coverage-counts": ("coverage-to-counts", "inclusive-exon-length", "estimated-counts"),
    "bedgraph-threshold-peaks": ("signal-threshold", "peak-merging", "boundary-semantics"),
    "sra-spot-export": ("spot-read-types", "mate-routing", "orphan-export"),
}
FORMATS = {
    "paf-query-coverage": ("paf", "csv-header"),
    "sam-pair-concordance": ("sam1.6",),
    "vcf-sample-qc": ("vcf4.3",),
    "fastqc-report-reconciliation": ("fastqc-data-text", "fastq-phred33", "csv-header"),
    "alignment-partitions": ("nexus-sets", "aligned-fasta"),
    "newick-split-support": ("newick", "csv-header"),
    "gtf-coverage-counts": ("gtf",),
    "bedgraph-threshold-peaks": ("bedgraph",),
    "sra-spot-export": ("csv-spot-ledger",),
}
RECIPES = tuple(
    Recipe(name, "1", Difficulty.MEDIUM, skills, FORMATS[name], (), partial(generate_repo_formats, operation=name))
    for name, skills in SKILLS.items()
)
