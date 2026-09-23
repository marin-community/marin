# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""VCF operations, strand-aware coding consequences, and genotype equilibrium."""

import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def vcf_text(records: list[list], samples: tuple[str, ...] = ("alice", "bob"), contig_length: int = 1000) -> str:
    header = (
        f"##fileformat=VCFv4.3\n##contig=<ID=chr1,length={contig_length}>\n"
        '##FILTER=<ID=q10,Description="Low quality">\n'
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">\n'
    )
    return (
        header
        + "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples])
        + "\n"
        + "".join("\t".join(map(str, row)) + "\n" for row in records)
    )


def generate_variants(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    expected, inputs = {}, {}
    if operation == "genotype-hwe":
        rows = []
        for i in range(3):
            aa, ab, bb = [rng.randint(5, 20) for _ in range(3)]
            rows.append({"locus": f"l{i}", "hom_ref": aa, "het": ab, "hom_alt": bb})
            total = aa + ab + bb
            p = (2 * aa + ab) / (2 * total)
            counts = [total * p * p, 2 * total * p * (1 - p), total * (1 - p) * (1 - p)]
            expected[f"l{i}"] = {
                "expected_het": counts[1],
                "chi_square": sum(
                    (observed - fit) ** 2 / fit for observed, fit in zip([aa, ab, bb], counts, strict=True)
                ),
                "n": total,
            }
        inputs = {"genotypes.csv": csv_text(rows)}
        columns = {
            "expected_het": Column(
                kind="number", unit="individuals", description="HWE expected heterozygote count", atol=1e-8, rtol=1e-8
            ),
            "chi_square": Column(
                kind="number",
                unit="statistic",
                description="Pearson statistic without continuity correction",
                atol=1e-8,
                rtol=1e-8,
            ),
            "n": Column(kind="integer", unit="diploid individuals", description="total called genotypes"),
        }
        prompt = (
            "For each biallelic diploid locus in genotypes.csv, estimate allele frequencies from "
            "genotype counts and calculate expected heterozygote count and the Pearson Hardy-Weinberg "
            "chi-square statistic over all three genotype classes. Do not apply continuity correction "
            "or report an asymptotic p-value. Use locus id."
        )
        wrong = [{"id": k, **v, "expected_het": v["expected_het"] / 2} for k, v in expected.items()]
        reason = "missing_factor_two_for_heterozygotes"
    elif operation == "variant-coding-consequences":
        genome = list("C" * 300)
        gff = ["##gff-version 3", "##sequence-region chr1 1 300"]
        records = []
        cases = [
            ("GCT", "GCC", "A", "A", "synonymous", 2),
            ("AAA", "TAA", "K", "*", "stop_gained", 0),
            ("ATG", "ATA", "M", "I", "missense", 2),
            ("TAA", "CAA", "*", "Q", "stop_lost", 0),
        ]
        rng.shuffle(cases)
        complement = dict(zip("ACGT", "TGCA", strict=True))
        for i, (ref, alt, aa, bb, effect, offset) in enumerate(cases):
            start = 20 + 50 * i + rng.randint(0, 5)
            strand = "+" if i % 2 == 0 else "-"
            # Stop loss acts on the terminal stop of a complete reference CDS.
            # The other substitutions are internal, after a separate initiation codon.
            cds = "ATGGGT" + ref if effect == "stop_lost" else "ATG" + ref + "GGTTAA"
            codon_offset = 6 if effect == "stop_lost" else 3
            genomic = cds if strand == "+" else "".join(complement[x] for x in cds[::-1])
            genome[start : start + len(cds)] = genomic
            for feature, phase, attributes in [
                ("gene", ".", f"ID=gene{i}"),
                ("mRNA", ".", f"ID=transcript{i};Parent=gene{i}"),
                ("exon", ".", f"ID=exon{i};Parent=transcript{i}"),
                ("CDS", "0", f"ID=tx{i};Parent=transcript{i}"),
            ]:
                gff.append(f"chr1\tfixture\t{feature}\t{start+1}\t{start+len(cds)}\t.\t{strand}\t{phase}\t{attributes}")
            index = codon_offset + offset
            position = start + index if strand == "+" else start + len(cds) - 1 - index
            refbase = ref[offset] if strand == "+" else complement[ref[offset]]
            altbase = alt[offset] if strand == "+" else complement[alt[offset]]
            records.append(["chr1", position + 1, f"v{i}", refbase, altbase, 60, "PASS", ".", "GT", "0/1", "0/0"])
            expected[f"v{i}:tx{i}"] = {
                "ref_codon": ref,
                "alt_codon": alt,
                "ref_aa": aa,
                "alt_aa": bb,
                "consequence": effect,
            }
        inputs = {
            "genome.fa": ">chr1\n" + "".join(genome) + "\n",
            "cds.gff3": "\n".join(gff) + "\n",
            "variants.vcf": vcf_text(records, contig_length=len(genome)),
        }
        columns = {
            name: Column(
                kind="text",
                unit="sequence" if name != "consequence" else "classification",
                description=name.replace("_", " "),
            )
            for name in ["ref_codon", "alt_codon", "ref_aa", "alt_aa", "consequence"]
        }
        prompt = (
            "Annotate every SNV in variants.vcf against overlapping CDS records in cds.gff3 and "
            "genome.fa. Each CDS is a complete phase-0 coding segment; use the standard code and "
            "transcript orientation, including reverse complementation of minus-strand alleles. "
            "Classify as synonymous, missense, stop_gained, or stop_lost. Use variant_ID:CDS_ID, and "
            "report oriented reference/alternate codons and amino acids (* for stop)."
        )
        wrong = [{"id": k, **v, "consequence": "missense"} for k, v in expected.items()]
        reason = "treated_all_coding_snvs_as_missense"
    else:
        records = []
        if operation == "vcf-allelic-depth":
            for i in range(3):
                depths = [rng.randint(3, 15) for _ in range(3)]
                records.append(
                    [
                        "chr1",
                        10 + i,
                        f"v{i}",
                        "A",
                        "C,G",
                        60,
                        "PASS",
                        "AF=0.2,0.3",
                        "DP:AD:GT",
                        f"99:{','.join(map(str,depths))}:1/2",
                        "99:0,0,0:0/0",
                    ]
                )
                for sample, values in [("alice", depths), ("bob", [0, 0, 0])]:
                    for alt in [1, 2]:
                        expected[f"v{i}:{sample}:{alt}"] = {
                            "alt_depth": values[alt],
                            "fraction": values[alt] / sum(values) if sum(values) else None,
                        }
            columns = {
                "alt_depth": Column(kind="integer", unit="reads", description="AD entry for this ALT allele"),
                "fraction": Column(
                    kind="number",
                    unit="fraction",
                    description="ALT AD divided by sum of all AD; null when zero",
                    nullable=True,
                    atol=1e-10,
                    rtol=1e-8,
                ),
            }
            prompt = (
                "For every sample and alternate allele in variants.vcf, report ALT read depth and allelic "
                "fraction ALT_AD/sum(AD). Parse FORMAT by name; DP can differ from sum(AD) and is not the "
                "denominator. Zero summed AD gives null fraction. Use variant_ID:sample:1-based_ALT_index."
            )
            wrong = [
                {"id": k, **v, "fraction": v["alt_depth"] / 99 if v["fraction"] is not None else None}
                for k, v in expected.items()
            ]
            reason = "used_dp_as_ad_denominator"
        elif operation == "vcf-site-filtering":
            cases = [
                (60, "PASS", "A", "C", 1),
                (30, "PASS", "G", "T", 1),
                (29, "PASS", "A", "G", 0),
                (80, "q10", "C", "T", 0),
                (80, ".", "G", "A", 0),
                (90, "PASS", "A", "AT", 0),
                (90, "PASS", "A", "C,G", 0),
                (None, "PASS", "A", "G", 0),
            ]
            rng.shuffle(cases)
            for i, (qual, status, ref, alt, keep) in enumerate(cases):
                records.append(
                    [
                        "chr1",
                        10 + i,
                        f"v{i}",
                        ref,
                        alt,
                        qual if qual is not None else ".",
                        status,
                        ".",
                        "GT",
                        "0/1",
                        "0/0",
                    ]
                )
                expected[f"v{i}"] = {"keep": keep}
            columns = {
                "keep": Column(kind="integer", unit="decision", description="1 for passing biallelic SNV, otherwise 0")
            }
            prompt = (
                "Classify every variants.vcf record for inclusion: keep only biallelic single-nucleotide "
                "A/C/G/T variants with FILTER exactly PASS and numeric QUAL >= 30. FILTER=. is unassessed, "
                "not PASS; missing QUAL fails. Report keep=0 or 1 for every variant ID."
            )
            wrong = [{"id": k, "keep": 1} for k in expected]
            reason = "ignored_site_filters"
        elif operation == "vcf-genotype-masking":
            cases = [("0/1", 10, 20), ("1|0", 9, 50), ("1/1", 30, 19), ("0/0", 15, None), ("./1", 12, 40)]
            rng.shuffle(cases)
            for i, (gt, dp, gq) in enumerate(cases):
                records.append(
                    [
                        "chr1",
                        10 + i,
                        f"v{i}",
                        "A",
                        "G",
                        60,
                        "PASS",
                        ".",
                        "GQ:GT:DP",
                        f'{gq if gq is not None else "."}:{gt}:{dp}',
                        "50:0/0:20",
                    ]
                )
                expected[f"v{i}:alice"] = {"gt": gt if dp >= 10 and gq is not None and gq >= 20 else "./."}
                expected[f"v{i}:bob"] = {"gt": "0/0"}
            columns = {
                "gt": Column(kind="text", unit="VCF genotype", description="original GT or unphased missing diploid ./.")
            }
            prompt = (
                "Mask low-quality genotypes in variants.vcf: retain the original GT string only when DP >= "
                "10 and GQ >= 20, both present. Otherwise output ./. regardless of original phasing. "
                "Preserve partially missing GT and phasing on retained calls. Parse reordered FORMAT "
                "fields. Return every variant_ID:sample."
            )
            wrong = [{"id": k, "gt": "0/0"} for k in expected]
            reason = "imputed_masked_calls_as_reference"
        elif operation == "vcf-multiallelic-splitting":
            for i in range(3):
                a, b, c = [rng.randint(2, 12) for _ in range(3)]
                records.append(
                    [
                        "chr1",
                        10 + i,
                        f"v{i}",
                        "A",
                        "C,G",
                        60,
                        "PASS",
                        "AF=0.2,0.3",
                        "GT:AD",
                        f"1/2:{a},{b},{c}",
                        f"0|2:{c},{b},{a}",
                    ]
                )
                for sample, depth, genotypes in [
                    ("alice", [a, b, c], ["1/.", "./1"]),
                    ("bob", [c, b, a], ["0|.", "0|1"]),
                ]:
                    for alt in [1, 2]:
                        expected[f"v{i}:{alt}:{sample}"] = {
                            "alt": "C" if alt == 1 else "G",
                            "gt": genotypes[alt - 1],
                            "ref_depth": depth[0],
                            "alt_depth": depth[alt],
                            "af": 0.2 if alt == 1 else 0.3,
                        }
            columns = {
                "alt": Column(kind="text", unit="allele", description="selected alternate allele"),
                "gt": Column(
                    kind="text", unit="VCF genotype", description="selected ALT as 1, REF as 0, other ALT as missing"
                ),
                "ref_depth": Column(kind="integer", unit="reads", description="original AD[0]"),
                "alt_depth": Column(kind="integer", unit="reads", description="selected AD entry"),
                "af": Column(
                    kind="number", unit="fraction", description="selected Number=A INFO/AF value", atol=1e-10, rtol=1e-8
                ),
            }
            prompt = (
                "Split each multiallelic variants.vcf record conceptually into one row per ALT and sample. "
                "Select the matching Number=A AF and Number=R AD values. Recode GT: reference=0, selected "
                "ALT=1, every other ALT=.; preserve allele order and / versus |. Do not merge discarded ALT"
                " depth into REF. Use variant_ID:original_1-based_ALT_index:sample."
            )
            wrong = [{"id": k, **v, "gt": v["gt"].replace(".", "0")} for k, v in expected.items()]
            reason = "other_alternate_became_reference"
        else:
            assert operation == "vcf-minimal-representation"
            for i, (ref, alt, trimmed_ref, trimmed_alt, offset) in enumerate(
                [
                    ("AAC", "ACC", "A", "C", 1),
                    ("CAAT", "CAT", "CA", "C", 0),
                    ("GTA", "GTTA", "G", "GT", 0),
                    ("T", "C", "T", "C", 0),
                ]
            ):
                pos = 20 + i * 30 + rng.randint(0, 8)
                records.append(["chr1", pos, f"v{i}", ref, alt, 60, "PASS", ".", "GT", "0/1", "0/0"])
                expected[f"v{i}"] = {"pos": pos + offset, "ref": trimmed_ref, "alt": trimmed_alt}
            columns = {
                "pos": Column(kind="integer", unit="1-based bases", description="VCF position after trimming"),
                "ref": Column(kind="text", unit="bases", description="minimally trimmed REF"),
                "alt": Column(kind="text", unit="bases", description="minimally trimmed ALT"),
            }
            prompt = (
                "Minimize the biallelic alleles in variants.vcf by removing common suffix bases first, then"
                " common prefix bases, retaining at least one base in each allele. Increment the 1-based "
                "POS for every removed prefix base. This task is trimming only, with no reference-based "
                "left alignment. Use variant ID."
            )
            wrong = [{"id": k, **v, "pos": v["pos"] - 1} for k, v in expected.items()]
            reason = "returned_zero_based_vcf_position"
        inputs = {"variants.vcf": vcf_text(records)}
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "vcf-allelic-depth": ("format-fields", "allele-indexing", "missing-denominators"),
    "vcf-site-filtering": ("filter-semantics", "variant-types", "missing-quality"),
    "vcf-genotype-masking": ("genotype-quality", "phasing", "missingness"),
    "vcf-multiallelic-splitting": ("number-a", "number-r", "genotype-recoding"),
    "vcf-minimal-representation": ("allele-normalization", "anchor-bases", "vcf-coordinates"),
    "variant-coding-consequences": ("strand", "coding-consequences", "genetic-code"),
    "genotype-hwe": ("allele-frequencies", "equilibrium", "expected-counts"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        (
            ("csv-header",)
            if name == "genotype-hwe"
            else ("vcf4.3", "gff3", "fasta") if name == "variant-coding-consequences" else ("vcf4.3",)
        ),
        ("https://samtools.github.io/hts-specs/VCFv4.3.pdf",),
        partial(generate_variants, operation=name),
    )
    for name, skills in SKILLS.items()
)
