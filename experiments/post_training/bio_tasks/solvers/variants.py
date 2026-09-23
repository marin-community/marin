# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""VCF parsers and independent input-derived answers."""

import re
from functools import partial
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta, reverse_complement, tab_rows, table, translate


def vcf_records(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text().splitlines()
    samples = next(line.split("\t")[9:] for line in lines if line.startswith("#CHROM"))
    return samples, [line.split("\t") for line in lines if line and not line.startswith("#")]


def solve_variants(inputs: Path, operation: str) -> list[dict]:
    if operation == "genotype-hwe":
        answer = []
        for row in table(inputs / "genotypes.csv"):
            observed = [int(row[key]) for key in ["hom_ref", "het", "hom_alt"]]
            n = sum(observed)
            q = (observed[1] + 2 * observed[2]) / (2 * n)
            predicted = [n * (1 - q) ** 2, n * 2 * q * (1 - q), n * q * q]
            answer.append(
                {
                    "id": row["locus"],
                    "expected_het": predicted[1],
                    "chi_square": sum(o * o / e for o, e in zip(observed, predicted, strict=True)) - n,
                    "n": n,
                }
            )
        return answer
    samples, records = vcf_records(inputs / "variants.vcf")
    answer = []
    for chrom, pos, name, ref, alt, qual, status, info, fmt, *calls in records:
        if operation == "vcf-site-filtering":
            answer.append(
                {
                    "id": name,
                    "keep": int(
                        ref in "ACGT"
                        and len(ref) == 1
                        and alt in "ACGT"
                        and len(alt) == 1
                        and status == "PASS"
                        and qual != "."
                        and float(qual) >= 30
                    ),
                }
            )
        elif operation == "vcf-minimal-representation":
            position = int(pos)
            while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
                ref, alt = ref[:-1], alt[:-1]
            while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
                ref, alt = ref[1:], alt[1:]
                position += 1
            answer.append({"id": name, "pos": position, "ref": ref, "alt": alt})
        elif operation == "variant-coding-consequences":
            genome = fasta(inputs / "genome.fa")
            for seqid, _, feature, start, end, _, strand, phase, attributes in tab_rows(inputs / "cds.gff3"):
                if feature != "CDS" or seqid != chrom or not int(start) <= int(pos) <= int(end):
                    continue
                assert phase == "0", "This recipe requires complete phase-zero CDS records"
                cds = genome[seqid][int(start) - 1 : int(end)]
                index = int(pos) - int(start)
                alternate = alt
                if strand == "-":
                    cds = reverse_complement(cds)
                    index = int(end) - int(pos)
                    alternate = reverse_complement(alt)
                codon_start = index // 3 * 3
                old = cds[codon_start : codon_start + 3]
                mutated = cds[:index] + alternate + cds[index + 1 :]
                new = mutated[codon_start : codon_start + 3]
                a, b = translate(old), translate(new)
                effect = (
                    "synonymous" if a == b else "stop_gained" if b == "*" else "stop_lost" if a == "*" else "missense"
                )
                identifier = dict(pair.split("=", 1) for pair in attributes.split(";"))["ID"]
                answer.append(
                    {
                        "id": name + ":" + identifier,
                        "ref_codon": old,
                        "alt_codon": new,
                        "ref_aa": a,
                        "alt_aa": b,
                        "consequence": effect,
                    }
                )
        else:
            for sample, call in zip(samples, calls, strict=True):
                fields = dict(zip(fmt.split(":"), call.split(":"), strict=True))
                if operation == "vcf-allelic-depth":
                    values = list(map(int, fields["AD"].split(",")))
                    for index in range(1, len(values)):
                        answer.append(
                            {
                                "id": f"{name}:{sample}:{index}",
                                "alt_depth": values[index],
                                "fraction": values[index] / sum(values) if sum(values) else None,
                            }
                        )
                elif operation == "vcf-genotype-masking":
                    keep = all(
                        fields[key] != "." and int(fields[key]) >= threshold
                        for key, threshold in [("DP", 10), ("GQ", 20)]
                    )
                    answer.append({"id": name + ":" + sample, "gt": fields["GT"] if keep else "./."})
                else:
                    assert operation == "vcf-multiallelic-splitting"
                    af = dict(pair.split("=", 1) for pair in info.split(";"))["AF"].split(",")
                    ad = list(map(int, fields["AD"].split(",")))
                    for index, allele in enumerate(alt.split(","), 1):
                        tokens = re.split(r"([/|])", fields["GT"])
                        gt = "".join(
                            token if token in {"/", "|", "0", "."} else "1" if int(token) == index else "."
                            for token in tokens
                        )
                        answer.append(
                            {
                                "id": f"{name}:{index}:{sample}",
                                "alt": allele,
                                "gt": gt,
                                "ref_depth": ad[0],
                                "alt_depth": ad[index],
                                "af": float(af[index - 1]),
                            }
                        )
    return answer


NAMES = (
    "vcf-allelic-depth",
    "vcf-site-filtering",
    "vcf-genotype-masking",
    "vcf-multiallelic-splitting",
    "vcf-minimal-representation",
    "variant-coding-consequences",
    "genotype-hwe",
)
SOLVERS = {name: partial(solve_variants, operation=name) for name in NAMES}
