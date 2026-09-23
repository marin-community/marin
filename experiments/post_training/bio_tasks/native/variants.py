# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Variant-tool calculations with explicit missingness and filtering rules."""

from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import tab_rows


def solve_bcftools(inputs: Path, work: Path) -> list[dict]:
    output = execute(
        ["bcftools", "query", "-f", "[%ID\\t%SAMPLE\\t%AD\\n]", str(inputs / "variants.vcf")],
        work,
        "depths.tsv",
    )
    answer = []
    # Native AD extraction preserves integer counts. Text VAF output rounds to six
    # significant digits, so this check derives the exact ratio from extracted AD.
    for name, sample, depths in tab_rows(output):
        values = list(map(int, depths.split(",")))
        total = sum(values)
        for alt in range(1, len(values)):
            answer.append(
                {
                    "id": f"{name}:{sample}:{alt}",
                    "alt_depth": values[alt],
                    "fraction": values[alt] / total if total else None,
                }
            )
    return answer


def solve_gatk(inputs: Path, work: Path) -> list[dict]:
    selected = work / "selected.vcf"
    execute(
        [
            "gatk",
            "--java-options",
            "-Xmx512m -XX:ActiveProcessorCount=1",
            "SelectVariants",
            "-V",
            str(inputs / "variants.vcf"),
            "-O",
            str(selected),
            "--create-output-variant-index",
            "false",
            "--select-type-to-include",
            "SNP",
            "--restrict-alleles-to",
            "BIALLELIC",
            "--select",
            "vc.hasLog10PError() && QUAL >= 30.0",
        ],
        work,
        "gatk.log",
    )
    # GATK's unfiltered '.' is not a literal PASS; preserve that contract distinction.
    passed = {row[2] for row in tab_rows(selected) if not row[0].startswith("#") and row[6] == "PASS"}
    return [
        {"id": row[2], "keep": int(row[2] in passed)}
        for row in tab_rows(inputs / "variants.vcf")
        if not row[0].startswith("#")
    ]
