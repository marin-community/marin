# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the fixed pilot's likelihood and classification targets."""

from collections.abc import Iterator
from dataclasses import dataclass

from experiments.evals.vep.prompts import HEALTHY_PREFIX, classification_prompt


@dataclass(frozen=True)
class Target:
    variant: str
    method: str
    strand: str
    mapping: str
    candidate: str
    prefix: str
    text: str
    single_token: bool


def targets(rows: list[dict], demonstrations: list[dict], format_check_ids: set[str]) -> Iterator[Target]:
    for row in rows:
        for strand in ("+", "-"):
            sequences = row["sequences"][strand]
            for method, prefix in (("null", ""), ("healthy", HEALTHY_PREFIX)):
                for allele in ("reference", "alternate"):
                    yield Target(row["id"], method, strand, "none", allele, prefix, sequences[allele], False)
            for method, examples in (("zero-shot", []), ("two-shot", demonstrations)):
                for mapping in ("primary", "reversed"):
                    prompt = classification_prompt(row, examples, strand, mapping)
                    for candidate in (" A", " B"):
                        yield Target(row["id"], method, strand, mapping, candidate, prompt, candidate, True)
            if row["id"] in format_check_ids:
                for mapping in ("primary", "reversed"):
                    original = classification_prompt(row, [], strand, mapping)
                    question = original[original.index("Which classification applies") :]
                    prompt = (
                        "Classify variant V01 for Mendelian disease.\n"
                        "The reference and alternate sequences are each 255 bases long.\n"
                        "The mutation is at position 128 (1-based) in the displayed sequences.\n\n"
                        f"Reference sequence: {sequences['reference']}\n"
                        f"Alternate sequence: {sequences['alternate']}\n"
                        f"Mutation V01: {sequences['reference'][127]}>{sequences['alternate'][127]} at position 128\n\n"
                        + question
                    )
                    for candidate in (" A", " B"):
                        yield Target(row["id"], "explicit-pair", strand, mapping, candidate, prompt, candidate, True)
