# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the issue's exact prompts and select teacher-forced target scores."""

from pathlib import Path

from experiments.evals.vep.protocol import VARIANT_INDEX

HEALTHY_PREFIX = "The following sequence is from a healthy individual:\n"
TEMPLATES = Path(__file__).parent / "templates"


def fields(row: dict, strand: str) -> dict[str, str]:
    sequences = row["sequences"][strand]
    return {
        "reference_sequence": sequences["reference"],
        "ref_base": sequences["reference"][VARIANT_INDEX],
        "alt_base": sequences["alternate"][VARIANT_INDEX],
    }


def classification_prompt(query: dict, demonstrations: list[dict], strand: str, mapping: str) -> str:
    assert mapping in {"primary", "reversed"}
    assert len(demonstrations) in {0, 2}
    template = (TEMPLATES / ("two-shot.txt" if demonstrations else "zero-shot.txt")).read_text().removesuffix("\n")
    if mapping == "reversed":
        template = template.replace("A: pathogenic\nB: benign", "A: benign\nB: pathogenic")
    values = fields(query, strand)
    for index, row in enumerate(demonstrations, 1):
        values.update({f"example_{index}_{key}": value for key, value in fields(row, strand).items()})
        pathogenic_letter = "A" if mapping == "primary" else "B"
        benign_letter = "B" if mapping == "primary" else "A"
        values[f"example_{index}_letter"] = pathogenic_letter if row["label"] else benign_letter
    return template.format(**values)
