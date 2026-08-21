# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert WSCollection.xml (Winograd Schema Challenge 273) -> jsonl
for lm-eval-harness consumption.

Output schema per row, matching MosaicML's gauntlet "schema" task type
(per llm-foundry's local_data/language_understanding/winograd_wsc.jsonl):
    {
        "prefix": <text up to (and not including) the pronoun>,
        "pronoun": <the pronoun substring, trimmed>,
        "suffix": <text after the pronoun>,
        "choices": [<continuation if pronoun = A>, <continuation if pronoun = B>],
        "gold": 0 or 1,
    }

Scoring (handled by the lm-eval YAML): for each candidate, the model sees
`prefix` as context and is scored on log-likelihood of `choices[i]`. Pick
the highest. The choices already incorporate the candidate noun phrase
substituted in.

This file is regenerated whenever WSCollection.xml changes — small enough
to ship the resulting JSONL in the repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).parent
XML = HERE / "WSCollection.xml"
JSONL = HERE / "wsc273.jsonl"


def _text(node, tag: str) -> str:
    el = node.find(tag)
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def main():
    tree = ET.parse(XML)
    root = tree.getroot()
    rows = []
    for schema in root.findall("schema"):
        text_node = schema.find("text")
        txt1 = _text(text_node, "txt1")
        pron = _text(text_node, "pron")
        txt2 = _text(text_node, "txt2")

        answers_node = schema.find("answers")
        answers = [(a.text or "").strip() for a in answers_node.findall("answer")]
        if len(answers) != 2:
            continue

        gold_letter = _text(schema, "correctAnswer").strip().rstrip(".").upper()
        if gold_letter not in ("A", "B"):
            continue
        gold = 0 if gold_letter == "A" else 1

        # MosaicML's schema format: each choice is (candidate_phrase + " " + suffix);
        # the prefix is the text up to (not including) the pronoun. We compute
        # log-likelihood of each choice given the prefix and pick the highest.
        prefix = txt1.rstrip()
        suffix = txt2.lstrip().rstrip()
        # Add a leading space so the choice flows naturally after the prefix.
        choices = [f" {a} {suffix}".rstrip() if suffix else f" {a}" for a in answers]

        rows.append(
            {
                "prefix": prefix,
                "pronoun": pron,
                "suffix": suffix,
                "choices": choices,
                "gold": gold,
            }
        )

    with JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {JSONL}")


if __name__ == "__main__":
    main()
