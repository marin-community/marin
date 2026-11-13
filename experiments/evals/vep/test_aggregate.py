# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict

import pytest

from experiments.evals.vep.aggregate import aggregate
from experiments.evals.vep.targets import targets


def test_score_signs_mapping_average_and_incomplete_run_rejection(tmp_path):
    row = {
        "id": "3:1000:A:C",
        "label": True,
        "subset": "missense_variant",
        "match_group": 1,
        "chrom": "3",
        "sequences": {
            "+": {"reference": "A" * 255, "alternate": "A" * 127 + "C" + "A" * 127},
            "-": {"reference": "T" * 255, "alternate": "T" * 127 + "G" + "T" * 127},
        },
    }
    panels_path = tmp_path / "panels.json"
    panels_path.write_text(json.dumps({"panels": {"pilot": [row], "demonstrations": [row, dict(row, label=False)]}}))
    records = [
        {
            "type": "start",
            "panels_sha256": hashlib.sha256(panels_path.read_bytes()).hexdigest(),
            "panel": "pilot",
            "variant_count": 1,
            "format_check_ids": [],
        }
    ]
    for target in targets([row], [row, dict(row, label=False)], set()):
        if not target.single_token:
            score = -10 if target.candidate == "reference" else (-14 if target.strand == "+" else -12)
        elif target.mapping == "primary":
            score = -1 if target.candidate == " A" else -3
        else:
            score = -5 if target.candidate == " A" else -2
        records.append(
            {
                "type": "score",
                **asdict(target),
                "logprob": score,
                "input_sha256": hashlib.sha256((target.prefix + target.text).encode()).hexdigest(),
            }
        )
    records.append({"type": "complete"})
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text("\n".join(json.dumps(row) for row in records))
    result = aggregate(panels_path, scores_path)[0]
    assert result["null"] == 3
    assert result["null/strand-difference"] == 2
    assert result["zero-shot"] == 2
    assert result["zero-shot/reversed"] == 3
    assert result["zero-shot/mapping-average"] == 2.5
    scores_path.write_text("\n".join(json.dumps(row) for row in records[:-1]))
    with pytest.raises(AssertionError, match="incomplete"):
        aggregate(panels_path, scores_path)
    scores_path.write_text("\n".join(json.dumps(row) for row in records[:2] + records[3:]))
    with pytest.raises(AssertionError, match="Missing scoring"):
        aggregate(panels_path, scores_path)
