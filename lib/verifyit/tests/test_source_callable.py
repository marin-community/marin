# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source callable transport passes private inputs to an installed scorer unchanged."""

import json

from verifyit.execution.source_callable import main


def test_source_callable_preserves_fractional_score_and_source_detail(tmp_path):
    source = tmp_path / "source_scorer.py"
    source.write_text(
        "def score(answer, contract):\n"
        "    return {'score': contract['partial'] if answer == contract['expected'] else 0.0, "
        "'diagnostics': {'matched': answer == contract['expected']}}\n"
    )
    descriptor = tmp_path / "invocation.json"
    descriptor.write_text(
        json.dumps(
            {
                "function": "source_scorer:score",
                "source_path": str(source),
                "args": ["answer", "contract"],
                "reward_key": "score",
                "detail_key": "diagnostics",
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"contract": {"expected": "correct", "partial": 0.375}}))
    answer = tmp_path / "answer.txt"
    result = tmp_path / "score.json"

    answer.write_text("correct")
    main(descriptor, config, answer, result)
    assert json.loads(result.read_text()) == {"reward": 0.375, "detail": {"matched": True}}

    answer.write_text("wrong")
    main(descriptor, config, answer, result)
    assert json.loads(result.read_text()) == {"reward": 0.0, "detail": {"matched": False}}
