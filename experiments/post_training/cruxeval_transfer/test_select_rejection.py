# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.cruxeval_transfer.select_rejection import select_first_correct


def test_selection_keeps_one_first_correct_sample_and_reports_uncovered_prompts():
    provenance = [
        {"custom_id": f"{prompt}::{sample}", "prompt_id": prompt, "sample": sample}
        for prompt in ("input", "output", "uncovered")
        for sample in reversed(range(8))
    ]
    audit = [
        row | {"correct": row["custom_id"] in ("input::2", "input::5", "output::0", "output::7")} for row in provenance
    ]
    selected, uncovered = select_first_correct(audit, provenance)
    assert [row["custom_id"] for row in selected] == ["input::2", "output::0"]
    assert uncovered == ["uncovered"]
