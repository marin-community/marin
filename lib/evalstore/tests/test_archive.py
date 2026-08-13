# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from evalstore.archive import (
    Choice,
    EvalSample,
    Grading,
    SampleKind,
    sample_from_archive_row,
    sample_to_archive_row,
)


def test_archive_row_round_trips_each_sample_kind():
    samples = (
        EvalSample(
            task="arc",
            doc_id="1",
            kind=SampleKind.MULTIPLE_CHOICE,
            choices=[Choice(label="A", text="a", loglikelihood=-1.0)],
            model_choice=0,
            target_choice=0,
        ),
        EvalSample(task="gsm8k", doc_id="2", kind=SampleKind.GENERATION, output="4", extracted="4"),
        EvalSample(
            task="aime",
            doc_id="3",
            kind=SampleKind.AGENTIC,
            trajectory_uri="finestore://blobs/t3/trajectory.json",
            grading=Grading(method="harbor:verifier", metric="reward", score=1.0, passed=True),
        ),
    )

    for sample in samples:
        row = sample_to_archive_row(sample, trial_id="t")
        assert row["trial_id"] == "t"
        assert sample_from_archive_row(row) == sample
