# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standing policy: data-curation jobs default to a three-day timeout."""

from rigging.timing import Duration

from experiments.post_training.task_curation import full_qc
from experiments.post_training.task_curation.difficulty import runtime as difficulty_runtime

THREE_DAYS = 3 * 24 * 3600


def test_data_job_timeouts_default_to_three_days():
    # difficulty_trials.ATTEMPT_TIMEOUT is the same 259200-second bound but its module
    # needs the harbor lane environment, so this test pins the importable constants.
    assert difficulty_runtime.JOB_TIMEOUT == Duration.from_seconds(THREE_DAYS)
    assert full_qc.FULL_QC_JOB_TIMEOUT == Duration.from_hours(72)
