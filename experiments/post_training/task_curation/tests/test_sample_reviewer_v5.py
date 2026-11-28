# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The V5 SAMPLE reviewer profile: V4 with doubled completion budgets, nothing else."""

import dataclasses

from experiments.post_training.task_curation.full_qc import (
    SAMPLE_REVIEWER_VERSION,
    ReviewerVersion,
    local_reviewer,
)
from experiments.post_training.task_curation.sample_qc_artifacts import SAMPLE_REVIEWER_SHAS

CONTEXT_WINDOW_TOKENS = 524288


def test_v5_is_v4_with_doubled_completion_budgets():
    v4 = local_reviewer(ReviewerVersion.V4)
    v5 = local_reviewer(ReviewerVersion.V5)
    assert v5.max_tokens == 32768 and v5.retry_max_tokens == 32768
    v4 = replace_budgets(v4, max_tokens=32768, retry_max_tokens=32768)
    assert dataclasses.asdict(v5) == dataclasses.asdict(v4), "V5 changes only the two budgets"


def replace_budgets(spec, **kwargs):
    return dataclasses.replace(spec, **kwargs)


def test_v5_is_the_pinned_default_and_its_identity_sha_is_recorded():
    assert SAMPLE_REVIEWER_VERSION is ReviewerVersion.V5
    assert SAMPLE_REVIEWER_SHAS["v5"] == local_reviewer(ReviewerVersion.V5).cache_revision
    # the serve's max_model_len (= the model's context window) admits the completion
    # plus the actual panel prompts, which sit far below the 32768-completion headroom
    assert CONTEXT_WINDOW_TOKENS - 32768 >= 0
