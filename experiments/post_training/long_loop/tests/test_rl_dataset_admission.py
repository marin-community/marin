# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.long_loop.rl_dataset_admission import (
    exclusion_reasons,
    matching_question,
    question_anchors,
    repository_ids,
    text_digest,
)


def test_repository_and_cross_prefix_identity_exclusion():
    index = {"ids": {"task-instance-123"}, "repos": {"django/django"}, "text_sha256": set()}
    assert repository_ids("django__django-1234") == {"django/django"}
    assert exclusion_reasons("other/task-instance-123", "unrelated", index, check_repo=False) == [
        "exact_or_basename_task_id"
    ]
    assert exclusion_reasons(
        "distinct", "git clone https://github.com/swesmith/django__django.abcdef .", index, check_repo=True
    ) == ["swebench_repository_overlap"]
    assert not exclusion_reasons(
        "distinct", "git clone https://github.com/swesmith/oauthlib__oauthlib.abcdef .", index, check_repo=True
    )


def test_question_match_requires_complete_question_not_only_prefix():
    question = "A cyclist travels twenty miles north before turning east for another ten miles."
    anchors = question_anchors([{"texts": {"question": question}}])
    assert matching_question("System instructions\n" + question.replace(" ", "  ") + "\nAnswer:", anchors)
    assert not matching_question(
        "A cyclist travels twenty miles north before turning west for another ten miles.", anchors
    )
    index = {"ids": set(), "repos": set(), "text_sha256": {text_digest(question)}}
    assert exclusion_reasons("distinct", question.replace(" ", "\n"), index, check_repo=False) == [
        "whitespace_normalized_instruction_sha256"
    ]
