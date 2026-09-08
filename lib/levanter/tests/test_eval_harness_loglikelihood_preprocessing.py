# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import haliax as hax

from levanter.eval_harness import _pack_requests, _record_loglikelihood_segments, _tokenize_loglikelihood_requests

pytest.importorskip("lm_eval", reason="lm_eval unavailable")


class _CharacterTokenizer:
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [[ord(char) for char in text] for text in texts]


def test_tokenize_loglikelihood_requests_normalizes_singleton_list_completion():
    from lm_eval.api.instance import Instance

    tokenizer = _CharacterTokenizer()
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question?", [" Answer"]),
            idx=0,
            metadata=("test_task", 0, None),
        )
    ]

    tokenized = _tokenize_loglikelihood_requests(requests, tokenizer, max_length=64, batch_size=8)

    assert tokenized.skipped_request_ids == set()
    assert tokenized.segment_to_request_id == {0: 0}
    assert len(tokenized.prompt_completions) == 1
    assert tokenized.prompt_completions[0].segment_id == 0
    assert tokenized.prompt_completions[0].prompt_length == len("Question?")
    assert tokenized.prompt_completions[0].ids == [ord(char) for char in "Question? Answer"]


def test_tokenize_loglikelihood_requests_expands_multi_reference_completion():
    from lm_eval.api.instance import Instance

    tokenizer = _CharacterTokenizer()
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question?", [" Answer A", " Answer B"]),
            idx=0,
            metadata=("test_task", 0, None),
        )
    ]

    tokenized = _tokenize_loglikelihood_requests(requests, tokenizer, max_length=64, batch_size=8)

    assert tokenized.skipped_request_ids == set()
    assert tokenized.segment_to_request_id == {0: 0, 1: 0}
    assert [request.segment_id for request in tokenized.prompt_completions] == [0, 1]
    assert [request.ids for request in tokenized.prompt_completions] == [
        [ord(char) for char in "Question? Answer A"],
        [ord(char) for char in "Question? Answer B"],
    ]


def test_tokenize_loglikelihood_requests_marks_empty_completion_as_skipped():
    from lm_eval.api.instance import Instance

    tokenizer = _CharacterTokenizer()
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question?", ""),
            idx=0,
            metadata=("test_task", 0, None),
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question?", " Answer"),
            idx=1,
            metadata=("test_task", 1, None),
        ),
    ]

    tokenized = _tokenize_loglikelihood_requests(requests, tokenizer, max_length=64, batch_size=8)

    assert tokenized.skipped_request_ids == {0}
    assert tokenized.segment_to_request_id == {0: 0, 1: 1}
    assert [request.segment_id for request in tokenized.prompt_completions] == [1]


def test_tokenize_loglikelihood_requests_skips_request_when_all_references_are_empty():
    from lm_eval.api.instance import Instance

    tokenizer = _CharacterTokenizer()
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question?", ["", ""]),
            idx=0,
            metadata=("test_task", 0, None),
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question?", " Answer"),
            idx=1,
            metadata=("test_task", 1, None),
        ),
    ]

    tokenized = _tokenize_loglikelihood_requests(requests, tokenizer, max_length=64, batch_size=8)

    assert tokenized.skipped_request_ids == {0}
    assert tokenized.segment_to_request_id == {0: 0, 1: 0, 2: 1}
    assert [request.segment_id for request in tokenized.prompt_completions] == [2]


def test_pack_requests_returns_skipped_empty_completion_ids():
    from lm_eval.api.instance import Instance

    tokenizer = _CharacterTokenizer()
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("A", ""),
            idx=0,
            metadata=("test_task", 0, None),
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("B", "C"),
            idx=1,
            metadata=("test_task", 1, None),
        ),
    ]

    packed, tokenized = _pack_requests(
        requests,
        tokenizer,
        hax.Axis("pos", 16),
        max_pack_size=2,
        pad_token_id=0,
        return_metadata=True,
    )

    assert tokenized.skipped_request_ids == {0}
    assert tokenized.segment_to_request_id == {0: 0, 1: 1}
    assert len(packed) == 1
    np.testing.assert_array_equal(packed[0].attn_mask.segment_ids[0].array[:2], np.array([1, 1]))


def test_record_loglikelihood_segments_uses_best_reference_and_preserves_request_order():
    result_probs = np.full(3, -np.inf)
    result_probs[0] = 0.0
    result_greedy = np.zeros(3)
    result_greedy[0] = True
    covered_points = np.array([True, False, False])

    _record_loglikelihood_segments(
        result_probs,
        result_greedy,
        covered_points,
        {0: 1, 1: 1, 2: 2},
        np.array([2, 0, 1]),
        np.array([-4.0, -8.0, -3.0]),
        np.array([True, True, False]),
    )

    np.testing.assert_array_equal(result_probs, np.array([0.0, -3.0, -4.0]))
    np.testing.assert_array_equal(result_greedy, np.array([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(covered_points, np.array([True, True, True]))
