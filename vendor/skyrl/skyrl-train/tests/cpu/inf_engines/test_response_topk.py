"""Behavior-policy response top-K capture contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from skyrl_train.inference_engines.response_topk import (
    select_flat_chat_response_topk,
    select_flat_response_topk,
    select_response_topk,
)


def test_flat_chat_head_matches_display_limit_for_ties_and_outside_sampled_tokens():
    rows = [[9, 2, 3, 4], [3, 8, 4, 3], [1, 8, 4, 6], [6, 8, 4, 6]]
    scores = [[-7, -2, -2, -2], [-2, -2, -2, -2], [-2, -2, -2, -2], [-2, -2, -2, -2]]
    flat = SimpleNamespace(
        start_indices=[0, 4, 8, 12],
        end_indices=[4, 8, 12, 16],
        token_ids=np.asarray(rows).reshape(-1).tolist(),
        logprobs=np.asarray(scores).reshape(-1).tolist(),
    )
    selected = select_flat_chat_response_topk(flat, [row[0] for row in rows], 2)
    for index, (tokens, values) in enumerate(zip(rows, scores, strict=True)):
        # The public formatter takes the first K+1 distinct dictionary entries.
        displayed = list(dict(zip(tokens, values, strict=True)).items())[:3]
        expected = sorted(displayed, key=lambda item: (-item[1], item[0]))[:2]
        np.testing.assert_array_equal(selected.topk_indices[index], [token for token, _ in expected])
        np.testing.assert_array_equal(selected.topk_logprobs[index], [score for _, score in expected])
        assert selected.sampled_logprobs[index] == values[0]


@pytest.mark.parametrize("low", [-10000.0, float("-inf")])
def test_flat_chat_head_preserves_display_floor_and_its_token_id_ties(low):
    # A distribution concentrated on token 1 can have zero-probability head
    # entries. Chat floors these scores before the native client orders ties.
    flat = SimpleNamespace(
        start_indices=[0],
        end_indices=[4],
        token_ids=[1, 1, 8, 4],
        logprobs=[0.0, 0.0, low, low - 1],
    )
    selected = select_flat_chat_response_topk(flat, [1], 2)
    assert selected.sampled_logprobs == [0.0]
    np.testing.assert_array_equal(selected.topk_indices, [[1, 4]])
    np.testing.assert_array_equal(selected.topk_logprobs, [[0.0, -9999.0]])
    assert flat.logprobs == [0.0, 0.0, low, low - 1]


def test_flat_capture_matches_dictionary_reference_with_duplicates_and_ties():
    ids = [[9, 2, 3, 4], [3, 2, 3, 4], [1, 8, 4, 6], [6, 8, 4, 6]]
    scores = [[-7, -0.2, -1.2, -2.2], [-1.2, -0.2, -1.2, -2.2], [-2, -2, -2, -2], [-2, -2, -2, -2]]
    flat = SimpleNamespace(
        start_indices=[0, 4, 8, 12],
        end_indices=[4, 8, 12, 16],
        token_ids=np.asarray(ids).reshape(-1).tolist(),
        logprobs=np.asarray(scores).reshape(-1).tolist(),
    )
    sampled, actual_ids, actual_scores = select_flat_response_topk(flat, [9, 3, 1, 6], 3)
    for row, (tokens, values) in enumerate(zip(ids, scores, strict=True)):
        expected_ids, expected_scores = select_response_topk(dict(zip(tokens, values, strict=True)), 3)
        np.testing.assert_array_equal(actual_ids[row], expected_ids)
        np.testing.assert_allclose(actual_scores[row], expected_scores, rtol=0, atol=1e-7)
        assert sampled[row] == pytest.approx(values[0])
    assert actual_ids.dtype == np.int32


def test_flat_capture_preserves_explicit_candidate_order_and_sampled_scores():
    flat = SimpleNamespace(
        start_indices=[0, 3],
        end_indices=[3, 6],
        token_ids=[9, 8, 3, 3, 8, 3],
        logprobs=[-7.0, -2.0, -1.0, -1.0, -2.0, -1.0],
    )
    sampled, ids, scores = select_flat_response_topk(flat, [9, 3], 2, [8, 3])
    assert sampled == [-7.0, -1.0]
    np.testing.assert_array_equal(ids, [[8, 3], [8, 3]])
    np.testing.assert_array_equal(scores, [[-2.0, -1.0], [-2.0, -1.0]])


def test_selected_token_outside_topk_is_not_forced_into_student_candidates():
    # vLLM includes the sampled token in addition to the requested top K.
    ids, scores = select_response_topk(
        {9: -7.0, 3: -0.2, 2: -0.1},
        2,
    )
    assert ids == [2, 3]
    assert scores == [-0.1, -0.2]


def test_tied_candidates_use_stable_token_id_order():
    assert select_response_topk({8: -0.5, 4: -0.5}, 2) == ([4, 8], [-0.5, -0.5])


@pytest.mark.parametrize("scores", [{3: float("nan")}, {3: 0.1}, {-1: -0.2}, {"3": -0.2}])
def test_invalid_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        select_response_topk(scores, 1)


def test_missing_candidates_fail_closed():
    with pytest.raises(ValueError, match="fewer response candidates"):
        select_response_topk({2: -0.1}, 2)
