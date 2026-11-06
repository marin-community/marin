# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math

import numpy as np
import pytest
from marin.testing.inference import snowball
from marin.testing.inference.snowball_position_parity import (
    MAX_LOGPROB_ERROR,
    MAX_P99_LOGPROB_ERROR,
    PAST_WINDOW_GROUP,
    PositionReference,
    ReferenceMetadata,
    ScoredPositions,
    case_reference_from_prompt_logprobs,
    check_reference_matches_prompts,
    compare_positions,
    dispatch_failures,
    expected_groups,
    logprob_failures,
    parse_position_reference,
    position_reference_payload,
    reference_id_grid,
)

METADATA = ReferenceMetadata(
    vllm_version="0.0.0.dev0",
    vllm_commit="0" * 40,
    vllm_release_tag="test",
    tensor_parallel_size=1,
    data_parallel_size=8,
    pipeline_parallel_size=1,
    expert_parallel=True,
    attention_backend="FLASH_ATTN",
    model_uri="s3://bucket/model/",
    model_sha256="1" * 64,
    prompt_fixture_sha256="2" * 64,
    top_k=20,
    min_prob=1e-3,
)


def _entry(scores: dict[int, float]) -> dict:
    return {
        str(token): {"logprob": value, "rank": 1, "decoded_token": f"token_id:{token}"}
        for token, value in scores.items()
    }


def _prompt_logprobs(token_ids: list[int]) -> list[dict | None]:
    """A vLLM-shaped response: top three plus the actual token at every position."""
    entries: list[dict | None] = [None]
    for position in range(1, len(token_ids)):
        scores = {7: math.log(0.6), 8: math.log(0.3), 9: math.log(0.0005)}
        scores.setdefault(token_ids[position], math.log(0.05))
        entries.append(_entry(scores))
    return entries


def _reference(cases: dict[str, list[int]]) -> PositionReference:
    return PositionReference(
        metadata=METADATA,
        cases=tuple(
            case_reference_from_prompt_logprobs(case_id, token_ids, _prompt_logprobs(token_ids))
            for case_id, token_ids in cases.items()
        ),
    )


def _matching_scores(reference: PositionReference, offset: float = 0.0) -> dict[str, ScoredPositions]:
    scored = {}
    for case in reference.cases:
        grid = reference_id_grid(case, reference.metadata.top_k)
        measured = np.full(grid.shape, np.nan)
        for position, values in enumerate(case.top_logprobs):
            measured[position, : len(values)] = np.asarray(values) + offset
        scored[case.id] = ScoredPositions(
            case_id=case.id,
            target_logprobs=np.asarray(case.target_logprobs),
            top1_token_ids=np.asarray([ids[0] for ids in case.top_token_ids], dtype=np.int32),
            reference_logprobs=measured,
        )
    return scored


def test_prompt_logprobs_keep_target_and_prune_improbable_tokens():
    case = case_reference_from_prompt_logprobs("math-gsm8k-01", [1, 7, 42], _prompt_logprobs([1, 7, 42]))

    assert case.num_entries == 2
    assert case.target_logprobs == pytest.approx((math.log(0.6), math.log(0.05)))
    # Token 9 is below MIN_PROB and outside the top two, so it is not stored.
    assert case.top_token_ids == ((7, 8), (7, 8, 42))
    assert case.token_ids_sha256 == hashlib.sha256(b"1,7,42").hexdigest()


def test_prompt_logprobs_reject_a_missing_target_or_misaligned_response():
    entries = _prompt_logprobs([1, 7, 42])
    entries[2] = _entry({7: -0.1, 8: -2.0})
    with pytest.raises(ValueError, match="omitted the prompt token"):
        case_reference_from_prompt_logprobs("case", [1, 7, 42], entries)
    with pytest.raises(ValueError, match="prompt log-prob entries"):
        case_reference_from_prompt_logprobs("case", [1, 7], _prompt_logprobs([1, 7, 42]))


def test_reference_payload_round_trips_and_fails_closed():
    reference = _reference({"code-humaneval-01": [1, 7, 8, 9], "tool-atlas-01": [3, 8]})
    payload = json.loads(json.dumps(position_reference_payload(reference)))

    assert parse_position_reference(payload) == reference
    with pytest.raises(ValueError, match="schema"):
        parse_position_reference({**payload, "schema_version": 0})
    payload["cases"][0]["top_logprobs"].pop()
    with pytest.raises(ValueError, match="entry counts"):
        parse_position_reference(payload)


def test_reference_must_match_the_prompt_token_ids():
    reference = _reference({"code-humaneval-01": [1, 7, 8]})

    check_reference_matches_prompts(reference, {"code-humaneval-01": [1, 7, 8]})
    with pytest.raises(ValueError, match="token ids differ"):
        check_reference_matches_prompts(reference, {"code-humaneval-01": [1, 7, 9]})
    with pytest.raises(ValueError, match="cases differ"):
        check_reference_matches_prompts(reference, {"other-case-01": [1, 7, 8]})


def test_matching_scores_pass_and_report_every_group():
    reference = _reference({"code-humaneval-01": [1, 7, 8, 9], "long-longbench-01": [1] + [7] * 9})
    stats = compare_positions(reference, _matching_scores(reference), sliding_window=4)

    assert set(stats) == {"code", "long", PAST_WINDOW_GROUP} == expected_groups(reference, sliding_window=4)
    # Entries 3..8 of the long case predict tokens 4..9, past the window of 4.
    assert stats[PAST_WINDOW_GROUP].positions == 6
    assert stats["code"].max_abs_logprob_diff == 0.0
    assert stats["code"].top1_agree == stats["code"].positions == 3
    assert logprob_failures(stats, expected_groups=expected_groups(reference, sliding_window=4)) == []


def test_logprob_gate_bounds_p99_and_max_and_fails_closed():
    reference = _reference({"code-humaneval-01": [1, 7, 8, 9]})
    groups = expected_groups(reference)

    small = compare_positions(reference, _matching_scores(reference, offset=MAX_P99_LOGPROB_ERROR / 2))
    assert logprob_failures(small, expected_groups=groups) == []

    shifted = compare_positions(reference, _matching_scores(reference, offset=1.0))
    assert [
        failure.split(":")[1].strip().split(" ")[0] for failure in logprob_failures(shifted, expected_groups=groups)
    ] == ["p99"]

    far = compare_positions(reference, _matching_scores(reference, offset=MAX_LOGPROB_ERROR + 0.5))
    failures = logprob_failures(far, expected_groups=groups)
    assert any("p99" in failure for failure in failures) and any("max" in failure for failure in failures)

    assert logprob_failures(small, expected_groups={*groups, "math"}) == ["math: no compared tokens"]


def test_unmeasured_reference_tokens_fail():
    reference = _reference({"code-humaneval-01": [1, 7, 8]})
    scored = _matching_scores(reference)
    scored["code-humaneval-01"].reference_logprobs[0, 1] = np.nan

    stats = compare_positions(reference, scored)
    assert stats["code"].missing == 1
    assert logprob_failures(stats, expected_groups={"code"}) == ["code: 1 reference tokens not measured"]


def test_top1_disagreement_inside_a_near_tie_is_counted_separately():
    token_ids = [1, 7]
    entries = [None, _entry({7: math.log(0.46), 8: math.log(0.44)})]
    reference = PositionReference(
        metadata=METADATA, cases=(case_reference_from_prompt_logprobs("math-gsm8k-01", token_ids, entries),)
    )
    scored = _matching_scores(reference)
    scored["math-gsm8k-01"].top1_token_ids[0] = 8

    stats = compare_positions(reference, scored)["math"]
    assert (stats.top1_agree, stats.top1_near_tie, stats.top1_disagree) == (0, 1, 0)


def test_dispatch_failures_report_every_dropping_layer():
    assert dispatch_failures([0, 0, 0], [0, 0, 0]) == []
    assert dispatch_failures([0, 2, 0], [0, 1, 4]) == [
        "layer 1: 3 valid assignments dropped (2 sender, 1 receiver)",
        "layer 2: 4 valid assignments dropped (0 sender, 4 receiver)",
    ]


def test_position_reference_reader_checks_the_pinned_sha(tmp_path, monkeypatch):
    reference = _reference({"code-humaneval-01": [1, 7, 8]})
    payload = json.dumps(position_reference_payload(reference)).encode()
    sha256 = hashlib.sha256(payload).hexdigest()
    (tmp_path / f"{sha256}.json").write_bytes(payload)
    monkeypatch.setattr(snowball, "VLLM_POSITION_REFERENCE_PREFIX", str(tmp_path))

    monkeypatch.setattr(snowball, "VLLM_POSITION_REFERENCE_SHA256", "")
    with pytest.raises(RuntimeError, match="not staged"):
        snowball.read_vllm_position_reference()

    monkeypatch.setattr(snowball, "VLLM_POSITION_REFERENCE_SHA256", sha256)
    assert snowball.read_vllm_position_reference() == reference

    (tmp_path / f"{sha256}.json").write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="sha256"):
        snowball.read_vllm_position_reference()
