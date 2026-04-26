# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.test_time_scaling import CandidateRecord, SelectorName, replay_selectors


def _candidate(
    *,
    prompt_id: str,
    sample_index: int,
    text: str,
    extracted_answer: str | None,
    is_correct: bool,
    normalized_logprob: float | None,
) -> CandidateRecord:
    return CandidateRecord(
        prompt_id=prompt_id,
        candidate_id=f"{prompt_id}-{sample_index}",
        sample_index=sample_index,
        raw_text=text,
        extracted_answer=extracted_answer,
        is_correct=is_correct,
        parse_valid=extracted_answer is not None,
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="stop",
        request_latency_seconds=0.1,
        generation_seed=42,
        logprob_sum=(normalized_logprob * 5) if normalized_logprob is not None else None,
        normalized_logprob=normalized_logprob,
    )


def test_replay_selectors_uses_same_candidate_pool():
    candidates = [
        _candidate(
            prompt_id="p0",
            sample_index=0,
            text="\\boxed{5}",
            extracted_answer="5",
            is_correct=False,
            normalized_logprob=-0.05,
        ),
        _candidate(
            prompt_id="p0",
            sample_index=1,
            text="\\boxed{4}",
            extracted_answer="4",
            is_correct=True,
            normalized_logprob=-0.20,
        ),
        _candidate(
            prompt_id="p0",
            sample_index=2,
            text="\\boxed{4}",
            extracted_answer="4",
            is_correct=True,
            normalized_logprob=-0.30,
        ),
    ]

    selections = replay_selectors(
        candidates,
        (
            SelectorName.FIRST_SAMPLE,
            SelectorName.MAJORITY_VOTE,
            SelectorName.NORMALIZED_LOGPROB,
        ),
    )
    by_selector = {selection.selector_name: selection for selection in selections}

    assert by_selector[SelectorName.FIRST_SAMPLE].chosen_candidate_id == "p0-0"
    assert by_selector[SelectorName.FIRST_SAMPLE].oracle_gap is True

    assert by_selector[SelectorName.MAJORITY_VOTE].chosen_candidate_id in {"p0-1", "p0-2"}
    assert by_selector[SelectorName.MAJORITY_VOTE].correctness is True

    assert by_selector[SelectorName.NORMALIZED_LOGPROB].chosen_candidate_id == "p0-0"
    assert by_selector[SelectorName.NORMALIZED_LOGPROB].correctness is False
