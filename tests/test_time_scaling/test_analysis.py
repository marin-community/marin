# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.test_time_scaling import (
    CandidateGenerationConfig,
    CandidateRecord,
    PromptManifest,
    PromptManifestRecord,
    PromptMessage,
    ScoringMode,
    SelectorName,
    TestTimeScalingConfig as TtsRunConfig,
    build_run_summary,
    replay_selectors,
)


def _candidate(
    *,
    prompt_id: str,
    sample_index: int,
    raw_text: str,
    extracted_answer: str | None,
    is_correct: bool,
    prompt_tokens: int,
    completion_tokens: int,
    request_latency_seconds: float,
) -> CandidateRecord:
    return CandidateRecord(
        prompt_id=prompt_id,
        candidate_id=f"{prompt_id}-{sample_index}",
        sample_index=sample_index,
        raw_text=raw_text,
        extracted_answer=extracted_answer,
        is_correct=is_correct,
        parse_valid=extracted_answer is not None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
        request_latency_seconds=request_latency_seconds,
        generation_seed=7,
        logprob_sum=-0.5 * completion_tokens,
        normalized_logprob=-0.5,
    )


def test_build_run_summary_dedupes_prompt_budget_per_request():
    manifest = PromptManifest(
        manifest_id="math-slice",
        task_name="math-demo",
        records=(
            PromptManifestRecord(
                prompt_id="p0",
                messages=(PromptMessage(role="user", content="What is 2 + 2?"),),
                expected_answer="\\boxed{4}",
                scoring_mode=ScoringMode.MATH_BOXED,
            ),
            PromptManifestRecord(
                prompt_id="p1",
                messages=(PromptMessage(role="user", content="What is 3 + 4?"),),
                expected_answer="\\boxed{7}",
                scoring_mode=ScoringMode.MATH_BOXED,
            ),
        ),
    )
    candidates = [
        _candidate(
            prompt_id="p0",
            sample_index=0,
            raw_text="\\boxed{4}",
            extracted_answer="4",
            is_correct=True,
            prompt_tokens=11,
            completion_tokens=5,
            request_latency_seconds=1.5,
        ),
        _candidate(
            prompt_id="p0",
            sample_index=1,
            raw_text="\\boxed{5}",
            extracted_answer="5",
            is_correct=False,
            prompt_tokens=11,
            completion_tokens=6,
            request_latency_seconds=1.5,
        ),
        _candidate(
            prompt_id="p1",
            sample_index=0,
            raw_text="\\boxed{7}",
            extracted_answer="7",
            is_correct=True,
            prompt_tokens=13,
            completion_tokens=4,
            request_latency_seconds=2.0,
        ),
        _candidate(
            prompt_id="p1",
            sample_index=1,
            raw_text="\\boxed{7}",
            extracted_answer="7",
            is_correct=True,
            prompt_tokens=13,
            completion_tokens=7,
            request_latency_seconds=2.0,
        ),
    ]
    run_config = TtsRunConfig(
        generation=CandidateGenerationConfig(num_candidates=2, temperature=0.7, seed=5),
        selectors=(SelectorName.FIRST_SAMPLE, SelectorName.MAJORITY_VOTE),
    )

    selections = replay_selectors(candidates, run_config.selectors)
    summary = build_run_summary(manifest, run_config, candidates, selections)

    assert summary.total_prompt_tokens == 24
    assert summary.total_completion_tokens == 22
    assert summary.total_request_latency_seconds == 3.5
    assert summary.oracle_accuracy == 1.0
