# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_usage import CompletionUsage

from marin.inference.chat_completions import ChatCompletionRequest
from marin.test_time_scaling import (
    CandidateGenerationConfig,
    PromptManifest,
    PromptManifestRecord,
    PromptMessage,
    ScoringMode,
    SelectorName,
    TestTimeScalingConfig as TtsRunConfig,
    build_run_summary,
    generate_candidates,
    load_prompt_manifest,
    read_candidate_records,
    read_selection_records,
    replay_selectors,
    write_candidate_records,
    write_prompt_manifest,
    write_run_summary,
    write_selection_records,
)


def _create_choice(tokenizer, response_text: str, logprob_values: list[float]) -> Choice:
    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(logprob_values) != len(token_ids):
        if len(logprob_values) == 1:
            logprob_values = logprob_values * len(token_ids)
        else:
            logprob_values = [logprob_values[0]] * len(token_ids)
    logprobs = []
    for token_id, logprob in zip(token_ids, logprob_values, strict=True):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.append(
            ChatCompletionTokenLogprob(
                token=token,
                logprob=logprob,
                bytes=list(token.encode("utf-8")),
                top_logprobs=[],
            )
        )

    return Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(role="assistant", content=response_text),
        logprobs=ChoiceLogprobs(content=logprobs),
    )


def _create_completion(tokenizer, responses: list[tuple[str, list[float]]]) -> ChatCompletion:
    choices = []
    completion_tokens = 0
    for index, (text, logprobs) in enumerate(responses):
        choice = _create_choice(tokenizer, text, logprobs)
        choice.index = index
        choices.append(choice)
        completion_tokens += len(choice.logprobs.content) if choice.logprobs and choice.logprobs.content else 0

    return ChatCompletion(
        id="chatcmpl-test",
        choices=choices,
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=12,
            completion_tokens=completion_tokens,
            total_tokens=12 + completion_tokens,
        ),
    )


class FakeCompletionProvider:
    def __init__(self, completions: list[ChatCompletion]):
        self._completions = completions
        self._request_index = 0

    def complete_messages(self, request: ChatCompletionRequest):
        assert request.num_completions == 3
        assert request.messages[0]["role"] == "user"
        completion = self._completions[self._request_index]
        self._request_index += 1
        return completion


def test_end_to_end_reasoning_tts_math_vertical_slice(tmp_path, gpt2_tokenizer):
    manifest = PromptManifest(
        manifest_id="math-slice",
        task_name="math-demo",
        records=(
            PromptManifestRecord(
                prompt_id="p0",
                messages=(PromptMessage(role="user", content="What is 2 + 2? Put the answer in \\boxed{}."),),
                expected_answer="\\boxed{4}",
                scoring_mode=ScoringMode.MATH_BOXED,
            ),
            PromptManifestRecord(
                prompt_id="p1",
                messages=(PromptMessage(role="user", content="What is 3 + 4? Put the answer in \\boxed{}."),),
                expected_answer="\\boxed{7}",
                scoring_mode=ScoringMode.MATH_BOXED,
            ),
        ),
    )
    provider = FakeCompletionProvider(
        [
            _create_completion(
                gpt2_tokenizer,
                [
                    ("The answer is \\boxed{5}", [-0.05, -0.05, -0.05, -0.05, -0.05]),
                    ("Working carefully gives \\boxed{4}", [-0.2, -0.2, -0.2, -0.2, -0.2]),
                    ("Checking again gives \\boxed{4}", [-0.3, -0.3, -0.3, -0.3, -0.3]),
                ],
            ),
            _create_completion(
                gpt2_tokenizer,
                [
                    ("By inspection we get \\boxed{7}", [-0.4, -0.4, -0.4, -0.4, -0.4]),
                    ("Another derivation gives \\boxed{7}", [-0.6, -0.6, -0.6, -0.6, -0.6]),
                    ("One more pass confirms \\boxed{7}", [-0.5, -0.5, -0.5, -0.5, -0.5]),
                ],
            ),
        ]
    )
    run_config = TtsRunConfig(
        generation=CandidateGenerationConfig(num_candidates=3, temperature=0.7, top_p=1.0, max_gen_toks=128, seed=11),
        selectors=(
            SelectorName.FIRST_SAMPLE,
            SelectorName.MAJORITY_VOTE,
            SelectorName.NORMALIZED_LOGPROB,
        ),
    )

    output_dir = tmp_path / "artifacts"
    write_prompt_manifest(str(output_dir), manifest)
    candidates = generate_candidates(manifest, provider, run_config.generation)
    write_candidate_records(str(output_dir), candidates)
    saved_candidates = read_candidate_records(str(output_dir / "candidates.jsonl"))
    selections = replay_selectors(saved_candidates, run_config.selectors)
    write_selection_records(str(output_dir), selections)
    summary = build_run_summary(manifest, run_config, saved_candidates, selections)
    write_run_summary(str(output_dir), summary)

    loaded_manifest = load_prompt_manifest(str(output_dir))
    loaded_selections = read_selection_records(str(output_dir / "selected.jsonl"))

    assert loaded_manifest.manifest_id == "math-slice"
    assert len(saved_candidates) == 6
    assert len(loaded_selections) == 6
    assert summary.oracle_accuracy == 1.0
    assert summary.total_candidates == 6

    selector_summaries = {selector.selector_name: selector for selector in summary.selector_summaries}
    assert selector_summaries[SelectorName.FIRST_SAMPLE].accuracy == 0.5
    assert selector_summaries[SelectorName.MAJORITY_VOTE].accuracy == 1.0
