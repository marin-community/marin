from types import SimpleNamespace

import pytest
from evalchemy_config import EvaluationConfig
from lm_eval.api.instance import Instance

from eval.limits import (
    ContextWindowExceededError,
    MissingContextLengthError,
    encoded_token_count,
    endpoint_prompt_token_count,
    ensure_context_window,
    format_key_value_args,
    message_content_token_count,
    parse_key_value_args,
    preflight_endpoint_generation,
    require_context_length,
    resolve_evaluation_limits,
    safe_generation_cap,
)
from eval.task import BaseBenchmark


def _args(**overrides):
    values = {
        "model_args": "pretrained=test/model,base_url=http://localhost/v1",
        "gen_kwargs": "temperature=0",
        "max_length": None,
        "max_tokens": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_limits_are_materialized_for_native_and_custom_paths():
    args = _args(max_length=32768, max_tokens=2048, gen_kwargs="temperature=0,max_new_tokens=2048")

    limits = resolve_evaluation_limits(args)

    assert limits.max_length == 32768
    assert limits.max_tokens == 2048
    assert parse_key_value_args(args.model_args)["max_length"] == "32768"
    native_kwargs = parse_key_value_args(args.gen_kwargs)
    assert native_kwargs == {"temperature": "0", "max_gen_toks": "2048"}
    assert args.max_tokens == 2048


@pytest.mark.parametrize("legacy_key", ["max_tokens", "max_new_tokens", "max_gen_toks"])
def test_legacy_output_aliases_resolve_to_one_contract(legacy_key):
    args = _args(model_args="pretrained=test/model,max_model_len=4096", gen_kwargs=f"{legacy_key}=512")

    limits = resolve_evaluation_limits(args)

    assert limits.max_length == 4096
    assert limits.max_tokens == 512
    assert parse_key_value_args(args.model_args)["max_length"] == "4096"
    assert parse_key_value_args(args.model_args)["max_model_len"] == "4096"
    assert parse_key_value_args(args.gen_kwargs) == {"max_gen_toks": "512"}


def test_conflicting_limit_spellings_fail_before_any_benchmark_runs():
    with pytest.raises(ValueError, match="conflicting max_tokens"):
        resolve_evaluation_limits(_args(max_tokens=512, gen_kwargs="max_gen_toks=1024"))

    with pytest.raises(ValueError, match="conflicting max_length"):
        resolve_evaluation_limits(_args(max_length=8192, model_args="pretrained=test/model,max_length=4096"))


def test_runner_and_portable_config_normalize_the_same_legacy_limit_aliases():
    args = _args(model_args="pretrained=test/model,max_model_len=4096", gen_kwargs="max_new_tokens=512")
    runner_limits = resolve_evaluation_limits(args)
    portable_config = EvaluationConfig.model_validate(
        {
            "extra_model_args": {"max_model_len": 4096},
            "gen_kwargs": "max_new_tokens=512",
        }
    )

    assert (portable_config.max_length, portable_config.max_tokens) == (
        runner_limits.max_length,
        runner_limits.max_tokens,
    )
    portable_model_args = parse_key_value_args(format_key_value_args(portable_config.extra_model_args))
    runner_model_args = parse_key_value_args(args.model_args)
    assert {key: portable_model_args[key] for key in portable_model_args} == {
        key: runner_model_args[key] for key in portable_model_args
    }
    assert parse_key_value_args(portable_config.gen_kwargs) == parse_key_value_args(args.gen_kwargs)


def source_items(*, seed_repeats=True, **options):
    from eval.contracts.eval_spec import SourceRef
    from eval.sources.common import SourceOptions, text_specs
    return text_specs(
        ({"id": str(index), "question": "question", "answer": "answer"} for index in range(5)),
        SourceOptions("fixture", SourceRef("fixture"), **options),
        prompt=lambda row: row["question"], target=lambda row: row["answer"], grader="exact_match",
        generation_kwargs={"max_gen_toks": 23},
        seed_repeats=seed_repeats,
    )


def test_request_cannot_escape_resolved_output_cap():
    spec = next(source_items(max_tokens=2048, context_length=32768))
    request = spec.requests[0]
    assert request.context_length == 32768
    assert request.generation == {"max_gen_toks": 2048}


def test_sample_limit_is_applied_before_expanding_trials():
    specs = list(source_items(limit=2, num_samples=3, seed_repeats=False))
    assert [spec.source_id for spec in specs] == ["0", "0", "0", "1", "1", "1"]
    assert [spec.trial for spec in specs] == [0, 1, 2, 0, 1, 2]
    # Explicit sample counts use independent trial seeds even when the source's
    # legacy repeat protocol omits them.
    assert [spec.requests[0].generation["seed"] for spec in specs] == [0, 1, 2, 0, 1, 2]


def test_sampling_seeds_default_and_sources_can_opt_out():
    from eval.contracts.eval_spec import SourceRef
    from eval.sources.common import SourceOptions, text_specs

    specs = list(text_specs(
        [{"id": "row"}], SourceOptions("fixture", SourceRef("fixture"), seed=7),
        prompt=lambda row: row["id"], target=lambda row: row["id"], grader="exact_match",
        generation_kwargs={"do_sample": True}, repeats=2,
    ))
    unseeded = list(text_specs(
        [{"id": "row"}], SourceOptions("fixture", SourceRef("fixture"), seed=7),
        prompt=lambda row: row["id"], target=lambda row: row["id"], grader="exact_match",
        generation_kwargs={"do_sample": True}, repeats=2, seed_repeats=False,
    ))

    assert [spec.requests[0].generation["seed"] for spec in specs] == [7, 8]
    assert all("seed" not in spec.requests[0].generation for spec in unseeded)


def test_limited_iterator_does_not_construct_later_prompts():
    assert len(list(source_items(limit=2))) == 2


class _Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(sum(len(message["content"].split()) for message in messages) + 3))


def test_shared_content_token_count_supports_text_and_messages():
    tokenizer = _Tokenizer()

    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    assert encoded_token_count(encode, "one two three") == 3
    assert (
        message_content_token_count(
            encode,
            [{"role": "user", "content": "one two"}, {"role": "assistant", "content": "three"}],
        )
        == 3
    )


def test_shared_context_contract_uses_typed_errors():
    with pytest.raises(MissingContextLengthError):
        require_context_length(None, task_name="long-context-task")

    with pytest.raises(ContextWindowExceededError) as error:
        ensure_context_window(context_length=128, prompt_tokens=100, output_tokens=20, safety_tokens=16)

    assert error.value.required_tokens == 136
    assert error.value.context_length == 128


def test_endpoint_preflight_caps_the_historical_tier2_overflow_before_transport():
    # 32,768 served tokens - 2,049 rendered prompt - 64-token wiggle room.
    assert safe_generation_cap(context_length=32768, prompt_tokens=2049, requested_max_tokens=30720) == 30655

    kwargs, prompt_tokens, cap = preflight_endpoint_generation(
        tokenizer=_Tokenizer(),
        payloads=["token " * 2049],
        gen_kwargs={"max_gen_toks": 30720, "temperature": 0},
        context_length=32768,
    )
    assert prompt_tokens == 2049
    assert cap == 30655
    assert kwargs == {"max_gen_toks": 30655, "temperature": 0}


def test_endpoint_preflight_uses_the_chat_template_and_is_a_noop_without_context():
    messages = [{"role": "user", "content": "one two"}]
    assert endpoint_prompt_token_count(_Tokenizer(), messages) == 5

    kwargs, prompt_tokens, cap = preflight_endpoint_generation(
        tokenizer=_Tokenizer(),
        payloads=[messages],
        gen_kwargs={"max_tokens": 128},
        context_length=None,
    )
    assert kwargs == {"max_tokens": 128}
    assert prompt_tokens is None
    assert cap is None


def test_endpoint_preflight_uses_the_endpoint_chat_template_controls():
    class ReasoningTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, reasoning_effort="high"):
            base = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            return {"input_ids": base + ([0] * 20 if reasoning_effort == "high" else [])}

    kwargs, prompt_tokens, cap = preflight_endpoint_generation(
        tokenizer=ReasoningTokenizer(),
        payloads=[[{"role": "user", "content": "one two"}]],
        gen_kwargs={"max_tokens": 20},
        context_length=100,
        chat_template_kwargs={"reasoning_effort": "low"},
    )

    assert prompt_tokens == 5
    assert cap == 20
    assert kwargs == {"max_tokens": 20}


def test_endpoint_preflight_skips_chat_count_without_a_client_tokenizer():
    kwargs, prompt_tokens, cap = preflight_endpoint_generation(
        tokenizer=None,
        payloads=[[{"role": "user", "content": "question"}]],
        gen_kwargs={"max_gen_toks": 128, "temperature": 0},
        context_length=4096,
    )

    assert kwargs == {"max_gen_toks": 128, "temperature": 0}
    assert prompt_tokens is None
    assert cap is None


def test_endpoint_preflight_still_bounds_token_ids_without_a_client_tokenizer():
    kwargs, prompt_tokens, cap = preflight_endpoint_generation(
        tokenizer=None,
        payloads=[[1] * 100],
        gen_kwargs={"max_gen_toks": 100},
        context_length=200,
    )

    assert prompt_tokens == 100
    assert cap == 36
    assert kwargs == {"max_gen_toks": 36}


def test_endpoint_preflight_preserves_typed_context_overflow():
    with pytest.raises(ContextWindowExceededError) as error:
        preflight_endpoint_generation(
            tokenizer=_Tokenizer(),
            payloads=["token " * 100],
            gen_kwargs={"max_tokens": 32},
            context_length=128,
        )

    assert error.value.context_length == 128
