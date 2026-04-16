# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 1 PrimeIntellectEnv verifier adapter."""

import sys
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset
from marin.rl.environments.inference_ctx import (
    LevanterInferenceContext,
    LevanterInferenceContextConfig,
    VLLMSamplingConfig,
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
)
from marin.rl.environments.inference_ctx.openai_compat import OpenAICompatClient
from marin.rl.environments.inference_ctx.vllm import InferenceMode
from marin.rl.environments.prime_intellect_env import PrimeIntellectEnv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob, Choice, ChoiceLogprobs
from openai.types.completion_usage import CompletionUsage


@dataclass
class DummyInferenceServer:
    """Minimal inference server for Levanter OpenAI client construction."""

    host: str = "localhost"
    port: int = 8000

    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def config(self):
        @dataclass
        class Config:
            model_name: str = "test-model"

        return Config()


class FakeVerifierEnv:
    """Verifier env test double with real Dataset behavior."""

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Dataset | None = None,
        *,
        message_type: str = "chat",
        oai_tools: list[dict[str, object]] | None = None,
        generate_result_factory=None,
    ):
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.message_type = message_type
        self.oai_tools = oai_tools
        self.generate_result_factory = generate_result_factory
        self.generate_calls: list[dict[str, object]] = []

    def get_dataset(self, n: int = -1):
        if n > 0:
            return self.dataset.select(range(min(n, len(self.dataset))))
        return self.dataset

    def get_eval_dataset(self, n: int = -1):
        dataset = self.eval_dataset if self.eval_dataset is not None else self.dataset
        if n > 0:
            return dataset.select(range(min(n, len(dataset))))
        return dataset

    def generate(self, *, inputs, client, model, sampling_args, max_concurrent):
        self.generate_calls.append(
            {
                "inputs": inputs,
                "client": client,
                "model": model,
                "sampling_args": dict(sampling_args),
                "max_concurrent": max_concurrent,
            }
        )
        return self.generate_result_factory(inputs=inputs)


def _prompt_dataset(example_ids: list[str], prefix: str) -> Dataset:
    return Dataset.from_dict(
        {
            "id": example_ids,
            "prompt": [[{"role": "user", "content": f"{prefix} prompt {example_id}"}] for example_id in example_ids],
            "answer": [""] * len(example_ids),
        }
    )


def _chat_completion(tokenizer, response_text: str, prompt_token_ids: list[int]) -> ChatCompletion:
    response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)
    logprobs_content = [
        ChatCompletionTokenLogprob(
            token=tokenizer.convert_ids_to_tokens(token_id),
            logprob=-0.1 * (index + 1),
            bytes=None,
            top_logprobs=[],
        )
        for index, token_id in enumerate(response_token_ids)
    ]

    completion = ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                logprobs=ChoiceLogprobs(content=logprobs_content),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=len(response_token_ids),
            prompt_tokens=len(prompt_token_ids),
            total_tokens=len(prompt_token_ids) + len(response_token_ids),
        ),
    )
    completion.choices[0].prompt_token_ids = prompt_token_ids
    completion.choices[0].response_token_ids = response_token_ids
    return completion


def _single_turn_generate_outputs(inputs, tokenizer, *, metric_scale: float = 1.0):
    counts_by_example_id: dict[str, int] = defaultdict(int)
    prompts = []
    completions = []
    states = []
    rewards = []
    metric_values = []

    for rollout_index, row in enumerate(inputs):
        example_id = str(row["id"])
        generation_index = counts_by_example_id[example_id]
        counts_by_example_id[example_id] += 1

        response_text = f"resp-{example_id}-{generation_index}"
        prompt_token_ids = [100 + rollout_index, 200 + rollout_index]
        response = _chat_completion(tokenizer, response_text, prompt_token_ids)

        prompts.append(row["prompt"])
        completions.append([{"role": "assistant", "content": response_text}])
        states.append({"responses": [response]})
        rewards.append(float(rollout_index + 1))
        metric_values.append(metric_scale * float((rollout_index + 1) * 10))

    return SimpleNamespace(
        prompt=prompts,
        completion=completions,
        state=states,
        reward=rewards,
        metrics={"score": metric_values},
    )


def _install_fake_verifiers(monkeypatch, loader):
    fake_verifiers = ModuleType("verifiers")
    load_calls: list[tuple[str, dict[str, object]]] = []

    def load_environment(env_id: str, **env_args):
        load_calls.append((env_id, dict(env_args)))
        return loader(env_id, env_args)

    fake_verifiers.load_environment = load_environment
    monkeypatch.setitem(sys.modules, "verifiers", fake_verifiers)
    return load_calls


def _levanter_inference_ctx(gpt2_tokenizer):
    ctx = LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=gpt2_tokenizer,
            stop_tokens=None,
            max_tokens=128,
            mesh=None,
            axis_mapping={},
        )
    )
    ctx._inference_server = DummyInferenceServer()
    return ctx


def _vllm_inference_ctx(monkeypatch, gpt2_tokenizer):
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_llm_engine",
        staticmethod(lambda _config: object()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.load_tokenizer",
        lambda _path: gpt2_tokenizer,
    )
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_renderer",
        staticmethod(lambda _model_name, _tokenizer: object()),
    )

    return vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name="test-model",
            canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            sampling_params=VLLMSamplingConfig(),
            mode=InferenceMode.SYNC,
        )
    )


@pytest.fixture(autouse=True)
def clear_prime_intellect_env_caches():
    PrimeIntellectEnv.INSTALLED_ENV_IDS.clear()
    PrimeIntellectEnv.LOADED_ENVIRONMENTS.clear()
    yield
    PrimeIntellectEnv.INSTALLED_ENV_IDS.clear()
    PrimeIntellectEnv.LOADED_ENVIRONMENTS.clear()


@pytest.fixture
def prime_cli(monkeypatch):
    subprocess_run = Mock()
    monkeypatch.setattr("marin.rl.environments.prime_intellect_env.shutil.which", lambda executable: "/usr/bin/prime")
    monkeypatch.setattr("marin.rl.environments.prime_intellect_env.subprocess.run", subprocess_run)
    return subprocess_run


def test_prime_intellect_env_sample_supports_levanter_single_turn_chat(monkeypatch, prime_cli, gpt2_tokenizer):
    train_dataset = _prompt_dataset(["train-0", "train-1"], "train")
    eval_dataset = _prompt_dataset(["eval-0", "eval-1"], "eval")
    verifier_env = FakeVerifierEnv(
        train_dataset,
        eval_dataset,
        generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(
            inputs, gpt2_tokenizer, metric_scale=0.5
        ),
    )
    load_calls = _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(
        env_id="primeintellect/gsm8k",
        env_args={"difficulty": "easy"},
        max_tokens=128,
        max_concurrent=7,
    )
    inference_ctx = _levanter_inference_ctx(gpt2_tokenizer)

    env.prepare()
    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=2,
        temperature=0.7,
        prng_key=None,
        mode="eval",
        top_k=11,
        stop=["<stop>"],
    )

    assert prime_cli.call_count == 1
    assert prime_cli.call_args.args == (["/usr/bin/prime", "env", "install", "primeintellect/gsm8k"],)
    assert prime_cli.call_args.kwargs == {"check": True}
    assert load_calls == [("gsm8k", {"difficulty": "easy"})]
    assert isinstance(verifier_env.generate_calls[0]["client"], AsyncOpenAI)
    assert verifier_env.generate_calls[0]["model"] == "marin-model"
    assert verifier_env.generate_calls[0]["max_concurrent"] == 7
    assert verifier_env.generate_calls[0]["sampling_args"] == {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_k": 11,
        "logprobs": True,
        "stop": ["<stop>"],
    }

    assert [rollout.env_example_id for rollout in rollout_groups[0].rollouts] == [
        "primeintellect/gsm8k:eval-0",
        "primeintellect/gsm8k:eval-0",
    ]
    assert [rollout.env_example_id for rollout in rollout_groups[1].rollouts] == [
        "primeintellect/gsm8k:eval-1",
        "primeintellect/gsm8k:eval-1",
    ]
    assert [gpt2_tokenizer.decode(rollout.response_tokens.tolist()) for rollout in rollout_groups[0].rollouts] == [
        "resp-eval-0-0",
        "resp-eval-0-1",
    ]
    assert [gpt2_tokenizer.decode(rollout.response_tokens.tolist()) for rollout in rollout_groups[1].rollouts] == [
        "resp-eval-1-0",
        "resp-eval-1-1",
    ]
    assert all(
        rollout.env_name == "prime_intellect:primeintellect/gsm8k"
        for group in rollout_groups
        for rollout in group.rollouts
    )
    assert metrics == {
        "primeintellect/gsm8k.score": pytest.approx(12.5),
        "primeintellect/gsm8k.mean_reward": pytest.approx(2.5),
        "primeintellect/gsm8k.total_rollouts": 4.0,
    }


def test_prime_intellect_env_sample_supports_vllm_single_turn_chat(monkeypatch, prime_cli, gpt2_tokenizer):
    train_dataset = _prompt_dataset(["0", "1"], "train")
    verifier_env = FakeVerifierEnv(
        train_dataset,
        generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    load_calls = _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k", max_tokens=64)
    inference_ctx = _vllm_inference_ctx(monkeypatch, gpt2_tokenizer)

    env.prepare()
    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=2,
        temperature=0.2,
        prng_key=None,
        mode="train",
    )

    assert load_calls == [("gsm8k", {})]
    assert isinstance(verifier_env.generate_calls[0]["client"], OpenAICompatClient)
    assert [rollout.prompt_tokens.tolist() for rollout in rollout_groups[0].rollouts] == [[100, 200], [102, 202]]
    assert [rollout.prompt_tokens.tolist() for rollout in rollout_groups[1].rollouts] == [[101, 201], [103, 203]]
    assert [gpt2_tokenizer.decode(rollout.response_tokens.tolist()) for rollout in rollout_groups[0].rollouts] == [
        "resp-0-0",
        "resp-0-1",
    ]
    assert [gpt2_tokenizer.decode(rollout.response_tokens.tolist()) for rollout in rollout_groups[1].rollouts] == [
        "resp-1-0",
        "resp-1-1",
    ]
    assert metrics["primeintellect/gsm8k.score"] == pytest.approx(25.0)
    assert metrics["primeintellect/gsm8k.mean_reward"] == pytest.approx(2.5)
    assert metrics["primeintellect/gsm8k.total_rollouts"] == 4.0


def test_prime_intellect_env_prepare_installs_once_per_env_id(monkeypatch, prime_cli, gpt2_tokenizer):
    load_calls = _install_fake_verifiers(
        monkeypatch,
        lambda _env_id, _env_args: FakeVerifierEnv(
            _prompt_dataset(["0"], "train"),
            generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(inputs, gpt2_tokenizer),
        ),
    )
    env_one = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "easy"})
    env_two = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "hard"})

    env_one.prepare()
    env_two.prepare()

    assert prime_cli.call_count == 1
    assert load_calls == []


def test_prime_intellect_env_load_cache_keys_include_env_args(monkeypatch, prime_cli, gpt2_tokenizer):
    load_calls = []

    def loader(_env_id: str, env_args: dict[str, object]):
        load_calls.append(env_args)
        return FakeVerifierEnv(
            _prompt_dataset([str(env_args["difficulty"])], "train"),
            generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(inputs, gpt2_tokenizer),
        )

    _install_fake_verifiers(monkeypatch, loader)
    inference_ctx = _levanter_inference_ctx(gpt2_tokenizer)

    easy_one = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "easy"})
    easy_two = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "easy"})
    hard = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "hard"})

    for env in (easy_one, easy_two, hard):
        env.prepare()
        env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )

    assert load_calls == [{"difficulty": "easy"}, {"difficulty": "hard"}]


def test_prime_intellect_env_prepare_rejects_non_primeintellect_ids(monkeypatch, prime_cli):
    _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: None)
    env = PrimeIntellectEnv(env_id="someone-else/gsm8k")

    with pytest.raises(ValueError, match="only supports 'primeintellect/\\*' IDs"):
        env.prepare()

    assert prime_cli.call_count == 0


def test_prime_intellect_env_prepare_requires_prime_cli(monkeypatch):
    _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: None)
    monkeypatch.setattr("marin.rl.environments.prime_intellect_env.shutil.which", lambda executable: None)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    with pytest.raises(RuntimeError, match="requires the 'prime' executable"):
        env.prepare()


def test_prime_intellect_env_sample_rejects_invalid_mode(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        _prompt_dataset(["0"], "train"),
        generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    load_calls = _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="Unsupported mode"):
        env.sample(
            inference_ctx=_levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
            mode="debug",
        )

    assert load_calls == []


def test_prime_intellect_env_sample_rejects_system_prompt(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        _prompt_dataset(["0"], "train"),
        generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    load_calls = _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="does not support Marin-level system prompts"):
        env.sample(
            inference_ctx=_levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
            system_prompt="You are helpful.",
        )

    assert load_calls == []


def test_prime_intellect_env_sample_rejects_non_chat_verifier_env(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        _prompt_dataset(["0"], "train"),
        message_type="completion",
        generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="only supports chat-format verifier environments"):
        env.sample(
            inference_ctx=_levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )


def test_prime_intellect_env_sample_rejects_tool_enabled_verifier_env(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        _prompt_dataset(["0"], "train"),
        oai_tools=[{"type": "function"}],
        generate_result_factory=lambda *, inputs: _single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="does not support tool-enabled verifier environments"):
        env.sample(
            inference_ctx=_levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )


def test_prime_intellect_env_sample_rejects_non_assistant_completion_turns(monkeypatch, prime_cli, gpt2_tokenizer):
    dataset = _prompt_dataset(["0"], "train")

    def generate_result_factory(*, inputs):
        output = _single_turn_generate_outputs(inputs, gpt2_tokenizer)
        output.completion[0] = [
            {"role": "assistant", "content": "resp-0-0"},
            {"role": "user", "content": "tool feedback"},
        ]
        return output

    verifier_env = FakeVerifierEnv(dataset, generate_result_factory=generate_result_factory)
    _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="does not support non-assistant turns in completions"):
        env.sample(
            inference_ctx=_levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )


def test_prime_intellect_env_sample_rejects_multiple_assistant_completion_turns(monkeypatch, prime_cli, gpt2_tokenizer):
    dataset = _prompt_dataset(["0"], "train")

    def generate_result_factory(*, inputs):
        output = _single_turn_generate_outputs(inputs, gpt2_tokenizer)
        output.completion[0] = [
            {"role": "assistant", "content": "resp-0-0"},
            {"role": "assistant", "content": "resp-0-1"},
        ]
        return output

    verifier_env = FakeVerifierEnv(dataset, generate_result_factory=generate_result_factory)
    _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="requires exactly one assistant completion turn"):
        env.sample(
            inference_ctx=_levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )


def test_prime_intellect_env_sample_rejects_multiple_response_objects(monkeypatch, prime_cli, gpt2_tokenizer):
    dataset = _prompt_dataset(["0"], "train")

    def generate_result_factory(*, inputs):
        output = _single_turn_generate_outputs(inputs, gpt2_tokenizer)
        response = output.state[0]["responses"][0]
        output.state[0] = {"responses": [response, response]}
        return output

    verifier_env = FakeVerifierEnv(dataset, generate_result_factory=generate_result_factory)
    _install_fake_verifiers(monkeypatch, lambda _env_id, _env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="requires exactly one response object per rollout"):
        env.sample(
            inference_ctx=_levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )
