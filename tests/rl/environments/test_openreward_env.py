# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from marin.rl.environments.openreward_env import OpenRewardEnv
from marin.rl.environments.inference_ctx.render import Qwen3Renderer
from marin.rl.openreward import (
    OpenRewardPromptBlock,
    OpenRewardPromptBlockType,
    OpenRewardTaskManifest,
    OpenRewardTaskManifestEntry,
    OpenRewardToolSpec,
    save_openreward_task_manifest,
)
from marin.rl.types import Rollout, RolloutGroup


class RendererTestTokenizer:
    """Tokenizer stub that preserves text and treats <|im_end|> as one token."""

    _END_MESSAGE_TOKEN = 99999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("<|im_end|>", cursor):
                tokens.append(self._END_MESSAGE_TOKEN)
                cursor += len("<|im_end|>")
                continue
            tokens.append(ord(text[cursor]))
            cursor += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        parts = []
        for token in tokens:
            if token == self._END_MESSAGE_TOKEN:
                parts.append("<|im_end|>")
            else:
                parts.append(chr(token))
        return "".join(parts)

    def get_vocab(self) -> dict[str, int]:
        return {"<|im_end|>": self._END_MESSAGE_TOKEN}


class FakeInferenceContext:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.tokenizer = RendererTestTokenizer()
        self.renderer = Qwen3Renderer(self.tokenizer)
        self.batch_calls: list[dict] = []

    def batch_completions(
        self,
        prompts,
        temperature,
        n,
        max_tokens=None,
        top_k=None,
        stop=None,
        system_prompt=None,
        tools=None,
    ) -> list[ChatCompletion]:
        self.batch_calls.append(
            {
                "prompts": prompts,
                "temperature": temperature,
                "n": n,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "stop": stop,
                "system_prompt": system_prompt,
                "tools": tools,
            }
        )

        choices = []
        for index, response_text in enumerate(self.responses[:n]):
            choice = Choice(
                finish_reason="stop",
                index=index,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                logprobs=None,
            )
            choice.response_token_ids = self.tokenizer.encode(response_text)
            choices.append(choice)

        return [
            ChatCompletion(
                id="chatcmpl-test",
                choices=choices,
                created=1234567890,
                model="test-model",
                object="chat.completion",
                usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
            )
        ]

    def assistant_turn_from_choice(self, choice: Choice):
        return self.renderer.parse_response(choice.response_token_ids)

    def create_rollout_from_choice(
        self,
        prompt,
        choice,
        env_name,
        env_example_id,
        reward,
        temperature,
        top_k=None,
        system_prompt=None,
        correctness_reward=None,
    ) -> Rollout:
        del system_prompt, correctness_reward
        prompt_tokens = np.array(self.tokenizer.encode(prompt), dtype=np.int32)
        response_tokens = np.array(choice.response_token_ids, dtype=np.int32)
        response_logprobs = np.full(len(response_tokens), -0.1, dtype=np.float32)
        token_rewards = jnp.full(len(response_tokens), reward, dtype=jnp.float32)

        return Rollout(
            env_name=env_name,
            env_example_id=env_example_id,
            prompt_tokens=jnp.array(prompt_tokens, dtype=jnp.int32),
            response_tokens=jnp.array(response_tokens, dtype=jnp.int32),
            response_logprobs=jnp.array(response_logprobs, dtype=jnp.float32),
            token_rewards=token_rewards,
            episode_reward=float(reward),
            temperature=temperature,
            top_k=top_k,
            is_truncated=False,
        )


class FakeTask:
    def __init__(self, environment_name: str, task_spec: dict):
        self.environment_name = environment_name
        self.task_spec = task_spec


class FakeToolResult:
    def __init__(self, reward: float, finished: bool):
        self.reward = reward
        self.finished = finished


class FakeSession:
    def __init__(self, calls: list[tuple[str, dict]], tool_results: dict[str, FakeToolResult], secrets):
        self.calls = calls
        self.tool_results = tool_results
        self.secrets = secrets

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def call_tool(self, name: str, arguments: dict) -> FakeToolResult:
        self.calls.append((name, dict(arguments), self.secrets))
        result = self.tool_results[name]
        if isinstance(result, Exception):
            raise result
        return result


class FakeEnvironment:
    def __init__(self, tool_results: dict[str, FakeToolResult]):
        self.tool_results = tool_results
        self.session_calls: list[tuple[str, dict, object]] = []

    def get_task(self, split: str, index: int) -> FakeTask:
        return FakeTask(environment_name="math-agent-v1", task_spec={"split": split, "index": index})

    def session(self, task=None, secrets=None, *, split=None, index=None):
        del split, index
        assert task is not None
        return FakeSession(self.session_calls, self.tool_results, secrets)


class FakeEnvironmentsAPI:
    def __init__(self, environment: FakeEnvironment, calls: dict):
        self.environment = environment
        self.calls = calls

    def get(self, name: str, variant: str | None = None):
        self.calls["deployment_name"] = name
        self.calls["variant"] = variant
        return self.environment


class FakeOpenRewardClient:
    def __init__(self, environment: FakeEnvironment, calls: dict, api_key: str | None, base_url: str | None):
        calls["api_key"] = api_key
        calls["base_url"] = base_url
        self.environments = FakeEnvironmentsAPI(environment, calls)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback


def _manifest(
    tmp_path,
    *,
    split: str = "train",
    prompt_blocks: list[OpenRewardPromptBlock] | None = None,
    tools: list[OpenRewardToolSpec] | None = None,
) -> str:
    manifest = OpenRewardTaskManifest(
        deployment_name="marin/openreward-math-agent",
        environment_name="math-agent-v1",
        split=split,
        tasks=[
            OpenRewardTaskManifestEntry(
                task_index=0,
                task_spec={"problem_id": f"{split}-0"},
                prompt_blocks=prompt_blocks
                or [OpenRewardPromptBlock(type=OpenRewardPromptBlockType.TEXT, text="What is 2+2?")],
                tools=tools
                or [
                    OpenRewardToolSpec(
                        name="submit_answer",
                        description="Submit the final answer.",
                        input_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
                    )
                ],
            )
        ],
    )
    output_path = tmp_path / f"{split}-manifest.json"
    save_openreward_task_manifest(manifest, str(output_path))
    return str(output_path)


def test_openreward_env_sample_executes_single_terminal_tool(monkeypatch, tmp_path):
    calls = {}
    environment = FakeEnvironment(tool_results={"submit_answer": FakeToolResult(reward=1.0, finished=True)})
    inference_ctx = FakeInferenceContext(['<tool_call>{"name":"submit_answer","arguments":{"answer":"4"}}</tool_call>'])

    monkeypatch.setattr(
        "marin.rl.environments.openreward_env.load_openreward_client",
        lambda: lambda api_key=None, base_url=None: FakeOpenRewardClient(environment, calls, api_key, base_url),
    )

    env = OpenRewardEnv(
        train_manifest_path=_manifest(tmp_path),
        base_url="https://openreward.example",
        api_key="api-key",
        variant="math",
        secrets={"OPENAI_API_KEY": "secret"},
    )

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=1,
        temperature=0.7,
        prng_key=jax.random.PRNGKey(0),
        mode="train",
    )

    assert calls["api_key"] == "api-key"
    assert calls["base_url"] == "https://openreward.example"
    assert calls["deployment_name"] == "marin/openreward-math-agent"
    assert calls["variant"] == "math"
    assert inference_ctx.batch_calls[0]["stop"] == ["</tool_call>"]
    assert inference_ctx.batch_calls[0]["prompts"] == ["What is 2+2?"]
    assert [tool.function.name for tool in inference_ctx.batch_calls[0]["tools"]] == ["submit_answer"]
    assert environment.session_calls == [("submit_answer", {"answer": "4"}, {"OPENAI_API_KEY": "secret"})]

    assert len(rollout_groups) == 1
    assert isinstance(rollout_groups[0], RolloutGroup)
    assert rollout_groups[0].rollouts[0].episode_reward == pytest.approx(1.0)
    assert rollout_groups[0].rollouts[0].env_name == "openreward:marin/openreward-math-agent"
    assert metrics["openreward.math-agent-v1.train.mean_reward"] == pytest.approx(1.0)
    assert metrics["openreward.math-agent-v1.train.finished_rate"] == pytest.approx(1.0)


def test_openreward_env_parse_failures_become_zero_reward(monkeypatch, tmp_path):
    environment = FakeEnvironment(tool_results={"submit_answer": FakeToolResult(reward=1.0, finished=True)})
    inference_ctx = FakeInferenceContext(["Final answer: 4"])

    monkeypatch.setattr(
        "marin.rl.environments.openreward_env.load_openreward_client",
        lambda: lambda api_key=None, base_url=None: FakeOpenRewardClient(environment, {}, api_key, base_url),
    )

    env = OpenRewardEnv(train_manifest_path=_manifest(tmp_path))

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=1,
        temperature=0.7,
        prng_key=jax.random.PRNGKey(0),
        mode="train",
    )

    assert len(rollout_groups) == 1
    assert rollout_groups[0].rollouts[0].episode_reward == pytest.approx(0.0)
    assert environment.session_calls == []
    assert metrics["openreward.math-agent-v1.train.parse_failure_rate"] == pytest.approx(1.0)
    assert metrics["openreward.math-agent-v1.train.mean_reward"] == pytest.approx(0.0)


def test_openreward_env_unfinished_tool_results_are_invalid(monkeypatch, tmp_path):
    environment = FakeEnvironment(tool_results={"submit_answer": FakeToolResult(reward=1.0, finished=False)})
    inference_ctx = FakeInferenceContext(['<tool_call>{"name":"submit_answer","arguments":{"answer":"4"}}</tool_call>'])

    monkeypatch.setattr(
        "marin.rl.environments.openreward_env.load_openreward_client",
        lambda: lambda api_key=None, base_url=None: FakeOpenRewardClient(environment, {}, api_key, base_url),
    )

    env = OpenRewardEnv(train_manifest_path=_manifest(tmp_path))

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=1,
        temperature=0.7,
        prng_key=jax.random.PRNGKey(0),
        mode="train",
    )

    assert len(rollout_groups) == 1
    assert rollout_groups[0].rollouts[0].episode_reward == pytest.approx(0.0)
    assert environment.session_calls == [("submit_answer", {"answer": "4"}, None)]
    assert metrics["openreward.math-agent-v1.train.invalid_tool_call_rate"] == pytest.approx(1.0)
    assert metrics["openreward.math-agent-v1.train.finished_rate"] == pytest.approx(0.0)


def test_openreward_env_invalid_tool_argument_errors_are_invalid(monkeypatch, tmp_path):
    environment = FakeEnvironment(tool_results={"submit_answer": ValueError("answer schema mismatch")})
    inference_ctx = FakeInferenceContext(['<tool_call>{"name":"submit_answer","arguments":{"answer":"4"}}</tool_call>'])

    monkeypatch.setattr(
        "marin.rl.environments.openreward_env.load_openreward_client",
        lambda: lambda api_key=None, base_url=None: FakeOpenRewardClient(environment, {}, api_key, base_url),
    )

    env = OpenRewardEnv(train_manifest_path=_manifest(tmp_path))

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=1,
        temperature=0.7,
        prng_key=jax.random.PRNGKey(0),
        mode="train",
    )

    assert len(rollout_groups) == 1
    assert rollout_groups[0].rollouts[0].episode_reward == pytest.approx(0.0)
    assert environment.session_calls == [("submit_answer", {"answer": "4"}, None)]
    assert metrics["openreward.math-agent-v1.train.invalid_tool_call_rate"] == pytest.approx(1.0)
    assert metrics["openreward.math-agent-v1.train.tool_execution_failure_rate"] == pytest.approx(0.0)


def test_openreward_env_raises_runtime_failures(monkeypatch, tmp_path):
    environment = FakeEnvironment(tool_results={"submit_answer": RuntimeError("tool backend unavailable")})
    inference_ctx = FakeInferenceContext(['<tool_call>{"name":"submit_answer","arguments":{"answer":"4"}}</tool_call>'])

    monkeypatch.setattr(
        "marin.rl.environments.openreward_env.load_openreward_client",
        lambda: lambda api_key=None, base_url=None: FakeOpenRewardClient(environment, {}, api_key, base_url),
    )

    env = OpenRewardEnv(train_manifest_path=_manifest(tmp_path))

    with pytest.raises(RuntimeError, match="OpenReward tool execution failed"):
        env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=0.7,
            prng_key=jax.random.PRNGKey(0),
            mode="train",
        )


def test_openreward_env_rejects_image_prompts(monkeypatch, tmp_path):
    environment = FakeEnvironment(tool_results={"submit_answer": FakeToolResult(reward=1.0, finished=True)})
    inference_ctx = FakeInferenceContext(['<tool_call>{"name":"submit_answer","arguments":{"answer":"4"}}</tool_call>'])

    monkeypatch.setattr(
        "marin.rl.environments.openreward_env.load_openreward_client",
        lambda: lambda api_key=None, base_url=None: FakeOpenRewardClient(environment, {}, api_key, base_url),
    )

    manifest_path = _manifest(
        tmp_path,
        prompt_blocks=[
            OpenRewardPromptBlock(type=OpenRewardPromptBlockType.TEXT, text="Look at the image."),
            OpenRewardPromptBlock(
                type=OpenRewardPromptBlockType.IMAGE,
                data="abc",
                mime_type="image/png",
            ),
        ],
    )
    env = OpenRewardEnv(train_manifest_path=manifest_path)

    with pytest.raises(ValueError, match="supports only text prompt blocks"):
        env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=0.7,
            prng_key=jax.random.PRNGKey(0),
            mode="train",
        )
