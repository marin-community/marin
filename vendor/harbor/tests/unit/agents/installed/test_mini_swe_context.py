import importlib
import json
import shlex
from copy import deepcopy
from pathlib import Path

import httpx
import litellm
import pytest
from minisweagent.run.mini import app
from typer.testing import CliRunner

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import ExecResult
from harbor.literal.proxy import LITERAL_LOG_FILENAME
from harbor.models.agent.context import AgentContext


def _compacting_model_class(monkeypatch):
    root = Path(__file__).parents[4] / "src" / "harbor"
    monkeypatch.syspath_prepend(str(root / "agents" / "installed"))
    monkeypatch.syspath_prepend(str(root / "llms"))
    module = importlib.import_module("mini_swe_model")
    return module.ContextCompactingModel


def _install_tokenizer(monkeypatch):
    def tokenizer(request):
        assert str(request.url) == "http://model/tokenize"
        return httpx.Response(200, json={"count": 128, "max_model_len": 32768})

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(tokenizer), **kwargs
        ),
    )


def _model_response(content: str) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        choices=[
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 128, "completion_tokens": 16, "total_tokens": 144},
    )


def _tool_response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-next",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"echo next"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        usage={"prompt_tokens": 128, "completion_tokens": 16, "total_tokens": 144},
    )


def _provider_characters(messages):
    return sum(
        len(
            json.dumps(
                {key: value for key, value in message.items() if key != "extra"},
                sort_keys=True,
            )
        )
        for message in messages
    )


def _model(compacting_model_class, threshold):
    return compacting_model_class(
        model_name="hosted_vllm/served-model",
        cost_tracking="ignore_errors",
        api_base="http://model/v1",
        context_window=32768,
        max_output_tokens=8192,
        summarization_threshold_chars=threshold,
    )


def _turn(label: str, output_characters: int) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"inspect {label}",
            "tool_calls": [
                {
                    "id": f"call-{label}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": f"cat {label}"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call-{label}",
            "content": label * output_characters,
        },
    ]


def test_compaction_activates_only_above_character_threshold(monkeypatch):
    compacting_model_class = _compacting_model_class(monkeypatch)
    _install_tokenizer(monkeypatch)
    history = _turn("a", 20)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *history,
    ]
    threshold = _provider_characters(history)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["messages"][0]["content"].startswith("You compact the work history"):
            return _model_response("summary")
        return _tool_response()

    monkeypatch.setattr(litellm, "completion", completion)
    _model(compacting_model_class, threshold).query(messages)
    assert len(calls) == 1

    calls.clear()
    _model(compacting_model_class, threshold - 1).query(messages)
    assert len(calls) == 2
    assert calls[0]["messages"][0]["content"].startswith("You compact the work history")


def test_compaction_preserves_full_transcript_and_complete_recent_turn(monkeypatch):
    compacting_model_class = _compacting_model_class(monkeypatch)
    _install_tokenizer(monkeypatch)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *_turn("old-a", 80),
        *_turn("old-b", 80),
        *_turn("recent", 20),
    ]
    original = deepcopy(messages)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) % 2 == 1:
            return _model_response("Files inspected: old-a and old-b.")
        return _tool_response()

    monkeypatch.setattr(litellm, "completion", completion)
    model = _model(compacting_model_class, 800)
    model.query(messages)

    assert messages == original
    request_messages = calls[1]["messages"]
    assert [message["role"] for message in request_messages[-2:]] == [
        "assistant",
        "tool",
    ]
    assert request_messages[-2]["tool_calls"][0]["id"] == "call-recent"
    assert request_messages[-1]["tool_call_id"] == "call-recent"
    assert _provider_characters(request_messages[2:]) <= 800
    event = model.serialize()["info"]["context_compaction"]["events"][0]
    assert event["history_chars_before"] > 800
    assert event["history_chars_after"] <= 800
    assert event["fallback_reason"] is None


def test_compaction_reuses_prior_summary_without_resummarizing_raw_history(
    monkeypatch,
):
    compacting_model_class = _compacting_model_class(monkeypatch)
    _install_tokenizer(monkeypatch)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *_turn("first", 300),
    ]
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["messages"][0]["content"].startswith("You compact the work history"):
            return _model_response(f"summary-{len(calls)}")
        return _tool_response()

    monkeypatch.setattr(litellm, "completion", completion)
    model = _model(compacting_model_class, 300)
    model.query(messages)
    messages.extend(_turn("second", 300))
    model.query(messages)

    summary_calls = [
        call
        for call in calls
        if call["messages"][0]["content"].startswith("You compact the work history")
    ]
    assert len(summary_calls) == 2
    assert "summary-1" in summary_calls[1]["messages"][1]["content"]
    assert "first" not in summary_calls[1]["messages"][1]["content"]
    state = model.serialize()["info"]["context_compaction"]
    assert len(state["events"]) == 2
    assert state["summarized_history_messages"] == 4


def test_compaction_falls_back_to_bounded_excerpt_when_summary_fails(monkeypatch):
    compacting_model_class = _compacting_model_class(monkeypatch)
    _install_tokenizer(monkeypatch)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *_turn("large", 2000),
    ]
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("summary unavailable")
        return _tool_response()

    monkeypatch.setattr(litellm, "completion", completion)
    model = _model(compacting_model_class, 300)
    model.query(messages)

    request_messages = calls[1]["messages"]
    assert _provider_characters(request_messages[2:]) <= 300
    event = model.serialize()["info"]["context_compaction"]["events"][0]
    assert event["fallback_reason"] == "RuntimeError"
    assert event["source_characters_omitted"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "prompt_tokens",
        "server_context",
        "requested",
        "expected",
        "custom_config",
        "model_format",
        "max_turns",
        "summarization_threshold_chars",
    ),
    [
        (32000, 65536, 16384, 768, True, "openai", None, 256),
        (16000, 16384, 16384, 384, True, "openai", None, None),
        (32000, 65536, 128, 128, True, "openai", None, None),
        (32000, 65536, 16384, 768, False, "openai", None, None),
        (32000, 65536, 16384, 768, True, "k2_horizon", None, None),
        (32000, 65536, 16384, 768, True, "openai", 1, None),
    ],
)
async def test_native_agent_enforces_total_context_and_keeps_tool_actions(
    tmp_path,
    monkeypatch,
    prompt_tokens,
    server_context,
    requested,
    expected,
    custom_config,
    model_format,
    max_turns,
    summarization_threshold_chars,
):
    root = Path(__file__).parents[4] / "src" / "harbor"
    monkeypatch.syspath_prepend(str(root / "agents" / "installed"))
    monkeypatch.syspath_prepend(str(root / "llms"))
    counts_sequence = [prompt_tokens]
    if summarization_threshold_chars is not None:
        counts_sequence.append(128)
    if max_turns is None:
        counts_sequence.append(min(32768, server_context))
    counts = iter(counts_sequence)
    tokenized = []
    tokenization_trial_ids = []
    completions = []

    def tokenizer(request):
        assert str(request.url) == "http://model/tokenize"
        tokenized.append(json.loads(request.content))
        tokenization_trial_ids.append(request.headers.get("x-ot-trial-id"))
        return httpx.Response(
            200, json={"count": next(counts), "max_model_len": server_context}
        )

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(tokenizer), **kwargs
        ),
    )

    def completion(**kwargs):
        completions.append(kwargs)
        (tmp_path / LITERAL_LOG_FILENAME).write_text(
            json.dumps(
                {
                    "literal": {
                        "prompt_token_ids": [101, 102],
                        "completion_token_ids": [301, 302],
                        "logprobs": [-0.2, -0.3],
                    }
                }
            )
            + "\n"
        )
        if kwargs["messages"][0]["content"].startswith("You compact the work history"):
            return _model_response("Inspected the repository and ran echo ready.")
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command": "echo ready"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": expected,
                "total_tokens": prompt_tokens + expected,
            },
        )

    monkeypatch.setattr(litellm, "completion", completion)
    config = tmp_path / "custom.yaml"
    config.write_text(
        f"model:\n  model_kwargs:\n    max_tokens: {requested}\n    temperature: 0.25\n    extra_body:\n      chat_template_kwargs:\n        enable_thinking: false\n"
    )
    trajectory = tmp_path / "mini-swe-agent.trajectory.json"

    class HarnessEnvironment:
        async def exec(self, *, command, env, **kwargs):
            words = shlex.split(command)
            if "mini-swe-agent" not in words:
                return ExecResult(return_code=0)
            start = words.index("mini-swe-agent") + 1
            argv = words[start : words.index("2>&1")]
            argv = [
                str(config)
                if word == "/tmp/mswea-config/custom.yaml"
                else f"--output={trajectory}"
                if word.startswith("--output=")
                else word
                for word in argv
            ]
            result = CliRunner().invoke(app, argv, env=env)
            assert result.exit_code == 0, result.output[-3000:]
            return ExecResult(return_code=result.exit_code, stdout=result.output)

    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="hosted_vllm/served-model",
        version="2.1.0",
        api_base="http://model/v1",
        config_file=str(config) if custom_config else None,
        model_info={"max_input_tokens": 32768, "max_output_tokens": 16384},
        collect_rollout_details=True,
        model_format=model_format,
        max_turns=max_turns,
        summarization_threshold_chars=summarization_threshold_chars,
    )
    context = AgentContext()
    await agent.run("Fix the issue", HarnessEnvironment(), context)
    agent.populate_context_post_run(context)

    saved = json.loads(trajectory.read_text())
    expected_status = "LimitsExceeded" if max_turns is not None else "ContextLimit"
    assert saved["info"]["exit_status"] == expected_status
    assert saved["messages"][-1]["extra"]["exit_status"] == expected_status
    assistant = next(
        message for message in saved["messages"] if message["role"] == "assistant"
    )
    assert assistant["extra"]["actions"][0]["command"] == "echo ready"
    agent_completions = [
        completion for completion in completions if completion.get("tools")
    ]
    assert len(agent_completions) == 1
    assert agent_completions[0]["max_tokens"] == expected
    assert (
        agent_completions[0]["extra_headers"]["x-ot-trial-id"]
        == context.metadata["rollout_correlation_id"]
    )
    assert agent_completions[0]["extra_headers"]["x-ot-task-agent"] == "mini-swe-agent"
    expected_requests = 1 if max_turns is not None else 2
    if summarization_threshold_chars is not None:
        expected_requests += 1
    assert (
        tokenization_trial_ids
        == [context.metadata["rollout_correlation_id"]] * expected_requests
    )
    assert context.rollout_details == [
        {
            "prompt_token_ids": [[101, 102]],
            "completion_token_ids": [[301, 302]],
            "logprobs": [[-0.2, -0.3]],
        }
    ]
    if custom_config:
        assert agent_completions[0]["temperature"] == 0.25
    assert tokenized[0]["messages"] == agent_completions[0]["messages"]
    assert tokenized[0]["tools"] == agent_completions[0]["tools"]
    assert tokenized[0]["chat_template_kwargs"] == (
        {"enable_thinking": False} if custom_config else {}
    )
    assert tokenized[0]["add_generation_prompt"] is True
    if max_turns is None and summarization_threshold_chars is None:
        previous_assistant = next(
            message
            for message in tokenized[1]["messages"]
            if message["role"] == "assistant"
        )
        if model_format == "k2_horizon":
            assert previous_assistant["reasoning"] == ""
        else:
            assert "reasoning" not in previous_assistant
    assert saved["info"]["config"]["model"]["context_window"] == 32768
    if summarization_threshold_chars is not None:
        assert (
            saved["info"]["config"]["model"]["summarization_threshold_chars"]
            == summarization_threshold_chars
        )
        compaction = saved["info"]["context_compaction"]
        assert compaction["events"][0]["history_chars_before"] > 256
        assert compaction["events"][0]["history_chars_after"] <= 256
        assert compaction["events"][0]["fallback_reason"] is None
        assert context.n_input_tokens == prompt_tokens + 128
        assert context.n_output_tokens == expected + 16
