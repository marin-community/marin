import base64
import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from jinja2 import TemplateError
from omegaconf import OmegaConf
from rolloutengine.models import ModelTurn
from skyrl_gym.source_task import source_task
from skyrl_gym.task_sessions import SearchTaskSession
from taskcompendium.chat import chat_conversation
from taskcompendium.convert.answers import answer_task
from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, FinalAction, FunctionDefinition, GradingAttempt, JsonValueAnswer, Source
from taskcompendium.pipeline.models import RawRow
from verifyit.spec import FunctionCall, PredictedActionSpec, StructuredExactSpec

from skyrl_train.inference_engines.chat_template import SINGLE_TOOL_CALL_TEMPLATE_ERROR
from skyrl_train.inference_engines.chat_continuation import render_exact_chat_continuation
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.trajectory_runners.model_clients import ContextLengthExceededError, DirectModelClient, ModelServerError


def _encoded_routes(rows):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(rows, dtype=np.uint16), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_direct_model_client_preserves_engine_tokens():
    engine_output = {
        "responses": ["answer"],
        "response_ids": [[3, 4]],
        "stop_reasons": ["stop"],
        "response_logprobs": [[-0.1, -0.2]],
        "prompt_logprobs": None,
    }
    engine = AsyncMock()
    engine.generate.return_value = engine_output

    output = await DirectModelClient(engine).generate({"prompt_token_ids": [[1, 2]]})

    assert output == {**engine_output, "token_provenance": "engine"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", ["", "<think>\nCalculate the answer.\n</think>\n\n"])
async def test_chat_answer_omits_stop_text_but_keeps_stop_token_evidence(load_tokenizer, reasoning):
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B", revision="c1899de")
    tokens = tokenizer.encode(reasoning + "20<|im_end|>", add_special_tokens=False)
    engine = AsyncMock()
    engine.model_name = "qwen"
    engine.tokenizer = tokenizer
    engine.tokenize.return_value = {"tokens": [1, 2]}

    async def respond(payload):
        content = "20<|im_end|>" if payload["json"].get("include_stop_str_in_output") else "20"
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "token_ids": tokens,
                    "logprobs": {"content": [{"logprob": -0.5} for _ in tokens]},
                }
            ]
        }

    engine.chat_completion.side_effect = respond
    result = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "What is ten plus ten?"}]],
            "chat_completion_params": [{}],
            "sampling_params": {"include_stop_str_in_output": True, "logprobs": 0},
        }
    )
    assert result["responses"] == ["20"]
    assert result["assistant_messages"] == [{"role": "assistant", "content": "20"}]
    assert result["response_ids"] == [tokens]
    assert result["response_logprobs"] == [[-0.5] * len(tokens)]


@pytest.mark.asyncio
async def test_direct_chat_client_preserves_server_error_identity_without_message():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenize.return_value = {"tokens": [1, 2]}

    async def server_error(payload):
        return {
            "error": {"message": "private prompt contents", "code": 500},
            "error_category": "constrained_decoding",
            "request_id": payload["headers"]["x-request-id"],
        }

    engine.chat_completion.side_effect = server_error
    with pytest.raises(ModelServerError) as raised:
        await DirectModelClient(engine).generate(
            {
                "prompts": [[{"role": "user", "content": "secret"}]],
                "chat_completion_params": [{}],
            }
        )

    error = raised.value
    assert error.request_id == engine.chat_completion.await_args.args[0]["headers"]["x-request-id"]
    assert error.category == "constrained_decoding"
    assert error.status_code == 500
    assert "private prompt contents" not in str(error)


@pytest.mark.asyncio
async def test_direct_chat_client_types_context_overflow_without_leaking_prompt():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenize.return_value = {"tokens": [1, 2]}
    engine.chat_completion.return_value = {
        "error": {"message": "private prompt contents", "code": 400},
        "error_category": "context_overflow",
        "request_id": "request-123",
    }

    with pytest.raises(ContextLengthExceededError) as raised:
        await DirectModelClient(engine).generate(
            {"prompts": [[{"role": "user", "content": "secret"}]], "chat_completion_params": [{}]}
        )

    assert raised.value.status_code == 400
    assert raised.value.request_id == "request-123"
    assert "private prompt contents" not in str(raised.value)


@pytest.mark.asyncio
async def test_direct_model_client_uses_vllm_chat_rendering_for_row_request_options():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "<tool-call tokens>"
    engine.tokenize.return_value = {"tokens": [11, 12, 13]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"x"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "token_ids": [21, 22],
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
            }
        ]
    }
    client = DirectModelClient(engine)

    output = await client.generate(
        {
            "prompts": [[{"role": "user", "content": "look it up"}]],
            "session_ids": ["trajectory-2"],
            "sampling_params": {"temperature": 0.7, "logprobs": 0},
            "chat_completion_params": [
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "Search",
                                "parameters": {"type": "object"},
                                "strict": True,
                            },
                        }
                    ],
                    "parallel_tool_calls": False,
                    "max_output_tokens": 128,
                }
            ],
        }
    )

    expected_tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
    ]
    tokenize_body = engine.tokenize.await_args.args[0]["json"]
    assert tokenize_body == {
        "model": "snowball",
        "messages": [{"role": "user", "content": "look it up"}],
        "tools": expected_tools,
        "add_generation_prompt": True,
    }
    chat_body = engine.chat_completion.await_args.args[0]["json"]
    assert chat_body == {
        "model": "snowball",
        "messages": [{"role": "user", "content": "look it up"}],
        "session_id": "trajectory-2",
        "_skyrl_exact_prompt_token_ids": [11, 12, 13],
        "temperature": 0.7,
        "tools": expected_tools,
        "parallel_tool_calls": False,
        "max_completion_tokens": 128,
        "include_stop_str_in_output": True,
        "logprobs": True,
    }
    assert output["prompt_ids"] == [[11, 12, 13]]
    assert output["response_ids"] == [[21, 22]]
    assert output["response_logprobs"] == [[-0.1, -0.2]]
    assert output["assistant_messages"] == [engine.chat_completion.return_value["choices"][0]["message"]]
    assert output["token_provenance"] == "engine"


@pytest.mark.asyncio
async def test_plain_chat_stop_marker_reaches_terminal_search_grading(task_lowering):
    engine = AsyncMock()
    engine.model_name = "plain-chat"
    engine.tokenizer = MagicMock()
    engine.tokenizer.eos_token_id = 99
    engine.tokenizer.decode.return_value = "<answer>Paris</answer>"
    engine.tokenize.return_value = {"tokens": [11, 12]}

    async def respond(payload):
        content = "<answer>Paris" + ("</answer>" if payload["json"]["include_stop_str_in_output"] else "")
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "token_ids": [21, 22],
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                }
            ]
        }

    engine.chat_completion.side_effect = respond
    prompt = [{"role": "user", "content": "What is the capital of France?"}]
    task = source_task(
        prompt,
        {"reward_spec": {"ground_truth": {"target": "Paris"}}},
        {"search_url": "http://127.0.0.1:1", "topk": 3, "timeout": 1, "log_requests": False},
        Source(dataset="search", revision="1", row="0", importer_revision="1"),
    )
    session = SearchTaskSession(task_lowering(task, "search"), None)
    try:
        start = await session.prepare()
        output = await DirectModelClient(engine).generate(
            {
                "prompts": [list(start.messages)],
                "chat_completion_params": [start.options],
                "sampling_params": {"stop": ["</answer>"], "logprobs": 0},
            }
        )
        transition = await session.advance(
            ModelTurn(
                output["assistant_messages"][0],
                tuple(output["prompt_ids"][0]),
                tuple(output["response_ids"][0]),
                tuple(output["response_logprobs"][0]),
                output["stop_reasons"][0],
                text=output["responses"][0],
            )
        )
        assert transition.done and transition.reward == 1.0
        assert transition.grade.passed is True
        assert output["responses"] == ["<answer>Paris</answer>"]
        assert output["response_ids"] == [[21, 22]]
        assert output["response_logprobs"] == [[-0.1, -0.2]]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_direct_model_client_recovers_strict_template_by_sequentializing_tool_calls():
    class WrappedValidationError(RuntimeError):
        def as_instanceof_cause(self):
            return TemplateError(SINGLE_TOOL_CALL_TEMPLATE_ERROR)

    engine = AsyncMock()
    engine.model_name = "strict-tool-model"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.side_effect = [WrappedValidationError(), {"tokens": [11, 12, 13]}]
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [21],
            }
        ]
    }
    first_call = {"id": "call-1", "type": "function", "function": {"name": "python", "arguments": "one"}}
    second_call = {"id": "call-2", "type": "function", "function": {"name": "python", "arguments": "two"}}
    messages = [
        {"role": "user", "content": "calculate"},
        {"role": "assistant", "content": None, "tool_calls": [first_call, second_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        {"role": "tool", "tool_call_id": "call-2", "content": "2"},
    ]

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [messages],
            "chat_completion_params": [{}],
        }
    )

    served_messages = engine.chat_completion.await_args.args[0]["json"]["messages"]
    assert served_messages == [
        {"role": "user", "content": "calculate"},
        {"role": "assistant", "content": None, "tool_calls": [first_call]},
        {"role": "assistant", "content": None, "tool_calls": [second_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        {"role": "tool", "tool_call_id": "call-2", "content": "2"},
    ]
    assert output["prompt_ids"] == [[11, 12, 13]]


@pytest.mark.asyncio
async def test_strict_template_recovery_preserves_exact_prefix_and_tool_results():
    class WrappedValidationError(RuntimeError):
        def as_instanceof_cause(self):
            return TemplateError(SINGLE_TOOL_CALL_TEMPLATE_ERROR)

    async def tokenize(request):
        messages = request["json"]["messages"]
        if any(len(message.get("tool_calls") or []) > 1 for message in messages):
            raise WrappedValidationError()
        if request["json"].get("continue_final_message"):
            return {"tokens": [10, 20, 77]}
        if len(messages) == 5:
            return {"tokens": [10, 20, 30, 99, 40, 50]}
        if len(messages) == 3 and messages[-1].get("tool_calls"):
            return {"tokens": [10, 20, 30, 99]}
        if len(messages) == 2:
            return {"tokens": [10, 20]}
        return {"tokens": [10, 20, 77, 99]}

    engine = AsyncMock()
    engine.model_name = "strict-tool-model"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.side_effect = tokenize
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [21],
            }
        ]
    }
    calls = [
        {"id": "call-1", "type": "function", "function": {"name": "python", "arguments": "one"}},
        {"id": "call-2", "type": "function", "function": {"name": "python", "arguments": "two"}},
    ]
    messages = [
        {"role": "user", "content": "calculate"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        {"role": "tool", "tool_call_id": "call-2", "content": "2"},
    ]

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [messages],
            "chat_completion_params": [{}],
            "chat_continuations": [{"served_prefix_token_ids": [7, 8], "assistant_message_index": 1}],
        }
    )

    served = engine.chat_completion.await_args.args[0]["json"]
    assert served["_skyrl_exact_prompt_token_ids"] == [7, 8, 99, 40, 50]
    assert [message.get("tool_call_id") for message in served["messages"] if message["role"] == "tool"] == [
        "call-1",
        "call-2",
    ]
    assert output["prompt_ids"] == [[7, 8, 99, 40, 50]]


@pytest.mark.asyncio
async def test_direct_chat_continuation_preserves_sampled_tool_call_tokens():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"

    async def tokenize(request):
        messages = request["json"]["messages"]
        if request["json"].get("continue_final_message"):
            return {"tokens": [11, 12, 90]}
        if len(messages) == 1:
            return {"tokens": [11, 12]}
        if len(messages) == 2 and not messages[1].get("tool_calls"):
            return {"tokens": [11, 12, 90, 30]}
        if len(messages) == 2:
            return {"tokens": [11, 12, 99, 22, 30]}
        return {"tokens": [11, 12, 99, 22, 30, 40, 41]}

    engine.tokenize.side_effect = tokenize
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [50],
            }
        ]
    }
    messages = [
        {"role": "user", "content": "run this"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "1"},
    ]

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [messages],
            "session_ids": ["trajectory-1"],
            "chat_completion_params": [{"tools": []}],
            "chat_continuations": [{"served_prefix_token_ids": [11, 12, 21, 22], "assistant_message_index": 1}],
        }
    )

    assert output["prompt_ids"] == [[11, 12, 21, 22, 30, 40, 41]]
    chat_body = engine.chat_completion.await_args.args[0]["json"]
    assert chat_body["_skyrl_exact_prompt_token_ids"] == [11, 12, 21, 22, 30, 40, 41]
    # A row with `tools: []` is served as a tool-free request.
    assert "tools" not in chat_body
    assert all("tools" not in call.args[0]["json"] for call in engine.tokenize.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_string", [False, True])
@pytest.mark.parametrize("enable_thinking", [False, True])
@pytest.mark.parametrize("observation_role", ["tool", "user"])
async def test_qwen_continuation_preserves_served_tokens_after_template_rewrites(
    load_tokenizer, enable_thinking, observation_role, stop_string
):
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B", revision="c1899de")
    history = [{"role": "user", "content": "Write 20 to a file."}]
    call = {"type": "function", "function": {"name": "shell", "arguments": '{"command":"echo 20"}'}}
    messages = [
        *history,
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": observation_role, "content": "Read the file."},
    ]
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    # The sampled JSON has different whitespace from the template's tool-call rendering.
    sampled = '<tool_call>\n{"name":"shell","arguments":{"command":"echo 20"}}\n</tool_call>'
    if not stop_string:
        sampled += "<|im_end|>"
    if enable_thinking:
        messages[1]["content"] = "<think>\nWrite the value.\n</think>\n\n"
        sampled = messages[1]["content"] + sampled
    served = tokenizer.encode(prompt + sampled, add_special_tokens=False)

    async def tokenize(payload):
        body = dict(payload["json"])
        chat_kwargs = body.pop("chat_template_kwargs", {})
        text = tokenizer.apply_chat_template(body.pop("messages"), tokenize=False, **body, **chat_kwargs)
        return {"tokens": tokenizer.encode(text, add_special_tokens=False)}

    actual = await render_exact_chat_continuation(
        tokenize,
        {
            "json": {
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
        },
        assistant_message_index=1,
        served_prefix_token_ids=served,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
    )
    observation = "Read the file."
    if observation_role == "tool":
        observation = f"<tool_response>\n{observation}\n</tool_response>"
    suffix = f"\n<|im_start|>user\n{observation}<|im_end|>\n<|im_start|>assistant\n"
    if not enable_thinking:
        suffix += "<think>\n\n</think>\n\n"
    if stop_string:
        suffix = "<|im_end|>" + suffix
    assert actual == [*served, *tokenizer.encode(suffix, add_special_tokens=False)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,revision",
    [
        pytest.param(
            "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.11",
            "50a201b16de975505e981a9e186b1f8aa5f53c91",
            id="snowball",
        ),
        pytest.param(
            "marin-community/marin-tokenizer", "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2", id="marin-tokenizer"
        ),
    ],
)
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("stop_string", [False, True])
async def test_grug_chat_continuation_keeps_served_tokens(load_tokenizer, model, revision, tool_call, stop_string):
    tokenizer = load_tokenizer(model, revision=revision)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    history = [{"role": "user", "content": "List files."}]
    assistant = {"role": "assistant", "content": "Paris"}
    observation = {"role": "user", "content": "Why?"}
    sampled = "Paris"
    if tool_call:
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"ls"}'}}
            ],
        }
        observation = {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}
        sampled = '{"name":"shell","arguments":{"command":"ls"}}'
        if model.startswith("open-athena/"):
            sampled = "<tool_call>\n" + sampled + "\n</tool_call>"
    if not stop_string:
        sampled += "<|eot_id|>"
    prompt = tokenizer.apply_chat_template(history, tools=tools, tokenize=False, add_generation_prompt=True)
    served = tokenizer.encode(prompt + sampled, add_special_tokens=False)

    async def tokenize(payload):
        body = dict(payload["json"])
        text = tokenizer.apply_chat_template(body.pop("messages"), tokenize=False, **body)
        return {"tokens": tokenizer.encode(text, add_special_tokens=False)}

    response_ids = tokenizer.encode("Done.<|eot_id|>", add_special_tokens=False)
    engine = AsyncMock()
    engine.model_name = model
    engine.tokenizer = tokenizer
    engine.tokenize.side_effect = tokenize
    engine.chat_completion.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop", "token_ids": response_ids}
        ]
    }
    result = await DirectModelClient(engine).generate(
        {
            "prompts": [[*history, assistant, observation]],
            "chat_completion_params": [{"tools": tools}],
            "chat_continuations": [{"assistant_message_index": 1, "served_prefix_token_ids": served}],
        }
    )

    suffix = "<|eot_id|>" if stop_string else ""
    if tool_call:
        suffix += '\n<|start_header_id|>tool<|end_header_id|>\n<tool_response id="c1">a.txt</tool_response><|eot_id|>\n'
    else:
        suffix += "<|start_header_id|>user<|end_header_id|>\nWhy?<|eot_id|>\n"
    suffix += "<|start_header_id|>assistant<|end_header_id|>\n"
    assert result["prompt_ids"] == [[*served, *tokenizer.encode(suffix, add_special_tokens=False)]]
    assert result["response_ids"] == [response_ids]


@pytest.mark.asyncio
@pytest.mark.parametrize("template_case", ["eom", "eom-with-literal", "omitted-tool-turn"])
@pytest.mark.parametrize("stop_string", [False, True])
async def test_chat_continuation_rejects_closers_outside_the_assistant_boundary(
    load_tokenizer, stop_string, template_case
):
    tokenizer = load_tokenizer(
        "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.11", revision="50a201b16de975505e981a9e186b1f8aa5f53c91"
    )
    template = tokenizer.chat_template.replace(
        "{%- endfor %}<|eot_id|>\n{% endgeneration %}", "{%- endfor %}<|eom_id|>{% endgeneration %}"
    )
    command = "ls"
    if template_case == "eom-with-literal":
        command = "echo <|eot_id|>"
    elif template_case == "omitted-tool-turn":
        template = tokenizer.chat_template.replace("{% elif has_tool_calls -%}", "{% elif has_tool_calls and false -%}")
    history = [{"role": "user", "content": "List files."}]
    messages = [
        *history,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "shell", "arguments": '{"command":"' + command + '"}'}}
            ],
        },
        {"role": "tool", "content": "a.txt"},
    ]
    prompt = tokenizer.apply_chat_template(history, tokenize=False, chat_template=template, add_generation_prompt=True)
    sampled = '<tool_call>\n{"name":"shell","arguments":{"command":"' + command + '"} }\n</tool_call>'
    if not stop_string:
        sampled += "<|eot_id|>" if template_case == "omitted-tool-turn" else "<|eom_id|>"
    served = tokenizer.encode(prompt + sampled, add_special_tokens=False)

    async def tokenize(payload):
        body = dict(payload["json"])
        text = tokenizer.apply_chat_template(body.pop("messages"), tokenize=False, **body)
        return {"tokens": tokenizer.encode(text, add_special_tokens=False)}

    result = await render_exact_chat_continuation(
        tokenize,
        {"json": {"messages": messages, "chat_template": template, "add_generation_prompt": True}},
        assistant_message_index=1,
        served_prefix_token_ids=served,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
    )

    assert result is None


@pytest.mark.asyncio
async def test_direct_chat_client_captures_exact_student_topk_ids():
    engine = AsyncMock()
    engine.model_name = "teacher"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.return_value = {"tokens": [1, 2]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [9, 10],
                "provider_specific_fields": {"routed_experts": _encoded_routes([[[1, 2]], [[3, 4]], [[4, 7]]])},
                "logprobs": {
                    "content": [
                        {
                            "logprob": -7.0,
                            "top_logprobs": [
                                {"token": "token_id:9", "logprob": -7.0},
                                {"token": "token_id:3", "logprob": -0.2},
                                {"token": "token_id:2", "logprob": -0.1},
                            ],
                        },
                        {
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "token_id:10", "logprob": -0.1},
                                {"token": "token_id:11", "logprob": -0.2},
                            ],
                        },
                    ]
                },
            }
        ]
    }

    output = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "session_ids": ["test"],
            "sampling_params": {"logprobs": 2},
            "chat_completion_params": [{}],
        }
    )

    body = engine.chat_completion.await_args.args[0]["json"]
    assert body["top_logprobs"] == 3
    assert body["return_tokens_as_token_ids"] is True
    assert output["student_topk_indices"] == [[[2, 3], [10, 11]]]
    assert output["behavior_topk_logprobs"] == [[[-0.1, -0.2], [-0.1, -0.2]]]
    np.testing.assert_array_equal(output["routed_experts"][0], [[[3, 4]], [[4, 7]]])
    assert output["routed_experts"][0].dtype == np.uint8


@pytest.mark.asyncio
@pytest.mark.parametrize("content,expected", [("final answer", "final answer"), (None, "")])
async def test_chat_grading_text_uses_parsed_final_content_and_preserves_raw_tokens(content, expected):
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "reasoning words and final answer"
    engine.tokenize.return_value = {"tokens": [1, 2]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "reasoning_content": "reasoning words"},
                "finish_reason": "stop",
                "token_ids": [3, 4, 5],
            }
        ]
    }
    result = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "chat_completion_params": [{}],
        }
    )
    assert result["responses"] == [expected]
    assert result["response_ids"] == [[3, 4, 5]]
    assert result["assistant_messages"][0]["reasoning_content"] == "reasoning words"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decoded,expected",
    [
        ('<|start_think|>not JSON<|end_think|>{"answer": 7}<|eot_id|>', '{"answer": 7}'),
        ("<|start_think|>a tentative answer is 7", ""),
        ("reasoning<|end_think|>7<|eot_id|>", "7"),
    ],
)
async def test_chat_grading_recovers_reasoning_boundaries_without_changing_replay_evidence(decoded, expected):
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = decoded
    engine.tokenize.return_value = {"tokens": [1, 2]}
    raw_message = {"role": "assistant", "content": "reasoning words mixed with answer", "tool_calls": []}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": raw_message,
                "finish_reason": "stop",
                "token_ids": [3, 4, 5],
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}, {"logprob": -0.3}]},
                "routed_experts": _encoded_routes([[[0, 0]], [[0, 0]], [[1, 2]], [[3, 4]]]),
            }
        ]
    }
    result = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "chat_completion_params": [{}],
        }
    )
    assert result["responses"] == [expected]
    assert result["response_ids"] == [[3, 4, 5]]
    assert result["response_logprobs"] == [[-0.1, -0.2, -0.3]]
    np.testing.assert_array_equal(result["routed_experts"][0], [[[0, 0]], [[1, 2]], [[3, 4]]])
    assert result["assistant_messages"] == [{**raw_message, "content": expected}]


@pytest.mark.asyncio
async def test_chat_output_budget_fits_the_exact_backend_rendered_prompt():
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "7"
    engine.tokenize.return_value = {"tokens": [1, 2, 3, 4]}

    async def serve(request):
        tokens = request["json"]["max_completion_tokens"]
        assert tokens == 1
        return {
            "choices": [{"message": {"role": "assistant", "content": "7"}, "finish_reason": "stop", "token_ids": [7]}]
        }

    engine.chat_completion.side_effect = serve
    result = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "a correction prompt"}]],
            "chat_completion_params": [{"max_output_tokens": 3}],
            "sampling_params": {"max_generate_length": 3},
            "max_context_length": 5,
        }
    )
    assert result["responses"] == ["7"]
    assert result["prompt_ids"] == [[1, 2, 3, 4]]
    assert result["generation_token_budgets"] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tokens,backend_stop,reason",
    [
        ([21, 22], None, "length"),
        ([21], None, "tool_calls"),
        ([21, 99], None, "tool_calls"),
        ([21, 98], 98, "tool_calls"),
    ],
)
async def test_chat_output_budget_stops_a_parser_reclassified_tool_call(tokens, backend_stop, reason):
    engine = AsyncMock()
    engine.model_name = "glm"
    engine.tokenizer = MagicMock(eos_token_id=99)
    engine.tokenizer.decode.return_value = "partial tool call"
    engine.tokenize.return_value = {"tokens": [1, 2]}
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call", "type": "function", "function": {"name": "shell", "arguments": '{"command":'}}],
    }
    engine.chat_completion.return_value = {
        "choices": [
            {"message": message, "finish_reason": "tool_calls", "token_ids": tokens, "stop_reason": backend_stop}
        ]
    }

    result = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "Run a command."}]],
            "chat_completion_params": [{}],
            "sampling_params": {"max_generate_length": 2},
        }
    )

    assert result["stop_reasons"] == [reason]
    assert result["response_ids"] == [tokens]
    assert result["assistant_messages"] == [message]


@pytest.mark.asyncio
async def test_chat_output_keeps_the_per_turn_limit_of_vllm_sampling_params():
    """Training passes vLLM-form sampling params; a large request window must not lift their per-turn limit."""
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "7"
    engine.tokenize.return_value = {"tokens": [1, 2, 3, 4]}
    served = []

    async def serve(request):
        served.append(request["json"])
        return {
            "choices": [{"message": {"role": "assistant", "content": "7"}, "finish_reason": "stop", "token_ids": [7]}]
        }

    engine.chat_completion.side_effect = serve
    sampling_params = get_vllm_sampling_params(
        OmegaConf.create(
            {"max_generate_length": 6528, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "logprobs": None}
        )
    )
    await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "chat_completion_params": [{}],
            "sampling_params": sampling_params,
            "max_context_length": 32768,
        }
    )
    assert served[0]["max_completion_tokens"] == 6528
    assert "max_tokens" not in served[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("native_call", [False, True])
async def test_chat_thinking_tokens_produce_gradeable_structured_answers_and_keep_native_calls(
    load_tokenizer, native_call
):
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B", revision="c1899de")
    tokens = tokenizer.encode('<|start_think|>This is not JSON.<|end_think|>{"answer":7}', add_special_tokens=False)
    message = {"role": "assistant", "content": 'This is not JSON.{"answer":7}'}
    if native_call:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-7", "type": "function", "function": {"name": "report", "arguments": '{"answer":7}'}}
            ],
        }
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = tokenizer
    engine.tokenize.return_value = {"tokens": [1, 2]}
    engine.chat_completion.return_value = {
        "choices": [{"message": message, "finish_reason": "tool_calls" if native_call else "stop", "token_ids": tokens}]
    }
    result = await DirectModelClient(engine).generate(
        {"prompts": [[{"role": "user", "content": "Return seven."}]], "chat_completion_params": [{}]}
    )
    task = answer_task(
        RawRow("structured", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {}),
        prompt="Return seven.",
        spec=StructuredExactSpec(expected={"answer": 7}),
    )
    task = task.model_copy(update={"answer_type": AnswerType.JSON, "answer_format": JsonValueAnswer()})
    if native_call:
        task = task.model_copy(
            update={
                "answer_type": AnswerType.NATIVE_ACTION,
                "answer_format": FinalAction(require_call=True, max_calls=1),
                "final_tools": (
                    FunctionDefinition(
                        name="report", parameters={"type": "object", "properties": {"answer": {"type": "integer"}}}
                    ),
                ),
                "grader": verifyit_package(
                    PredictedActionSpec(expected_calls=(FunctionCall(name="report", arguments={"answer": 7}),))
                ).grader,
            }
        )
    trace = chat_conversation([{"role": "user", "content": "Return seven."}, result["assistant_messages"][0]])
    grade = grade_answer(task, GradingAttempt(trace))
    assert (grade.status, grade.reward) == (Outcome.GRADED, 1.0)
    assert result["response_ids"] == [tokens]
    if native_call:
        assert result["assistant_messages"][0]["tool_calls"] == message["tool_calls"]
