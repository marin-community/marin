"""Generation settings observed at an OpenAI-compatible endpoint."""

import asyncio
import base64
import json
import subprocess
import sys
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers
import yaml
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.openai_completions import (
    LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion, OpenAICompletionsAPI,
)
from lm_eval.tasks import TaskManager
from lm_eval.utils import simple_parse_args_string
from finestore.eval import ARCHIVE_SAMPLES_TABLE
from finestore.reader import ReadView
from tokenizers import Tokenizer, models, pre_tokenizers

from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark
from eval.chat_benchmarks.MMLUPro.eval_instruct import MMLUProBenchmark, generate_cot_prompt
from eval.chat_benchmarks.LiveBench.eval_instruct import LiveBenchBenchmark
from eval.chat_benchmarks.RepoBench import eval_instruct as repobench
from eval.contracts.finestore_output import FineStoreOutput, read_finestore_output
from eval.eval import LM_EVAL_ROUTE, evaluate
from eval.execution.backend import LMBackend
from eval.execution.engine import CollectSamples, Evaluator
from eval.contracts.eval_spec import ModelRequest, RequestKind
from eval.sources.common import SourceOptions, text_specs
from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.execution.grading import Graders
from eval.graders.specs import GRADERS
from eval.limits import ContextWindowExceededError
from tests.contracts.source_harness import MemoryCheckpoints

LOCAL_CHAT_COMPLETIONS = "local-chat-completions"
LOCAL_COMPLETIONS = "local-completions"


class _Endpoint(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append(json.loads(body))
        output = self.server.outputs.pop(0) if self.server.outputs else "\\boxed{42}"
        completion = {"index": 0, "text": output, "message": {"content": output}}
        response = json.dumps({"choices": [completion]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        pass


@pytest.fixture
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), _Endpoint)
    server.requests = []
    server.outputs = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _model(endpoint, model_class, model_name="served", **kwargs):
    path = "chat/completions" if issubclass(model_class, LocalChatCompletion) else "completions"
    return model_class(
        model=model_name,
        base_url=f"http://127.0.0.1:{endpoint.server_port}/v1/{path}",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    )


@pytest.mark.parametrize("model_class", [
    LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion, OpenAICompletionsAPI,
])
@pytest.mark.parametrize("overrides, official_temperature, local_temperature", [
    ({}, 0.0, None),
    ({"temperature": 0.0, "seed": 1234}, 0.0, 0.0),
    ({"temperature": 0.85, "seed": 27}, 0.85, 0.85),
    ({"do_sample": False, "temperature": 0.85, "seed": 27}, 0.85, 0.85),
    ({"do_sample": True}, 0.0, None),
])
def test_math_source_preserves_api_sampling_policy(
    endpoint, tmp_path, model_class, overrides, official_temperature, local_temperature
):
    data = tmp_path / "math.jsonl"
    data.write_text(json.dumps({"id": "one", "problem": "Six times seven?", "answer": "42"}) + "\n")
    benchmark = MATH500Benchmark(data_file=str(data))
    model = _model(endpoint, model_class, header={"Authorization": "Bearer offline-test"})
    args = Namespace(model="api", model_args="", gen_kwargs=overrides, limit=1, max_tokens=8)

    result = evaluate(
        model, SimpleNamespace(get_benchmark=lambda _task: benchmark), None,
        ["MATH500"], {"MATH500": "Evalchemy chat benchmark"}, [1], args=args,
    )

    assert result["results"]["MATH500"]["accuracy"] == 1.0
    [payload] = endpoint.requests
    official = model_class in (OpenAIChatCompletion, OpenAICompletionsAPI)
    expected_temperature = official_temperature if official else local_temperature
    if expected_temperature is None:
        assert "temperature" not in payload
    else:
        assert payload["temperature"] == expected_temperature
    if "seed" in overrides or official:
        assert payload["seed"] == overrides.get("seed", 1234)
    else:
        assert "seed" not in payload


@pytest.mark.parametrize("model_class", [
    LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion, OpenAICompletionsAPI,
])
@pytest.mark.parametrize("model_name, max_tokens, overrides, local_cap, official_cap", [
    ("gpt-4o-mini", None, {}, 16384, 16384),
    ("gpt-4o-mini", 8192, {}, 8192, 8192),
    ("gpt-4o-mini", 32768, {}, 32768, 16384),
    ("gpt-4o-mini", None, {"max_gen_toks": 32768}, 32768, 16384),
    ("served", None, {}, 32768, 32768),
])
def test_math_source_preserves_4o_output_cap_and_caller_precedence(
    endpoint, tmp_path, model_class, model_name, max_tokens, overrides, local_cap, official_cap
):
    data = tmp_path / "math.jsonl"
    data.write_text(json.dumps({"id": "one", "problem": "Six times seven?", "answer": "42"}) + "\n")
    benchmark = MATH500Benchmark(data_file=str(data))
    model = _model(endpoint, model_class, model_name, header={"Authorization": "Bearer offline-test"})
    args = Namespace(model="api", model_args="", gen_kwargs=overrides, limit=1, max_tokens=max_tokens)

    result = evaluate(
        model, SimpleNamespace(get_benchmark=lambda _task: benchmark), None,
        ["MATH500"], {"MATH500": "Evalchemy chat benchmark"}, [1], args=args,
    )

    assert result["results"]["MATH500"]["accuracy"] == 1.0
    [payload] = endpoint.requests
    expected_cap = official_cap if model_class in (OpenAIChatCompletion, OpenAICompletionsAPI) else local_cap
    token_key = "max_completion_tokens" if model_class is OpenAIChatCompletion else "max_tokens"
    assert payload[token_key] == expected_cap


@pytest.mark.parametrize("model_class", [
    LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion, OpenAICompletionsAPI,
])
def test_sampling_source_preserves_trial_seeds_before_api_caller_override(endpoint, tmp_path, model_class):
    data = tmp_path / "aime.jsonl"
    data.write_text(json.dumps({"id": "one", "problem": "Six times seven?", "expected_answer": "42"}) + "\n")
    benchmark = AIME24Benchmark(data_file=data)
    model = _model(endpoint, model_class, header={"Authorization": "Bearer offline-test"})
    args = Namespace(model="api", model_args="", gen_kwargs={"do_sample": False}, limit=1, max_tokens=8)

    result = evaluate(
        model, SimpleNamespace(get_benchmark=lambda _task: benchmark), None,
        ["AIME24"], {"AIME24": "Evalchemy chat benchmark"}, [1], args=args,
    )

    assert result["results"]["AIME24"]["accuracy_avg"] == 1.0
    assert len(endpoint.requests) == 10
    for trial, payload in enumerate(endpoint.requests):
        if model_class in (OpenAIChatCompletion, OpenAICompletionsAPI):
            assert payload["temperature"] == 0.7
            assert payload["seed"] == trial
        else:
            assert payload["temperature"] == 0.0
            assert "seed" not in payload


@pytest.mark.parametrize("model_class", [
    LocalChatCompletion, LocalCompletionsAPI, OpenAIChatCompletion, OpenAICompletionsAPI,
])
def test_native_request_keeps_4o_output_budget(endpoint, model_class):
    model = _model(endpoint, model_class, "gpt-4o-mini", header={"Authorization": "Bearer offline-test"})
    request = ModelRequest(
        RequestKind.GENERATE, [{"role": "user", "content": "Six times seven?"}],
        {"max_gen_toks": 32768, "do_sample": False, "temperature": 0.7, "seed": 27},
    )

    assert generate(model, request) == "\\boxed{42}"

    [payload] = endpoint.requests
    token_key = "max_completion_tokens" if model_class is OpenAIChatCompletion else "max_tokens"
    assert payload[token_key] == 32768
    assert payload["temperature"] == 0.7
    assert payload["seed"] == 27


@pytest.mark.parametrize(("model_context", "request_context"), [(32768, 32768), (2061, 2061), (32768, 2061), (32768, 2049)])
def test_repobench_wire_preserves_raw_completion_and_native_truncation(endpoint, tmp_path, monkeypatch, model_context, request_context):
    tokenizer = Tokenizer(models.WordLevel({
        "[UNK]": 0, "[EOS]": 1, "user": 2, "assistant": 3, "discard": 4,
        "keep": 5, "=": 6, "1": 7,
    }, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "TokenizersBackend", "unk_token": "[UNK]", "eos_token": "[EOS]",
        "chat_template": "{% for message in messages %}{{ message['role'] + ' ' + message['content'] }}{% endfor %} assistant",
    }))
    model = LocalCompletionsAPI(
        model="served", tokenizer=str(tmp_path), tokenizer_backend="huggingface", tokenized_requests=True,
        base_url=f"http://127.0.0.1:{endpoint.server_port}/v1/completions",
        max_length=model_context, num_concurrent=256,
    )
    row = {
        "repo_name": "example", "file_path": "src/example.py", "context": [
            {"path": "dependencies.py", "snippet": "discard " * 2050},
        ],
        "import_statement": "", "cropped_code": "keep = 1   ", "next_line": "print(keep)",
    }
    monkeypatch.setattr(repobench, "load_dataset", lambda *_args, **_kwargs: {"cross_file_first": [row]})
    source = repobench.RepoBenchmark(languages=["python"], subsets=["cross_file_first"], max_tokens=2048)
    options = SourceOptions(
        "RepoBench", source.SOURCE, max_tokens=2048, context_length=request_context,
        generation={"temperature": 0.0, "seed": 1234}, encode=model.tokenizer.encode,
    )
    [completion] = source.iter_specs(options)
    native_prompt = "# Repo Name: example\n# Path: dependencies.py\n\n# Path: src/example.py\n\nkeep = 1\n"
    [chat] = text_specs(
        [row], options, prompt=lambda _row: native_prompt, target=lambda record: record["next_line"],
        grader="code_line", generation_kwargs={"max_gen_toks": 2048},
    )

    async def run():
        async with LMBackend(model) as backend:
            return [await backend.execute(spec.requests[0]) for spec in (completion, chat)]

    if request_context == 2049:
        with pytest.raises(ContextWindowExceededError):
            asyncio.run(run())
        assert endpoint.requests == []
        return
    assert asyncio.run(run()) == ["\\boxed{42}", "\\boxed{42}"]
    raw_request, chat_request = endpoint.requests
    original_model = LocalCompletionsAPI(
        model="served", tokenizer=str(tmp_path), tokenizer_backend="huggingface", tokenized_requests=True,
        base_url=f"http://127.0.0.1:{endpoint.server_port}/v1/completions", max_length=min(model_context, request_context),
        num_concurrent=256,
    )
    original = Instance("generate_until", row, (native_prompt, dict(completion.requests[0].generation)), 0)
    assert original_model.generate_until([original]) == ["\\boxed{42}"]
    original_request = endpoint.requests[-1]
    assert completion.requests[0].prompt == native_prompt
    assert completion.target == row["next_line"]
    assert raw_request == original_request
    input_budget = min(model_context, request_context) - 2048 - 1
    assert raw_request["prompt"] == [model.tok_encode(native_prompt)[-input_budget:]]
    assert chat_request["prompt"] == [model.tok_encode("user " + native_prompt + " assistant")[-input_budget:]]
    assert raw_request["prompt"] != chat_request["prompt"]
    assert raw_request["max_tokens"] == 2048
    assert raw_request["temperature"] == 0.0
    assert raw_request["seed"] == 1234


@pytest.mark.parametrize("system_instruction", [None, "Follow the evaluation rules."])
def test_livebench_wire_preserves_fastchat_template_and_cleaned_turn_history(endpoint, tmp_path, system_instruction):
    source = LiveBenchBenchmark(question_source="jsonl", dataset_name="fixture")
    source.data_path = str(tmp_path)
    data = tmp_path / "fixture" / "category"
    data.mkdir(parents=True)
    question = {
        "question_id": "two-turns", "turns": ["Turn one", "Turn two"],
        "task": "coding_completion", "category": "coding", "livebench_release_date": "2024-08-31",
        "livebench_removal_date": "",
    }
    (data / "question.jsonl").write_text(json.dumps(question) + "\n")
    options = SourceOptions(task="LiveBench", source=source.SOURCE, limit=1, system_instruction=system_instruction)
    model = _model(endpoint, LocalChatCompletion)
    model.model_identifier = "model=owner__Qwen-Qwen3-0.6B,tokenizer_backend=None"
    endpoint.outputs = ["  First answer<|endoftext|> discarded  ", "  Second answer  "]
    sink = CollectSamples()

    async def run():
        async with LMBackend(model) as backend:
            evaluator = Evaluator(backend, None, sink, workers=1, predict_only=True)
            return [result async for result in evaluator.run(source.iter_specs(options))]

    [completed] = asyncio.run(run())
    assert completed.error is None
    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    first = prefix + "<|im_start|>user\nTurn one<|im_end|>\n<|im_start|>assistant\n"
    second = first + "First answer<|im_end|>\n<|im_start|>user\nTurn two<|im_end|>\n<|im_start|>assistant\n"
    outer_system = [{"role": "system", "content": system_instruction}] if system_instruction else []
    assert [request["messages"] for request in endpoint.requests] == [
        [*outer_system, {"role": "user", "content": first}],
        [*outer_system, {"role": "user", "content": second}],
    ]
    assert sink.samples["LiveBench"][0]["filtered_resps"] == ["First answer", "Second answer"]


@pytest.mark.parametrize("answer, expected", [(42, 1.0), (41, 0.0)])
def test_model_request_progresses_while_humaneval_sandbox_waits(endpoint, answer, expected):
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def sandbox_connection(reader, writer):
            try:
                assert await reader.readline() == b"grading\n"
                started.set()
                await release.wait()
                writer.write(b"42")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        async with await asyncio.start_server(sandbox_connection, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            completion = (
                "```python\ndef answer():\n    import socket\n"
                f"    with socket.create_connection(('127.0.0.1', {port}), timeout=3) as connection:\n"
                "        connection.sendall(b'grading\\n')\n"
                f"        return int(connection.recv(2)) - {42 - answer}\n```"
            )
            problem = {"task_id": "python/0", "prompt": "def answer():\n", "test": "assert answer() == 42"}
            spec = EvalSpec(
                task="HumanEval", source=SourceRef("fixture"), source_id="0", ordinal=0, requests=(),
                grader=GraderSpec("program_suite", {"suite": "HumanEval", "language": "python",
                                                      "problem": problem, "timeout": 5.0}), doc=problem,
            )
            graders = Graders(GRADERS, workers=1)
            grading = asyncio.create_task(graders(spec, [completion]))
            try:
                await asyncio.wait_for(started.wait(), timeout=15)
                async with LMBackend(_model(endpoint, LocalChatCompletion)) as backend:
                    output = await backend.execute(ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "Next example"}]))
                assert output == "\\boxed{42}"
                assert not grading.done()
            finally:
                # The real sandbox cannot finish until the driver completes this
                # separate HTTP model operation and releases its socket response.
                release.set()
                result = await grading
                graders.close()
            return result

    assert asyncio.run(run()).metrics["accuracy"] == expected


def test_model_request_progresses_during_checkpoint_commit_and_cancellation_joins_the_write(endpoint):
    started = threading.Event()
    release = threading.Event()

    class Storage(MemoryCheckpoints):
        def record(self, row):
            started.set()
            # A deadlock guard, not a performance threshold: only the test's
            # separate HTTP response releases this simulated storage commit.
            if not release.wait(timeout=5):
                raise TimeoutError("model response could not progress during checkpoint commit")
            super().record(row)

    class SavedOnly:
        async def execute(self, request):
            raise AssertionError("the committed response must replay without inference")

    store = Storage()
    spec = EvalSpec(
        task="checkpoint", source=SourceRef("fixture"), source_id="one", ordinal=0,
        requests=(ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "Stored question"}]),),
        grader=GraderSpec("unused"),
    )

    async def run():
        async with LMBackend(_model(endpoint, LocalChatCompletion)) as backend:
            engine = Evaluator(backend, None, CollectSamples(), workers=1, checkpoints={"checkpoint": store}, predict_only=True)
            stream = engine.run([spec])
            pending = asyncio.create_task(anext(stream))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                pending.cancel()
                response = await backend.execute(ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "Other question"}]))
                assert response == "\\boxed{42}"
                pending.cancel()
                assert not pending.done()
            finally:
                release.set()
                try:
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                finally:
                    await stream.aclose()
        assert store.closed
        sink = CollectSamples()
        restored = Evaluator(SavedOnly(), None, sink, workers=1, checkpoints={"checkpoint": store}, predict_only=True)
        [completed] = [result async for result in restored.run([spec])]
        assert completed.error is None
        assert sink.samples["checkpoint"][0]["filtered_resps"] == ["\\boxed{42}"]

    asyncio.run(run())


def test_endpoint_admission_exceeds_aiohttp_default_connection_limit():
    async def run():
        received = []
        admitted = asyncio.Event()
        release = asyncio.Event()

        async def respond(reader, writer):
            try:
                headers = (await reader.readuntil(b"\r\n\r\n")).decode()
                content_length = next(int(line.split(":", 1)[1]) for line in headers.split("\r\n") if line.lower().startswith("content-length:"))
                received.append(json.loads(await reader.readexactly(content_length)))
                if len(received) == 101:
                    admitted.set()
                await release.wait()
                body = b'{"choices":[{"index":0,"message":{"content":"done"}}]}'
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        async with await asyncio.start_server(respond, "127.0.0.1", 0, backlog=256) as server:
            model = LocalChatCompletion(
                model="served", base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/chat/completions",
                tokenizer_backend=None, tokenized_requests=False,
            )
            async with LMBackend(model) as backend:
                tasks = [asyncio.create_task(backend.execute(ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": f"Example {index}"}]))) for index in range(101)]
                try:
                    await asyncio.wait_for(admitted.wait(), timeout=10)
                finally:
                    release.set()
                outputs = await asyncio.gather(*tasks)
            return received, outputs

    received, outputs = asyncio.run(run())
    assert {request["messages"][0]["content"] for request in received} == {f"Example {index}" for index in range(101)}
    assert outputs == ["done"] * 101


class _PlainLM(LM):
    def __init__(self):
        super().__init__()
        self.requests = []

    def generate_until(self, requests):
        self.requests.extend(requests)
        return ["\\boxed{42}"] * len(requests)

    def loglikelihood(self, _requests):
        raise AssertionError("generation task requested loglikelihood")

    def loglikelihood_rolling(self, _requests):
        raise AssertionError("generation task requested rolling loglikelihood")


@pytest.mark.parametrize("batch_sizes", [[2, 3], ["auto", 3]])
def test_driver_preserves_per_task_local_batches(tmp_path, batch_sizes):
    class BatchedModel(_PlainLM):
        def __init__(self):
            super().__init__()
            self.batches = []
            self.batch_size = 1

        def tok_encode(self, text):
            return list(text.encode())

        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

        def generate_until(self, requests):
            task = requests[0].metadata[0]
            assert all(request.metadata[0] == task for request in requests)
            size = self.batch_size
            self.batches.append((task, size, len(requests)))
            return super().generate_until(requests)

    sources = {}
    for task in ("first", "second"):
        data = tmp_path / f"{task}.jsonl"
        data.write_text("".join(json.dumps({"id": index, "problem": f"{task} {index}", "answer": "42"}) + "\n" for index in range(6)))
        sources[task] = MATH500Benchmark(data_file=str(data))
    model = BatchedModel()
    args = Namespace(model="fixture", model_args="", seed=[0, 0, 0, 0], max_batch_size=4)

    result = evaluate(
        model, SimpleNamespace(get_benchmark=sources.__getitem__), None,
        list(sources), {task: "Evalchemy chat benchmark" for task in sources}, batch_sizes, args=args,
    )

    assert all(outcome["status"] == "succeeded" for outcome in result["task_outcomes"].values())
    assert max(count for _, _, count in model.batches) > 1
    for task, size in zip(sources, batch_sizes, strict=True):
        batches = [(configured, count) for name, configured, count in model.batches if name == task]
        assert sum(count for _, count in batches) == 6
        assert all(configured == size for configured, _ in batches)


@pytest.mark.parametrize(
    "checkpoint",
    ["moonshotai/Kimi-Linear-48B-A3B-Instruct", "google/gemma-4-26B-A4B-it"],
)
def test_served_chat_checkpoint_generates_without_client_tokenizer(endpoint, checkpoint, monkeypatch):
    def reject_local_tokenizer(*_args, **_kwargs):
        raise AssertionError("chat evaluation must use the server's tokenizer")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", reject_local_tokenizer)
    model = _model(endpoint, LocalChatCompletion, checkpoint)
    request = Instance("generate_until", {"question": "Question"}, ([{"role": "user", "content": "Question"}], {}), 0)

    normalized = ModelRequest(RequestKind.GENERATE, request.args[0], request.args[1])
    assert generate(model, normalized) == "\\boxed{42}"
    assert endpoint.requests[0]["model"] == checkpoint
    assert endpoint.requests[0]["messages"] == [{"role": "user", "content": "Question"}]


def test_served_chat_forwards_template_kwargs_and_extra_body(endpoint):
    template_kwargs = {"enable_thinking": False, "add_generation_prompt": True}
    encoded_kwargs = base64.urlsafe_b64encode(json.dumps(template_kwargs).encode()).decode()
    model_args = simple_parse_args_string(
        "model=served,"
        f"base_url=http://127.0.0.1:{endpoint.server_port}/v1/chat/completions,"
        "tokenizer_backend=None,tokenized_requests=False,"
        f'chat_template_kwargs=base64:{encoded_kwargs},extra_body={{"priority":7}}'
    )
    model = LocalChatCompletion(**model_args)
    request = Instance("generate_until", {"question": "Question"}, ([{"role": "user", "content": "Question"}], {}), 0)

    normalized = ModelRequest(RequestKind.GENERATE, request.args[0], request.args[1])
    assert generate(model, normalized) == "\\boxed{42}"
    assert endpoint.requests[0]["chat_template_kwargs"] == template_kwargs
    assert endpoint.requests[0]["priority"] == 7
    assert "extra_body" not in endpoint.requests[0]


def test_served_completions_loads_custom_tokenizer_with_remote_code(monkeypatch):
    def load_custom_tokenizer(_checkpoint, *, trust_remote_code, **_kwargs):
        if not trust_remote_code:
            raise ValueError("custom tokenizer requires trust_remote_code=True")

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, _text, **_kwargs):
                return SimpleNamespace(input_ids=[1, 2])

        return Tokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_custom_tokenizer)
    model = LocalCompletionsAPI(
        model="custom/checkpoint", tokenizer="custom/checkpoint",
        base_url="http://127.0.0.1:8000/v1/completions",
        tokenizer_backend="huggingface", trust_remote_code=True,
    )

    assert model.tok_encode("Hello") == [1, 2]


def test_served_completions_loads_transformers5_tokenizer_config(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "<|video|>": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "unk_token": "[UNK]",
                "extra_special_tokens": ["<|video|>"],
            }
        )
    )

    model = LocalCompletionsAPI(
        model="served",
        tokenizer=str(tmp_path),
        base_url="http://127.0.0.1:8000/v1/completions",
        tokenizer_backend="huggingface",
    )

    assert model.tok_encode("hello") == [1]
    assert model.tok_encode("<|video|>") == [2]


@pytest.mark.parametrize(
    ("num_fewshot", "budget_shots", "headroom", "expected_shots"),
    [(2, 2, 1, 2), (2, 2, 0, 1), (2, 1, 0, 0), (0, 2, 1, 0)],
)
def test_mmlupro_chat_budget_preserves_fitting_demonstrations(
    endpoint, tmp_path, monkeypatch, num_fewshot, budget_shots, headroom, expected_shots
):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "unk_token": "[UNK]"})
    )
    question = {
        "question": "Held-out math question?", "options": ["test-a", "test-b"],
        "answer": "B", "cot_content": "A: Let's think step by step. test-b", "category": "math",
    }
    demonstrations = [
        {**question, "question": "First math demonstration?"},
        {**question, "question": "Unrelated history demonstration?", "category": "history"},
        {**question, "question": "Second math demonstration?"},
        {**question, "question": "Third unused math demonstration?"},
    ]
    monkeypatch.setattr(
        "eval.chat_benchmarks.MMLUPro.eval_instruct.load_dataset",
        lambda *_args, **_kwargs: {
            "validation": demonstrations,
            "test": [question, {**question, "question": "Held-out question beyond the limit?"}],
        },
    )
    budget_prompt = generate_cot_prompt(demonstrations, question, budget_shots)
    input_budget = len(tokenizer.encode(budget_prompt).ids) + headroom
    assert len(budget_prompt) > input_budget
    benchmark = MMLUProBenchmark(num_fewshot=num_fewshot)
    model = LocalChatCompletion(
        model=str(tokenizer_dir if num_fewshot else tmp_path / "no-tokenizer-needed"),
        base_url=f"http://127.0.0.1:{endpoint.server_port}/v1/chat/completions",
        tokenizer_backend=None, tokenized_requests=False, auth_token="offline-test",
    )
    args = Namespace(
        model=LOCAL_CHAT_COMPLETIONS, model_args="", gen_kwargs=None, limit=1, num_fewshot=num_fewshot,
        seed=[0, 1234, 1234, 1234], max_tokens=8, max_length=input_budget + 8,
    )

    result = evaluate(
        model, SimpleNamespace(get_benchmark=lambda _task: benchmark), None,
        ["MMLUPro"], {"MMLUPro": "Evalchemy chat benchmark"}, [1], args=args,
    )

    assert result["task_outcomes"]["MMLUPro"]["status"] == "succeeded"
    assert len(endpoint.requests) == 1
    payload = endpoint.requests[0]
    prompt = payload["messages"][0]["content"]
    assert payload["messages"] == [{"role": "user", "content": prompt}]
    assert "Held-out math question?" in prompt
    assert ("First math demonstration?" in prompt) is (expected_shots >= 1)
    assert ("Second math demonstration?" in prompt) is (expected_shots == 2)
    assert "Unrelated history demonstration?" not in prompt
    assert "Third unused math demonstration?" not in prompt
    assert "Held-out question beyond the limit?" not in prompt
    assert prompt == generate_cot_prompt(demonstrations, question, expected_shots)
    assert payload["max_tokens"] == 8




















def test_evaluation_driver_streams_lm_eval_samples_without_result_list(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({"question": "What is 6 times 7?", "answer": "\\boxed{42}"}) + "\n")
    task = {
        "task": "streaming_driver_probe",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(dataset)}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{question}}",
        "doc_to_target": "{{answer}}",
        "metric_list": [{"metric": "exact_match"}],
    }
    (tmp_path / "streaming_driver_probe.yaml").write_text(yaml.safe_dump(task))
    root = str(tmp_path / "archive")
    model = _PlainLM()
    args = Namespace(
        model=model,
        model_args="",
        num_fewshot=0,
        max_batch_size=None,
        device=None,
        limit=1,
        check_integrity=False,
        write_out=False,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        gen_kwargs=None,
        verbosity="INFO",
        predict_only=False,
        confirm_run_unsafe_code=False,
        seed=[0, 0, 0, 0],
        task_preparations={},
        resume_manager_factory=None,
    )
    task_name = "streaming_driver_probe"

    with FineStoreOutput(root, task_name, [task_name]) as output:
        results = evaluate(
            lm=model,
            task_manager=SimpleNamespace(tasks={}),
            pretrain_task_manager=TaskManager(include_path=str(tmp_path)),
            task_list=[task_name],
            task_routes={task_name: LM_EVAL_ROUTE},
            batch_sizes_list=[1],
            args=args,
            output_sink=output,
        )
        assert "samples" not in results
        staged = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
        assert staged is not None
        assert staged.num_rows == 1
        assert read_finestore_output(root, task_name) is None
        output.finish(results)

    assert results["task_outcomes"][task_name]["status"] == "succeeded"
    assert len(model.requests) == 1




# Cold lm-eval imports/catalog discovery need headroom; the child must time out before pytest.
@pytest.mark.timeout(120)
@pytest.mark.parametrize("task", ["AIME24", "MATH500"])
def test_cli_output_cap_wins_over_math_benchmark_defaults(endpoint, tmp_path, task, math500_jsonl):
    root = Path(__file__).parents[2]
    dataset_cache = tmp_path / "datasets"
    staged_math = dataset_cache / MATH500Benchmark.SOURCE.dataset / MATH500Benchmark.SOURCE.revision / "test.jsonl"
    staged_math.parent.mkdir(parents=True)
    staged_math.write_bytes(math500_jsonl.read_bytes())
    staged_math.with_name("test.jsonl.complete").touch()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.eval",
            "--model",
            "local-chat-completions",
            "--model_args",
            f"model=served,base_url=http://127.0.0.1:{endpoint.server_port}/v1/chat/completions,tokenizer_backend=None,tokenized_requests=False",
            "--tasks",
            task,
            "--dataset_cache",
            str(dataset_cache),
            "--debug",
            "--limit",
            "1",
            "--max_length",
            "65536",
            "--max_tokens",
            "16000",
            "--gen_kwargs",
            "temperature=0.85,top_p=0.91,seed=27",
            "--finestore_output_path",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stderr
    assert endpoint.requests, result.stdout + result.stderr
    for payload in endpoint.requests:
        assert payload["max_tokens"] == 16000
        assert "max_gen_toks" not in payload
        assert payload["temperature"] == 0.85
        assert payload["seed"] == 27
        assert payload["top_p"] == 0.91


def test_cli_generation_limits_win_over_config_defaults(endpoint, tmp_path):
    root = Path(__file__).parents[2]
    config = tmp_path / "tasks.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "tasks": [{"task_name": "AIME24", "batch_size": "auto"}],
                "max_length": 32768,
                "max_tokens": 8192,
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.eval",
            "--model",
            "local-chat-completions",
            "--model_args",
            f"model=served,base_url=http://127.0.0.1:{endpoint.server_port}/v1/chat/completions,tokenizer_backend=None,tokenized_requests=False,max_length=65536",
            "--config",
            str(config),
            "--debug",
            "--limit",
            "1",
            "--gen_kwargs",
            "max_gen_toks=16000",
            "--finestore_output_path",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert endpoint.requests, result.stdout + result.stderr
    assert all(payload["max_tokens"] == 16000 for payload in endpoint.requests)
    assert all("max_gen_toks" not in payload for payload in endpoint.requests)


def test_unified_driver_persists_mixed_sources_through_one_endpoint(endpoint, tmp_path):
    math_data = tmp_path / "math.jsonl"
    math_data.write_text(json.dumps({"id": "one", "problem": "What is 6 times 7?", "answer": "42"}) + "\n")
    native_data = tmp_path / "native.jsonl"
    native_data.write_text(json.dumps({"question": "What is 6 times 7?", "answer": "\\boxed{42}"}) + "\n")
    native_task = {
        "task": "mixed_probe", "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(native_data)}},
        "test_split": "test", "output_type": "generate_until",
        "doc_to_text": "{{question}}", "doc_to_target": "{{answer}}",
        "metric_list": [{"metric": "exact_match"}], "num_fewshot": 0,
    }
    (tmp_path / "mixed_probe.yaml").write_text(yaml.safe_dump(native_task))
    benchmark = MATH500Benchmark(data_file=str(math_data))
    args = Namespace(
        model="local-chat-completions", model_args="num_concurrent=2", gen_kwargs=None,
        limit=1, num_fewshot=0, seed=[0, 1234, 1234, 1234], max_tokens=8,
        apply_chat_template=True,
    )
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "mixed", ["MATH500", "mixed_probe"]) as output:
        result = evaluate(
            lm=_model(endpoint, LocalChatCompletion),
            task_manager=SimpleNamespace(get_benchmark=lambda _task: benchmark),
            pretrain_task_manager=TaskManager(include_path=str(tmp_path)),
            task_list=["MATH500", "mixed_probe"],
            task_routes={"MATH500": "Evalchemy chat benchmark", "mixed_probe": "lm-eval"},
            batch_sizes_list=[1, 1], args=args, output_sink=output,
        )
        output.finish_task()
        samples = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
        assert samples is not None and samples.num_rows == 2
        assert read_finestore_output(root, "mixed") is None
        assert not result.get("samples")
        assert result["results"]["MATH500"]["accuracy"] == 1.0
        assert result["results"]["mixed_probe"]["exact_match,none"] == 1.0
        assert all(outcome["status"] == "succeeded" for outcome in result["task_outcomes"].values())
        output.finish(result)
    assert len(endpoint.requests) == 2
    assert all(request["max_tokens"] == 8 for request in endpoint.requests)


@pytest.mark.parametrize("model_class", [LocalChatCompletion, LocalCompletionsAPI])
@pytest.mark.parametrize("overrides", [
    {}, {"temperature": 0.0, "seed": 1234}, {"temperature": 0.85, "seed": 27, "top_p": 0.91},
    {"max_gen_toks": 16000}, {"do_sample": False}, {"do_sample": True},
    {"do_sample": False, "temperature": 0.85, "seed": 27},
])
def test_native_source_preserves_defaults_and_applies_caller_settings(endpoint, tmp_path, model_class, overrides):
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({"question": "Six times seven?", "answer": "\\boxed{42}"}) + "\n")
    config = {
        "task": "generation_probe", "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(dataset)}}, "test_split": "test",
        "output_type": "generate_until", "doc_to_text": "{{question}}", "doc_to_target": "{{answer}}",
        "metric_list": [{"metric": "exact_match"}], "num_fewshot": 0,
        "generation_kwargs": {"temperature": 0.0, "top_p": 0.4, "do_sample": False},
    }
    (tmp_path / "generation_probe.yaml").write_text(yaml.safe_dump(config))
    args = Namespace(model=LOCAL_CHAT_COMPLETIONS if model_class is LocalChatCompletion else LOCAL_COMPLETIONS,
                     model_args="", gen_kwargs=overrides, num_fewshot=0, limit=1,
                     seed=[0, 1234, 1234, 1234], apply_chat_template=model_class is LocalChatCompletion)
    result = evaluate(
        _model(endpoint, model_class), SimpleNamespace(tasks={}), TaskManager(include_path=str(tmp_path)),
        ["generation_probe"], {"generation_probe": LM_EVAL_ROUTE}, [1], args=args,
    )
    assert result["results"]["generation_probe"]["exact_match,none"] == 1.0
    payload = endpoint.requests[0]
    if "temperature" in overrides:
        assert payload["temperature"] == overrides["temperature"]
    elif overrides.get("do_sample") is False:
        assert payload["temperature"] == 0.0
    else:
        assert "temperature" not in payload
    assert payload["top_p"] == overrides.get("top_p", 0.4)
    if "seed" in overrides:
        assert payload["seed"] == overrides["seed"]
    else:
        assert "seed" not in payload
    if "max_gen_toks" in overrides:
        assert payload["max_tokens"] == overrides["max_gen_toks"]


def test_context_preflight_caps_the_effective_generation_request(endpoint):
    class Tokenizer:
        def apply_chat_template(self, _messages, **_kwargs):
            return list(range(900))
    model = _model(endpoint, LocalChatCompletion)
    model.tokenizer = Tokenizer()
    model.max_length = 999
    request = ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "Question"}], {"max_gen_toks": 4096})
    assert generate(model, request) == "\\boxed{42}"
    assert endpoint.requests[0]["max_tokens"] == 36


def generate(model, request):
    async def run():
        async with LMBackend(model) as backend:
            return await backend.execute(request)
    return asyncio.run(run())
