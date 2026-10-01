# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Workplace import and mutable provider behavior."""

import asyncio
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.request import urlopen

import pytest
from tasktrove_verify.spec import MathType

from taskcompendium import release_audit
from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.importers import nemo_workplace
from taskcompendium.importers.nemo_workplace import (
    DATASET_REVISION,
    DATASET_SPLIT_ROW_COUNTS,
    DATASET_SPLIT_SHA256,
    PROVIDER,
    PROVIDER_GIT_REVISION,
    PROVIDER_REPOSITORY,
    ROW_SHA256,
    ROW_SHA256_BY_ID,
    SOURCE_EXAMPLE_MAX_BYTES,
    SOURCE_EXAMPLE_SHA256,
    SOURCE_EXAMPLE_URL,
    import_dataset_split,
    import_row,
    select_row_zero,
    select_rows,
    workplace_environment_config,
)
from taskcompendium.lowering import lower_to_harbor, provider_class
from taskcompendium.mixed_release import PublishedRow, SourceProof
from taskcompendium.models import AnswerType, ConversationInput, ConversationTrace, TaskSpec, TextMessage, VerifierKind
from taskcompendium.provider_sources import ToolProviderCache, stage_git_provider
from taskcompendium.public_release import build_workplace_candidate
from taskcompendium.release_audit import audit_demonstration
from taskcompendium.submission import GradingAttempt, PlainText, SubmissionConvention
from taskcompendium.verifier_registry import grade_answer
from taskcompendium.verifiers.mathematical import mathematical_answer
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


@pytest.fixture(scope="module")
def source_example() -> bytes:
    """Resolve the pinned upstream source in trusted test setup, before trials."""
    with urlopen(SOURCE_EXAMPLE_URL, timeout=30) as response:
        return response.read(SOURCE_EXAMPLE_MAX_BYTES + 1)


@pytest.fixture(scope="module")
def source_row(source_example: bytes) -> bytes:
    return select_row_zero(source_example)


@pytest.fixture(scope="module")
def trusted_provider_checkout(tmp_path_factory) -> Path:
    """Resolve the pinned source once, before any exported Harbor trial starts."""
    checkout = tmp_path_factory.mktemp("workplace-source") / "nemo_workplace"
    subprocess.run(["git", "clone", "--quiet", PROVIDER_REPOSITORY, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", PROVIDER_GIT_REVISION], check=True)
    return checkout


@pytest.fixture(scope="module")
def provider_source(tmp_path_factory, trusted_provider_checkout) -> Path:
    source = tmp_path_factory.mktemp("workplace-snapshot") / "provider"
    stage_git_provider(PROVIDER, trusted_provider_checkout, source)
    return source


@pytest.fixture(scope="module")
def workplace_provider(provider_source):
    binding = workplace_environment_config(provider_source).tool_providers["workplace"]
    with ToolProviderCache() as cache:
        yield provider_class(binding, provider_source, cache=cache)


@contextmanager
def _serve_endpoint(endpoint: type[BaseHTTPRequestHandler]) -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _send_completion(handler: BaseHTTPRequestHandler, message: dict) -> None:
    body = json.dumps({"choices": [{"message": message}]}).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _provider(workplace_provider):
    return workplace_provider(seed_sha256=workplace_provider.SEED_SHA256)


def _source(data: bytes) -> tuple[bytes, dict]:
    return data, json.loads(data)


async def _reward(specification: TaskSpec, convention: SubmissionConvention, provider) -> float:
    conversation = ConversationTrace(
        events=(*specification.context.events, TextMessage(role="assistant", content="Done."))
    )
    result = await grade_answer(
        specification,
        convention,
        GradingAttempt(conversation=conversation, tool_providers={"workplace": provider}, workspace=None),
    )
    assert result.reward is not None
    return result.reward


def test_workplace_import_pins_row_tool_surface_and_private_state(
    source_example: bytes, source_row: bytes, provider_source
):
    assert hashlib.sha256(source_example).hexdigest() == SOURCE_EXAMPLE_SHA256
    assert hashlib.sha256(source_row).hexdigest() == ROW_SHA256
    data, row = _source(source_row)
    specification, convention, binding = import_row(data, provider_source)
    provider = binding.tool_providers["workplace"]
    assert specification.source.revision == DATASET_REVISION
    assert specification.answer_type.value == convention.answer_format.value == "state"
    assert convention.provider == "workplace"
    assert specification.verifier.kind == VerifierKind.STRUCTURED_EXACT
    assert len(provider.tools) == len(row["responses_create_params"]["tools"]) == 27
    visible = specification.context.model_dump_json()
    assert [(event.role, event.content) for event in specification.context.events] == [
        (message["role"], message["content"]) for message in row["responses_create_params"]["input"]
    ]
    assert "ground_truth" not in visible
    assert provider.provider.endswith(f"@{PROVIDER_GIT_REVISION}:nemo_workplace.provider:NemoWorkplaceProvider")


def test_workplace_import_converts_every_pinned_example_row(source_example: bytes, provider_source):
    rows = select_rows(source_example)
    imports = [import_row(row, provider_source) for row in rows]
    specifications = [item.specification for item in imports]

    assert [spec.source.row for spec in specifications] == ["0", "1", "2", "3", "4"]
    assert [spec.id for spec in specifications] == [f"nemo-workplace-{row_id}" for row_id in range(5)]
    assert all(spec.answer_type is AnswerType.STATE for spec in specifications)
    assert all(spec.verifier.kind is VerifierKind.STRUCTURED_EXACT for spec in specifications)
    assert all("ground_truth" not in spec.model_dump_json() for spec in specifications)


def test_workplace_candidate_exports_public_rows_from_provider_snapshot(
    monkeypatch, tmp_path, source_row: bytes, provider_source
):
    digest = hashlib.sha256(source_row).hexdigest()
    sources = {}
    for split in ("train", "validation"):
        monkeypatch.setitem(DATASET_SPLIT_ROW_COUNTS, split, 1)
        monkeypatch.setitem(DATASET_SPLIT_SHA256, split, digest)
        sources[split] = tmp_path / f"{split}.jsonl"
        sources[split].write_bytes(source_row)
    output = build_workplace_candidate(
        sources["train"],
        sources["validation"],
        tmp_path / "candidate",
        provider_source=provider_source,
        builder_revision="a" * 40,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    provider = manifest["sources"]["nemo_workplace"]
    assert provider["provider_binding"]["provider"] == PROVIDER
    assert len(provider["provider_binding"]["tools"]) == 27
    for split in ("train", "validation"):
        data = (output / "data" / f"{split}.jsonl").read_bytes()
        record = json.loads(data)
        assert record["id"] == f"nemo-workplace-{split}-0"
        assert record["tool_providers"]["workplace"] == {
            "action_interface": provider["action_interface"],
            "seed_sha256": provider["seed_sha256"],
        }
        assert manifest["data_files"][split]["sha256"] == hashlib.sha256(data).hexdigest()
        assert "ground_truth" not in record and "verifier" not in record
    assert {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} == {
        "data/train.jsonl",
        "data/validation.jsonl",
        "manifest.json",
        "README.md",
    }


async def test_workplace_import_gradeable_empty_ground_truth_split_row(
    monkeypatch, source_row: bytes, provider_source, workplace_provider
):
    row = json.loads(source_row)
    row["category"] = "workplace_assistant_analytics"
    row["ground_truth"] = []
    data = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    monkeypatch.setitem(DATASET_SPLIT_ROW_COUNTS, "train", 1)
    monkeypatch.setitem(DATASET_SPLIT_SHA256, "train", hashlib.sha256(data).hexdigest())

    (workplace_import,) = import_dataset_split(data, "train", provider_source)
    specification, convention, _ = workplace_import
    provider = _provider(workplace_provider)

    assert specification.id == "nemo-workplace-train-0"
    assert specification.source.row == f"train:0:{hashlib.sha256(data).hexdigest()}"
    assert await _reward(specification, convention, provider) == 1.0


def test_workplace_import_rejects_unpinned_row_and_changed_tools(
    monkeypatch, source_example: bytes, source_row: bytes, provider_source
):
    data, row = _source(source_row)
    with pytest.raises(ValueError, match="pinned digest"):
        select_row_zero(source_example + b" ")
    with pytest.raises(ValueError, match="size limit"):
        select_row_zero(b" " * (SOURCE_EXAMPLE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="pinned raw digest"):
        import_row(data + b" ", provider_source)
    row["responses_create_params"]["tools"][0]["name"] = "wrong_tool"
    changed = json.dumps(row).encode()
    monkeypatch.setitem(ROW_SHA256_BY_ID, 0, hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="pinned provider"):
        import_row(changed, provider_source)


async def test_workplace_success_wrong_and_noop_state(source_row: bytes, provider_source, workplace_provider):
    _, row = _source(source_row)
    specification, convention, _ = import_row(source_row, provider_source)
    gold = row["ground_truth"]
    success, wrong, noop = (_provider(workplace_provider) for _ in range(3))
    action = gold[0]
    await success.dispatch_action(action["name"], action["arguments"], "call-1")
    await wrong.dispatch_action(
        action["name"],
        '{"email_id":"00000057","body":"Thanks for the update - I will not follow up."}',
        "call-1",
    )
    await noop.dispatch_action(
        "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}', "call-1"
    )
    assert await _reward(specification, convention, success) == 1.0
    assert await _reward(specification, convention, wrong) == 0.0
    assert await _reward(specification, convention, noop) == 0.0


async def test_workplace_tool_error_recovers_and_retains_call_order(
    source_row: bytes, provider_source, workplace_provider
):
    _, row = _source(source_row)
    specification, convention, _ = import_row(source_row, provider_source)
    action = row["ground_truth"][0]
    provider = _provider(workplace_provider)
    error = await provider.dispatch_action(action["name"], '{"email_id":"00000057","unknown":"x"}', "call-bad")
    success = await provider.dispatch_action(action["name"], action["arguments"], "call-good")
    assert [entry.call_id for entry in provider.trace] == ["call-bad", "call-good"]
    assert [entry.output for entry in provider.trace] == [error, success]
    assert await _reward(specification, convention, provider) == 1.0


async def test_workplace_concurrent_trials_start_from_fresh_seed(source_row: bytes, provider_source, workplace_provider):
    _, row = _source(source_row)
    specification, convention, _ = import_row(source_row, provider_source)
    action = row["ground_truth"][0]
    first, second = _provider(workplace_provider), _provider(workplace_provider)
    await asyncio.gather(
        first.dispatch_action(action["name"], action["arguments"], "first"),
        second.dispatch_action(
            "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}', "second"
        ),
    )
    assert (await _reward(specification, convention, first), await _reward(specification, convention, second)) == (
        1.0,
        0.0,
    )


async def test_workplace_harbor_scripted_endpoint_recovers_after_tool_error(
    tmp_path, trusted_provider_checkout, source_row: bytes, provider_source
):
    data, row = _source(source_row)
    specification, convention, binding = import_row(data, provider_source)
    task_dir = lower_to_harbor(
        specification,
        convention,
        binding,
        tmp_path / "task",
        trusted_provider_sources={"workplace": trusted_provider_checkout},
    )
    gold = row["ground_truth"][0]
    calls = [
        ('{"email_id":"00000057","unknown":"x"}', "bad"),
        (gold["arguments"], "good"),
    ]
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) <= len(calls):
                arguments, label = calls[len(requests) - 1]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{label}",
                            "type": "function",
                            "function": {"name": gold["name"], "arguments": arguments},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Done."}
            _send_completion(self, message)

    with _serve_endpoint(Endpoint) as server:
        result = await run_trial(
            task_dir,
            binding,
            ChatLaunch(
                model="fixture",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                temperature=1.0,
                parallel_tool_calls=False,
                max_turns=4,
            ),
            tmp_path / "trials",
            "workplace",
        )
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(requests) == 3
    assert len(requests[0]["tools"]) == 27
    assert all(request["temperature"] == 1.0 and request["parallel_tool_calls"] is False for request in requests)
    assert requests[1]["messages"][-1]["tool_call_id"] == "call-bad"
    assert requests[2]["messages"][-1]["tool_call_id"] == "call-good"
    assert "ground_truth" not in json.dumps(requests)
    metadata = result.agent_result.metadata
    assert len(metadata["tool_definitions"]) == 27
    assert [action["call_id"] for action in metadata["tools"]] == ["call-bad", "call-good"]
    assert [action["observation"] for action in metadata["tools"]] == [
        requests[1]["messages"][-1]["content"],
        requests[2]["messages"][-1]["content"],
    ]
    assert [message["role"] for message in metadata["all_messages"]] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_workplace_chat_tools_can_answer_text_from_observation(
    tmp_path, trusted_provider_checkout, source_row: bytes, provider_source
):
    imported, _, binding = import_row(source_row, provider_source)
    subject = "Task Update on Develop prototype for report generation"
    specification = TaskSpec(
        id="workplace-subject-answer",
        context=ConversationInput(
            events=(TextMessage(role="user", content="Use the available tools to find the subject of email 00000057."),)
        ),
        verifier=exact_answer(subject),
        source=imported.source,
        environment_requirements=imported.environment_requirements,
        tool_providers=imported.tool_providers,
        answer_type=AnswerType.TEXT,
    )
    convention = PlainText(id="plain")
    task_dir = lower_to_harbor(
        specification,
        convention,
        binding,
        tmp_path / "task",
        trusted_provider_sources={"workplace": trusted_provider_checkout},
    )
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-subject",
                            "type": "function",
                            "function": {
                                "name": "email_get_email_information_by_id",
                                "arguments": '{"email_id":"00000057","field":"subject"}',
                            },
                        }
                    ],
                }
            else:
                observation = json.loads(payload["messages"][-1]["content"])
                message = {"role": "assistant", "content": observation["output"]["subject"]}
            _send_completion(self, message)

    with _serve_endpoint(Endpoint) as server:
        result = await run_trial(
            task_dir,
            binding,
            ChatLaunch(
                model="fixture",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                temperature=1.0,
                parallel_tool_calls=False,
                max_turns=2,
            ),
            tmp_path / "trials",
            "subject",
        )
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(requests) == 2
    assert len(requests[0]["tools"]) == 27
    assert requests[1]["messages"][-1]["tool_call_id"] == "call-subject"
    assert result.agent_result.metadata["assistant_final"]["content"] == subject


async def test_workplace_harbor_trials_are_fresh_and_concurrent(
    tmp_path, trusted_provider_checkout, source_row: bytes, provider_source
):
    specification, convention, binding = import_row(source_row, provider_source)
    task_dir = lower_to_harbor(
        specification,
        convention,
        binding,
        tmp_path / "task",
        trusted_provider_sources={"workplace": trusted_provider_checkout},
    )
    gold = json.loads(source_row)["ground_truth"][0]
    turns: dict[str, int] = {}
    lock = Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = payload["model"]
            with lock:
                turns[model] = turns.get(model, 0) + 1
                turn = turns[model]
            if turn == 1:
                if model.startswith("good"):
                    name, arguments = gold["name"], gold["arguments"]
                else:
                    name, arguments = "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}'
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"{model}-call",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Done."}
            _send_completion(self, message)

    with _serve_endpoint(Endpoint) as server:

        async def trial(model: str):
            launch = ChatLaunch(
                model=model,
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                temperature=1.0,
                parallel_tool_calls=False,
                max_turns=3,
            )
            return await run_trial(task_dir, binding, launch, tmp_path / "trials", model)

        good, noop = await asyncio.gather(trial("good-1"), trial("noop"))
        good_again = await trial("good-2")

    assert all(result.exception_info is None for result in (good, noop, good_again))
    assert [result.verifier_result.rewards for result in (good, noop, good_again)] == [
        {"reward": 1.0},
        {"reward": 0.0},
        {"reward": 1.0},
    ]


async def test_demonstration_gate_grades_reconstructed_workplace_and_answer_rows(
    tmp_path, source_row: bytes, provider_source, trusted_provider_checkout
):
    imported = import_row(source_row, provider_source)
    ready = tmp_path / "ready"
    data = ready / "data"
    (data / "workplace").mkdir(parents=True)
    (data / "tasktrove_clean").mkdir()
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_bytes(source_row)
        row = PublishedRow(
            **imported.specification.model_copy(
                update={
                    "source": imported.specification.source.model_copy(
                        update={"row": f"{split}:0:{hashlib.sha256(source_row).hexdigest()}"}
                    )
                }
            ).model_dump(),
            source_category=json.loads(source_row)["category"],
            provenance=SourceProof(
                source_row=imported.specification.source.row,
                input_file=f"{split}.jsonl",
                input_object_pin="sha256:" + hashlib.sha256(source_row).hexdigest(),
            ),
        )
        (data / "workplace" / f"{split}.jsonl").write_text(row.model_dump_json() + "\n")
    for cohort, verifier in (
        ("mcqa", multiple_choice_answer("B", 3)),
        ("prism_math", mathematical_answer("1/2", MathType.SCALAR)),
    ):
        task = imported.specification.model_copy(
            update={"id": cohort, "tool_providers": {}, "answer_type": AnswerType.TEXT, "verifier": verifier}
        )
        row = PublishedRow(
            **task.model_dump(),
            provenance=SourceProof(
                source_row=task.source.row, input_file="source.parquet", input_object_pin="sha256:" + "a" * 64
            ),
        )
        (data / "tasktrove_clean" / f"{cohort}.jsonl").write_text(row.model_dump_json() + "\n")
    report = await audit_demonstration(ready, tmp_path, provider_source, trusted_provider_checkout)
    assert report["workplace_expected_state_matches"] == {"train": 1, "validation": 1}
    assert report["harbor_rewards"] == {
        "workplace/train": [1.0, 0.0],
        "workplace/validation": [1.0, 0.0],
        "tasktrove_clean/mcqa": [1.0, 0.0],
        "tasktrove_clean/prism_math": [1.0, 0.0],
    }


@pytest.mark.parametrize("corrupted", [True, 1.0])
async def test_demo_audit_rejects_boolean_and_float_state_coercion(
    monkeypatch, tmp_path, synthetic_workplace_source, corrupted
):
    provider_source, source_row, trusted_provider_checkout, locator = synthetic_workplace_source
    monkeypatch.setattr(release_audit, "PROVIDER", locator)
    imported = import_row(source_row, provider_source)
    digest = hashlib.sha256(source_row).hexdigest()
    task = imported.specification.model_copy(
        update={
            "source": imported.specification.source.model_copy(update={"row": f"train:0:{digest}"}),
            "verifier": structured_exact({"value": corrupted}),
        }
    )
    row = PublishedRow(
        **task.model_dump(),
        provenance=SourceProof(
            source_row=task.source.row, input_file="train.jsonl", input_object_pin="sha256:" + digest
        ),
    )
    ready = tmp_path / "ready"
    (ready / "data/workplace").mkdir(parents=True)
    (ready / "data/workplace/train.jsonl").write_text(row.model_dump_json() + "\n")
    (tmp_path / "train.jsonl").write_bytes(source_row)
    with pytest.raises(ValueError, match="source canonical state"):
        await audit_demonstration(ready, tmp_path, provider_source, trusted_provider_checkout)


@pytest.fixture
def synthetic_workplace_source(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    package = checkout / "src" / "synthetic_workplace"
    package.mkdir(parents=True)
    schema = {"name": "finish", "parameters": {"type": "object"}}
    definitions = [{"type": "function", "function": schema}]
    (package / "__init__.py").write_text(
        "import importlib\nimport json\nfrom pathlib import Path\n"
        "ACTION_INTERFACE = 'synthetic:v1'\nSEED_SHA256 = 'a' * 64\n"
        "PROVIDER_REVISION = 'synthetic-v1'\nTOOLS_SHA256 = 'b' * 64\n"
        f"TOOL_DEFINITIONS = {definitions!r}\n"
        "REQUEST_PARALLEL_TOOL_CALLS = False\nREQUEST_TEMPERATURE = 0\n"
        "class ToolProvider: pass\n"
        "def _seed_digest(): return SEED_SHA256\n"
        f"def get_tools(): return {{'schemas': [{schema!r}]}}\n"
        "def expected_state_json(actions):\n"
        "    assert importlib.import_module(__name__) is not None\n"
        "    return json.dumps({'value': int(Path(__file__).is_file())})\n"
    )
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "https://github.com/example/synthetic"], check=True
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Synthetic provider",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    locator = f"python+git+https://github.com/example/synthetic@{revision}:synthetic_workplace:ToolProvider"
    monkeypatch.setattr(nemo_workplace, "PROVIDER", locator)
    source = tmp_path / "source"
    stage_git_provider(locator, checkout, source)
    row = json.dumps(
        {
            "id": 0,
            "environment_name": "workplace_assistant",
            "category": "workplace_assistant_calendar",
            "responses_create_params": {
                "input": [{"role": "user", "content": "Finish the task."}],
                "tools": [schema],
                "parallel_tool_calls": False,
                "temperature": 0,
            },
            "ground_truth": [],
        }
    ).encode()
    monkeypatch.setattr(nemo_workplace, "ROW_SHA256_BY_ID", {0: hashlib.sha256(row).hexdigest()})
    return source, row, checkout, locator


def test_import_retains_provider_source_until_expected_state_is_built(synthetic_workplace_source):
    source, row, _, locator = synthetic_workplace_source
    before = set(sys.modules)
    imported = import_row(row, source)
    assert json.loads(imported.specification.verifier.parameters_json)["expected"] == {"value": 1}
    assert imported.environment_config.tool_providers["workplace"].provider == locator
    assert not any(name.startswith("_taskcompendium_providers_") for name in set(sys.modules) - before)
