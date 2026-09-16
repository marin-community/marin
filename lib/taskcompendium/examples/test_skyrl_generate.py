# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral coverage using pinned SkyRL token helpers and native Harbor trials."""

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from tasktrove_verify.spec import Mode
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from taskcompendium.execution import HarborExecutionConfig, HarborLaunchConfig, HarborTaskBinding, NoEnvironment
from taskcompendium.importers.nemo_predicted_action import import_row, rendering, replay_action
from taskcompendium.importers.nemo_workplace_multistep import build_multistep_sample
from taskcompendium.importers.sequential import sentence_revision_task
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AssistantFinal,
    BoxedLatex,
    JsonPath,
    Outcome,
    Rendering,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    TaskTroveVerifier,
    XmlPath,
)
from taskcompendium.skyrl import TaskCompendiumTrajectoryRunner, UngradedBatchError, request_batch

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/nemo"


class CrashingVerifier(BaseVerifier):
    """Represent an unavailable verifier service at Harbor's plugin boundary."""

    async def verify(self):
        raise RuntimeError("fixture verifier service unavailable")


class MalformedDiagnosticVerifier(BaseVerifier):
    """Represent a truncated verifier artifact at the filesystem boundary."""

    async def verify(self):
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "taskcompendium-result.json").write_text('{"status":')
        return VerifierResult(rewards={"reward": 1.0})


@pytest.fixture(scope="session")
def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["hello answer observation next done task user assistant tool system"],
        trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<|unk|>", "<|eos|>", "<|pad|>", "<|start|>", "<|tools|>"],
        ),
    )
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<|unk|>", eos_token="<|eos|>", pad_token="<|pad|>"
    )
    result.chat_template = (
        "{% if tools %}{{ '<|tools|>' }}{{ tools | tojson }}{% endif %}"
        "{% for message in messages %}"
        "{{ '<|start|>' + message.role + '\\n' }}"
        "{% if message.content is string %}{{ message.content }}"
        "{% elif message.content is defined and message.content is not none %}{{ message.content | tojson }}{% endif %}"
        "{% if message.tool_calls is defined %}{{ message.tool_calls | tojson }}{% endif %}"
        "{{ '<|eos|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start|>assistant\\n' }}{% endif %}"
    )
    return result


def _math_specification():
    return TaskSpec(
        id="consumer/fraction",
        requirements=TaskRequirements(),
        resources=(),
        metadata=TaskMetadata(Source("consumer-fixture", "1", "fraction", "1")),
        steps=(
            StepSpecification(
                instructions="Compute three quarters as a fraction.",
                verifier=TaskTroveVerifier(Mode.MATH, {"expected": "3/4"}),
            ),
        ),
    )


def _row(root, specification, renderings, *, agent="replay", agent_kwargs=None, binding=None):
    binding = binding or HarborTaskBinding(NoEnvironment())
    task = lower_to_harbor(
        specification,
        renderings,
        binding,
        root,
        reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig(agent)),
        agent_kwargs=agent_kwargs,
        model_name="scripted-model" if agent != "replay" else None,
    )
    return {
        "uid": root.name,
        "task_dir": str(task),
        "execution": json.loads((task / "reference-execution.json").read_text()),
    }


def _archive(output):
    archives = list(output.glob("attempts-*.jsonl"))
    assert len(archives) == 1
    return [json.loads(line) for line in archives[0].read_text().splitlines()]


def _active_text(tokenizer, batch, index):
    return tokenizer.decode(
        [token for token, active in zip(batch["response_ids"][index], batch["loss_masks"][index], strict=True) if active]
    )


@contextmanager
def _endpoint(response):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            body = json.dumps({"choices": [{"message": response(payload)}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_mixed_renderings_preserve_rewards_identity_archives_and_reconstructed_tokens(tmp_path, tokenizer):
    specification = _math_specification()
    formats = [
        ("plain", AssistantFinal(), "3/4"),
        ("json", AssistantFinal(JsonPath()), '{"answer":"3/4"}'),
        ("xml", AssistantFinal(XmlPath()), "<answer>3/4</answer>"),
        ("boxed", AssistantFinal(BoxedLatex()), r"\boxed{\frac{3}{4}}"),
        ("wrong", AssistantFinal(), "1/2"),
    ]
    rows = [
        _row(tmp_path / name, specification, (Rendering(name, submission),), agent_kwargs={"response": answer})
        for name, submission, answer in formats
    ]
    output = tmp_path / "output"
    runner = TaskCompendiumTrajectoryRunner(tokenizer, output, concurrency=2)
    request = request_batch(rows, repetitions=2)
    batch = await runner.run(request)
    archived = _archive(output)

    expected_ids = [(name, repetition) for name, _, _ in formats for repetition in range(2)]
    assert [(identity.instance_id, identity.repetition_id) for identity in batch["trajectory_ids"]] == expected_ids
    assert [(attempt["instance_id"], attempt["repetition_id"]) for attempt in archived] == expected_ids
    assert batch["rewards"] == batch["unshaped_rewards"] == [1.0] * 8 + [0.0] * 2
    assert [attempt["reward"] for attempt in archived] == batch["rewards"]
    assert all(attempt["status"] == Outcome.GRADED for attempt in archived)
    assert len({attempt["trial_dir"] for attempt in archived}) == 10
    assert len({attempt["specification_sha256"] for attempt in archived}) == 1
    assert all(attempt["source"]["dataset"] == "consumer-fixture" for attempt in archived)
    assert all(attempt["token_provenance"] == "reconstructed" for attempt in archived)
    assert batch["rollout_logprobs"] is None
    assert batch["rollout_routed_experts"] is None
    for index, attempt in enumerate(archived):
        ids, mask = batch["response_ids"][index], batch["loss_masks"][index]
        assert len(ids) == len(mask) and sum(mask) > 0
        assert attempt["messages"][-1]["content"] in _active_text(tokenizer, batch, index)
        assert Path(attempt["trial_dir"], "verifier/taskcompendium-result.json").is_file()


async def test_ordered_chat_retains_prior_conversation_and_excludes_user_tokens(tmp_path, tokenizer):
    def response(payload):
        messages = payload["messages"]
        if "Thursday" in messages[-1]["content"]:
            previous = next(message["content"] for message in reversed(messages[:-1]) if message["role"] == "assistant")
            return {"role": "assistant", "content": previous.replace("Tuesday", "Thursday")}
        return {"role": "assistant", "content": "Mira will meet Leo on Tuesday."}

    with _endpoint(response) as (endpoint, requests):
        row = _row(
            tmp_path / "ordered",
            sentence_revision_task(),
            (Rendering("first", AssistantFinal()), Rendering("revision", AssistantFinal())),
            agent="chat",
            agent_kwargs={"api_base": endpoint},
        )
        runner = TaskCompendiumTrajectoryRunner(tokenizer, tmp_path / "output", concurrency=2)
        batch = await runner.run(request_batch([row], repetitions=2))

    assert batch["rewards"] == [1.0, 1.0]
    assert all([step.reward for step in attempt.step_results] == [1.0, 1.0] for attempt in runner.last_attempts)
    revisions = [request for request in requests if "Thursday" in request["messages"][-1]["content"]]
    assert len(revisions) == 2
    assert all(request["messages"][-2]["content"] == "Mira will meet Leo on Tuesday." for request in revisions)
    for index, attempt in enumerate(_archive(tmp_path / "output")):
        assert [message["role"] for message in attempt["messages"]] == ["user", "assistant", "user", "assistant"]
        full_text = tokenizer.decode(batch["response_ids"][index])
        active_text = _active_text(tokenizer, batch, index)
        assert "Revise your previous sentence" in full_text
        assert "Revise your previous sentence" not in active_text
        assert "Mira will meet Leo on Tuesday." in active_text
        assert "Mira will meet Leo on Thursday." in active_text


async def test_native_action_replay_retains_function_call_as_trainable_output(tmp_path, tokenizer):
    source = json.loads((FIXTURES / "predicted-action.json").read_text())
    provenance = json.loads((FIXTURES / "predicted-action.provenance.json").read_text())
    expected = source["expected_action"]
    row = _row(
        tmp_path / "action",
        import_row(source, provenance["canonical_json_sha256"]),
        (rendering(source, provenance["canonical_json_sha256"]),),
        agent_kwargs={"actions": [replay_action(expected["name"], expected["arguments"])]},
    )
    runner = TaskCompendiumTrajectoryRunner(tokenizer, tmp_path / "output", concurrency=1)
    batch = await runner.run(request_batch([row], repetitions=2))
    attempts = _archive(tmp_path / "output")

    assert batch["rewards"] == [1.0, 1.0]
    for index, attempt in enumerate(attempts):
        action = attempt["messages"][-1]["tool_calls"][0]["function"]
        assert action["name"] == expected["name"]
        assert json.loads(action["arguments"]) == json.loads(expected["arguments"])
        assert expected["name"] in _active_text(tokenizer, batch, index)
        prompt_text = tokenizer.decode(batch["prompt_token_ids"][index])
        assert "<|tools|>" in prompt_text
        assert expected["name"] in prompt_text
        assert '"parameters"' in prompt_text
        assert not any(message["role"] == "tool" for message in attempt["messages"])
    assert batch["rollout_logprobs"] is None


async def test_workplace_provider_retains_state_history_and_masks_observations(tmp_path, tokenizer):
    sample = build_multistep_sample(FIXTURES)
    responses = []
    for step_index, calls in enumerate(sample.all_good):
        responses.extend(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"step-{step_index}-call-{call_index}",
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                ],
            }
            for call_index, call in enumerate(calls)
        )
        responses.append({"role": "assistant", "content": "Completed."})

    def response(payload):
        turn = sum(message["role"] == "assistant" for message in payload["messages"])
        return responses[turn]

    with _endpoint(response) as (endpoint, requests):
        row = _row(
            tmp_path / "workplace",
            sample.specification,
            sample.renderings,
            binding=sample.binding,
            agent="provider_chat",
            agent_kwargs={"api_base": endpoint, "max_turns": 4},
        )
        runner = TaskCompendiumTrajectoryRunner(tokenizer, tmp_path / "output", concurrency=2)
        batch = await runner.run(request_batch([row], repetitions=2))

    assert batch["rewards"] == [1.0, 1.0]
    attempts = _archive(tmp_path / "output")
    assert all([step["reward"] for step in attempt["step_results"]] == [1.0, 1.0, 1.0] for attempt in attempts)
    for index, attempt in enumerate(attempts):
        calls = [call for message in attempt["messages"] for call in message.get("tool_calls", [])]
        assert [call["function"]["name"] for call in calls] == [call.name for step in sample.all_good for call in step]
        assert len([message for message in attempt["messages"] if message["role"] == "user"]) == 3
        active_text = _active_text(tokenizer, batch, index)
        assert "email_reply_email" in active_text and "project_management_update_task" in active_text
        assert "The prototype work is underway" not in active_text
        assert "The prototype work is underway" in tokenizer.decode(batch["response_ids"][index])
        observed = [message for message in attempt["messages"] if message["role"] == "tool"]
        assert len(observed) == 5
        full_text = tokenizer.decode(batch["response_ids"][index])
        assert "Email replied successfully." in full_text
        assert "Task updated successfully." in full_text
        assert "Email replied successfully." not in active_text
        assert "Task updated successfully." not in active_text
        advertised = {tool["function"]["name"] for tool in attempt["tools"]}
        assert all(call["function"]["name"] in advertised for call in calls)
        assert "<|tools|>" in tokenizer.decode(batch["prompt_token_ids"][index])
        assert "<|tools|>" not in active_text
    assert any(any(message["role"] == "tool" for message in request["messages"]) for request in requests)
    assert batch["rollout_logprobs"] is None


async def test_extraction_error_is_zeroed_for_training_but_retained_as_null_semantic_reward(tmp_path, tokenizer):
    row = _row(
        tmp_path / "malformed",
        _math_specification(),
        (Rendering("json", AssistantFinal(JsonPath())),),
        agent_kwargs={"response": "not JSON"},
    )
    durable = tmp_path / "durable"
    runner = TaskCompendiumTrajectoryRunner(
        tokenizer,
        tmp_path / "output",
        concurrency=1,
        archive_uri=durable.as_uri(),
    )
    batch = await runner.run(request_batch([row], repetitions=2))
    attempts = _archive(tmp_path / "output")

    assert batch["rewards"] == [0.0, 0.0]
    assert batch["exception_types"] == [Outcome.EXTRACTION_ERROR, Outcome.EXTRACTION_ERROR]
    assert batch["error_treatments"] == ["zero", "zero"]
    assert [attempt["status"] for attempt in attempts] == [Outcome.EXTRACTION_ERROR, Outcome.EXTRACTION_ERROR]
    assert [attempt["reward"] for attempt in attempts] == [None, None]
    durable_archives = list(durable.glob("attempts-*.jsonl"))
    assert len(durable_archives) == 1
    assert [json.loads(line)["reward"] for line in durable_archives[0].read_text().splitlines()] == [None, None]


@pytest.mark.parametrize("failure", ["verifier_crash", "malformed_artifact"])
async def test_infrastructure_failures_retain_null_rewards_without_numeric_batch(tmp_path, tokenizer, failure):
    row = _row(
        tmp_path / failure,
        _math_specification(),
        (Rendering("json", AssistantFinal(JsonPath())),),
        agent_kwargs={"response": '{"answer":"3/4"}'},
    )
    if failure == "verifier_crash":
        row["execution"]["verifier"]["import_path"] = "test_skyrl_generate:CrashingVerifier"
    elif failure == "malformed_artifact":
        row["execution"]["verifier"]["import_path"] = "test_skyrl_generate:MalformedDiagnosticVerifier"
    graded = _row(
        tmp_path / "graded",
        _math_specification(),
        (Rendering("plain", AssistantFinal()),),
        agent_kwargs={"response": "3/4"},
    )
    runner = TaskCompendiumTrajectoryRunner(tokenizer, tmp_path / "output", concurrency=2)
    with pytest.raises(UngradedBatchError):
        await runner.run(request_batch([graded, row], repetitions=2))
    attempts = _archive(tmp_path / "output")

    assert [attempt["status"] for attempt in attempts] == [
        Outcome.GRADED,
        Outcome.GRADED,
        Outcome.INFRA_ERROR,
        Outcome.INFRA_ERROR,
    ]
    assert [attempt["reward"] for attempt in attempts] == [1.0, 1.0, None, None]
    assert all(attempt["exception"] is not None for attempt in attempts[2:])
    assert all(attempt["messages"][-1]["role"] == "assistant" for attempt in attempts)
    assert [(attempt["instance_id"], attempt["repetition_id"]) for attempt in attempts] == [
        ("graded", 0),
        ("graded", 1),
        (failure, 0),
        (failure, 1),
    ]
