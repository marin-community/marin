# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A TaskCompendium run that reads a growing task feed and writes each graded group back to it."""

import asyncio
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.models import (
    ConversationRecord,
    GradedRollout,
    ModelTurn,
    Rollout,
    RolloutFailure,
    RolloutStep,
    Transition,
)
from shellbox.machine import CommandSemantics, PackageSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import ExactSpec

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from marinskyrl import task_machine_factories as machine_factories
from skyrl_train.dataset.task_feed import (
    FeedTaskDataset,
    GradedGroup,
    graded_group,
    lower_feed_task,
    write_graded_group,
)
from skyrl_train.rollouts.buffer import BatchPolicy, RolloutBufferConfig, RolloutLease, RolloutTask
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.rollouts.loader import FeedPromptLoader
from skyrl_train.rollouts.payloads import MemoryPayloads
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.trajectory_runners.types import TokenProvenance

SESSION = {
    "max_turns": 2,
    "max_total_tokens": None,
    "model_turn_timeout": None,
    "tool_turn_timeout": 600,
    "total_turn_timeout": None,
    "attempt_timeout": None,
    "verifier_timeout": None,
    "cleanup_timeout": 30,
    "command_timeout": 60,
    "shell_tool": {"name": "shell", "command_parameter": "command"},
}
SCRIPT_GRADER = ScriptGrader(
    argv=("python3", "/tests/grade.py"),
    environment=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, package_spec=PackageSpec()),
)


class CharTokenizer:
    """One token for each four characters of the rendered conversation."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **options):
        self.calls.append(options)
        text = json.dumps(messages) + json.dumps(options.get("tools"))
        return list(range(len(text) // 4))


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_task(content="Say yes.", *, events=None, grader=None, requirements=None) -> TaskSpec:
    return TaskSpec(
        id="feed-task",
        context=ConversationInput(events=events or (TextMessage(role="user", content=content),)),
        environment_requirements=requirements or EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader or verifyit_package(ExactSpec(expected=("yes",))).grader,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def assistant_prefix_task() -> TaskSpec:
    return make_task(
        events=(TextMessage(role="user", content="Close it."), TextMessage(role="assistant", content="[1, [2")),
        grader=verifyit_package(ExactSpec(expected=("]]",))).grader,
    )


def write_chunk(root: Path, first: int, tasks: list[str]) -> None:
    path = root / "chunks" / f"{first:012d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    seqs = pa.array(range(first, first + len(tasks)), pa.int64())
    pq.write_table(pa.table({"seq": seqs, "task": pa.array(tasks, pa.string())}), path)


def feed(root: Path, *, min_rows=0, clock=None, sleep=None, max_prompt_length=10_000) -> FeedTaskDataset:
    return FeedTaskDataset(
        str(root),
        CharTokenizer(),
        max_prompt_length,
        session=SESSION,
        poll_seconds=10,
        min_rows=min_rows,
        clock=clock or Clock(),
        sleep=sleep or (lambda seconds: pytest.fail("the feed waited")),
    )


def uids(loader: FeedPromptLoader, live=()) -> list[str]:
    offered = []
    while (prompt := loader.next_prompt(live)) is not None:
        offered.append(prompt["uid"])
    return offered


def test_feed_offers_each_row_once_in_order_and_sees_rows_after_start(tmp_path):
    write_chunk(tmp_path, 0, [make_task().model_dump_json()] * 3)
    clock = Clock()
    loader = FeedPromptLoader(feed(tmp_path, min_rows=3, clock=clock))
    first = loader.next_prompt(())
    assert first["uid"] == "0" and first["input"].spec.harness.startswith("shellbox:")
    loader.retry(first)
    assert uids(loader, live={"0"}) == ["1", "2"]
    assert uids(loader) == ["0"]

    write_chunk(tmp_path, 3, [make_task().model_dump_json()] * 2)
    asyncio.run(loader.refresh())
    assert uids(loader) == [], "polls at most once each poll_seconds"
    clock.now = 10
    asyncio.run(loader.refresh())
    assert uids(loader) == ["3", "4"]
    assert loader.observe(()) == {"feed/backlog": 0.0}


def test_feed_waits_for_min_rows(tmp_path):
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        write_chunk(tmp_path, 0, [make_task().model_dump_json()] * 2)

    assert len(feed(tmp_path, min_rows=2, sleep=sleep)) == 2
    assert waits == [10]


def test_bad_rows_are_skipped_and_resume_continues_at_the_next_seq(tmp_path):
    good = make_task().model_dump_json()
    long = make_task("x" * 4000).model_dump_json()
    write_chunk(tmp_path, 0, [good, "{not a task", good, long, good])
    loader = FeedPromptLoader(feed(tmp_path, max_prompt_length=500))
    assert [loader.next_prompt(())["uid"] for _ in range(2)] == ["0", "2"]
    skipped = {path.name: json.loads(path.read_text()) for path in (tmp_path / "skipped").iterdir()}
    assert set(skipped) == {"000000000001.json", "000000000003.json"}
    assert skipped["000000000001.json"]["uid"] == "1" and "cannot lower" in skipped["000000000001.json"]["reason"]
    assert "more than 500" in skipped["000000000003.json"]["reason"]

    state = loader.state_dict()
    assert state.order == {"next_seq": 4}
    resumed = FeedPromptLoader(feed(tmp_path, max_prompt_length=500))
    resumed.load_state_dict(state)
    assert uids(resumed) == ["4"]


def test_dispatch_wakes_on_a_timer_while_the_feed_is_empty(tmp_path):
    async def scenario():
        dataset = feed(tmp_path, clock=asyncio.get_running_loop().time)
        dataset.poll_seconds = 0.05
        buffer_config = RolloutBufferConfig(8, None, 1, BatchPolicy.FULL_BATCH, None, None)
        context = TrainingContext(
            FeedPromptLoader(dataset),
            buffer_config,
            object(),
            None,
            None,
            None,
            MemoryPayloads(),
            rollout_spans=False,
            prompt_wait=0.05,
        )
        waiting = asyncio.create_task(context._next_prompt())
        await asyncio.sleep(0.2)
        assert not waiting.done()
        write_chunk(tmp_path, 0, [make_task().model_dump_json()])
        prompt = await asyncio.wait_for(waiting, 2)
        assert prompt["uid"] == "0" and context._live_uids["0"] == 1

    asyncio.run(scenario())


def test_graded_group_round_trip_strips_token_data(tmp_path):
    turn = ModelTurn(
        message={"role": "assistant", "content": "yes"},
        prompt_token_ids=(1, 2),
        response_token_ids=(3,),
        logprobs=(-0.1,),
        stop_reason="stop",
    )
    messages = ({"role": "user", "content": "q"}, turn.message)
    step = RolloutStep(turn, Transition(done=True), 0, messages[:1])
    conversation = ConversationRecord(messages, (1, 2), (3,), (1,), (-0.1,), (step,))
    failure = RolloutFailure("TrialTimeoutError", {"operation": "attempt"})
    rollout = Rollout("7_0", (conversation,), "error", {"turns": 1}, {"/app/out": b"\xff\x00"}, failure)
    graded = GradedRollout(rollout, GradeResult(Outcome.GRADED, 1.0))
    write_graded_group(str(tmp_path), graded_group("7", 3, [graded]))

    raw = json.loads((tmp_path / "graded/000000000007.json").read_text())
    assert raw["uid"] == "7" and raw["policy_step"] == 3
    (back,) = GradedGroup.model_validate_json(json.dumps(raw)).rollouts
    (kept,) = back.rollout.conversations
    assert (kept.prompt_token_ids, kept.response_token_ids, kept.loss_mask, kept.logprobs, kept.steps) == (
        (),
        (),
        (),
        None,
        (),
    )
    assert kept.messages == messages
    assert back.grade == graded.grade and back.rollout.failure == failure
    assert back.rollout.metrics == {"turns": 1} and back.rollout.outputs == rollout.outputs


@pytest.mark.parametrize(
    "task,harness,task_backend,verifier_backend",
    [
        (assistant_prefix_task(), ASSISTANT_PREFIX_HARNESS, None, None),
        (make_task(), "shellbox:", None, None),
        (make_task(grader=SCRIPT_GRADER), "shellbox:", None, "local"),
        (
            make_task(requirements=EnvironmentRequirements(capabilities=("files",), command_semantics="linux_process")),
            "shellbox:",
            "local",
            None,
        ),
        (
            make_task(
                requirements=EnvironmentRequirements(setup_commands=("touch a",), command_semantics="shell_simulator")
            ),
            "shellbox:",
            "shellsim",
            None,
        ),
    ],
)
def test_lowering_rules(task, harness, task_backend, verifier_backend):
    lowered = lower_feed_task(task, SESSION)
    assert lowered.harness.startswith(harness)
    assert (lowered.task_machine and lowered.task_machine.backend) == task_backend
    assert (lowered.verifier_machine and lowered.verifier_machine.backend) == verifier_backend
    assert all(machine.network == "deny" for machine in (lowered.task_machine, lowered.verifier_machine) if machine)
    assert lowered.limits.max_turns == SESSION["max_turns"] and lowered.limits.cleanup_timeout == 30


class FakeWriter:
    def __init__(self):
        self.written, self.skipped = [], []

    async def write_rollout(self, lease, group):
        self.written.append(group)

    async def skip_rollout(self, lease, prompt):
        self.skipped.append(prompt["uid"])


class FakeModel:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return {
            "responses": ["]]"],
            "assistant_messages": [{"role": "assistant", "content": "]]"}],
            "prompt_ids": [[1, 2, 3]],
            "response_ids": [[7, 8]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.5, -0.5]],
            "token_provenance": TokenProvenance.ENGINE,
        }


def feed_worker(tmp_path, tokenizer, model) -> TaskRolloutWorker:
    recipe = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml").read_text())
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    topology = LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    generator = compose_skyrl_config(parse_rl_config(str(path)), {}, topology).config.generator
    return TaskRolloutWorker(
        generator,
        WholeTaskProjection(generator, tokenizer),
        model,
        {},
        shutdown_timeout=1.0,
        feed_root=str(tmp_path),
    )


def feed_task(tmp_path, task: TaskSpec, samples: int) -> RolloutTask:
    write_chunk(tmp_path, 0, [task.model_dump_json()])
    prompt = FeedPromptLoader(feed(tmp_path)).next_prompt(())
    request, _ = prepare_trajectory_request([prompt], samples, {}, "train", 5)
    return RolloutTask(RolloutLease("lease", 5, 5), prompt, request)


def test_worker_skips_a_row_that_fails_the_engine_check(tmp_path):
    # No factory serves the script grader's bubblewrap machine.
    task = feed_task(tmp_path, make_task(grader=SCRIPT_GRADER), 2)
    writer = FakeWriter()
    assert asyncio.run(feed_worker(tmp_path, CharTokenizer(), FakeModel()).run_task(task, writer)) == 0
    assert writer.skipped == ["0"] and not writer.written
    skipped = json.loads((tmp_path / "skipped/000000000000.json").read_text())
    assert skipped["uid"] == "0" and skipped["reason"]
    assert not (tmp_path / "graded").exists()


def test_worker_continues_an_assistant_prefix_and_writes_the_group_before_selection(tmp_path):
    task = feed_task(tmp_path, assistant_prefix_task(), 2)
    tokenizer, model, writer = CharTokenizer(), FakeModel(), FakeWriter()
    worker = feed_worker(tmp_path, tokenizer, model)

    async def training_batch(request, rollouts):
        return {"response_ids": [list(rollout.response_token_ids) for rollout in rollouts]}

    worker.training_batch = training_batch
    assert asyncio.run(worker.run_task(task, writer)) == 4
    assert len(model.requests) == 2
    for request in model.requests:
        (options,) = request["chat_completion_params"]
        assert options["extra_body"] == {"continue_final_message": True, "add_generation_prompt": False}
    group = GradedGroup.model_validate_json((tmp_path / "graded/000000000000.json").read_text())
    assert group.uid == "0" and group.policy_step == 5 and len(group.rollouts) == 2
    for graded in group.rollouts:
        assert graded.grade.status == Outcome.GRADED and graded.grade.reward == 1.0
        (conversation,) = graded.rollout.conversations
        assert conversation.messages[-1]["content"] == "]]" and conversation.response_token_ids == ()
    assert [group.uid for group in writer.written] == ["0"] and not writer.skipped


def test_local_settings_mount_the_host_python_and_bound_open_machines(tmp_path, monkeypatch):
    created = []

    class Machine:
        async def close(self):
            pass

    class LocalFactory:
        backend = "local"

        def __init__(self, *, read_only, bin_dirs):
            created.append((read_only, bin_dirs))

        async def create(self, spec):
            return Machine()

    monkeypatch.setattr(machine_factories, "LocalMachineFactory", LocalFactory)
    settings = {"read_only": [str(tmp_path)], "python_environment": "host", "max_machines": 1}
    factory = machine_factories.local_machine_factory(settings)
    ((read_only, bin_dirs),) = created
    assert read_only[0] == tmp_path and Path(sys.prefix) in read_only
    assert bin_dirs == (Path(sys.prefix) / "bin",)

    async def bounded():
        machine = await factory.create(None)
        second = asyncio.create_task(factory.create(None))
        await asyncio.sleep(0.05)
        assert not second.done()
        await machine.close()
        await asyncio.wait_for(second, 1)

    asyncio.run(bounded())
    with pytest.raises(ValueError, match="Unknown"):
        machine_factories.local_machine_factory({"mounts": []})
