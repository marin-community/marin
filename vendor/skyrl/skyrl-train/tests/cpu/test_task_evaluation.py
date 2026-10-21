# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete evaluations preserve graders and bound the shared verifier account."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from taskcompendium.models import ScriptGrader, ConversationInput, TextMessage
from skyrl_gym.verification import VerificationResult
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import write_tasks
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl_train.evaluate import evaluate, evaluate_step_wise, evaluation_session
from skyrl_train.rollouts.buffer import write_trajectory_batch
from skyrl_train.rollouts.finalization import finalize_trajectory_batch
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    parse_trajectory_retention_config,
    execute_publication,
)
from skyrl_train.trajectory_runners.trajectory_retention_publisher import InlineTrajectoryPublisher
from skyrl_train.trajectory_runners.types import TrajectoryBatch
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, text_task, lowered_task, task_input


@pytest.fixture
def dummy_config():
    return get_default_config()


class FixtureRunner:
    trajectory_runner_cfg = {}
    trajectory_sink = None

    async def run(self, input_batch) -> TrajectoryBatch:
        output = await self._run(input_batch)
        return await finalize_trajectory_batch(input_batch, output, self.trajectory_runner_cfg, self.trajectory_sink)

    async def run_task(self, task, writer) -> int:
        output = await self.run(task.request)
        return await write_trajectory_batch(task, writer, output)

    def set_trajectory_sink(self, sink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def _run(self, input_batch) -> TrajectoryBatch:
        raise NotImplementedError

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None:
        pass

    async def stop_eval_session(self) -> None:
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def configure_eval(cfg, tmp_path):
    cfg.generator.backend = "vllm"
    cfg.generator.eval_sampling_params = OmegaConf.create(
        {
            "max_generate_length": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "logprobs": None,
            "stop": None,
        }
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.environment = OmegaConf.create({"env_class": "gsm8k"})
    cfg.trainer.dump_eval_results = False
    cfg.trainer.export_path = str(tmp_path)
    return cfg


class DummyStatefulDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class DummyRunner(FixtureRunner):
    def __init__(self, outputs: dict[str, TrajectoryBatch]):
        self.outputs = outputs
        self.seen_inputs = []

    async def _run(self, input_batch):
        self.seen_inputs.append(input_batch)
        return {**self.outputs[input_batch.metadata[0].uid], "trajectory_ids": list(input_batch.trajectory_ids)}


@pytest.mark.asyncio
@pytest.mark.parametrize("step_wise", [False, True])
@pytest.mark.parametrize("scored", [False, True])
async def test_required_coverage_persists_failed_attempts_and_accepts_valid_zero(
    dummy_config, tmp_path, step_wise, scored
):
    cfg = configure_eval(dummy_config, tmp_path)
    cfg.trainer.dump_eval_results = True
    cfg.trainer.eval_required_score_coverage = 1.0
    verdict = VerificationResult.verified(0.0) if scored else VerificationResult.unavailable("provider unavailable")
    runner = DummyRunner(
        {
            "paired-task": {
                "prompt_token_ids": [[101]],
                "response_ids": [[201]],
                "rewards": [0.0],
                "loss_masks": [[1]],
                "stop_reasons": ["stop"],
                "rollout_logprobs": None,
                "verification_results": [verdict],
                "is_last_step": [True],
            }
        }
    )
    loader = DummyStatefulDataLoader(
        [[{"input": task_input(lowered_task(text_task("paired-task", source="panel"))), "uid": "paired-task"}]]
    )
    evaluation = evaluate_step_wise if step_wise else evaluate
    tokenizer = SimpleNamespace(decode=lambda _: "response")
    sink = TrajectorySink(
        parse_trajectory_retention_config(cfg.generator.trajectory_retention),
        tokenizer,
        publisher=InlineTrajectoryPublisher(execute_publication),
    )
    arguments = dict(
        eval_dataloader=loader,
        trajectory_runner=runner,
        cfg=cfg,
        global_step=0,
        tokenizer=tokenizer,
        trajectory_sink=sink,
    )
    if scored:
        metrics = await evaluation(**arguments)
        assert metrics["eval/all/avg_verifier_score"] == 0.0
        assert metrics["eval/all/verifier_score_coverage"] == 1.0
    else:
        with pytest.raises(ValueError, match="Evaluation verifier coverage"):
            await evaluation(**arguments)
    record = json.loads((tmp_path / "dumped_evals/global_step_0_evals/panel.jsonl").read_text())
    assert record["uid"] == ("eval-0-0--paired-task" if step_wise else "paired-task")
    assert record["verification"]["status"] == ("verified" if scored else "unavailable")
    assert record["verification"]["score"] == (0.0 if scored else None)


class EvaluationLifecycle:
    def __init__(self):
        self.events = []

    async def pause_and_drain(self):
        self.events.append("drained")

    def resume(self):
        self.events.append("resumed")

    def set_trajectory_sink(self, sink):
        pass

    async def start_eval_session(self, **kwargs):
        self.events.append("started")

    async def stop_eval_session(self):
        self.events.append("stopped")


@pytest.mark.asyncio
@pytest.mark.parametrize("eval_limit,expected_peak", [(2, 2), (None, 3)])
async def test_evaluation_session_caps_shared_work_independently_of_training(eval_limit, expected_peak):
    cfg = get_default_config()
    cfg.trajectory_runner.max_concurrent_tasks = 3
    cfg.trajectory_runner.eval_max_concurrent_tasks = eval_limit
    ready = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def work():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == expected_peak:
            ready.set()
        await release.wait()
        active -= 1

    runner = EvaluationLifecycle()
    async with evaluation_session(runner, cfg, 0, object()) as pool:
        async with asyncio.TaskGroup() as tasks:
            for source in ("native", "sandbox", "callback"):
                tasks.create_task(pool.run(source, work))
            async with asyncio.timeout(5):
                await ready.wait()
            release.set()
    assert peak == expected_peak
    assert runner.events == ["started", "stopped"]


def test_eval_only_task_dataset_preserves_private_grader_and_prefill_contract(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    grader = ScriptGrader(argv=("sh", "/tests/private-grader.sh"), environment=NATIVE_ENVIRONMENT)
    shell_task = text_task("shell-private").model_copy(update={"grader": grader})
    native_task = text_task("native-private").model_copy(
        update={
            "grader": grader,
            "context": ConversationInput(
                events=(
                    TextMessage(role="user", content="Complete this"),
                    TextMessage(role="assistant", content="prefix"),
                ),
            ),
        }
    )
    expected = [
        lowered_task(shell_task, verifier_backend="daytona"),
        lowered_task(native_task, "prefill", max_turns=1, verifier_backend="daytona"),
    ]
    path = tmp_path / "validation.parquet"
    write_tasks(path, expected)
    experiment = object.__new__(EvalOnlyEntrypoint)
    experiment.cfg = get_default_config()
    experiment.cfg.trainer.evaluation_runner = "task"
    experiment.cfg.trainer.max_prompt_length = 8
    experiment.cfg.data.val_data = [str(path)]
    experiment.tokenizer = tokenizer

    dataset = experiment.get_eval_dataset()
    selected = dataset.collate_fn(list(dataset))

    assert [row["input"].spec for row in selected] == expected
    assert [row["uid"] for row in selected] == ["0", "1"]
