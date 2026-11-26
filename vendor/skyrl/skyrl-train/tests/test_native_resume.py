"""Recovery through the trainer loader and the existing CPU checkpoint strategy."""

import asyncio
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist

from marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from skyrl_train.config.utils import get_default_config
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
from skyrl_train.rollouts.buffer import BatchPolicy, RolloutBuffer, RolloutBufferConfig, RolloutContentPolicy
from skyrl_train.rollouts.context import RolloutRequestSpec, TrainingContext, TrainingContextState
from skyrl_train.rollouts.loader import DatasetEntrySelector, DatasetEntrySelectorState, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, ObjectStorePayloads
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode
from tests.cpu.task_specs import lowered_task, task_input, text_task
from tests.cpu.tiny_training.cpu_backend import CPUStrategy


class LocalPolicyGroup:
    """In-process policy RPC boundary backed by the real CPU save/load strategy."""

    def __init__(self, config):
        self.config = config
        self.strategy = CPUStrategy(max_grad_norm=1.0)
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(1.0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

    def async_run_ray_method(self, _dispatch, _method, *, ckpt_dir, load_training_state):
        return self.strategy.load_checkpoint(
            self.model, ckpt_dir, self.optimizer, self.scheduler, load_training_state=load_training_state
        )

    def update(self):
        loss = self.model(torch.ones(1, 1)).square().sum()
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

    def save(self, root: Path, step: int, *, data_position: int = 1) -> Path:
        checkpoint = root / f"global_step_{step}"
        self.strategy.save_checkpoint(self.model, str(checkpoint / "policy"), 0, self.optimizer, self.scheduler)
        torch.save(
            {"config": self.config, "global_step": step, "distillation_scored_tokens_total": step * 17},
            checkpoint / TRAINER_STATE_FILENAME,
        )
        torch.save(
            TrainingContextState(DatasetEntrySelectorState({"epoch": 0, "position": data_position}, []), [], None),
            checkpoint / "data.pt",
        )
        (root / LATEST_CHECKPOINT_FILE).write_text(str(step))
        return checkpoint


@pytest.fixture
def trainer_factory(tmp_path, monkeypatch):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'gloo'}", rank=0, world_size=1)
    # Keep the RPC boundary local; checkpoint serialization and restoration are real.
    monkeypatch.setattr(ray, "get", lambda result: result)

    def make(output: Path, seed: Path | None = None, mode: str = "latest", *, reset: bool = False):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = get_default_config()
        trainer.cfg.trainer.ckpt_path = str(output)
        trainer.cfg.trainer.resume_path = str(seed) if seed else None
        trainer.cfg.trainer.ckpt_interval = 1
        trainer.cfg.trainer.reset_global_step_on_resume = reset
        trainer.cfg.trainer.reset_distillation_token_count_on_resume = reset
        trainer.resume_mode = ResumeMode(mode)
        trainer.colocate_all = True
        trainer.policy_model = LocalPolicyGroup(trainer.cfg)
        trainer._domain_balancer = None
        trainer._restored_rollout_state = None
        trainer.distillation_scored_tokens_total = 0
        return trainer

    yield make
    dist.destroy_process_group()


def next_entry_index(trainer) -> int:
    order = SeededPasses(2, seed=0, shuffle=False)
    if trainer._restored_rollout_state is not None:
        order.load_state_dict(trainer._restored_rollout_state.selector.order)
    return order.next_index()


class CheckpointTasks:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return index

    def uid(self, index: int) -> str:
        return f"task-{index}"

    label = uid

    def collate_fn(self, indices: list[int]) -> list[dict]:
        return [
            {"uid": self.uid(index), "input": task_input(lowered_task(text_task(self.uid(index)), "text"))}
            for index in indices
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("object_store", [False, True])
async def test_old_rollout_state_restores_an_admitted_group_and_selector_cursor(monkeypatch, object_store):
    fixture = Path(__file__).parent / "fixtures/legacy_rollout_state"
    state = torch.load(fixture / "rollout_state.pt", map_location="cpu", weights_only=False)
    if object_store:
        group = replace(state.submitted[0], payload=str(fixture / "rollout_group.pickle"))
        state = replace(state, submitted=[group], object_store_root=str(fixture))
        payloads = ObjectStorePayloads(str(fixture))
    else:
        # Keep Ray's object-store boundary in-process.
        def put(group):
            future = asyncio.get_running_loop().create_future()
            future.set_result(group)
            return future

        monkeypatch.setattr(ray, "put", put)
        payloads = MemoryPayloads()

    dataset = CheckpointTasks()
    selector = DatasetEntrySelector(
        dataset,
        SeededPasses(len(dataset), seed=0, shuffle=False),
        batch_size=1,
        minimum_pool_size=2,
        max_stale_retries=1,
        max_unusable_groups=2,
        families={dataset.uid(index): "example" for index in range(len(dataset))},
    )
    config = RolloutBufferConfig(1, 1, 0, BatchPolicy.FULL_BATCH, None, None, {"example": 1})
    buffer = RolloutBuffer(config)
    context = TrainingContext(
        selector,
        config,
        SimpleNamespace(restore=SimpleNamespace(remote=buffer.restore)),
        RolloutContentPolicy(
            GroupAdmissionPolicy(
                GroupAdvantageInvariant.exact_physical(physical_group_size=1), rollout_logprobs_required=False
            ),
            GroupSelectionPolicy(None),
        ),
        RolloutRequestSpec(1, {}),
        None,
        payloads,
        rollout_spans=False,
        wait_labels=("lease", "task"),
    )

    await context.load_state_dict(state)
    await buffer.publish(1)
    (admitted,) = (await buffer.admit(timeout=1)).admitted
    (group,) = await payloads.fetch(buffer.payload_refs(1, [admitted.index]))
    assert (admitted.uid, admitted.sample_count, admitted.response_tokens) == ("task-0", 1, 2)
    assert group.entry["input"].spec.task.id == "task-0"
    assert group.trajectory_batch["response_ids"] == [[3, 4]]
    assert [selector.next_entry(live=(), family="example")["uid"] for _ in range(2)] == ["task-1", "task-2"]


@pytest.mark.parametrize("reset", [False, True])
def test_continuation_recovers_own_progress_and_next_update(tmp_path, trainer_factory, reset):
    original = trainer_factory(tmp_path / "seed")
    assert original.load_checkpoints() == (0, None)
    for _ in range(2):
        original.policy_model.update()
    # The starting run's cursor lies beyond the continuation's smaller dataset.
    seed = original.policy_model.save(tmp_path / "seed", 2, data_position=3)
    output = tmp_path / "continuation"

    continuation = trainer_factory(output, seed, reset=reset)
    assert continuation.load_checkpoints() == (0 if reset else 2, str(seed))
    assert continuation.distillation_scored_tokens_total == (0 if reset else 34)
    assert next_entry_index(continuation) == 0
    for _ in range(3):
        original.policy_model.update()
        continuation.policy_model.update()
    own_step = 3 if reset else 5
    own = continuation.policy_model.save(output, own_step)

    recovered = trainer_factory(output, seed, reset=reset)
    assert recovered.load_checkpoints() == (own_step, str(own))
    assert recovered.distillation_scored_tokens_total == own_step * 17
    assert next_entry_index(recovered) == 1
    original.policy_model.update()
    for policy in (continuation.policy_model, recovered.policy_model):
        policy.update()
        torch.testing.assert_close(policy.model.state_dict(), original.policy_model.model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(
            policy.optimizer.state_dict(), original.policy_model.optimizer.state_dict(), rtol=0, atol=0
        )
        assert policy.scheduler.state_dict() == original.policy_model.scheduler.state_dict()


@pytest.mark.parametrize(
    ("mode", "seed_exists", "expected_step", "expected_path"),
    [
        ("latest", False, 5, "continuation/global_step_5"),
        ("from_path", True, 2, "seed/global_step_2"),
        ("none", True, 0, None),
    ],
)
def test_resume_modes_preserve_selection(tmp_path, trainer_factory, mode, seed_exists, expected_step, expected_path):
    output = tmp_path / "continuation"
    seed = tmp_path / "seed/global_step_2"
    writer = trainer_factory(tmp_path / "writer")
    if seed_exists:
        writer.policy_model.save(seed.parent, 2)
    writer.policy_model.save(output, 5)
    selected = trainer_factory(output, seed, mode)
    assert selected.load_checkpoints() == (expected_step, str(tmp_path / expected_path) if expected_path else None)


@pytest.mark.parametrize(
    ("path", "contents", "error"),
    [
        (LATEST_CHECKPOINT_FILE, b"not-a-step", ValueError),
        (LATEST_CHECKPOINT_FILE, b"99", FileNotFoundError),
        (LATEST_CHECKPOINT_FILE, b"1", ValueError),
        ("global_step_5/policy/rank_0.pt", b"corrupt", pickle.UnpicklingError),
    ],
)
def test_bad_own_checkpoint_never_falls_back_to_seed(tmp_path, trainer_factory, path, contents, error):
    writer = trainer_factory(tmp_path / "writer")
    seed = writer.policy_model.save(tmp_path / "seed", 2)
    output = tmp_path / "continuation"
    writer.policy_model.save(output, 5)
    (output / path).write_bytes(contents)

    with pytest.raises(error):
        trainer_factory(output, seed).load_checkpoints()


def test_missing_starting_checkpoint_is_an_error(tmp_path, trainer_factory):
    with pytest.raises(FileNotFoundError):
        trainer_factory(tmp_path / "continuation", tmp_path / "seed/global_step_2").load_checkpoints()
