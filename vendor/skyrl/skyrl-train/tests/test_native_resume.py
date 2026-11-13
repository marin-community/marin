"""Recovery through the trainer loader and the existing CPU checkpoint strategy."""

import pickle
from pathlib import Path

import pytest
import ray
import torch
import torch.distributed as dist

from marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from skyrl_train.config.utils import get_default_config
from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode
from tests.cpu.tiny_training.cpu_backend import CPUStrategy


class LocalPolicyGroup:
    """In-process policy RPC boundary backed by the real CPU save/load strategy."""

    def __init__(self):
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

    def save(self, root: Path, step: int) -> Path:
        checkpoint = root / f"global_step_{step}"
        self.strategy.save_checkpoint(self.model, str(checkpoint / "policy"), 0, self.optimizer, self.scheduler)
        torch.save(
            {"global_step": step, "distillation_scored_tokens_total": step * 17},
            checkpoint / TRAINER_STATE_FILENAME,
        )
        (root / LATEST_CHECKPOINT_FILE).write_text(str(step))
        return checkpoint


@pytest.fixture
def trainer_factory(tmp_path, monkeypatch):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'gloo'}", rank=0, world_size=1)
    # Keep the RPC boundary local; checkpoint serialization and restoration are real.
    monkeypatch.setattr(ray, "get", lambda result: result)

    def make(output: Path, seed: Path | None = None, mode: str = "latest"):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = get_default_config()
        trainer.cfg.trainer.ckpt_path = str(output)
        trainer.cfg.trainer.resume_path = str(seed) if seed else None
        trainer.cfg.trainer.ckpt_interval = 1
        trainer.cfg.trainer.restore_dataloader_state = False
        trainer.resume_mode = ResumeMode(mode)
        trainer.colocate_all = True
        trainer.policy_model = LocalPolicyGroup()
        trainer.critic_model = None
        trainer._domain_balancer = None
        trainer.distillation_scored_tokens_total = 0
        return trainer

    yield make
    dist.destroy_process_group()


def test_continuation_recovers_own_progress_and_next_update(tmp_path, trainer_factory):
    original = trainer_factory(tmp_path / "seed")
    assert original.load_checkpoints() == (0, None)
    for _ in range(2):
        original.policy_model.update()
    seed = original.policy_model.save(tmp_path / "seed", 2)
    output = tmp_path / "continuation"

    continuation = trainer_factory(output, seed)
    assert continuation.load_checkpoints() == (2, str(seed))
    assert continuation.distillation_scored_tokens_total == 34
    for _ in range(3):
        original.policy_model.update()
        continuation.policy_model.update()
    own = continuation.policy_model.save(output, 5)

    recovered = trainer_factory(output, seed)
    assert recovered.load_checkpoints() == (5, str(own))
    assert recovered.distillation_scored_tokens_total == 85
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
