"""Recovery through the trainer loader and the existing CPU checkpoint strategy."""

import pickle
from pathlib import Path

import pytest
import ray
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
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
        trainer.cfg = OmegaConf.create(
            {
                "trainer": {
                    "ckpt_path": str(output),
                    "resume_path": str(seed) if seed else None,
                    "ckpt_interval": 1,
                    "restore_dataloader_state": False,
                }
            }
        )
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
    for _ in range(2):
        original.policy_model.update()
    seed = original.policy_model.save(tmp_path / "seed", 2)
    output = tmp_path / "continuation"

    continuation = trainer_factory(output, seed)
    assert continuation.load_checkpoints() == (2, str(seed))
    assert continuation.distillation_scored_tokens_total == 34
    for _ in range(3):
        continuation.policy_model.update()
    own = continuation.policy_model.save(output, 5)

    recovered = trainer_factory(output, seed)
    assert recovered.load_checkpoints() == (5, str(own))
    assert recovered.distillation_scored_tokens_total == 85
    assert recovered.policy_model.scheduler.last_epoch == 5
    assert recovered.policy_model.scheduler.get_last_lr() == continuation.policy_model.scheduler.get_last_lr()
    for policy in (continuation.policy_model, recovered.policy_model):
        policy.update()
    torch.testing.assert_close(recovered.policy_model.model.weight, continuation.policy_model.model.weight, rtol=0, atol=0)
    for recovered_state, expected_state in zip(
        recovered.policy_model.optimizer.state.values(), continuation.policy_model.optimizer.state.values(), strict=True
    ):
        torch.testing.assert_close(recovered_state["momentum_buffer"], expected_state["momentum_buffer"], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mode", "own_step", "seed_exists", "expected_step"),
    [("latest", None, False, 0), ("latest", 5, False, 5), ("from_path", 5, True, 2), ("none", 5, True, 0)],
)
def test_resume_modes_preserve_selection(tmp_path, trainer_factory, mode, own_step, seed_exists, expected_step):
    output = tmp_path / "continuation"
    seed = tmp_path / "seed/global_step_2"
    writer = trainer_factory(tmp_path / "writer")
    if seed_exists:
        writer.policy_model.save(seed.parent, 2)
    if own_step:
        writer.policy_model.save(output, own_step)
    selected = trainer_factory(output, seed if mode != "latest" or own_step else None, mode)

    step, path = selected.load_checkpoints()

    assert step == expected_step
    assert selected.distillation_scored_tokens_total == expected_step * 17
    assert path == (str(seed if mode == "from_path" else output / f"global_step_{own_step}") if step else None)


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("marker", ValueError),
        ("missing", FileNotFoundError),
        ("inconsistent", ValueError),
        ("policy", pickle.UnpicklingError),
    ],
)
def test_bad_own_checkpoint_never_falls_back_to_seed(tmp_path, trainer_factory, corruption, error):
    writer = trainer_factory(tmp_path / "writer")
    seed = writer.policy_model.save(tmp_path / "seed", 2)
    output = tmp_path / "continuation"
    own = writer.policy_model.save(output, 5)
    if corruption == "marker":
        (output / LATEST_CHECKPOINT_FILE).write_text("not-a-step")
    elif corruption == "missing":
        (output / LATEST_CHECKPOINT_FILE).write_text("99")
    elif corruption == "inconsistent":
        (output / LATEST_CHECKPOINT_FILE).write_text("1")
    else:
        (own / "policy/rank_0.pt").write_bytes(b"corrupt")

    with pytest.raises(error):
        trainer_factory(output, seed).load_checkpoints()


def test_missing_starting_checkpoint_is_an_error(tmp_path, trainer_factory):
    with pytest.raises(FileNotFoundError):
        trainer_factory(tmp_path / "continuation", tmp_path / "seed/global_step_2").load_checkpoints()
