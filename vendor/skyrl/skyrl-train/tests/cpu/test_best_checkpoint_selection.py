import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import create_callbacks_from_config
from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import RayPPOTrainer


class CheckpointPolicy:
    def __init__(self, root: Path, upload_failures: int):
        self.root = root
        self.upload_failures = upload_failures

    def async_run_ray_method(self, dispatch, method, **kwargs):
        if method == "save_checkpoint":
            directory = Path(kwargs["ckpt_dir"])
            marker = self.root / LATEST_CHECKPOINT_FILE
            if marker.exists() and directory.parent.name == f"global_step_{marker.read_text()}":
                raise FileExistsError("Cannot overwrite committed policy shards")
            directory.mkdir(parents=True, exist_ok=True)
            torch.save({"weight": torch.tensor([1.0])}, directory / "weights.pt")
        elif method == "wait_checkpoint_upload":
            if self.upload_failures:
                self.upload_failures -= 1
                raise OSError("Policy shard upload failed")
        else:
            raise ValueError(method)
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_checkpoint", ["committed", "uploading", "missing", "failed_upload"])
async def test_best_selection_reuses_committed_shards_and_recovers_failed_uploads(
    tmp_path, monkeypatch, existing_checkpoint
):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create({"trainer": {"ckpt_path": str(tmp_path), "max_ckpts_to_keep": -1}})
    trainer.global_step = 10
    trainer._last_saved_step = None
    trainer._pending_checkpoint_upload = None
    trainer._checkpoint_save_failures = 0
    trainer.policy_model = CheckpointPolicy(tmp_path, int(existing_checkpoint == "failed_upload"))
    trainer.critic_model = None
    trainer.tokenizer = None
    trainer.colocate_all = False
    trainer.distillation_scored_tokens_total = 0
    trainer._ftpo_stopped = False
    trainer._domain_balancer = None
    trainer.all_timings = {}
    trainer.all_metrics = {}
    rollout_state = TrainingContextState(PromptLoaderState({"position": 42}, []), [], None)

    async def state_dict():
        return rollout_state

    trainer.context = SimpleNamespace(state_dict=state_dict)
    # Replace only the accelerator/Ray boundary; publication, selection and files are real.
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda references: references)
    state = TrainerState(10, 0, 20, 20)
    if existing_checkpoint == "committed":
        await trainer.save_checkpoints()
    elif existing_checkpoint in ("uploading", "failed_upload"):
        snapshot = trainer._snapshot_checkpoint(rollout_state)
        trainer._start_checkpoint_upload(snapshot, state)

    (tmp_path / "validation_selection_history.json").write_text(
        json.dumps([{"step": step, "reward": step / 20, "attempted": 1, "scored": 1} for step in (2, 4, 6, 8)])
    )
    config = OmegaConf.create({"trainer": {"callbacks": [{
        "type": "best_checkpoint", "initial_model_uri": "base-model", "expected_tasks": 1,
        "minimum_scored_tasks": 1, "ema_window": 5, "evaluation_interval": 2,
    }]}})
    callback = create_callbacks_from_config(config)[0]
    await callback.on_evaluate_async(
        state, TrainerControl(), trainer=trainer,
        metrics={"eval/all/num_attempted": 1, "eval/all/num_scored": 1, "eval/all/avg_verifier_score": 0.5},
    )

    selected = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert selected["step"] == 10
    assert selected["model_format"] == "native"
    assert (tmp_path / LATEST_CHECKPOINT_FILE).read_text() == "10"
    assert torch.load(Path(selected["model_uri"]) / "weights.pt", weights_only=True)["weight"].tolist() == [1.0]
    restored = torch.load(tmp_path / "global_step_10/data.pt", weights_only=False)
    assert restored == rollout_state
    if existing_checkpoint == "failed_upload":
        assert trainer.all_metrics["trainer/checkpoint_save_failures"] == 1


@pytest.mark.asyncio
async def test_preservation_gates_reject_ineligible_step_zero_evaluation(tmp_path):
    config = OmegaConf.create(
        {
            "trainer": {
                "callbacks": [
                    {
                        "type": "best_checkpoint",
                        "initial_model_uri": "base-model",
                        "expected_tasks": 40,
                        "minimum_scored_tasks": 40,
                        "selection_metric": "identity",
                        "preservation_max_drop": {"preservation": 0.0},
                    }
                ]
            }
        }
    )
    callback = create_callbacks_from_config(config)[0]
    trainer = SimpleNamespace(cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path))))

    with pytest.raises(ValueError, match="eligible step-zero evaluation"):
        await callback.on_evaluate_async(
            TrainerState(0, 0, 20, 20),
            TrainerControl(),
            trainer=trainer,
            metrics={
                "eval/all/num_attempted": 40,
                "eval/all/num_scored": 39,
                "identity": 0.1,
                "preservation": 0.9,
            },
        )


@pytest.mark.asyncio
async def test_delayed_best_selection_preserves_baseline_gates_after_resume(tmp_path):
    saved_steps = []
    current_step = 0

    async def save_checkpoints():
        saved_steps.append(current_step)
        (tmp_path / f"weights-{current_step}").write_text("checkpoint")

    trainer = SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path))), save_checkpoints=save_checkpoints
    )
    parameters = dict(
        initial_model_uri="base-model",
        expected_tasks=40,
        minimum_scored_tasks=40,
        selection_metric="identity",
        save_after_fraction=0.5,
        preservation_max_drop={"preservation": 0.0},
    )
    config = OmegaConf.create({"trainer": {"callbacks": [{"type": "best_checkpoint", **parameters}]}})
    callback = create_callbacks_from_config(config)[0]

    async def evaluate(step, identity, preservation):
        nonlocal current_step
        current_step = step
        return await callback.on_evaluate_async(
            TrainerState(step, 0, 20, 20),
            TrainerControl(),
            trainer=trainer,
            metrics={
                "eval/all/num_attempted": 40,
                "eval/all/num_scored": 40,
                "identity": identity,
                "preservation": preservation,
            },
        )

    await evaluate(0, 0.1, 0.9)
    await evaluate(5, 1.0, 0.9)
    assert saved_steps == []
    await evaluate(10, 0.5, 0.9)
    assert saved_steps == [10]
    callback = create_callbacks_from_config(config)[0]
    await evaluate(15, 0.8, 0.85)
    assert saved_steps == [10]
    await evaluate(20, 0.7, 0.9)
    assert saved_steps == [10, 20]
    await evaluate(20, 0.7, 0.9)
    assert saved_steps == [10, 20]
    selected = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert selected["step"] == 20
    assert selected["accuracy"] == 0.7
    assert selected["preservation_baseline"] == {"preservation": 0.9}
    assert selected["model_format"] == "native"
