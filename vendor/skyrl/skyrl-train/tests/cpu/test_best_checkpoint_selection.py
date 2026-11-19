import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import create_callbacks_from_config
from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode


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
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path), export_path=str(tmp_path / "exports")))
    )

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
        cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path), export_path=str(tmp_path / "exports"))),
        save_checkpoints=save_checkpoints,
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

    async def evaluate(step, identity, preservation, *, attempted=40):
        nonlocal current_step
        current_step = step
        return await callback.on_evaluate_async(
            TrainerState(step, 0, 20, 20),
            TrainerControl(),
            trainer=trainer,
            metrics={
                "eval/all/num_attempted": attempted,
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
    await evaluate(12, 0.99, 0.9, attempted=41)
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
    (tmp_path / "best_checkpoint.json").unlink()
    assert json.loads((tmp_path / "exports/best_checkpoint.json").read_text()) == selected


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True], ids=["same-trainer", "after-restart"])
async def test_periodic_checkpoint_can_be_selected_on_write_once_storage(tmp_path, monkeypatch, restart):
    class PolicyStorage:
        def async_run_ray_method(self, _dispatch, method, **kwargs):
            if method == "save_checkpoint":
                path = Path(kwargs["ckpt_dir"]) / "rank_0.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as shard:
                    shard.write(b"policy")
            return []

    async def rollout_state():
        return TrainingContextState(PromptLoaderState({}, []), [], None)

    config = OmegaConf.create(
        {
            "trainer": {
                "ckpt_path": str(tmp_path),
                "export_path": str(tmp_path / "exports"),
                "ckpt_interval": 1,
                "max_ckpts_to_keep": -1,
                "offload_optimizer_during_rollouts": False,
                "restore_dataloader_state": False,
                "callbacks": [
                    {
                        "type": "best_checkpoint",
                        "initial_model_uri": "base-model",
                        "expected_tasks": 1,
                        "minimum_scored_tasks": 1,
                        "selection_metric": "score",
                    }
                ],
            }
        }
    )
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)

    def make_trainer():
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = config
        trainer.policy_model = PolicyStorage()
        trainer.critic_model = None
        trainer.context = SimpleNamespace(state_dict=rollout_state)
        trainer.tokenizer = None
        trainer.all_timings = {}
        trainer._last_saved_step = None
        trainer._ftpo_stopped = False
        trainer._domain_balancer = None
        trainer.distillation_scored_tokens_total = 0
        trainer.colocate_all = False
        trainer.resume_mode = ResumeMode.LATEST
        trainer.callback_handler = CallbackHandler(create_callbacks_from_config(config))
        trainer._control = TrainerControl()
        return trainer

    trainer = make_trainer()
    coverage = {"eval/all/num_attempted": 1, "eval/all/num_scored": 1}
    trainer.global_step = 1
    await trainer._save_intermediate_checkpoint(TrainerState(1, 0, 1, 1))
    if restart:
        await trainer._drain_checkpoint_upload()
        trainer = make_trainer()
        trainer.global_step, _ = trainer.load_checkpoints()
    callback = trainer.callback_handler.callbacks[0]
    await callback.on_evaluate_async(
        TrainerState(trainer.global_step, 0, 1, 1), TrainerControl(), trainer=trainer, metrics={**coverage, "score": 1.0}
    )

    selected = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert (selected["step"], selected["accuracy"]) == (1, 1.0)
    assert (Path(selected["model_uri"]) / "rank_0.pt").read_bytes() == b"policy"
    assert json.loads((tmp_path / "exports/best_checkpoint.json").read_text()) == selected
