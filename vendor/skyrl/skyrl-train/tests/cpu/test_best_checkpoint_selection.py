import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from marinskyrl.checkpoint_paths import BEST_CHECKPOINT_FILENAME

from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import create_callbacks_from_config
from skyrl_train.checkpoint_listing import ROLLOUT_STATE_FILENAME
from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode


def test_rolling_cleanup_keeps_best_periodic_and_recent_checkpoints(tmp_path, monkeypatch):
    for step in range(1, 7):
        checkpoint = tmp_path / f"global_step_{step}"
        checkpoint.mkdir()
        (checkpoint / TRAINER_STATE_FILENAME).touch()
        (checkpoint / ROLLOUT_STATE_FILENAME).touch()
    (tmp_path / BEST_CHECKPOINT_FILENAME).write_text(json.dumps({"step": 1, "model_format": "native"}))
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {"trainer": {"ckpt_path": str(tmp_path), "max_ckpts_to_keep": 2, "checkpoint_keep_every": 2}}
    )
    trainer._node_ids = ["worker"]
    monkeypatch.setattr("skyrl_train.trainer.run_on_each_node", lambda *_args: None)

    trainer._cleanup_old_checkpoints(6)

    assert {path.name for path in tmp_path.glob("global_step_*")} == {
        "global_step_1",
        "global_step_2",
        "global_step_4",
        "global_step_5",
        "global_step_6",
    }


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
@pytest.mark.parametrize("resume", ["same-trainer", "after-restart", "imported"])
async def test_periodic_checkpoint_can_be_selected_on_write_once_storage(tmp_path, monkeypatch, resume):
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
    output = tmp_path
    if resume != "same-trainer":
        await trainer._drain_checkpoint_upload()
        if resume == "imported":
            output = tmp_path / "resumed"
            config.trainer.ckpt_path = str(output)
            config.trainer.export_path = str(output / "exports")
            config.trainer.resume_path = str(tmp_path / "global_step_1")
        trainer = make_trainer()
        trainer.global_step, _ = trainer.load_checkpoints()
    callback = trainer.callback_handler.callbacks[0]
    await callback.on_evaluate_async(
        TrainerState(trainer.global_step, 0, 1, 1),
        TrainerControl(),
        trainer=trainer,
        metrics={**coverage, "score": 1.0},
    )

    selected = json.loads((output / "best_checkpoint.json").read_text())
    assert (selected["step"], selected["accuracy"]) == (1, 1.0)
    model = Path(selected["model_uri"])
    assert model.is_relative_to(output)
    assert (model / "rank_0.pt").read_bytes() == b"policy"
    assert json.loads((output / "exports/best_checkpoint.json").read_text()) == selected


@pytest.mark.asyncio
@pytest.mark.parametrize("last_upper_bound,last_pass_rate", [(8, 0.81), (7, 0.70)])
async def test_ftpo_pilot_stops_at_plateau_or_correctness_breach_after_resume(tmp_path, last_upper_bound, last_pass_rate):
    config = OmegaConf.create({"trainer": {"callbacks": [{"type": "ftpo_pilot",
        "patience_evaluations": 2, "max_pass_rate_drop": 0.05}]}})
    trainer = SimpleNamespace(cfg=SimpleNamespace(trainer=SimpleNamespace(export_path=str(tmp_path))))
    callback = create_callbacks_from_config(config)[0]

    async def evaluate(step, upper_bound, pass_rate):
        return await callback.on_evaluate_async(TrainerState(step, 0, 12, 12), TrainerControl(), trainer=trainer,
            metrics={"eval/all/loop_upper_bound": upper_bound, "eval/all/fail_closed_pass_rate": pass_rate})

    assert not (await evaluate(0, 10, 0.80)).should_training_stop
    assert not (await evaluate(3, 8, 0.78)).should_training_stop
    assert not (await evaluate(6, 8, 0.80)).should_training_stop
    callback = create_callbacks_from_config(config)[0]
    result = await evaluate(9, last_upper_bound, last_pass_rate)
    assert result.should_training_stop is True
    record = json.loads((tmp_path / "ftpo_pilot_state.json").read_text())
    assert record["baseline_pass_rate"] == 0.80
    assert record["stopped"] is True
