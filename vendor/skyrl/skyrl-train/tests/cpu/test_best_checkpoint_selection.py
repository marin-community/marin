import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import create_callbacks_from_config
from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import RayPPOTrainer


class CheckpointPolicy:
    """Local policy RPC boundary that rejects overwriting an existing shard."""

    def async_run_ray_method(self, _dispatch, _method, *, ckpt_dir, tokenizer):
        directory = Path(ckpt_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "weights.pt").open("xb") as shard:
            shard.write(b"policy weights")
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("cadence_save", ["absent", "pending", "committed"])
async def test_best_selection_reuses_cadence_checkpoint_without_overwriting_shards(tmp_path, monkeypatch, cadence_save):
    # Keep only accelerator RPCs local; snapshot serialization, asynchronous
    # marker publication, and best-checkpoint selection use production code.
    monkeypatch.setattr(ray, "get", lambda result: result)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.trainer.ckpt_path = str(tmp_path)
    trainer.cfg.trainer.max_ckpts_to_keep = -1
    trainer.global_step = 2
    trainer._last_saved_step = None
    trainer._pending_checkpoint_upload = None
    trainer.colocate_all = False
    trainer.policy_model = CheckpointPolicy()
    trainer.tokenizer = None
    trainer.distillation_scored_tokens_total = 0
    trainer._ftpo_stopped = False
    trainer._domain_balancer = None
    trainer.all_timings = {}
    trainer.callback_handler = CallbackHandler()
    trainer._control = TrainerControl()

    async def rollout_state():
        return TrainingContextState(PromptLoaderState({"epoch": 0, "position": 2}, []), [], None)

    trainer.context = SimpleNamespace(state_dict=rollout_state)
    state = TrainerState(2, 0, 20, 20)
    if cadence_save != "absent":
        await trainer._save_intermediate_checkpoint(state)
        if cadence_save == "committed":
            await trainer._drain_checkpoint_upload()

    callback = create_callbacks_from_config(
        OmegaConf.create({"trainer": {"callbacks": [{
            "type": "best_checkpoint", "initial_model_uri": "base-model",
            "expected_tasks": 40, "minimum_scored_tasks": 40,
        }]}})
    )[0]
    await callback.on_evaluate_async(
        state, TrainerControl(), trainer=trainer,
        metrics={"eval/all/num_attempted": 40, "eval/all/num_scored": 40, "eval/all/avg_verifier_score": 0.5},
    )

    checkpoint = tmp_path / "global_step_2"
    selected = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert selected["model_uri"] == str(checkpoint / "policy")
    assert selected["model_format"] == "native"
    assert (checkpoint / "trainer_state.pt").is_file()
    assert (checkpoint / "data.pt").is_file()
    assert (checkpoint / "policy/weights.pt").read_bytes() == b"policy weights"

    # A later optimizer step must still save its own weights.
    trainer.global_step = 3
    await trainer.save_checkpoints()
    assert (tmp_path / "global_step_3/policy/weights.pt").read_bytes() == b"policy weights"


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
        checkpoint = tmp_path / f"global_step_{current_step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.pt").write_bytes(b"committed")

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
