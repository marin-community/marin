import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import CheckpointCallback, EvaluationCallback, create_default_callbacks
from skyrl_train.callbacks.loop_evidence import LoopEvidenceCallback


@pytest.mark.parametrize("explicit", [False, True])
def test_loop_evidence_preserves_lifecycle_and_default_callbacks(tmp_path, explicit):
    cfg = OmegaConf.create(
        {
            "trainer": {
                "loop_evidence_enabled": True,
                "export_path": str(tmp_path),
                "callbacks": [{"type": "checkpoint", "save_steps": 1}, {"type": "evaluation", "eval_steps": 1}]
                if explicit
                else None,
                "ckpt_interval": 1,
                "eval_interval": 1,
                "enable_db_registration": False,
            },
            "generator": {"backend": "vllm", "inference_stats_interval": 0},
        }
    )
    callbacks = create_default_callbacks(cfg)
    assert any(isinstance(callback, CheckpointCallback) for callback in callbacks)
    assert any(isinstance(callback, EvaluationCallback) for callback in callbacks)
    evidence = next(callback for callback in callbacks if isinstance(callback, LoopEvidenceCallback))
    handler = CallbackHandler([evidence])
    trainer = SimpleNamespace(
        tracker=SimpleNamespace(logger={"wandb": SimpleNamespace(url="https://wandb.ai/team/project/runs/exact-id")})
    )
    state = TrainerState(global_step=0, epoch=0, total_steps=1, num_steps_per_epoch=1)
    control = TrainerControl()
    handler.call_event("on_train_begin", state, control, trainer=trainer)
    path = tmp_path / "loop-evidence.json"
    startup = json.loads(path.read_text())
    assert startup["expected_steps"] == 1
    assert startup["steps"] == []
    assert startup["wandb_url"] == "https://wandb.ai/team/project/runs/exact-id"
    metrics = {"eval/scored": 100, "eval/attempted": 100, "eval/reward": 0.2, "answer": "private answer"}
    handler.call_event("on_evaluate", state, control, metrics=metrics, trainer=trainer)
    state = replace(
        state,
        global_step=1,
        metrics={
            "policy/policy_update_steps": 1,
            "policy/skipped_steps": 0,
            "policy/raw_grad_norm": 0.1,
            "policy/policy_loss": 0.03,
            "text": "private transcript",
        },
    )
    handler.call_event("on_step_end", state, control, trainer=trainer)
    handler.call_event("on_evaluate", state, control, metrics=metrics | {"eval/reward": 0.3}, trainer=trainer)
    record = json.loads(path.read_text())
    assert record["evaluations"]["0"]["eval/reward"] == 0.2
    assert record["evaluations"]["1"]["eval/reward"] == 0.3
    assert record["steps"] == [
        {"step": 1, "metrics": {key: value for key, value in state.metrics.items() if key != "text"}}
    ]
    assert "private" not in path.read_text()
    assert record["observations_only"] is True
    assert "status" not in record
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Nonfinite loop observation"):
        handler.call_event(
            "on_step_end", replace(state, metrics={"policy/raw_grad_norm": float("nan")}), control, trainer=trainer
        )
    assert path.read_bytes() == before
