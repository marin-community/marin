"""Whole-task provenance survives learner conversion and the first optimizer dispatch."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.training_batch import TrainingInputBatch


def test_whole_task_rollouts_keep_order_through_padding_dump_and_training(tmp_path, monkeypatch):
    cfg = get_default_config()
    cfg.trainer.step_wise_training = False
    cfg.trainer.dump_training_order = True
    cfg.trainer.dump_data_batch = True
    cfg.trainer.export_path = str(tmp_path)
    cfg.trainer.algorithm.advantage_estimator = "rloo_n"
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.algorithm.off_policy_correction = "tis"
    cfg.trainer.algorithm.advantage_batch_normalize = False
    cfg.generator.n_samples_per_prompt = 2

    # Replace accelerator workers only; assembly, collation, padding, correction,
    # advantage calculation and both retained files use the production methods.
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.distillation_plan = None
    trainer.trajectory_selector = None
    trainer.group_advantage_invariant = GroupAdvantageInvariant.minimum_baseline_eligible(
        physical_group_size=2, minimum_group_size=2
    )
    trainer.context = SimpleNamespace(config=SimpleNamespace(batch_size=2, max_staleness_steps=2))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids: str(ids))
    trainer.global_step = 1
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.policy_model = SimpleNamespace(
        actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=6))],
        async_run_ray_method=lambda *args: [],
    )
    trainer.critic_model = None
    trainer.ref_model = None
    received = []

    def forward(batch):
        batch["action_log_probs"] = batch["rollout_logprobs"].clone()
        batch["values"] = None
        return batch

    def optimize(batch):
        received.append(batch)
        return {"policy_update_steps": 1.0}

    monkeypatch.setattr(trainer, "fwd_logprobs_values_reward", forward)
    monkeypatch.setattr(trainer, "train_critic_and_policy", optimize)
    groups = [
        RolloutGroup(
            trajectory_batch={
                "prompt_token_ids": [[11], [12]],
                "response_ids": [[21, 31, 41], [22, 42]],
                "rewards": [0.0, 1.0],
                "loss_masks": [[1, 0, 1], [1, 1]],
                "rollout_logprobs": [np.array([-0.2, 0.0, -0.4]), np.array([-0.3, -0.5])],
                "trajectory_ids": [TrajectoryID(source, 0), TrajectoryID(source, 1)],
                "exclude_from_baseline": [False, False],
            },
            uid=source, policy_step=0, prompt={},
        )
        for source in ("swe", "math")
    ]
    batch = trainer.convert_rollout_groups_to_training_input(groups)
    status = asyncio.run(trainer._run_training(batch))
    order = json.loads((tmp_path / "training_order/step-1.json").read_text())
    assert order["trajectory_ids"] == ["swe_0", "swe_1", "math_0", "math_1", "pad0", "pad1"]
    assert order["uids"] == ["swe", "swe", "math", "math", "pad0", "pad1"]
    assert order["pad_size"] == 2
    retained = TrainingInputBatch({}).load(tmp_path / "dumped_data/global_step_1_training_input.pkl")
    assert retained.metadata["trajectory_ids"] == order["trajectory_ids"]
    assert retained["loss_mask"].tolist() == [[1, 0, 1], [1, 1, 0]] * 2 + [[0, 0, 0]] * 2
    assert retained.metadata["exclude_from_baseline"].tolist() == [False] * 4 + [True] * 2
    assert torch.isfinite(received[0]["advantages"]).all()
    assert status == {"policy_update_steps": 1.0}
