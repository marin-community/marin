# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

COMPOSE = """
import json
import sys
from pathlib import Path

from cloud.iris.launch_config import load_launch_config
from skyrl_train.checkpoint_schedule import CheckpointSchedule
from skyrl_train.config.training_budget import training_budget

def matched(path):
    cfg = load_launch_config(Path(path)).skyrl
    CheckpointSchedule.from_config(cfg)  # the trainer's own start-up check of the checkpoint fields
    budget = training_budget(cfg)
    optimizer = cfg.trainer.policy.optimizer_config
    return {
        "sequences_per_step": budget.sequences_per_step,
        "sequences_per_mini_batch": budget.sequences_per_mini_batch,
        "max_input_tokens": budget.max_input_tokens,
        "response_caps": dict(budget.response_caps.by_data_source),
        "optimizer": [optimizer.optimizer, optimizer.lr, list(optimizer.adam_betas), optimizer.weight_decay,
                      optimizer.max_grad_norm, optimizer.num_warmup_steps, optimizer.scheduler],
        "steps": [cfg.trainer.max_steps, cfg.trainer.epochs, cfg.trainer.update_epochs_per_batch],
        "seed": cfg.trainer.seed,
        "packing": cfg.trainer.use_sample_packing,
        "order": [cfg.data.shuffle, cfg.data.sampling.kind],
        "checkpoints": [list(cfg.trainer.checkpoint_steps), list(cfg.trainer.optimizer_checkpoint_steps)],
        "router_telemetry": cfg.trainer.router_telemetry.enabled,
        "strategy": cfg.trainer.strategy,
        "env_class": cfg.environment.env_class,
        "resume": [cfg.trainer.resume_mode, cfg.trainer.resume_path],
    }

print(json.dumps([matched(path) for path in sys.argv[1:]]))
"""
