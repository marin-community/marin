# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Unified RL job interface for configuring and running RL training.

This module provides a high-level interface that abstracts away worker management
and infrastructure concerns, letting users focus on the RL algorithm and hyperparameters.

The dataclass configs and pure worker-config builder live in
:mod:`marin.rl.job_config`; this module just adds the :class:`RLJob` wrapper
that submits a job through :mod:`marin.rl.orchestration`.
"""

import logging

from fray.client import JobHandle
from marin.rl.job_config import (
    RLJobConfig,
    RunConfig,
    TrainParams,
    build_worker_configs,
    make_tokenizer,
)
from marin.rl.orchestration import submit_rl_job
from marin.rl.rollout_worker import RolloutWorkerConfig
from marin.rl.train_worker import TrainWorkerConfig

__all__ = [
    "RLJob",
    "RLJobConfig",
    "RunConfig",
    "TrainParams",
    "build_worker_configs",
    "make_tokenizer",
]

logger = logging.getLogger(__name__)


class RLJob:
    """High-level interface for RL training jobs.

    Handles worker creation, coordination, and lifecycle management.
    """

    def __init__(self, config: RLJobConfig):
        self.config = config

    @staticmethod
    def make_step_fn():
        return lambda config: RLJob(config).run(config.run_id)

    def run(self, name: str) -> JobHandle:
        """Submit the RL job via the v2 orchestration layer.

        Submits a single coordinator job that creates all shared actors
        and child jobs (trainer + rollout workers). The coordinator runs
        inside the cluster with proper job hierarchy.
        """
        handle = submit_rl_job(self.config)
        handle.wait(raise_on_failure=True)
        return handle

    def to_worker_configs(self) -> tuple[TrainWorkerConfig, RolloutWorkerConfig]:
        """Export worker configurations for inspection/testing."""
        return build_worker_configs(self.config)
