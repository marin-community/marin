# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import StepContext

from experiments.post_training.snowball_eagle_speculators import _qa_rollout_steps


def test_qa_rollouts_inherit_the_root_cluster() -> None:
    step = _qa_rollout_steps()[0]
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.accelerator.target_cluster is None
