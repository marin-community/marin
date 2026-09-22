# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from experiments.post_training.snowball_eagle_speculators import _agentic_rollout_steps, _qa_rollout_steps


def test_rollouts_use_requested_version_and_inherit_root_cluster() -> None:
    with build_context(BuildContext(versions=VersionCodex(default="2099.01.02"))):
        qa_steps = _qa_rollout_steps()
        agentic_steps = _agentic_rollout_steps()

    step = qa_steps[0]
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert {step.version for step in qa_steps} == {"2099.01.02"}
    assert [step.version for step in agentic_steps] == ["2099.01.02.1", "2099.01.02.2", "2099.01.02.3"]
    assert config.accelerator.target_cluster is None
    assert step.run.max_retries_failure == 1
