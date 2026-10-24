# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect

import cloudpickle
from fray.iris_backend import resolve_coscheduling

from experiments.evaluation.serving_parity.launch import Arm


def test_gang_entrypoint_preserves_the_arm_argument():
    arm = Arm("canary", "v4-32", "gs://bucket/fresh", "gs://bucket/model", True)
    fn, args, kwargs = cloudpickle.loads(arm.entrypoint().workdir_files["_callable.pkl"])
    bound = inspect.signature(fn).bind(*args, **kwargs)
    assert bound.arguments == {"arm": arm}
    resource = arm.resources()
    assert resource.replicas == 4
    assert resolve_coscheduling(resource, resource.replicas).group_by == "tpu-name"
