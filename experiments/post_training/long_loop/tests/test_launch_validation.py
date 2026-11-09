# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from experiments.post_training.long_loop.eval_runtime import judge


@pytest.mark.timeout(120)
def test_invalid_launch_fails_before_judge_allocation(monkeypatch):
    allocated = []

    def allocate(*args, **kwargs):
        allocated.append(True)
        raise AssertionError("Invalid config reached judge allocation")

    monkeypatch.setattr(judge, "remote_inference", allocate)
    config = SimpleNamespace(skyrl=SimpleNamespace(launch_config_yaml="unexpected_root_field: true\n"))
    with pytest.raises(ValueError, match="Strict SkyRL composition failed"):
        judge.run_judged(config)
    assert not allocated
