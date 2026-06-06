# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.inference.vllm_smoke_test import REMOTE_VLLM_SMOKE_EXTRAS

from experiments.evals.evals import EVAL_DEPENDENCY_GROUPS, EVALCHEMY_DEPENDENCY_GROUPS


def test_tpu_vllm_eval_dependency_groups_do_not_request_generic_tpu() -> None:
    assert EVAL_DEPENDENCY_GROUPS == ["eval", "vllm"]
    assert EVALCHEMY_DEPENDENCY_GROUPS == ["evalchemy", "vllm"]


def test_remote_vllm_smoke_uses_self_contained_vllm_extra() -> None:
    assert REMOTE_VLLM_SMOKE_EXTRAS == ["eval", "vllm"]
