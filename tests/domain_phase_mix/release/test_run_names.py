# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix.release.run_names import training_run_name_from_uri


def test_training_run_name_from_uri():
    uri = "gs://.../lwspufp_uncheatable_seed666200_lwspu_u_snc_cap06-8cf646/checkpoints/eval_metrics.jsonl"
    assert training_run_name_from_uri(uri) == "lwspufp_uncheatable_seed666200_lwspu_u_snc_cap06-8cf646"
    with pytest.raises(ValueError):
        training_run_name_from_uri("gs://.../uncheatable_seed666200/lwspu_u_snc_cap06/checkpoints/eval_metrics.jsonl")
