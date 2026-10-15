# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix.release.run_names import seeds_from_uri, training_run_name_from_uri


def test_training_run_name_from_uri():
    uri = "gs://.../lwspufp_uncheatable_seed666200_lwspu_u_snc_cap06-8cf646/checkpoints/eval_metrics.jsonl"
    assert training_run_name_from_uri(uri) == "lwspufp_uncheatable_seed666200_lwspu_u_snc_cap06-8cf646"
    with pytest.raises(ValueError):
        training_run_name_from_uri("gs://.../uncheatable_seed666200/lwspu_u_snc_cap06/checkpoints/eval_metrics.jsonl")


def test_seeds_from_uri():
    # The seeds come from the run's group directory, never from a seed token inside the run name.
    run = "lwspu_table9_seed1_lwspu_t9_snc_cap08-8cf646/checkpoints/eval_metrics.jsonl"
    assert seeds_from_uri(f"gs://.../table9_seed662009_t2/{run}") == (662009, 2)
    assert seeds_from_uri(f"gs://.../table9_seed662009/{run}") == (662009, 0)  # older launches omit trainer seed 0
    with pytest.raises(ValueError):
        seeds_from_uri(f"gs://.../table9/{run}")
