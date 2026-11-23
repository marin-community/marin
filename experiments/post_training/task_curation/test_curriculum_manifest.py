# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from marin.rl.curriculum import PoolRow, consumed_route_quotas, draw_validation_split, validation_payload

from experiments.post_training.task_curation.curriculum_manifest import curriculum_from_manifest, rollout_source_routes
from experiments.post_training.task_curation.exp4_curriculum import TARGET_SHARES as EXP4_SHARES
from experiments.post_training.task_curation.exp4_curriculum import exp4_rl_curriculum
from experiments.post_training.task_curation.exp5_curriculum import TARGET_SHARES as EXP5_SHARES
from experiments.post_training.task_curation.exp5_curriculum import exp5_rl_curriculum


@pytest.mark.parametrize(
    "shares,builder,quotas",
    [
        (EXP4_SHARES, exp4_rl_curriculum, {"agentic": 22, "nonagentic": 42}),
        (EXP5_SHARES, exp5_rl_curriculum, {"agentic": 13, "nonagentic": 51}),
    ],
)
def test_pool_curriculum_round_trip_preserves_holdout_and_route_quotas(shares, builder, quotas):
    rows = {label: tuple(PoolRow(id=f"{label}-{i}", sha256=f"{i:064x}") for i in range(80)) for label in shares}
    validation = draw_validation_split(dict(shares), rows, size=200, seed=20261020).validation
    original = builder(validation)
    manifest = {"curriculum": original.identity, "validation": validation_payload(validation)}
    rebuilt = curriculum_from_manifest(manifest)
    assert rebuilt == original
    routes = {label: "agentic" if label == "long-loop-smartie-swesmith" else "nonagentic" for label in shares}
    assert consumed_route_quotas(rebuilt, rollout_source_routes(routes), 64) == quotas
    changed = copy.deepcopy(manifest)
    changed["validation"]["rows"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from pool manifest"):
        curriculum_from_manifest(changed)


def test_exp5_runtime_routes_every_raw_source_in_the_stamped_pools():
    teacher_routes = {
        label: "agentic" if label == "long-loop-smartie-swesmith" else "nonagentic" for label in EXP5_SHARES
    }
    routes = rollout_source_routes(teacher_routes)
    raw_sources = {
        "long-loop-smartie-swesmith": "agentic",
        "PRIME-RL/Eurus-2-RL-Data": "nonagentic",
        "nvidia/Nemotron-RL-Ultra-Training-Blends": "nonagentic",
        "nvidia/OpenScience": "nonagentic",
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": "nonagentic",
        "open-thoughts/TaskTrove": "nonagentic",
        "https://files.pythonhosted.org/packages/42/21/"
        "bb9f6d2f76424f8b516fe83f8026e52e466b0202341bb4ee7a9861e06277/"
        "reasoning_gym-0.1.25-py3-none-any.whl": "nonagentic",
        "nvidia/Llama-Nemotron-Post-Training-Dataset": "nonagentic",
        "allenai/IF_multi_constraints_upto5": "nonagentic",
        "OpenResearcher/OpenResearcher-Dataset": "nonagentic",
        "pafitis/HARDMath_processed_training": "nonagentic",
    }
    assert {source: routes[source] for source in raw_sources} == raw_sources
    assert {label: routes[label] for label in teacher_routes} == teacher_routes
