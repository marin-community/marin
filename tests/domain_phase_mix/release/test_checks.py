# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from experiments.domain_phase_mix.release import checks, keys, paths


def _run(oid: str, bpb: float = 0.8, w: float = 1.0, pf: float = 1.0, mixture_id: str | None = None) -> dict:
    """A one-bucket run whose content ids hash its weight and pool fraction, unless `mixture_id` overrides one."""
    return {
        "observation_id": oid,
        "mixture_id": mixture_id or keys.mixture_id({"arxiv": w}),
        "exposure_id": keys.exposure_id({"arxiv": w}, {"arxiv": pf}, paths.PARTITION_REVISION),
        "w__arxiv": w,
        "pf__arxiv": pf,
        "bpb__arc_challenge": bpb,
    }


@pytest.mark.parametrize("column, field", [("bpb__arc_challenge", "bpb"), ("w__arxiv", "w"), ("pf__arxiv", "pf")])
def test_same_observation_same_values(column, field):
    tables = {
        "validation_3e18": pd.DataFrame([_run("obs_a", **{field: 0.5})]),
        "heldout": pd.DataFrame([_run("obs_a", **{field: 0.25}), _run("obs_b", **{field: 0.5})]),
    }
    assert checks.consistency(tables) == [f"heldout: obs_a: {column} = 0.25 but validation_3e18 has 0.5"]
    tables["heldout"] = pd.DataFrame([_run("obs_a", **{field: 0.5 + 1e-10}), _run("obs_b", **{field: 0.5})])
    assert checks.consistency(tables) == []


def test_a_value_missing_from_one_table_is_not_a_disagreement():
    # heldout lacks the OlmoBaseEval Easy components that the 160M swarm table has for the same deletion runs.
    tables = {
        "swarm_llama_160m_1p2b": pd.DataFrame([_run("obs_a", bpb=0.8)]),
        "heldout": pd.DataFrame([_run("obs_a", bpb=float("nan"))]),
    }
    assert checks.consistency(tables) == []


@pytest.mark.parametrize(
    "seed, problem",
    [
        ("data_seed", "heldout: obs_a: data_seed = 1.0 but validation_3e18 has 7.0"),
        ("trainer_seed", "heldout: obs_a: trainer_seed = 1.0 but validation_3e18 has 0.0"),
    ],
    ids=["data_seed", "trainer_seed"],
)
def test_seeds_agree_where_both_tables_have_them(seed, problem):
    seeds = {"data_seed": 7, "trainer_seed": 0}
    tables = {
        "validation_3e18": pd.DataFrame([_run("obs_a") | seeds, _run("obs_b") | seeds]),
        # heldout records no seeds for obs_b, which is no disagreement.
        "heldout": pd.DataFrame([_run("obs_a") | seeds | {seed: 1}, _run("obs_b") | dict.fromkeys(seeds)]),
    }
    assert checks.consistency(tables) == [problem]


def test_content_ids_must_hash_the_row():
    wrong = pd.DataFrame([_run("obs_a", mixture_id="mix_wrong")])
    assert any("obs_a" in p and "mix_wrong" in p for p in checks.consistency({"validation_3e18": wrong}))
    # Pool fractions set the epochs: the exposure id changes with them, the mixture id does not.
    halved = pd.DataFrame([_run("obs_a") | {"pf__arxiv": 0.5}])
    problems = checks.consistency({"heldout": halved})
    assert len(problems) == 1 and "obs_a" in problems[0] and "exposure_id" in problems[0]


def test_run_sets_membership():
    tables = {
        "swarm_qwen3_360m_1p6b": pd.DataFrame(
            [{"observation_id": "obs_a", "fit_inclusion": True}, {"observation_id": "obs_r", "fit_inclusion": False}]
        ),
        "heldout": pd.DataFrame([{"observation_id": "obs_h", "cohort": "frozen", "swarm": "qwen3_360m_1p6b"}]),
        "validation_3e18": pd.DataFrame([{"observation_id": "obs_h", "launch": "frozen_procedure"}]),
        "ladder": pd.DataFrame([{"observation_id": "obs_h"}]),
        # Every run of the StarCoder and domain-sweep configs, under any of their tables' id columns, is one set.
        "starcoder_ramp_designs": pd.DataFrame(
            [{"observation_id": "obs_s", "source_observation_id": "obs_t", "stage": stage} for stage in range(2)]
        ),
        "domain_sweeps": pd.DataFrame([{"observation_id": "obs_d", "evaluation": e} for e in ("sciq", "gsm8k")]),
        # Derived tables reference runs of other tables and form no set.
        "fits_proposal_predictions": pd.DataFrame([{"observation_id": "obs_f", "predictor": "mariner"}]),
    }
    sets = checks.run_sets(tables)
    assert list(sets.columns) == ["observation_id", "run_set"]
    assert sorted(map(tuple, sets.to_numpy())) == sorted(
        [
            ("obs_a", "fit_swarm_qwen3_360m_1p6b"),
            ("obs_r", "proportional_repeats_qwen3_360m_1p6b"),
            ("obs_h", "heldout_frozen"),
            ("obs_h", "validation/frozen_procedure"),
            ("obs_h", "ladder"),
            ("obs_s", "starcoder"),
            ("obs_t", "starcoder"),
            ("obs_d", "domain_sweeps"),
        ]
    )
