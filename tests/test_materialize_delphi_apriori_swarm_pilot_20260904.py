# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from experiments.domain_phase_mix import launch_delphi_apriori_swarm_3e18 as launcher
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_apriori_swarm_pilot_20260904 as materializer,
)


def _canary_run(*, pool_fraction: float = 0.5, subset_seed: int = 662_009):
    return SimpleNamespace(
        id="canary",
        tags=[
            f"source_run={launcher.CANARY_RUN_NAME}",
            f"tpu_type={launcher.TARGET_TPU_TYPE}",
            f"tpu_zone={launcher.DEFAULT_TPU_ZONE}",
        ],
        config={
            "data_seed": 662_009,
            "trainer": {"seed": 0, "num_train_steps": 3007},
            "data": {
                "simulated_epoch_subset_seed": subset_seed,
                "simulated_epoch_pool_fractions": {"dolmino_synth_qa": pool_fraction},
            },
        },
    )


def test_runtime_config_matches_frozen_canary_and_expands_pool_fractions() -> None:
    design = materializer.read_design()
    row = design[design["run_name"].eq(launcher.CANARY_RUN_NAME)].iloc[0].to_dict()

    observed = materializer.validate_runtime_config(
        _canary_run(),
        row,
        expected_tpu_type=launcher.TARGET_TPU_TYPE,
        expected_tpu_zone=launcher.DEFAULT_TPU_ZONE,
    )

    assert observed["data_seed"] == observed["subset_seed"] == 662_009
    assert observed["pool_fraction::dolmino_synth_qa"] == 0.5
    assert all(
        value == 1.0
        for key, value in observed.items()
        if key.startswith("pool_fraction::") and key != "pool_fraction::dolmino_synth_qa"
    )


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (_canary_run(pool_fraction=0.25), "pool fraction"),
        (_canary_run(subset_seed=662_010), "subset_seed"),
    ],
)
def test_runtime_config_rejects_drift_from_frozen_canary(run, message: str) -> None:
    design = materializer.read_design()
    row = design[design["run_name"].eq(launcher.CANARY_RUN_NAME)].iloc[0].to_dict()

    with pytest.raises(ValueError, match=message):
        materializer.validate_runtime_config(
            run,
            row,
            expected_tpu_type=launcher.TARGET_TPU_TYPE,
            expected_tpu_zone=launcher.DEFAULT_TPU_ZONE,
        )
