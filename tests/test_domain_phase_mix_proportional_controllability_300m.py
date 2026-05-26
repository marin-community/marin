# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix.launch_proportional_controllability_300m import (
    DEFAULT_EVAL_DATASETS_CACHE_PATH,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    build_launch_artifacts,
    build_run_specs,
)


def test_subset_retry_filters_training_steps_by_run_name():
    full_specs = build_run_specs()
    selected_run_names = (full_specs[0].run_name, full_specs[-1].run_name)

    artifacts = build_launch_artifacts(
        base_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m",
        tpu_type=DEFAULT_TPU_TYPE,
        tpu_regions=(DEFAULT_TPU_REGION,),
        tpu_zone=DEFAULT_TPU_ZONE,
        eval_datasets_cache_path=DEFAULT_EVAL_DATASETS_CACHE_PATH,
        include_eval_harness=False,
        only_run_names=selected_run_names,
    )

    assert [run_spec.run_name for run_spec in artifacts.run_specs] == list(selected_run_names)
    assert [intervention.run_name for intervention in artifacts.interventions] == list(selected_run_names)
    assert len(artifacts.training_steps) == len(selected_run_names)
    assert all(run_spec.run_id in {full_specs[0].run_id, full_specs[-1].run_id} for run_spec in artifacts.run_specs)


def test_subset_retry_rejects_unknown_run_name():
    with pytest.raises(ValueError, match="Unknown proportional-controllability run names"):
        build_launch_artifacts(
            base_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m",
            tpu_type=DEFAULT_TPU_TYPE,
            tpu_regions=(DEFAULT_TPU_REGION,),
            tpu_zone=DEFAULT_TPU_ZONE,
            eval_datasets_cache_path=DEFAULT_EVAL_DATASETS_CACHE_PATH,
            include_eval_harness=False,
            only_run_names=("pctrl_missing",),
        )
