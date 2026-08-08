# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.data.text.datasets import ConcatDatasetComponent
from marin.execution.lazy import StepContext

from experiments.june_tpu_67b_a2b.moe.launch_tied_experts import JuneTiedPhase, build_tied_expert_runs
from experiments.june_tpu_67b_a2b.moe.optimizer import TiedExpertLrScale

_PREFIX = "gs://marin-us-central2/test"


def _component_cache_paths(component):
    if isinstance(component, ConcatDatasetComponent):
        return [path for child in component.children.values() for path in _component_cache_paths(child)]
    return [component.cache_dir]


def _fingerprint_config(run):
    return run.build_config(StepContext.for_fingerprint(run.runtime_args.keys(), run.deps))


def test_june_67b_tied_comparison_is_matched_and_central2_only():
    runs = build_tied_expert_runs(version="dev", phase=JuneTiedPhase.SMOKE)
    configs = [_fingerprint_config(run) for run in runs]
    baseline, tied = configs

    for run, config in zip(runs, configs, strict=True):
        resources = run.runtime_args["train_resources"]
        assert resources.regions == ["us-central2"]
        assert resources.device.variant == "v4-2048"
        assert run.path(_PREFIX).startswith("gs://marin-us-central2/")
        cache_paths = [
            path for component in config.data.components.values() for path in _component_cache_paths(component)
        ]
        datakit_paths = [path for path in cache_paths if path.startswith("gs://")]
        assert datakit_paths
        assert all(path.startswith("gs://marin-us-central2/") for path in datakit_paths)

    assert baseline.steps == tied.steps == 100
    assert baseline.batch_size == tied.batch_size == 8_192
    assert baseline.model.max_seq_len == tied.model.max_seq_len == 8_192
    assert baseline.seed == tied.seed == 0
    assert baseline.optimizer.schedule_num_train_steps_override == 150_000
    assert tied.optimizer.tied_expert_lr_scale is TiedExpertLrScale.UNSCALED
    assert baseline.model.resolved_expert_bank_for_layer == tuple(range(26))
    assert tied.model.expert_bank_group_sizes == (1, 1, 4, 4, 4, 4, 4, 2, 1, 1)


def test_june_67b_milestone_keeps_full_schedule_and_changes_only_run_horizon():
    smoke = build_tied_expert_runs(version="dev", phase=JuneTiedPhase.SMOKE, variant_names=("middle_groups_unscaled",))[
        0
    ]
    milestone = build_tied_expert_runs(
        version="dev", phase=JuneTiedPhase.MILESTONE, variant_names=("middle_groups_unscaled",)
    )[0]
    smoke_config = _fingerprint_config(smoke)
    milestone_config = _fingerprint_config(milestone)

    assert smoke_config.steps == 100
    assert milestone_config.steps == 3_000
    assert smoke_config.model == milestone_config.model
    assert smoke_config.optimizer == milestone_config.optimizer
    assert smoke_config.data.train_weights == milestone_config.data.train_weights
