# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.data.text.datasets import ConcatDatasetComponent
from marin.execution.lazy import materialized_config

from experiments.june_tpu_67b_a2b.moe.launch_merge_recovery import build_merge_matching_pipeline

_PREFIX = "gs://marin-us-central2/test"


def _component_cache_paths(component):
    if isinstance(component, ConcatDatasetComponent):
        return [path for child in component.children.values() for path in _component_cache_paths(child)]
    return [component.cache_dir]


def test_large_merge_graph_keeps_checkpoint_data_compute_and_outputs_in_central2():
    pipeline = build_merge_matching_pipeline(version="dev")
    calibration = materialized_config(pipeline.calibration, _PREFIX)
    matching = materialized_config(pipeline.matching, _PREFIX)

    assert calibration.resources.regions == ["us-central2"]
    assert matching.resources.regions == ["us-central2"]
    assert calibration.source.checkpoint_dir.startswith("gs://marin-us-central2/")
    assert calibration.output_path.startswith("gs://marin-us-central2/")
    assert matching.calibration_path.startswith("gs://marin-us-central2/")
    assert matching.output_path.startswith("gs://marin-us-central2/")
    cache_paths = [
        path for component in calibration.data.components.values() for path in _component_cache_paths(component)
    ]
    assert cache_paths
    assert all(path.startswith("gs://marin-us-central2/") for path in cache_paths)


def test_large_merge_uses_middle_pair_and_stage_specific_meshes():
    pipeline = build_merge_matching_pipeline(version="dev")
    calibration = materialized_config(pipeline.calibration, _PREFIX)
    matching = materialized_config(pipeline.matching, _PREFIX)

    assert calibration.layers == (12, 13)
    assert matching.representative_layer == 12
    assert matching.source_layer == 13
    assert calibration.source.model.num_layers == 26
    assert calibration.source.model.max_seq_len == 8_192
    assert calibration.source.model.qk_mult == 1.57
    assert calibration.batch_size == 128
    assert calibration.trace_sample_size == 131_072
    assert calibration.resources.device.variant == "v4-256"
    assert (calibration.replica_axis_size, calibration.expert_axis_size, calibration.model_axis_size) == (1, 1, 1)
    assert matching.resources.device.variant == "v4-64"
    assert (matching.replica_axis_size, matching.expert_axis_size, matching.model_axis_size) == (2, 1, 16)


def test_graph_construction_does_not_read_the_remote_checkpoint(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("checkpoint storage was accessed during graph construction")

    monkeypatch.setattr("levanter.checkpoint.latest_checkpoint_path", fail_if_called)
    pipeline = build_merge_matching_pipeline(version="dev")
    pipeline.matching.fingerprint()
    pipeline.matching.lower()
