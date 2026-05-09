# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from levanter.callbacks import profiler as profiler_mod
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, profile


def test_profile_options_config_builds_jax_profile_options():
    options = ProfileOptionsConfig(
        host_tracer_level=1,
        device_tracer_level=0,
        python_tracer_level=0,
        enable_hlo_proto=False,
        include_dataset_ops=False,
        advanced_configuration={
            "gpu_num_chips_to_profile_per_task": 1,
            "gpu_max_callback_api_events": 131072,
            "gpu_max_activity_api_events": 131072,
            "gpu_max_annotation_strings": 65536,
            "tpu_trace_mode": "TRACE_ONLY_HOST",
        },
    ).build_jax_profile_options()

    assert options is not None
    assert options.host_tracer_level == 1
    assert options.advanced_configuration == {
        "device_tracer_level": 0,
        "gpu_num_chips_to_profile_per_task": 1,
        "gpu_max_callback_api_events": 131072,
        "gpu_max_activity_api_events": 131072,
        "gpu_max_annotation_strings": 65536,
        "tpu_trace_mode": "TRACE_ONLY_HOST",
    }
    assert options.python_tracer_level == 0
    assert options.enable_hlo_proto is False
    assert options.include_dataset_ops is False


def test_profile_passes_profile_options_and_uses_timed_stop(monkeypatch):
    start_calls = []
    stop_calls = []
    artifacts = []

    class FakeTracker:
        def log_artifact(self, artifact_path, *, name=None, type=None):
            artifacts.append((artifact_path, name, type))

    def fake_start_trace(path, *, create_perfetto_link, create_perfetto_trace, profiler_options):
        start_calls.append((path, create_perfetto_link, create_perfetto_trace, profiler_options))

    def fake_stop_trace_with_timing():
        stop_calls.append(True)
        return 0.0

    monkeypatch.setattr(profiler_mod.jax, "process_index", lambda: 0)
    monkeypatch.setattr(profiler_mod.jax.profiler, "start_trace", fake_start_trace)
    monkeypatch.setattr(profiler_mod, "stop_trace_with_timing", fake_stop_trace_with_timing)
    monkeypatch.setattr(profiler_mod.levanter.tracker, "current_tracker", lambda: FakeTracker())
    monkeypatch.setattr(profiler_mod, "barrier_sync", lambda: None)

    options = ProfilerConfig(profile_options=ProfileOptionsConfig(host_tracer_level=1)).build_jax_profile_options()
    callback = profile(
        "/tmp/profiler",
        start_step=5,
        num_steps=1,
        create_perfetto_link=False,
        profiler_options=options,
    )

    callback(SimpleNamespace(step=4))
    callback(SimpleNamespace(step=5))

    assert start_calls == [("/tmp/profiler", False, True, options)]
    assert stop_calls == [True]
    assert artifacts == [("/tmp/profiler", "jax-profile-step-5-6", "jax_profile")]
