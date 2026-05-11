# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from levanter.callbacks import LambdaCallback
from levanter.callbacks import profiler as profiler_module
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


def test_profile_passes_profile_options_and_ignores_duplicate_forced_stop(monkeypatch):
    calls = []
    artifacts = []

    class FakeTracker:
        def log_artifact(self, artifact_path: str, *, name: str | None = None, type: str | None = None) -> None:
            artifacts.append((artifact_path, name, type))

    def start_trace(path: str, *, create_perfetto_link: bool, create_perfetto_trace: bool, profiler_options) -> None:
        calls.append(("start", path, create_perfetto_link, create_perfetto_trace, profiler_options))

    def stop_trace() -> None:
        calls.append(("stop",))

    monkeypatch.setattr(profiler_module.jax, "process_index", lambda: 0)
    monkeypatch.setattr(profiler_module.jax.profiler, "start_trace", start_trace)
    monkeypatch.setattr(profiler_module.jax.profiler, "stop_trace", stop_trace)
    monkeypatch.setattr(profiler_module.levanter.tracker, "current_tracker", lambda: FakeTracker())
    monkeypatch.setattr(profiler_module, "barrier_sync", lambda: calls.append(("barrier",)))

    options = ProfilerConfig(profile_options=ProfileOptionsConfig(host_tracer_level=1)).build_jax_profile_options()
    callback = LambdaCallback(
        profile(
            "/tmp/profiler",
            start_step=5,
            num_steps=1,
            create_perfetto_link=False,
            profiler_options=options,
        )
    )

    callback.on_step(SimpleNamespace(step=4))
    callback.on_step(SimpleNamespace(step=5), force=True)
    callback.on_step(SimpleNamespace(step=5), force=True)

    assert calls == [
        ("start", "/tmp/profiler", False, True, options),
        ("stop",),
        ("barrier",),
    ]
    assert artifacts == [("/tmp/profiler", "jax-profile-step-5-6", "jax_profile")]


def test_profile_callback_stress_repeated_start_stop_finalization(monkeypatch):
    calls = []

    class FakeTracker:
        def log_artifact(self, artifact_path: str, *, name: str | None = None, type: str | None = None) -> None:
            calls.append(("artifact", artifact_path, name, type))

    monkeypatch.setattr(
        profiler_module.jax.profiler,
        "start_trace",
        lambda *_args, **_kwargs: calls.append(("start",)),
    )
    monkeypatch.setattr(profiler_module.jax.profiler, "stop_trace", lambda: calls.append(("stop",)))
    monkeypatch.setattr(profiler_module.levanter.tracker, "current_tracker", lambda: FakeTracker())
    monkeypatch.setattr(profiler_module, "barrier_sync", lambda: calls.append(("barrier",)))

    callback = LambdaCallback(profile("profile-dir", start_step=10, num_steps=2, create_perfetto_link=False))
    for _ in range(50):
        callback.on_step(SimpleNamespace(step=9))
        callback.on_step(SimpleNamespace(step=10))
        callback.on_step(SimpleNamespace(step=10), force=True)

    assert calls.count(("start",)) == 50
    assert calls.count(("stop",)) == 50
    assert calls.count(("barrier",)) == 50
    assert calls.count(("artifact", "profile-dir", "jax-profile-step-10-12", "jax_profile")) == 50
