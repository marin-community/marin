# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import levanter.callbacks as callbacks_module
from levanter.callbacks import LambdaCallback
from levanter.callbacks import profile_ctx
from levanter.callbacks import profiler as profiler_module
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, profile


def test_profile_writes_trace_to_run_dir_and_ignores_duplicate_forced_stop(monkeypatch, tmp_path):
    calls = []

    def start_trace(path: str, *, create_perfetto_link: bool, create_perfetto_trace: bool, profiler_options) -> None:
        calls.append(("start", path, create_perfetto_link, create_perfetto_trace, profiler_options))

    def stop_trace() -> None:
        calls.append(("stop",))

    monkeypatch.setattr(profiler_module.jax, "process_index", lambda: 0)
    monkeypatch.setattr(profiler_module.jax.profiler, "start_trace", start_trace)
    monkeypatch.setattr(profiler_module.jax.profiler, "stop_trace", stop_trace)
    monkeypatch.setattr(profiler_module, "barrier_sync", lambda: calls.append(("barrier",)))

    options = ProfilerConfig(profile_options=ProfileOptionsConfig(host_tracer_level=1)).build_jax_profile_options()
    profile_dir = tmp_path / "run" / "profiler"
    callback = LambdaCallback(
        profile(
            str(profile_dir),
            start_step=5,
            num_steps=1,
            create_perfetto_link=False,
            profiler_options=options,
        )
    )

    assert profile_dir.exists()

    callback.on_step(SimpleNamespace(step=4))
    callback.on_step(SimpleNamespace(step=4), force=True)
    callback.on_step(SimpleNamespace(step=4), force=True)

    assert calls == [
        ("start", str(profile_dir), False, True, options),
        ("stop",),
        ("barrier",),
    ]
    assert profile_dir.exists()


def test_profile_callback_stress_repeated_start_stop_finalization(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        profiler_module.jax.profiler,
        "start_trace",
        lambda path, *_args, **_kwargs: calls.append(("start", path)),
    )
    monkeypatch.setattr(profiler_module.jax.profiler, "stop_trace", lambda: calls.append(("stop",)))
    monkeypatch.setattr(profiler_module, "barrier_sync", lambda: calls.append(("barrier",)))

    profile_dir = tmp_path / "stress" / "profiler"
    callback = LambdaCallback(profile(str(profile_dir), start_step=10, num_steps=2, create_perfetto_link=False))
    for _ in range(50):
        callback.on_step(SimpleNamespace(step=9))
        callback.on_step(SimpleNamespace(step=10))
        callback.on_step(SimpleNamespace(step=10), force=True)

    assert calls.count(("start", str(profile_dir))) == 50
    assert calls.count(("stop",)) == 50
    assert calls.count(("barrier",)) == 50
    assert profile_dir.exists()


def test_profile_ctx_writes_host_profile_files_without_tracker_upload(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(callbacks_module, "barrier_sync", lambda: calls.append(("barrier",)))

    profile_dir = tmp_path / "ctx" / "profiler"
    with profile_ctx(str(profile_dir), device_profile=False, host_profile=True, host_profile_topn=10):
        sum(range(1000))

    assert (profile_dir / "host_profile.pstats").exists()
    assert (profile_dir / "host_profile.txt").exists()
    assert calls == [("barrier",)]
