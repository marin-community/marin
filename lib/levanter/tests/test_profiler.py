# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import levanter.callbacks as callbacks_module
from levanter.callbacks import LambdaCallback
from levanter.callbacks import profile_ctx
from levanter.callbacks import profiler as profiler_module
from levanter.callbacks.profiler import profile


def test_profile_callback_stress_repeated_start_stop_finalization(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        profiler_module.jax.profiler,
        "start_trace",
        lambda path, *_args, **_kwargs: calls.append(("start", path)),
    )
    monkeypatch.setattr(profiler_module.jax.profiler, "stop_trace", lambda: calls.append(("stop",)))
    monkeypatch.setattr(profiler_module, "barrier_sync", lambda *, timeout: calls.append(("barrier",)))

    profile_dir = tmp_path / "stress" / "profiler"
    callback = LambdaCallback(profile(str(profile_dir), start_step=10, num_steps=2, create_perfetto_link=False))
    process_dir = profile_dir / "process_00000"
    for _ in range(50):
        callback.on_step(SimpleNamespace(step=9))
        callback.on_step(SimpleNamespace(step=10))
        callback.on_step(SimpleNamespace(step=10), force=True)

    assert calls.count(("start", str(process_dir))) == 50
    assert calls.count(("stop",)) == 50
    assert calls.count(("barrier",)) == 50
    assert profile_dir.exists()


def test_profile_ctx_writes_host_profile_files_without_tracker_upload(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(callbacks_module, "barrier_sync", lambda: calls.append(("barrier",)))

    profile_dir = tmp_path / "ctx" / "profiler"
    with profile_ctx(str(profile_dir), device_profile=False, host_profile=True, host_profile_topn=10):
        sum(range(1000))

    process_dir = profile_dir / "process_00000"
    assert (process_dir / "host_profile.pstats").exists()
    assert (process_dir / "host_profile.txt").exists()
    assert calls == [("barrier",)]
