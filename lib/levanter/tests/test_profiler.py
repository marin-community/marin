# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

import levanter.callbacks as callbacks_module
from levanter.callbacks import LambdaCallback
from levanter.callbacks import profile_ctx
from levanter.callbacks import profiler as profiler_module
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, XprofUploadConfig, profile


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
        ("start", str(profile_dir), False, False, options),
        ("stop",),
        ("barrier",),
    ]
    assert profile_dir.exists()


def test_profile_uploads_new_xplane_session_and_logs_viewer_link(monkeypatch, tmp_path, caplog):
    profile_dir = tmp_path / "run" / "profiler"
    upload_dir = tmp_path / "uploaded"
    old_session = profile_dir / "plugins" / "profile" / "old-session"
    old_session.mkdir(parents=True)
    (old_session / "worker-0.xplane.pb").write_bytes(b"old")

    monkeypatch.setattr(profiler_module.jax, "process_index", lambda: 0)
    monkeypatch.setattr(profiler_module.jax.profiler, "start_trace", lambda *_args, **_kwargs: None)

    def stop_trace() -> None:
        session = profile_dir / "plugins" / "profile" / "2026_07_23_12_00_00"
        session.mkdir(parents=True)
        (session / "worker-0.xplane.pb").write_bytes(b"xplane")
        (session / "worker-0.hlo_proto.pb").write_bytes(b"hlo")

    monkeypatch.setattr(profiler_module.jax.profiler, "stop_trace", stop_trace)
    monkeypatch.setattr(profiler_module, "barrier_sync", lambda: None)

    callback = LambdaCallback(
        profile(
            str(profile_dir),
            start_step=5,
            num_steps=1,
            create_perfetto_link=False,
            upload_uri=f"file://{upload_dir}",
            xprof_service_url="https://iris.example/proxy/xprof",
        )
    )
    with caplog.at_level("INFO", logger=profiler_module.__name__):
        callback.on_step(SimpleNamespace(step=4))
        callback.on_step(SimpleNamespace(step=5))

    uploaded_session = upload_dir / "plugins" / "profile" / "steps-5-to-6"
    assert (uploaded_session / "worker-0.xplane.pb").read_bytes() == b"xplane"
    assert (uploaded_session / "worker-0.hlo_proto.pb").read_bytes() == b"hlo"
    assert not (upload_dir / "plugins" / "profile" / "old-session").exists()
    link_record = next(record for record in caplog.records if record.message.startswith("XProf profile:"))
    link = link_record.message.removeprefix("XProf profile: ")
    assert urlparse(link).path == "/proxy/xprof/open"
    assert parse_qs(urlparse(link).query) == {"uri": [f"file://{upload_dir}"]}


def test_upload_destination_uses_xprof_ttl_path(monkeypatch):
    calls = []

    def temp_bucket(ttl_days: int, prefix: str) -> str:
        calls.append((ttl_days, prefix))
        return f"gs://marin-us-east5/tmp/ttl={ttl_days}d/{prefix}"

    monkeypatch.setattr(profiler_module, "marin_temp_bucket", temp_bucket)

    assert XprofUploadConfig().destination_for_run("run-123") == "gs://marin-us-east5/tmp/ttl=30d/xprof/run-123"
    assert calls == [(30, "xprof/run-123")]


def test_profiler_upload_can_be_disabled(monkeypatch, tmp_path):
    captured = {}

    def fake_profile(path, **kwargs):
        captured.update(path=path, **kwargs)
        return lambda _step: None

    monkeypatch.setattr(profiler_module, "profile", fake_profile)
    config = ProfilerConfig(upload=XprofUploadConfig(enabled=False))
    config.build(str(tmp_path / "capture"), run_id="local-run")

    assert captured["upload_uri"] is None
    assert captured["xprof_service_url"] is None


@pytest.mark.parametrize(
    "fallback",
    [
        "file://{tmp_path}/marin-tmp/{prefix}",
        "s3://some-other-bucket/marin/tmp/{prefix}",
    ],
)
def test_non_ttl_marin_fallback_does_not_copy_or_log_hosted_link(monkeypatch, tmp_path, fallback):
    captured = {}

    def fake_profile(path, **kwargs):
        captured.update(path=path, **kwargs)
        return lambda _step: None

    monkeypatch.setattr(profiler_module, "profile", fake_profile)
    monkeypatch.setattr(
        profiler_module,
        "marin_temp_bucket",
        lambda _ttl_days, prefix: fallback.format(tmp_path=tmp_path, prefix=prefix),
    )
    ProfilerConfig().build(str(tmp_path / "capture"), run_id="local-run")

    assert captured["upload_uri"] is None
    assert captured["xprof_service_url"] is None


def test_upload_error_reaches_second_barrier_before_propagating(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(profiler_module.jax, "process_index", lambda: 0)
    monkeypatch.setattr(profiler_module.jax.profiler, "start_trace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(profiler_module.jax.profiler, "stop_trace", lambda: None)
    monkeypatch.setattr(profiler_module, "barrier_sync", lambda: calls.append("barrier"))

    callback = profile(
        str(tmp_path / "capture"),
        start_step=5,
        num_steps=1,
        create_perfetto_link=False,
        upload_uri=f"file://{tmp_path}/upload",
    )
    callback(SimpleNamespace(step=4))
    with pytest.raises(RuntimeError, match="Failed to upload XProf profile"):
        callback(SimpleNamespace(step=5))

    assert calls == ["barrier", "barrier"]


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
