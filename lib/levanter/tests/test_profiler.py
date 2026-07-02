# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import SimpleNamespace
from types import TracebackType
import uuid

from levanter.callbacks import LambdaCallback
from levanter.callbacks import profile_ctx
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, mirror_process_profile_dir, profile
from rigging.filesystem import url_to_fs


@dataclass(frozen=True)
class RecordedProfile:
    path: str
    create_perfetto_link: bool
    profiler_options: object | None
    stop_barrier_timeout: float
    remote_profile_dir: str | None


class RecordingProfileContext(AbstractContextManager[None]):
    def __init__(self, profile_record: RecordedProfile, events: list[tuple[str, RecordedProfile]]):
        self.profile_record = profile_record
        self.events = events

    def __enter__(self) -> None:
        self.events.append(("enter", self.profile_record))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.events.append(("exit", self.profile_record))


class RecordingProfileFactory:
    def __init__(self):
        self.events: list[tuple[str, RecordedProfile]] = []

    def __call__(
        self,
        path: str,
        create_perfetto_link: bool,
        *,
        profiler_options: object | None,
        stop_barrier_timeout: float,
        remote_profile_dir: str | None,
    ) -> RecordingProfileContext:
        profile_record = RecordedProfile(
            path=path,
            create_perfetto_link=create_perfetto_link,
            profiler_options=profiler_options,
            stop_barrier_timeout=stop_barrier_timeout,
            remote_profile_dir=remote_profile_dir,
        )
        return RecordingProfileContext(profile_record, self.events)

    def records_for(self, event: str) -> Iterator[RecordedProfile]:
        for observed_event, profile_record in self.events:
            if observed_event == event:
                yield profile_record


def test_profile_callback_starts_once_and_forced_stop_is_idempotent(tmp_path):
    profile_dir = tmp_path / "run" / "profiler"
    remote_profile_dir = f"memory://levanter-profile/{uuid.uuid4()}/profiler"
    options = ProfilerConfig(profile_options=ProfileOptionsConfig(host_tracer_level=1)).build_jax_profile_options()
    context_factory = RecordingProfileFactory()

    callback = LambdaCallback(
        profile(
            str(profile_dir),
            start_step=5,
            num_steps=1,
            create_perfetto_link=False,
            profiler_options=options,
            stop_barrier_timeout=123,
            remote_profile_dir=remote_profile_dir,
            profile_context_factory=context_factory,
        )
    )

    callback.on_step(SimpleNamespace(step=4))
    callback.on_step(SimpleNamespace(step=4), force=True)
    callback.on_step(SimpleNamespace(step=4), force=True)

    assert context_factory.events == [
        (
            "enter",
            RecordedProfile(str(profile_dir), False, options, 123, remote_profile_dir),
        ),
        (
            "exit",
            RecordedProfile(str(profile_dir), False, options, 123, remote_profile_dir),
        ),
    ]
    assert (profile_dir / "process_00000").is_dir()


def test_profile_callback_can_profile_repeated_windows_after_stop(tmp_path):
    profile_dir = tmp_path / "stress" / "profiler"
    context_factory = RecordingProfileFactory()
    callback = LambdaCallback(
        profile(
            str(profile_dir),
            start_step=10,
            num_steps=2,
            create_perfetto_link=True,
            profile_context_factory=context_factory,
        )
    )

    for _ in range(3):
        callback.on_step(SimpleNamespace(step=9))
        callback.on_step(SimpleNamespace(step=9))
        callback.on_step(SimpleNamespace(step=10), force=True)

    assert len(list(context_factory.records_for("enter"))) == 3
    assert len(list(context_factory.records_for("exit"))) == 3
    assert all(profile_record.path == str(profile_dir) for profile_record in context_factory.records_for("enter"))
    assert all(profile_record.create_perfetto_link for profile_record in context_factory.records_for("enter"))
    assert (profile_dir / "process_00000").is_dir()


def test_profile_context_mirrors_process_dir_to_remote_profile_dir(tmp_path):
    profile_dir = tmp_path / "ctx" / "profiler"
    remote_profile_dir = f"memory://levanter-profile/{uuid.uuid4()}/profiler"

    with profile_ctx(str(profile_dir), device_profile=False, remote_profile_dir=remote_profile_dir):
        trace_path = profile_dir / "process_00000" / "plugins" / "profile" / "2026_07_02" / "host.xplane.pb"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_bytes(b"profile")

    fs, fs_path = url_to_fs(f"{remote_profile_dir}/process_00000/plugins/profile/2026_07_02/host.xplane.pb")
    assert fs.exists(fs_path)


def test_mirror_process_profile_dir_preserves_process_subdirectory(tmp_path):
    profile_dir = tmp_path / "run" / "profiler"
    trace_path = profile_dir / "process_00007" / "plugins" / "profile" / "2026_07_02" / "trace.xplane.pb"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_bytes(b"profile")
    remote_profile_dir = f"memory://levanter-profile/{uuid.uuid4()}/profiler"

    mirrored_profile_dir = mirror_process_profile_dir(profile_dir, remote_profile_dir, process_index=7)

    fs, fs_path = url_to_fs(f"{remote_profile_dir}/process_00007/plugins/profile/2026_07_02/trace.xplane.pb")
    assert mirrored_profile_dir == remote_profile_dir
    assert fs.exists(fs_path)


def test_profile_ctx_writes_host_profile_files_without_tracker_upload(tmp_path):
    profile_dir = tmp_path / "ctx" / "profiler"

    with profile_ctx(str(profile_dir), device_profile=False, host_profile=True, host_profile_topn=10):
        sum(range(1000))

    assert (profile_dir / "process_00000" / "host_profile.pstats").exists()
    assert (profile_dir / "process_00000" / "host_profile.txt").exists()
