import glob
import threading

from cloud.iris import task_runtime


class _RecordingFilesystem:
    def __init__(self, *, wait_for_parallel_upload: bool = False):
        self._lock = threading.Lock()
        self._parallel_upload = threading.Event()
        self._wait_for_parallel_upload = wait_for_parallel_upload
        self.active_uploads = 0
        self.max_active_uploads = 0

    def put(self, local_path, _remote_path):
        with self._lock:
            self.active_uploads += 1
            self.max_active_uploads = max(self.max_active_uploads, self.active_uploads)
            if self.active_uploads >= 2:
                self._parallel_upload.set()
        if self._wait_for_parallel_upload:
            self._parallel_upload.wait(timeout=1)
        with open(local_path, "rb") as source:
            source.read()
        with self._lock:
            self.active_uploads -= 1


class _BlockingFilesystem:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def put(self, _local_path, _remote_path):
        self.started.set()
        self.release.wait()


class _FirstUploadBlockingFilesystem:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.uploads = []
        self._calls = 0

    def put(self, local_path, remote_path):
        with self._lock:
            self._calls += 1
            call = self._calls
        if call == 1:
            self.started.set()
            self.release.wait()
        with open(local_path, "rb") as source:
            payload = source.read()
        with self._lock:
            self.uploads.append((remote_path, payload))


def _ray_log_tree(tmp_path, monkeypatch, *payloads: bytes):
    log_dir = tmp_path / "session_test" / "logs"
    log_dir.mkdir(parents=True)
    paths = []
    for index, payload in enumerate(payloads):
        path = log_dir / f"worker-{index}.out"
        path.write_bytes(payload)
        paths.append(path)
    monkeypatch.setattr(glob, "glob", lambda _pattern: [str(log_dir)])
    return paths


def test_ray_log_sync_uploads_to_node_directory_concurrently(tmp_path, monkeypatch):
    _ray_log_tree(tmp_path, monkeypatch, b"first", b"second", b"third")
    filesystem = _RecordingFilesystem(wait_for_parallel_upload=True)
    resolved_uris = []

    def resolve_destination(uri):
        resolved_uris.append(uri)
        return filesystem, "bucket/logs"

    monkeypatch.setattr(task_runtime, "fs_and_path", resolve_destination)
    sync_session = task_runtime.RayLogSyncSession("s3://logs", "node-0")

    result = sync_session.sync("periodic")

    assert result.uploaded_files == 3
    assert resolved_uris == ["s3://logs/node-0"]
    assert filesystem.max_active_uploads >= 2


def test_ray_log_sync_skips_unchanged_files(tmp_path, monkeypatch):
    first_path, _ = _ray_log_tree(tmp_path, monkeypatch, b"first", b"second")
    filesystem = _RecordingFilesystem()
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (filesystem, "bucket/logs"))
    sync_session = task_runtime.RayLogSyncSession("s3://logs", "node-0")

    first = sync_session.sync("periodic")
    second = sync_session.sync("periodic")
    first_path.write_bytes(b"first changed")
    third = sync_session.sync("final")

    assert (first.uploaded_files, first.unchanged_files) == (2, 0)
    assert (second.uploaded_files, second.unchanged_files) == (0, 2)
    assert (third.uploaded_files, third.unchanged_files) == (1, 1)


def test_ray_log_sync_continues_when_a_file_disappears_during_rotation(tmp_path, monkeypatch):
    (first_path,) = _ray_log_tree(tmp_path, monkeypatch, b"first")
    (first_path.parent / "missing.out").symlink_to(tmp_path / "already-gone.out")
    filesystem = _RecordingFilesystem()
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (filesystem, "bucket/logs"))

    result = task_runtime.RayLogSyncSession("s3://logs", "node-0").sync("periodic")

    assert result.uploaded_files == 1
    assert result.failed_files == 1


def test_final_ray_log_sync_timeout_does_not_block_teardown(tmp_path, monkeypatch):
    _ray_log_tree(tmp_path, monkeypatch, b"blocked")
    filesystem = _BlockingFilesystem()
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (filesystem, "bucket/logs"))
    monkeypatch.setenv("OT_AGENT_RAY_LOG_FINAL_SYNC_TIMEOUT_S", "0.01")
    sync_session = task_runtime.RayLogSyncSession("s3://logs", "node-0")

    status = sync_session.sync_bounded("timeout")
    assert filesystem.started.wait(timeout=1)
    filesystem.release.set()

    assert status == task_runtime.RayLogSyncWaitStatus.TIMED_OUT
    assert sync_session.sync("cleanup").unchanged_files == 1


def test_final_ray_log_sync_preempts_a_stuck_periodic_pass(tmp_path, monkeypatch):
    (ordinary_log,) = _ray_log_tree(tmp_path, monkeypatch, b"periodic upload")
    filesystem = _FirstUploadBlockingFilesystem()
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (filesystem, "bucket/logs"))
    sync_session = task_runtime.RayLogSyncSession("s3://logs", "node-0")
    periodic = threading.Thread(target=sync_session.sync, args=("periodic",), daemon=True)
    periodic.start()
    assert filesystem.started.wait(timeout=1)
    fatal_log = ordinary_log.parent / "worker-fatal.err"
    fatal_log.write_bytes(b"the original actor traceback")

    status = sync_session.sync_bounded("driver failure")

    filesystem.release.set()
    periodic.join(timeout=1)
    assert status == task_runtime.RayLogSyncWaitStatus.COMPLETED
    assert any(
        path.endswith("worker-fatal.err") and payload == fatal_log.read_bytes() for path, payload in filesystem.uploads
    )


def test_ray_log_sync_prioritizes_diagnostics_within_each_pass_budget(tmp_path, monkeypatch):
    ordinary_log, _ = _ray_log_tree(tmp_path, monkeypatch, b"ordinary-1", b"ordinary-2")
    fatal_log = ordinary_log.parent / "worker-fatal.err"
    fatal_log.write_bytes(b"fatal")
    filesystem = _RecordingFilesystem()
    uploaded_paths = []

    def record_put(local_path, remote_path):
        uploaded_paths.append(remote_path)
        _RecordingFilesystem.put(filesystem, local_path, remote_path)

    filesystem.put = record_put
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (filesystem, "bucket/logs"))
    monkeypatch.setattr(task_runtime, "RAY_LOG_SYNC_MAX_FILES_PER_PASS", 1)

    result = task_runtime.RayLogSyncSession("s3://logs", "node-0").sync("periodic")

    assert result.uploaded_files == 1
    assert result.deferred_files == 2
    assert uploaded_paths[0].endswith("worker-fatal.err")
