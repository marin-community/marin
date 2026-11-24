# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation import harbor_hub
from experiments.post_training.task_curation.harbor_packages import package_archive

RELEASE_DIGEST = "a" * 64
RELEASE = f"fixture/tasks@sha256:{RELEASE_DIGEST}"
# Computed with the vendored Harbor Packager on the two original fixture files.
TASK_DIGEST = "7c3c451d09512efd56ecfc6acfb5c89b3d108a934c643440c37c116f2f4f7935"
TASK_NAMES = ("one", "two")
REFS = tuple(f"fixture/{name}@sha256:{TASK_DIGEST}" for name in TASK_NAMES)


@dataclass
class RegistryState:
    archive: bytes
    modified_archive: bytes
    archive_with_extra_file: bytes
    requests: Counter[str] = field(default_factory=Counter)
    fail_once: set[str] = field(default_factory=set)


def registry_handler(state: RegistryState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path)
            if route.path == "/rest/v1/dataset_version":
                filters = parse_qs(route.query)
                rows = [{"id": "release", "content_hash": RELEASE_DIGEST}]
                if filters["content_hash"] != [f"eq.{RELEASE_DIGEST}"]:
                    rows = []
            elif route.path == "/rest/v1/dataset_version_file":
                rows = []
            elif route.path == "/rest/v1/dataset_version_task":
                rows = [
                    {
                        "task_version": {
                            "content_hash": TASK_DIGEST,
                            "archive_path": f"{name}.tar.gz",
                            "package": {"name": name, "org": {"name": "fixture"}},
                        }
                    }
                    for name in TASK_NAMES
                ]
            else:
                name = route.path.rsplit("/", 1)[-1].removesuffix(".tar.gz")
                state.requests[name] += 1
                if name in state.fail_once:
                    state.fail_once.remove(name)
                    self.send_error(503)
                    return
                self._reply(state.archive)
                return
            self._reply(json.dumps(rows).encode())

        def _reply(self, payload: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


@pytest.fixture
def registry(tmp_path, monkeypatch) -> Iterator[RegistryState]:
    package = tmp_path / "package"
    (package / "tests").mkdir(parents=True)
    verifier = package / "tests/test.sh"
    verifier.write_text("exit 0\n")
    verifier.chmod(0o755)
    instruction = package / "instruction.md"
    instruction.write_text("Solve the task.\n")
    archive = package_archive(package)
    ignored = package / "tests/extra.pyc"
    ignored.write_bytes(b"unexpected file")
    archive_with_extra_file = package_archive(package)
    ignored.unlink()
    instruction.write_text("Modified instruction")
    state = RegistryState(archive, package_archive(package), archive_with_extra_file)

    with ThreadingHTTPServer(("127.0.0.1", 0), registry_handler(state)) as server:
        monkeypatch.setattr(harbor_hub, "REGISTRY_URL", f"http://127.0.0.1:{server.server_port}")
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield state
        finally:
            server.shutdown()
            thread.join()


def test_acquisition_preserves_native_archives_and_resumes_downloads(tmp_path, registry):
    registry.fail_once.add("two")
    output = tmp_path / "output"
    with pytest.raises(requests.HTTPError):
        harbor_hub.acquire_release(RELEASE, StoragePath(str(output)), ())
    assert not (output / "release.json").exists()
    harbor_hub.acquire_release(RELEASE, StoragePath(str(output)), ())

    expected = {hashlib.sha256(ref.encode()).hexdigest() + ".tar.gz" for ref in REFS}
    archives = set((output / "tasks").glob("*.tar.gz"))
    assert {path.name for path in archives} == expected
    assert all(path.read_bytes() == registry.archive for path in archives)
    assert json.loads((output / "release.json").read_text())["selected"] == list(REFS)
    assert registry.requests == {"one": 1, "two": 2}


@pytest.mark.parametrize("extra_file", [False, True], ids=["modified-file", "extra-file"])
def test_acquisition_rejects_unverified_archive_without_publishing(tmp_path, registry, extra_file):
    registry.archive = registry.archive_with_extra_file if extra_file else registry.modified_archive
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        harbor_hub.acquire_release(RELEASE, StoragePath(str(output)), REFS[:1])
    assert not output.exists()
