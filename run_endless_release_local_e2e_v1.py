"""Run the FULL Endless oracle-release path locally end to end against a local Hub fixture.

Every real guard runs: the frozen-pin byte checks, the license lineage gates, the
upstream tree walk and downloads, the card render, both exact-pin license checks
(the upstream pin and the published pin through its wrapper chain), and the upload
readback. Only the Hub WRITES are a local fixture; the reads resolve the real
upstream pins, and the published pin's own card is served from the fixture.
"""
import hashlib
import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import run_endless_oracle_release_v1 as worker
from experiments.post_training.task_curation import license_publication, license_resolution

COMMIT = "e" * 40
REPO = worker.REPO
REAL_HF_API = license_publication.HfApi
REAL_HF_DOWNLOAD = license_publication.hf_hub_download
REAL_RESOLVE_FETCH = license_resolution._fetch


class FixtureHub:
    """The Hub writes land in a local folder; the reads serve the written bytes."""

    def __init__(self, store: Path):
        self.store = store
        self.writes = []

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        assert repo_id == REPO and repo_type == "dataset" and private is True
        (self.store / repo_id).mkdir(parents=True, exist_ok=exist_ok)
        (self.store / repo_id / ".gitattributes").write_bytes(b"")
        self.writes.append("create")

    def list_repo_tree(self, repo_id, repo_type, revision, recursive):
        if repo_id == REPO:
            folder = self.store / repo_id
            return [SimpleNamespace(path=str(path.relative_to(folder)), size=path.stat().st_size)
                    for path in sorted(folder.rglob("*")) if path.is_file()]
        return REAL_HF_API().list_repo_tree(repo_id, repo_type=repo_type, revision=revision, recursive=recursive)

    def repo_info(self, repo_id, **kwargs):
        if repo_id == REPO:
            return SimpleNamespace(private=True, sha=COMMIT)
        return REAL_HF_API().repo_info(repo_id, **kwargs)

    def upload_folder(self, repo_id, repo_type, folder_path, allow_patterns, parent_commit, commit_message):
        folder = self.store / repo_id
        folder.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            source = Path(folder_path) / name
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        self.writes.append("upload")
        return SimpleNamespace(oid=COMMIT)

    def dataset_info(self, dataset, *, revision):
        if dataset == REPO:
            folder = self.store / REPO
            files = [SimpleNamespace(rfilename=str(path.relative_to(folder)))
                     for path in sorted(folder.rglob("*")) if path.is_file()]
            return SimpleNamespace(private=True, sha=revision or "f" * 40, siblings=files)
        return REAL_HF_API().dataset_info(dataset, revision=revision)

    def serve_resolve(self, url):
        """Serve a published-pin resolve URL from the local store (the card readback)."""
        marker = "/resolve/"
        head, tail = url.split(marker, 1)
        assert head.endswith(REPO), url
        name = tail.split("?")[0].rsplit("/", 1)[-1]
        return (self.store / REPO / name).read_bytes()

    def download(self, repo_id, name, repo_type, revision):
        folder = self.store / repo_id
        path = folder / name
        if path.is_file():
            return str(path)
        return REAL_HF_DOWNLOAD(repo_id, name, repo_type=repo_type, revision=revision)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="endless-release-e2e-"))
    store = tmp / "hub"
    store.mkdir()
    hub = FixtureHub(store)

    def routed_download(repo_id, name, repo_type=None, revision=None):
        path = store / repo_id / name
        if path.is_file():
            return str(path)
        return REAL_HF_DOWNLOAD(repo_id, name, repo_type=repo_type, revision=revision)

    worker.get_job_info = lambda: SimpleNamespace(
        attempt_id=0, attempt_uid="local-e2e", job_id=SimpleNamespace(to_wire=lambda: "/local/endless-release-e2e"))
    license_publication.HfApi = lambda: hub
    license_publication.hf_hub_download = routed_download
    license_resolution.HfApi = lambda: hub
    license_resolution.hf_hub_download = routed_download
    license_resolution._fetch = lambda url, **kwargs: (
        hub.serve_resolve(url) if "/open-athena/fweindel-endless-oracle-filtered-20261124-v1/" in url
        else REAL_RESOLVE_FETCH(url, **kwargs))
    worker.OUT = str(tmp / "oracle-release-local")
    sys.argv = ["run_endless_oracle_release_v1.py"]
    worker.main()
    result = json.loads((Path(worker.OUT) / "result.json").read_bytes())
    assert result["status"] == "ENDLESS_ORACLE_RELEASE_COMPLETE" and result["admitted"] == 1976
    published_card = (store / REPO / "README.md").read_bytes()
    assert b"license: mit" in published_card and b"| endless_v2 |" in published_card
    evidence = json.loads((Path(worker.OUT) / "filtration-evidence.json").read_bytes())
    assert evidence["private"] is True and evidence["filtered_release"]["repo"] == REPO
    verification = json.loads((Path(worker.OUT) / "verification.json").read_bytes())
    publication = json.loads((Path(worker.OUT) / "publication.json").read_bytes())
    assert publication["license_check"]["verdict"] == "PASS"
    assert publication["repo"] == REPO and publication["revision"]
    gate = json.loads(json.dumps(publication["license_check"]))
    assert gate["receipt"]["dataset_id"] in (REPO, "FWeindel/validated-tasks")
    print(json.dumps({"status": "PASS", "writes": hub.writes, "admitted": result["admitted"],
                      "repo": REPO, "revision": result["release"]["revision"][:12],
                      "output": worker.OUT}, indent=1))


if __name__ == "__main__":
    main()
