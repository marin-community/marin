"""Publish completed Hugging Face model directories."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from glob import escape
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, RepoFile
from loguru import logger

from marinskyrl.checkpoint_paths import policy_export_path
from marinskyrl.hf_model import hugging_face_hub_online
from marinskyrl.hugging_face_retry import call_with_hugging_face_retry
from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, ModelManifest, ModelManifestFile, sha256_file
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.hf_export_schema import DEFAULT_HF_HUB_REVISION, DEFAULT_HF_UPLOAD_MODE, HFExportRequest, HFUploadMode
from skyrl_train.hf_model_io import verify_completed_hf_model_export
from skyrl_train.io import io


def publish_verified_hf_export(request: HFExportRequest, staging_dir: str) -> None:
    """Upload an object-store export from a CPU job without loading model tensors."""
    if request.hf_hub_repo_id is None:
        raise ValueError("Hub publication requires hf_hub_repo_id")
    export_path = policy_export_path(request.export_path, request.step)
    verify_completed_hf_model_export(export_path)
    started = time.monotonic()
    logger.info("Starting Hub publication of {} to {}", export_path, request.hf_hub_repo_id)
    HuggingFacePublisher(
        repo_id=request.hf_hub_repo_id,
        private=request.hf_hub_private,
        revision=request.hf_hub_revision,
        upload_mode=request.hf_upload_mode,
        staging_dir=staging_dir,
    ).publish(export_path, request.step)
    logger.info("Hub publication completed in {:.3f} seconds", time.monotonic() - started)


@dataclass(frozen=True)
class HuggingFacePublisher:
    """Publish one completed model at the repository root and optional step archive."""

    repo_id: str
    private: bool
    revision: str = DEFAULT_HF_HUB_REVISION
    upload_mode: HFUploadMode = DEFAULT_HF_UPLOAD_MODE
    staging_dir: str | None = None

    def publish(self, export_path: str, step: int) -> None:
        if not io.exists(export_path):
            raise FileNotFoundError(f"HF export not found: {export_path}")
        manifest_path = join_resource_path(export_path, MODEL_MANIFEST_FILENAME)
        if not io.exists(manifest_path):
            raise ValueError(f"Hugging Face publication requires a model manifest: {manifest_path}")
        api = HfApi()
        with hugging_face_hub_online():
            call_with_hugging_face_retry(
                lambda: api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=self.private,
                    exist_ok=True,
                ),
                operation=f"create Hugging Face repository {self.repo_id}",
            )
            repo = call_with_hugging_face_retry(
                lambda: api.model_info(self.repo_id), operation=f"read Hugging Face repository {self.repo_id}"
            )
            if repo.private != self.private:
                raise ValueError(
                    f"Hugging Face repository {self.repo_id} visibility differs from hf_hub_private={self.private}"
                )
            staging_started = time.monotonic()
            with io.local_read_dir(export_path, staging_dir=self.staging_dir) as local_dir:
                upload_started = time.monotonic()
                logger.info("Hub publication staging completed in {:.3f} seconds", upload_started - staging_started)
                root = Path(local_dir)
                marker = root / MODEL_MANIFEST_FILENAME
                manifest = ModelManifest.from_mapping(json.loads(marker.read_bytes()), str(marker))
                files = (
                    *manifest.files,
                    ModelManifestFile(path=marker.name, size=marker.stat().st_size, sha256=sha256_file(marker)),
                )
                self._upload(api, root, files, "", f"Upload checkpoint at step {step}")
                if self.upload_mode is HFUploadMode.ALL:
                    self._upload(
                        api, root, files, f"checkpoints/step_{step}", f"Archive checkpoint at step {step}"
                    )
                logger.info("Hub upload completed in {:.3f} seconds", time.monotonic() - upload_started)

    def _upload(
        self,
        api: HfApi,
        root: Path,
        files: tuple[ModelManifestFile, ...],
        destination: str,
        message: str,
    ) -> None:
        commit = call_with_hugging_face_retry(
            lambda: api.upload_folder(
                folder_path=str(root),
                repo_id=self.repo_id,
                path_in_repo=destination,
                repo_type="model",
                revision=self.revision,
                commit_message=message,
                allow_patterns=[escape(file.path) for file in files],
            ),
            operation=f"upload Hugging Face checkpoint {self.repo_id}@{self.revision}",
        )
        expected = {str(PurePosixPath(destination, file.path)): file for file in files}
        tree = call_with_hugging_face_retry(
            lambda: api.get_paths_info(self.repo_id, list(expected), repo_type="model", revision=commit.oid),
            operation=f"read Hugging Face checkpoint {self.repo_id}@{commit.oid}",
        )
        uploaded = {file.path: file for file in tree if isinstance(file, RepoFile)}
        for path, source in expected.items():
            remote = uploaded.get(path)
            if remote is None or remote.size != source.size or self._sha256(api, remote, commit.oid) != source.sha256:
                raise ValueError(f"Hugging Face file differs from the model manifest: {self.repo_id}@{commit.oid}:{path}")

    def _sha256(self, api: HfApi, file: RepoFile, revision: str) -> str:
        if file.lfs is not None:
            return file.lfs.sha256
        downloaded = call_with_hugging_face_retry(
            lambda: api.hf_hub_download(self.repo_id, file.path, repo_type="model", revision=revision),
            operation=f"read Hugging Face file {self.repo_id}@{revision}:{file.path}",
        )
        return sha256_file(Path(downloaded))
