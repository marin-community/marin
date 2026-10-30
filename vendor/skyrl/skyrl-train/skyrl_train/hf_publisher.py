"""Publish completed Hugging Face model directories."""

from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import HfApi

from marinskyrl.hf_model import hugging_face_hub_online
from marinskyrl.hugging_face_retry import call_with_hugging_face_retry
from skyrl_train.hf_export_schema import DEFAULT_HF_HUB_REVISION, DEFAULT_HF_UPLOAD_MODE, HFUploadMode
from skyrl_train.io import io


@dataclass(frozen=True)
class HuggingFacePublisher:
    """Publish one completed model at the repository root and optional step archive."""

    repo_id: str
    private: bool = False
    revision: str = DEFAULT_HF_HUB_REVISION
    upload_mode: HFUploadMode = DEFAULT_HF_UPLOAD_MODE
    api: HfApi | None = None

    def publish(self, export_path: str, step: int) -> None:
        if not io.exists(export_path):
            raise FileNotFoundError(f"HF export not found: {export_path}")
        api = self.api or HfApi()
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
            with io.local_read_dir(export_path) as local_dir:
                call_with_hugging_face_retry(
                    lambda: api.upload_folder(
                        folder_path=str(local_dir),
                        repo_id=self.repo_id,
                        path_in_repo="",
                        repo_type="model",
                        revision=self.revision,
                        commit_message=f"Upload checkpoint at step {step}",
                    ),
                    operation=f"upload Hugging Face checkpoint {self.repo_id}@{self.revision}",
                )
                if self.upload_mode is HFUploadMode.ALL:
                    call_with_hugging_face_retry(
                        lambda: api.upload_folder(
                            folder_path=str(local_dir),
                            repo_id=self.repo_id,
                            path_in_repo=f"checkpoints/step_{step}",
                            repo_type="model",
                            revision=self.revision,
                            commit_message=f"Archive checkpoint at step {step}",
                        ),
                        operation=f"archive Hugging Face checkpoint {self.repo_id}@{self.revision}",
                    )
