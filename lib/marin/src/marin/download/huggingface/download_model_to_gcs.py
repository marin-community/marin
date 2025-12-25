"""
Script that downloads a model from huggingface and then uploads to GCS
"""

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import shutil
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Hugging Face model repo and upload its files to a GCS path."
        )
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model repo id (e.g., 'org/model').",
    )
    parser.add_argument(
        "--gcs-uri",
        type=str,
        default="gs://marin-us-central2/models/qwen2.5-7b-instruct",
        help="Destination GCS URI (e.g., 'gs://bucket/path').",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional revision (branch, tag, or commit sha).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token; defaults to HF_TOKEN env var if set.",
    )
    parser.add_argument(
        "--extra-gsutil-args",
        type=str,
        default="",
        help="Extra arguments passed to gsutil rsync (e.g. '-d').",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def validate_gcs_uri(gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs-uri must start with 'gs://'")


def run_gsutil_rsync(src_dir: Path, gcs_uri: str, extra_args: Optional[str] = None) -> None:
    args = ["gsutil", "-m", "rsync", "-r"]
    if extra_args:
        args.extend(extra_args.split())
    args.extend([str(src_dir), gcs_uri])
    logging.info("Running: %s", " ".join(args))
    subprocess.run(args, check=True)


def download_to_directory(
    model_id: str, destination_dir: Path, revision: Optional[str], hf_token: Optional[str]
) -> None:
    logging.info("Starting download: %s", model_id)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(destination_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        token=hf_token,
    )
    logging.info("Completed download to: %s", destination_dir)


def main() -> None:
    args = parse_args()
    configure_logging()

    validate_gcs_uri(args.gcs_uri)

    with tempfile.TemporaryDirectory(prefix="hf_model_") as tmpdir:
        tmp_path = Path(tmpdir)
        # Download the model files directly into the temp directory
        download_to_directory(
            model_id=args.model_id,
            destination_dir=tmp_path,
            revision=args.revision,
            hf_token=args.hf_token,
        )

        # Sync directory to GCS
        run_gsutil_rsync(src_dir=tmp_path, gcs_uri=args.gcs_uri, extra_args=args.extra_gsutil_args)

        logging.info("Upload complete: %s", args.gcs_uri)

        # Clean up the temporary directory
        shutil.rmtree(tmp_path)
        logging.info("Temporary directory cleaned up: %s", tmp_path)

if __name__ == "__main__":
    main()


