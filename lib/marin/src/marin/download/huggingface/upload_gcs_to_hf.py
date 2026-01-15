# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Upload GCS to Hugging Face (HF) Script

This script transfers model checkpoints or other content from Google Cloud Storage (GCS)
to Hugging Face repositories. It handles:
- Finding checkpoint directories in GCS buckets
- Downloading the content locally (to a temporary directory)
- Uploading to a specified Hugging Face repository with appropriate versioning
- Supporting dry-run mode to preview what would be uploaded

Usage as a script:
  python upload_gcs_to_hf.py --repo-id="organization/model-name" [--dry-run] [--directory="gs://bucket/path"]

Usage as an ExecutorStep:
  upload_step = upload_gcs_to_hf_step(
      hf_repo_id="organization/model-name",
      gcs_directories=["gs://bucket/path/to/model"],
      dry_run=False
  )
"""

import argparse
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

from google.cloud import storage
from google.cloud.storage import transfer_manager
from huggingface_hub import HfApi, create_repo

from marin.execution import ExecutorStep, step, deferred

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class UploadConfig:
    """Configuration for uploading from GCS to Hugging Face."""

    hf_repo_id: str
    gcs_directories: list[str] = field(default_factory=list)
    dry_run: bool = False
    wait_for_completion: bool = True  # Added for compatibility with other configs


# Default GCS directories to check if none specified
DEFAULT_GCS_DIRS = [
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/hf/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/hf/",
    "gs://marin-us-central2/checkpoints/tootsie-8b-soft-raccoon-3/hf/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-adept-phoenix/hf/",
    "gs://marin-us-central2/checkpoints/tootsie-8b-sensible-starling/hf/",
    "gs://marin-us-central1/checkpoints/tootsie-8b-deeper-starling/hf/",
]


def list_gcs_directories(gcs_path: str) -> list[tuple[str, int]]:
    """List subdirectories by examining full blob paths."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path = gcs_path[5:]  # Remove "gs://"
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])

    logger.info(f"Checking: {gcs_path}")

    # Get the bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List blobs with this prefix (without delimiter to get all)
    blobs = bucket.list_blobs(prefix=prefix)

    # Extract potential directories from blob paths
    directories = set()
    step_pattern = re.compile(r"step-\d+")

    for blob in blobs:
        # Remove the prefix to get the relative path
        relative_path = blob.name[len(prefix) :]

        # Skip if there's no relative path
        if not relative_path:
            continue

        # Extract the first directory level
        parts = relative_path.strip("/").split("/")
        if parts:
            first_dir = parts[0]

            # Check if it's a step directory
            if step_pattern.match(first_dir):
                directories.add(first_dir)

    # Process the directories we found
    step_dirs_local = []
    for dir_name in directories:
        if step_pattern.match(dir_name):
            try:
                step_number = int(dir_name.split("-")[1])
                full_path = f"{gcs_path}{dir_name}/"
                step_dirs_local.append((full_path, step_number))
                logger.info(f"Found step directory: {full_path} with step {step_number}")
            except (IndexError, ValueError) as e:
                logger.error(f"Error parsing step number from {dir_name}: {e}")

    logger.info(f"Found {len(step_dirs_local)} step directories in {gcs_path}")
    return step_dirs_local


def download_from_gcs(gcs_path: str, local_path: str) -> bool:
    """Download contents from a GCS path to a local directory using the GCS transfer manager."""
    logger.info(f"Downloading {gcs_path} to {local_path}...")

    # Parse the GCS path (format: gs://bucket-name/path/to/files)
    if not gcs_path.startswith("gs://"):
        logger.error(f"Invalid GCS path format: {gcs_path}")
        return False

    bucket_name = gcs_path[5:].split("/")[0]
    prefix = "/".join(gcs_path[5:].split("/")[1:])

    # Handle wildcard at the end (the original had f"{gcs_path}*")
    if prefix.endswith("*"):
        prefix = prefix[:-1]

    # Initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all matching blobs
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logger.error(f"No files found in {gcs_path}")
        return False

    total_files = len(blobs)
    logger.info(f"Found {total_files} files to download from {gcs_path}")

    # Get the blob names to download (excluding directory placeholders)
    blob_names = []
    for blob in blobs:
        if not blob.name.endswith("/"):
            blob_names.append(blob.name)

    if len(blob_names) < total_files:
        logger.info(f"Filtered out {total_files - len(blob_names)} directory markers")

    # Ensure local directory exists
    os.makedirs(local_path, exist_ok=True)

    # Log the first few blob names to debug issues
    if blob_names:
        logger.info(f"Sample blob names (first 3): {', '.join(blob_names[:3])}")

    # Use transfer manager to download all blobs in parallel
    logger.info(f"Starting parallel download of {len(blob_names)} files...")

    transfer_manager.download_many_to_path(
        bucket=bucket,
        blob_names=blob_names,
        destination_directory=local_path,
        max_workers=8,
        create_directories=True,
        worker_type="process",
        raise_exception=True,
    )

    logger.info(f"Download completed successfully. Downloaded {len(blob_names)} files.")
    return True


def checkpoint_exists(repo_id: str, step: int, version_name: str) -> bool:
    """Check if a specific revision exists in a Hugging Face repository."""
    try:
        api = HfApi()
        commits = api.list_repo_commits(repo_id=repo_id)
        for commit in commits:
            if f"step {step}" in commit.title:
                return True
        return False
    except Exception:
        return False


def extract_version_from_path(gcs_path: str) -> str:
    """Extract the version name from a GCS path."""
    # Extract model name from path like "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/"
    parts = gcs_path.strip("/").split("/")
    return parts[-3]


def upload_to_huggingface(local_path: str, repo_id: str, step: int, version_name: str) -> bool:
    """Upload a local directory to Hugging Face as a specific revision."""
    logger.info(f"Uploading checkpoint {version_name}, step {step} to Hugging Face")

    # Check if repo exists, create if not
    api = HfApi()
    create_repo(repo_id=repo_id, exist_ok=True)
    # Upload the directory
    result = api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        commit_message=f"Upload checkpoint for step {step} ({version_name})",
    )
    try:
        api.delete_tag(repo_id=repo_id, tag=version_name)
    except Exception:
        logger.info("Creating tag for the first time")
    api.create_tag(repo_id=repo_id, tag=version_name)
    logger.info("Upload completed successfully.")
    logger.info(f"Commit URL: {result.commit_url}")
    return True


def upload_gcs_to_hf(cfg: UploadConfig) -> None:
    """Main function to upload model checkpoints from GCS to Hugging Face."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Collect all step directories
    all_step_dirs = []

    # Determine which directories to process
    directories_to_process = cfg.gcs_directories if cfg.gcs_directories else DEFAULT_GCS_DIRS

    # Process each directory
    for directory in directories_to_process:
        try:
            step_dirs = list_gcs_directories(directory)
            all_step_dirs.extend(step_dirs)
        except Exception as e:
            logger.error(f"Error listing {directory}: {e}")

    # Sort all step directories by step number
    if all_step_dirs:
        all_step_dirs.sort(key=lambda x: x[1])

        # Print sorted step directories
        logger.info("\nAll step directories sorted by step number:")
        logger.info("-" * 50)
        for full_path, _step_number in all_step_dirs:
            logger.info(f"- {full_path}")

        logger.info(f"\nTotal: {len(all_step_dirs)} step directories")

        # Upload to Hugging Face
        if not cfg.dry_run:
            logger.info(f"\nUploading to Hugging Face repo: {cfg.hf_repo_id}")

            for full_path, step_number in all_step_dirs:
                # Extract version name from the path
                version_name = extract_version_from_path(full_path)

                # Check if this checkpoint already exists
                if checkpoint_exists(cfg.hf_repo_id, step_number, version_name):
                    logger.info(
                        f"Step {step_number} for {version_name} already exists in HF repo {cfg.hf_repo_id}, skipping"
                    )
                    continue

                # Create a temporary directory for downloading
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger.info(f"\nProcessing step {step_number} from {full_path} ({version_name})")

                    # Download from GCS
                    if download_from_gcs(full_path, temp_dir):
                        # Upload to HF
                        if upload_to_huggingface(temp_dir, cfg.hf_repo_id, step_number, version_name):
                            logger.info(
                                f"Successfully uploaded step {step_number} ({version_name}) to HF repo {cfg.hf_repo_id}"
                            )
                        else:
                            logger.error(f"Failed to upload step {step_number}")
                    else:
                        logger.error(f"Failed to download step {step_number}")

            logger.info("\nUpload process completed.")
        else:
            logger.info("\nDry run - showing what would be uploaded:")
            logger.info("-" * 50)

            for i, (full_path, step_number) in enumerate(all_step_dirs):
                version_name = extract_version_from_path(full_path)
                logger.info(f"\nCheckpoint {i + 1}/{len(all_step_dirs)}:")
                logger.info(f"  Source: {full_path}")
                logger.info(f"  Target repo: {cfg.hf_repo_id}")
                logger.info(f"  Revision: {version_name}")
                logger.info(f"  Commit message: Upload checkpoint for step {step_number} ({version_name})")

                # Try to estimate what files would be uploaded
                try:
                    # Use gsutil to list files in the directory
                    cmd = ["gsutil", "ls", f"{full_path}"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        files = result.stdout.strip().split("\n")
                        # Filter out empty strings and limit to 5 for display
                        files = [f for f in files if f]

                        if files:
                            logger.info(
                                f"  Example files that would be uploaded ({min(len(files), 5)} of {len(files)}):"
                            )
                            for file in files[:5]:
                                logger.info(f"    - {os.path.basename(file)}")
                            if len(files) > 5:
                                logger.info(f"    - ... and {len(files) - 5} more")
                except Exception as e:
                    logger.error(f"  Could not list files: {e}")

            logger.info("\nDry run completed - no actual uploads performed.")
    else:
        logger.warning("\nNo step directories found in any of the paths.")
        logger.warning("You might want to check if:")
        logger.warning("1. The paths are correct")
        logger.warning("2. You have permissions to access these buckets")
        logger.warning("3. There are step directories in these locations")


def upload_gcs_to_hf_step(
    hf_repo_id: str,
    gcs_directories: list[str] | None = None,
    dry_run: bool = False,
) -> ExecutorStep:
    """
    Factory function to create an ExecutorStep for uploading GCS content to Hugging Face.

    Args:
        hf_repo_id: Target Hugging Face repository ID (e.g., "username/model-name")
        gcs_directories: List of GCS directories to process. If None, uses DEFAULT_GCS_DIRS
        dry_run: If True, only lists checkpoints without uploading

    Returns:
        ExecutorStep: An executor step that performs the upload
    """
    upload_gcs_to_hf_deferred = deferred(upload_gcs_to_hf)

    @step(name="upload_gcs_to_hf")
    def _step():
        return upload_gcs_to_hf_deferred(
            UploadConfig(
                hf_repo_id=hf_repo_id,
                gcs_directories=gcs_directories or [],
                dry_run=dry_run,
            )
        )

    return _step()


def main():
    """Command line entry point for direct script usage."""
    parser = argparse.ArgumentParser(description="Upload checkpoints from GCS to Hugging Face")
    parser.add_argument(
        "--repo-id", required=True, help='Target Hugging Face repository ID (e.g., "username/model-name")'
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list checkpoints without uploading")
    parser.add_argument(
        "--directories",
        nargs="+",
        help="Process specific GCS directories instead of the built-in list. Multiple directories can be provided.",
    )
    args = parser.parse_args()

    # Create config from args
    config = UploadConfig(
        hf_repo_id=args.repo_id, gcs_directories=args.directories if args.directories else [], dry_run=args.dry_run
    )

    # Check if application default credentials are set
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        logger.warning("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        logger.warning("Make sure you're authenticated with Google Cloud before running this script.")
        logger.warning("You can authenticate using: gcloud auth application-default login")

    # Run the upload function
    upload_gcs_to_hf(config)


if __name__ == "__main__":
    main()
