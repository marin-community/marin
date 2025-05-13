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
  upload_step = ExecutorStep(
      name="upload_model_to_hf",
      fn=upload_gcs_to_hf,
      config=UploadConfig(
          hf_repo_id="organization/model-name",
          gcs_directories=["gs://bucket/path/to/model"],
          dry_run=False
      )
  )
"""

from google.cloud import storage
import os
import re
import subprocess
import tempfile
import shutil
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from huggingface_hub import HfApi, create_repo
import argparse

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class UploadConfig:
    """Configuration for uploading from GCS to Hugging Face."""

    hf_repo_id: str
    gcs_directories: List[str] = field(default_factory=list)
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


def list_gcs_directories(gcs_path: str) -> List[Tuple[str, int]]:
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
    """Download contents from a GCS path to a local directory using gsutil."""
    try:
        logger.info(f"Downloading {gcs_path} to {local_path}...")
        cmd = ["gsutil", "-m", "cp", "-r", f"{gcs_path}*", local_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Download completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading from {gcs_path}: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def revision_exists(repo_id: str, revision: int) -> bool:
    """Check if a specific revision exists in a Hugging Face repository."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.repo_info(repo_id=repo_id, revision=f"step-{revision}")
            return True
        except Exception:
            return False
    except Exception:
        return False


def extract_version_from_path(gcs_path: str) -> str:
    """Extract the version name from a GCS path."""
    # Extract model name from path like "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/"
    parts = gcs_path.strip("/").split("/")
    return parts[-3]


def upload_to_huggingface(local_path: str, repo_id: str, revision: int, version_name: str) -> bool:
    """Upload a local directory to Hugging Face as a specific revision."""
    try:
        logger.info(f"Uploading checkpoint {version_name} to Hugging Face as revision: {revision}")

        # Check if repo exists, create if not
        api = HfApi()
        create_repo(repo_id=repo_id, exist_ok=True)
        version_info = f" ({version_name})" if version_name else ""
        # Upload the directory
        result = api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            revision=f"step-{revision}",
            commit_message=f"Upload checkpoint for step {revision}{version_info}",
        )
        logger.info(f"Upload completed successfully.")
        logger.info(f"Commit URL: {result.commit_url}")
        return True
    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        return False


def upload_gcs_to_hf(cfg: UploadConfig) -> None:
    """Main function to upload model checkpoints from GCS to Hugging Face."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the GCS client
    client = storage.Client()

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
        for full_path, step_number in all_step_dirs:
            logger.info(f"- {full_path}")

        logger.info(f"\nTotal: {len(all_step_dirs)} step directories")

        # Upload to Hugging Face
        if not cfg.dry_run:
            logger.info(f"\nUploading to Hugging Face repo: {cfg.hf_repo_id}")

            for full_path, step_number in all_step_dirs:
                # Check if this revision already exists
                if revision_exists(cfg.hf_repo_id, step_number):
                    logger.info(f"Step {step_number} already exists in HF repo {cfg.hf_repo_id}, skipping")
                    continue

                # Extract version name from the path
                version_name = extract_version_from_path(full_path)

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
                logger.info(f"\nCheckpoint {i+1}/{len(all_step_dirs)}:")
                logger.info(f"  Source: {full_path}")
                logger.info(f"  Target repo: {cfg.hf_repo_id}")
                logger.info(f"  Revision: step-{step_number}")
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
