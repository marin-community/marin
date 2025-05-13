from google.cloud import storage
import os
import re
import subprocess
import tempfile
import shutil
from collections import defaultdict
import argparse

# List of GCS directories to check
dirs = [
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/hf/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/hf/",
    "gs://marin-us-central2/checkpoints/tootsie-8b-soft-raccoon-3/hf/",
    "gs://marin-us-central2/checkpoints/llama-8b-tootsie-adept-phoenix/hf/",
    "gs://marin-us-central2/checkpoints/tootsie-8b-sensible-starling/hf/",
    "gs://marin-us-central1/checkpoints/tootsie-8b-deeper-starling/hf/",
]


def list_gcs_directories(gcs_path):
    """List subdirectories by examining full blob paths."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path = gcs_path[5:]  # Remove "gs://"
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])

    print(f"\nChecking: {gcs_path}")

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
                print(f"Found step directory: {full_path} with step {step_number}")
            except (IndexError, ValueError) as e:
                print(f"Error parsing step number from {dir_name}: {e}")

    print(f"Found {len(step_dirs_local)} step directories in {gcs_path}")
    return step_dirs_local


def download_from_gcs(gcs_path, local_path):
    """Download contents from a GCS path to a local directory using gsutil."""
    try:
        print(f"Downloading {gcs_path} to {local_path}...")
        cmd = ["gsutil", "-m", "cp", "-r", f"{gcs_path}*", local_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Download completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from {gcs_path}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def upload_to_huggingface(local_path, repo_id, revision):
    """Upload a local directory to Hugging Face as a specific revision."""
    try:
        print(f"Uploading checkpoint to Hugging Face as revision: {revision}")
        # Make sure huggingface_hub is installed
        subprocess.run(["pip", "install", "huggingface_hub"], check=True, capture_output=True)

        # Use the Hugging Face CLI to upload
        cmd = [
            "python",
            "-m",
            "huggingface_hub",
            "upload",
            local_path,
            "--repo-id",
            repo_id,
            "--revision",
            f"step-{revision}",
            "--commit-message",
            f"Upload checkpoint for step {revision}",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Upload completed successfully.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error uploading to Hugging Face: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints from GCS to Hugging Face")
    parser.add_argument(
        "--repo-id", required=True, help='Target Hugging Face repository ID (e.g., "username/model-name")'
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list checkpoints without uploading")
    args = parser.parse_args()

    # Initialize the GCS client
    client = storage.Client()

    # Collect all step directories
    all_step_dirs = []

    # Main execution
    for directory in dirs:
        try:
            step_dirs = list_gcs_directories(directory)
            all_step_dirs.extend(step_dirs)
        except Exception as e:
            print(f"Error listing {directory}: {e}")

    # Sort all step directories by step number
    if all_step_dirs:
        all_step_dirs.sort(key=lambda x: x[1])

        # Print sorted step directories
        print("\nAll step directories sorted by step number:")
        print("-" * 50)
        for full_path, step_number in all_step_dirs:
            print(f"- {full_path}")

        print(f"\nTotal: {len(all_step_dirs)} step directories")

        # Upload to Hugging Face
        if not args.dry_run:
            print(f"\nUploading to Hugging Face repo: {args.repo_id}")

            for full_path, step_number in all_step_dirs:
                # Create a temporary directory for downloading
                with tempfile.TemporaryDirectory() as temp_dir:
                    print(f"\nProcessing step {step_number} from {full_path}")

                    # Download from GCS
                    if download_from_gcs(full_path, temp_dir):
                        # Upload to HF
                        if upload_to_huggingface(temp_dir, args.repo_id, step_number):
                            print(f"Successfully uploaded step {step_number} to HF repo {args.repo_id}")
                        else:
                            print(f"Failed to upload step {step_number}")
                    else:
                        print(f"Failed to download step {step_number}")

            print("\nUpload process completed.")
        else:
            print("\nDry run - showing what would be uploaded:")
            print("-" * 50)

            for i, (full_path, step_number) in enumerate(all_step_dirs):
                print(f"\nCheckpoint {i+1}/{len(all_step_dirs)}:")
                print(f"  Source: {full_path}")
                print(f"  Target repo: {args.repo_id}")
                print(f"  Revision: step-{step_number}")
                print(f"  Commit message: Upload checkpoint for step {step_number}")

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
                            print(f"  Example files that would be uploaded ({min(len(files), 5)} of {len(files)}):")
                            for file in files[:5]:
                                print(f"    - {os.path.basename(file)}")
                            if len(files) > 5:
                                print(f"    - ... and {len(files) - 5} more")
                except Exception as e:
                    print(f"  Could not list files: {e}")

            print("\nDry run completed - no actual uploads performed.")
    else:
        print("\nNo step directories found in any of the paths.")
        print("You might want to check if:")
        print("1. The paths are correct")
        print("2. You have permissions to access these buckets")
        print("3. There are step directories in these locations")


if __name__ == "__main__":
    # Check if application default credentials are set
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        print("Make sure you're authenticated with Google Cloud before running this script.")
        print("You can authenticate using: gcloud auth application-default login")

    # Check if logged in to Hugging Face
    try:
        result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True)
        if "Not logged in" in result.stdout:
            print("Not logged in to Hugging Face. Please login using:")
            print("huggingface-cli login")
            exit(1)
    except FileNotFoundError:
        print("Hugging Face CLI not found. Installing huggingface_hub...")
        subprocess.run(["pip", "install", "huggingface_hub"], check=True)
        print("Please login to Hugging Face after installation:")
        print("huggingface-cli login")
        exit(1)

    main()
