# This is a special case of copy_hf_dataset_to_gcs.py, where we have to manually get the URLs of the files to copy.
#
# Usage:
# python operations/download/legal/download_hupd.py --destination_path gs://somewhere/ --urls_dir gs://somewhere_publicly_readable
import argparse

from scripts.copy_hf_dataset_to_gcs import create_transfer_job, get_gcloud_project, save_urls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer a Hugging Face dataset to Google Cloud Storage.")
    parser.add_argument('--dataset_name', required=False, help="The name of the Hugging Face dataset.",
                        default="HUPD/hupd")
    parser.add_argument('--destination_path', required=True, help="The name of the destination in a GCS bucket.")
    parser.add_argument('--urls_dir', required=True,
                        help="The GCS path to save the URLs file. Should be an HTTP accessible location.")
    args = parser.parse_args()

    # Automatically determine the project ID
    project_id = get_gcloud_project()

    # Step 1: Get URLs from the Hugging Face dataset
    base_url = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/data"
    urls = [f"{base_url}/{i}.tar.gz" for i in range(2004, 2019)]

    url_tsv_name = f'{args.dataset_name.replace("/", "-")}.tsv'

    # Step 2: Save URLs to a TSV file
    url_path_on_gcs = save_urls(urls, args.urls_dir, url_tsv_name)

    # Step 4: Create a Google Storage Transfer Service job
    # STS only reads from a public url, so we need to get the public url of the file
    public_url = f"https://storage.googleapis.com/{url_path_on_gcs}"
    create_transfer_job(args.dataset_name, project_id, public_url, args.destination_path)
