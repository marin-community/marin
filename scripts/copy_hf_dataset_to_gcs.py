import os
import datasets
import argparse
from datasets import DownloadConfig
from huggingface_hub import HfFileSystem, hf_hub_url
from google.auth import default
from google.cloud import storage, storage_transfer_v1
from google.protobuf import duration_pb2
from datetime import datetime, timedelta

def get_urls_to_copy(dataset_name: str, **kwargs):
    fs = HfFileSystem()
    ds_builder = datasets.load_dataset_builder(dataset_name, **kwargs)
    config = ds_builder.config
    download_config = DownloadConfig(token=ds_builder.token, storage_options=ds_builder.storage_options)
    config._resolve_data_files(base_path=ds_builder.base_path, download_config=download_config)

    data_files = config.data_files
    if data_files is None:
        raise ValueError("Data files are not available for this dataset")

    urls = []

    for split, files in data_files.items():
        for file in files:
            resolved_path = fs.resolve_path(file)
            url = hf_hub_url(
                resolved_path.repo_id,
                resolved_path.path_in_repo,
                repo_type=resolved_path.repo_type,
                revision=resolved_path.revision,
            )
            urls.append(url)
    return urls

def save_urls_to_tsv(urls, file_path):
    with open(file_path, 'w') as f:
        f.write("TsvHttpData-1.0\n")
        for url in urls:
            f.write(f"{url}\n")
    print(f"URLs have been saved to {file_path}")

def upload_file_to_gcs(bucket_name, source_file, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file)

    print(f"File {source_file} uploaded to {destination_blob_name} in bucket {bucket_name}.")

def create_transfer_job(project_id, source_file, gcs_bucket_name):
    client = storage_transfer_v1.StorageTransferServiceClient()

    transfer_job = {
        'project_id': project_id,
        'transfer_spec': {
            'http_data_source': {
                'list_url': source_file
            },
            'gcs_data_sink': {
                'bucket_name': gcs_bucket_name
            },
        },
        'status': 'ENABLED',
        'schedule': {
            'schedule_start_date': {
                'year': datetime.now().year,
                'month': datetime.now().month,
                'day': datetime.now().day
            },
            'schedule_end_date': {
                'year': (datetime.now() + timedelta(days=1)).year,
                'month': (datetime.now() + timedelta(days=1)).month,
                'day': (datetime.now() + timedelta(days=1)).day
            }
        },
    }

    response = client.create_transfer_job(transfer_job)
    print(f"Created transfer job: {response.name}")

def get_gcloud_project():
    credentials, project_id = default()
    if project_id:
        return project_id
    else:
        raise EnvironmentError("Google Cloud project ID is not set. Use 'gcloud config set project PROJECT_ID' to set it.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer a Hugging Face dataset to Google Cloud Storage.")
    parser.add_argument('--dataset_name', required=True, help="The name of the Hugging Face dataset.")
    parser.add_argument('--destination_bucket', required=True, help="The name of the destination GCS bucket.")
    parser.add_argument('--http_bucket', required=True, help="The name of the GCS bucket that is publicly accessible via HTTP, for the url file")
    parser.add_argument('--urls_file_name', default='urls.tsv', help="The name of the URLs file to save.")
    args = parser.parse_args()

    # Automatically determine the project ID
    project_id = get_gcloud_project()

    # Step 1: Get URLs from the Hugging Face dataset
    urls = get_urls_to_copy(args.dataset_name)

    # Step 2: Save URLs to a TSV file
    save_urls_to_tsv(urls, args.urls_file_name)

    # Step 3: Upload the URLs file to public GCS bucket so STS can access it
    upload_file_to_gcs(args.http_bucket, args.urls_file_name, args.urls_file_name)


    # Step 4: Create a Google Storage Transfer Service job
    create_transfer_job(project_id, f"gs://{args.source_bucket_name}/{args.urls_file_name}", args.destination_bucket_name)
