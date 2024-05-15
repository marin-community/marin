# For a given Hugging Face dataset, this script will:
# 1. Get the URLs of the dataset files
# 2. Save the URLs to a TSV file
# 3. Upload the TSV file to a GCS bucket
# 4. Create a Google Storage Transfer Service job to transfer the dataset files to a GCS bucket
#
# Usage:
# python copy_hf_dataset_to_gcs.py --dataset_name <DATASET_NAME> --destination_path gs://somewhere/ --urls_dir gs://somewhere_publicly_readable
import os
import tempfile
import urllib.parse

import datasets
import argparse
from datasets import DownloadConfig
from huggingface_hub import HfFileSystem, hf_hub_url
from google.auth import default
from google.cloud import storage, storage_transfer_v1



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

def upload_file_to_gcs(source_file, destination_path):
    storage_client = storage.Client()
    bucket_name, destination_blob_name = _split_gcs_path(destination_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file)

    print(f"File {source_file} uploaded to {destination_blob_name} in bucket {bucket_name}.")

    return f"{bucket_name}/{destination_blob_name}"

def create_transfer_job(dataset_name, project_id, source_file, gcs_dest_path):
    client = storage_transfer_v1.StorageTransferServiceClient()
    gcs_bucket_name, gcs_dest_path_in_bucket = _split_gcs_path(gcs_dest_path)


    transfer_job = storage_transfer_v1.TransferJob(
        description=f"Transfer job for Hugging Face dataset {dataset_name}",
        project_id=project_id,
        transfer_spec=storage_transfer_v1.TransferSpec(
            # object_conditions=storage_transfer_v1.ObjectConditions(max_time_elapsed_since_last_modification="86400s"),
            http_data_source=storage_transfer_v1.HttpData(
                list_url=source_file
            ),

            gcs_data_sink=storage_transfer_v1.GcsData(
                bucket_name=gcs_bucket_name,
                path=f"{gcs_dest_path_in_bucket}/"
            )
        ),
        status="ENABLED",
    )

    request = storage_transfer_v1.CreateTransferJobRequest(
        transfer_job=transfer_job
    )

    response = client.create_transfer_job(request)
    print(f"Created transfer job: {response.name}")
    # now we have to start the transfer job

    run_response = client.run_transfer_job({"job_name": response.name, "project_id": project_id})
    print("You can check the status of the transfer job at:")
    # they look like https://console.cloud.google.com/transfer/jobs/transferJobs%2F3385472833789201271/monitoring?authuser=1&hl=en&project=hai-gcp-models
    # job names have slashes in them, so we need to escape them
    escaped_name = urllib.parse.quote_plus(response.name)
    print(f"https://console.cloud.google.com/transfer/jobs/{escaped_name}/monitoring?hl=en&project={project_id}")




def _split_gcs_path(path):
    if path.startswith('gs://'):
        gcs_bucket_name = path.split('/')[2]
        gcs_dest_path_in_bucket = '/'.join(path.split('/')[3:])
    else:
        gcs_bucket_name = path.split('/')[0]
        gcs_dest_path_in_bucket = '/'.join(path.split('/')[1:])
    # strip leading /
    if gcs_dest_path_in_bucket.startswith('/'):
        gcs_dest_path_in_bucket = gcs_dest_path_in_bucket[1:]

    return gcs_bucket_name, gcs_dest_path_in_bucket


def get_gcloud_project():
    credentials, project_id = default()
    if project_id:
        return project_id
    else:
        raise EnvironmentError("Google Cloud project ID is not set. Use 'gcloud config set project PROJECT_ID' to set it.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer a Hugging Face dataset to Google Cloud Storage.")
    parser.add_argument('--dataset_name', required=True, help="The name of the Hugging Face dataset.")
    parser.add_argument('--destination_path', required=True, help="The name of the destination in a GCS bucket.")
    parser.add_argument('--urls_dir', required=True, help="The GCS path to save the URLs file. Should be an HTTP accessible location.")
    args = parser.parse_args()

    # Automatically determine the project ID
    project_id = get_gcloud_project()

    # Step 1: Get URLs from the Hugging Face dataset
    urls = get_urls_to_copy(args.dataset_name)

    url_tsv_name = f'{args.dataset_name.replace("/", "-")}.tsv'

    # Step 2: Save URLs to a TSV file
    with tempfile.TemporaryDirectory() as tmp_dir:
        urls_file_name = os.path.join(tmp_dir, url_tsv_name)
        save_urls_to_tsv(urls, urls_file_name)
        base_name = os.path.basename(urls_file_name)
        url_path_on_gcs = upload_file_to_gcs(urls_file_name, f"{args.urls_dir}/{base_name}")

    # Step 4: Create a Google Storage Transfer Service job
    # STS only reads from a public url, so we need to get the public url of the file
    public_url = f"https://storage.googleapis.com/{url_path_on_gcs}"
    create_transfer_job(args.dataset_name, project_id, public_url, args.destination_path)
