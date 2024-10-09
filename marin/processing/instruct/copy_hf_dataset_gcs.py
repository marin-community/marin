import os
import tempfile
import urllib.parse
import re
import datasets
import argparse
from datasets import DownloadConfig
from huggingface_hub import HfFileSystem, hf_hub_url
from google.auth import default
from google.cloud import storage, storage_transfer_v1

def get_urls_to_copy(dataset_name: str, **kwargs):
    print(f"Fetching URLs for dataset: {dataset_name}")
    fs = HfFileSystem()
    ds_builder = datasets.load_dataset_builder(dataset_name, trust_remote_code=True, **kwargs)
    config = ds_builder.config
    download_config = DownloadConfig(token=ds_builder.token, storage_options=ds_builder.storage_options)
    config._resolve_data_files(base_path=ds_builder.base_path, download_config=download_config)

    data_files = config.data_files
    if data_files is None:
        if hasattr(config, "data_url"):
            print(f"Single data URL found: {config.data_url}")
            return [(config.data_url, f"{dataset_name.replace('/', '-')}.parquet")]
        else:
            raise ValueError("There is no data available for this dataset (data_files and data_url are None).")
    else:
        url_pairs = []
        for split, files in data_files.items():
            print(f"Processing split: {split}")
            for file in files:
                if file.endswith('.parquet'):
                    resolved_path = fs.resolve_path(file)
                    url = hf_hub_url(
                        resolved_path.repo_id,
                        resolved_path.path_in_repo,
                        repo_type=resolved_path.repo_type,
                        revision=resolved_path.revision,
                    )
                    file_name = os.path.basename(resolved_path.path_in_repo)
                    new_path = f"{dataset_name}/{file_name}"
                    url_pairs.append((url, new_path))
                    print(f"Added URL pair: ({url}, {new_path})")
    print(f"Total URL pairs: {len(url_pairs)}")
    return url_pairs

def save_urls_to_tsv(url_pairs, file_path):
    print(f"Saving URLs to TSV file: {file_path}")
    with open(file_path, 'w') as f:
        f.write("TsvHttpData-1.0\n")
        for url, new_path in url_pairs:
            f.write(f"{url}\t{new_path}\n")
    print(f"URLs have been saved to {file_path}")
    
    # Print the contents of the file for debugging
    print("Contents of the TSV file:")
    with open(file_path, 'r') as f:
        print(f.read())

def upload_file_to_gcs(source_file, destination_path):
    print(f"Uploading file {source_file} to {destination_path}")
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

    print(f"Creating transfer job for dataset {dataset_name}")
    print(f"Source file: {source_file}")
    print(f"Destination: GCS bucket {gcs_bucket_name}, path {gcs_dest_path_in_bucket}")

    transfer_job = storage_transfer_v1.TransferJob(
        description=f"Transfer job for Hugging Face dataset {dataset_name}",
        project_id=project_id,
        transfer_spec=storage_transfer_v1.TransferSpec(
            http_data_source=storage_transfer_v1.HttpData(
                list_url=source_file
            ),
            gcs_data_sink=storage_transfer_v1.GcsData(
                bucket_name=gcs_bucket_name,
                path=gcs_dest_path_in_bucket
            ),
            transfer_options=storage_transfer_v1.TransferOptions(
                overwrite_objects_already_existing_in_sink=True,
                delete_objects_unique_in_sink=False,
                delete_objects_from_source_after_transfer=False,
            )
        ),
        status="ENABLED",
    )

    request = storage_transfer_v1.CreateTransferJobRequest(
        transfer_job=transfer_job
    )

    response = client.create_transfer_job(request)
    print(f"Created transfer job: {response.name}")

    run_response = client.run_transfer_job({"job_name": response.name, "project_id": project_id})
    print(f"Transfer job run response: {run_response}")
    
    print("You can check the status of the transfer job at:")
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
        raise EnvironmentError(
            "Google Cloud project ID is not set. Use 'gcloud config set project PROJECT_ID' to set it.")

def save_urls(url_pairs, urls_dir, url_tsv_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        urls_file_name = os.path.join(tmp_dir, url_tsv_name)
        save_urls_to_tsv(url_pairs, urls_file_name)
        base_name = os.path.basename(urls_file_name)
        return upload_file_to_gcs(urls_file_name, f"{urls_dir}/{base_name}")

def simplify_dataset_name(dataset_name):
    # Remove any prefix like 'allenai/'
    name = dataset_name.split('/')[-1]
    # Remove version numbers and other extraneous information
    name = re.sub(r'-v\d+.*', '', name)
    return name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer a Hugging Face dataset to Google Cloud Storage.")
    parser.add_argument('--dataset_name', required=True, help="The name of the Hugging Face dataset.")
    parser.add_argument('--config', required=False, help="The Hugging Face dataset config to use.", default="default")
    parser.add_argument('--revision', required=False, help="The Hugging Face dataset revision to use.", default="main")
    parser.add_argument('--destination_path', required=True, help="The name of the destination in a GCS bucket.")
    parser.add_argument('--urls_dir', required=True,
                        help="The GCS path to save the URLs file. Should be an HTTP accessible location.")
    args = parser.parse_args()

    print(f"Starting transfer for dataset: {args.dataset_name}")
    print(f"Destination path: {args.destination_path}")
    print(f"URLs directory: {args.urls_dir}")

    project_id = get_gcloud_project()
    print(f"Using Google Cloud project ID: {project_id}")

    url_pairs = get_urls_to_copy(args.dataset_name, name=args.config, revision=args.revision)

    url_tsv_name = f'{args.dataset_name.replace("/", "-")}.tsv'
    print(f"TSV file name: {url_tsv_name}")

    url_path_on_gcs = save_urls(url_pairs, args.urls_dir, url_tsv_name)
    print(f"URL file saved to GCS: {url_path_on_gcs}")

    public_url = f"https://storage.googleapis.com/{url_path_on_gcs}"
    print(f"Public URL for transfer job: {public_url}")

    create_transfer_job(args.dataset_name, project_id, public_url, args.destination_path)

    print("Script execution completed.")