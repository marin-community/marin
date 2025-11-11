#!/usr/bin/env python3
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
Use BigQuery to deduplicate outlinks, such that the remaining records
are all guaranteed to have different link targets.

First, copy the service account key file for
marin-data-browser@hai-gcp-models.iam.gserviceaccount.com to
gcs-key.json in this directory:

```
gcloud iam service-accounts keys create bigquery-gcs-key.json --iam-account=marin-crawl-bigquery@hai-gcp-models.iam.gserviceaccount.com
mv bigquery-gcs-key.json marin/crawl/
```

Deduplicating open-web-math outlinks (~2.5 mins):

```
export AUTHENTICATION_JSON="$(jq -c . ./marin/crawl/bigquery-gcs-key.json)"

python marin/run/ray_run.py \
    --pip_deps 'google-cloud-bigquery' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks.py \
        --gcs_input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
        --gcs_output_prefix 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-unique/unique_links' \
        --bq_table_id 'open_web_math_outlinks'
```

Deduplicating open-web-math-cc-deduplicated outlinks (~2.5 mins):

```
export AUTHENTICATION_JSON="$(jq -c . ./marin/crawl/bigquery-gcs-key.json)"

python marin/run/ray_run.py \
    --pip_deps 'google-cloud-bigquery' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks.py \
        --gcs_input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/*_links.jsonl.gz' \
        --gcs_output_prefix 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-unique/unique_links' \
        --bq_table_id 'open_web_math_cc_deduplicated_outlinks'
```

Deduplicating fineweb-edu outlinks (~36 mins):

```
export AUTHENTICATION_JSON="$(jq -c . ./marin/crawl/bigquery-gcs-key.json)"

python marin/run/ray_run.py \
    --pip_deps 'google-cloud-bigquery' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks.py \
        --gcs_input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/*_links.jsonl.gz' \
        --gcs_output_prefix 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-unique/unique_links' \
        --bq_table_id 'fineweb_edu_outlinks'
```

Deduplicating fineweb-edu-cc-deduplicated outlinks (~8 mins):

```
export AUTHENTICATION_JSON="$(jq -c . ./marin/crawl/bigquery-gcs-key.json)"

python marin/run/ray_run.py \
    --pip_deps 'google-cloud-bigquery' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks.py \
        --gcs_input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated/*_links.jsonl.gz' \
        --gcs_output_prefix 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-unique/unique_links' \
        --bq_table_id 'fineweb_edu_cc_deduplicated_outlinks'
```
"""  # noqa: E501
import json
import logging
import os
import uuid
from dataclasses import dataclass

import draccus
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeduplicateOutlinksConfig:
    """
    Configuration dataclass for deduplicating outlinks using BigQuery.

    This configuration specifies the input and output paths on Google Cloud Storage (GCS)
    as well as the BigQuery table and dataset to be used. It is consumed by
    `deduplicate_and_shuffle_with_bq_driver` to perform a deduplication job that:

    1. Loads JSONL data from GCS into a BigQuery table.
    2. Deduplicates records based on the "link_target" field.
    3. Randomly shuffles the deduplicated data.
    4. Exports the shuffled data back to GCS in multiple shards.

    Attributes:
        gcs_input_pattern (str):
            A GCS URI pattern (e.g., "gs://bucket/path/*.jsonl.gz") that points to
            the gzipped JSONL files containing outlinks to be deduplicated.
        gcs_output_prefix (str):
            A GCS URI prefix where deduplicated and shuffled JSONL files will be
            written (e.g., "gs://bucket/path/deduped_").
        bq_table_id (str):
            The name of the BigQuery table (within the specified dataset) to be
            created or overwritten during the deduplication process.
        bq_dataset_id (str, optional):
            The BigQuery dataset ID in which the table will be created or overwritten.
            Defaults to "marin_crawl".
    """

    gcs_input_pattern: str
    gcs_output_prefix: str
    bq_table_id: str
    bq_dataset_id: str = "marin_crawl_links"


def deduplicate_and_shuffle_with_bq(
    dataset_id: str,
    table_id: str,
    gcs_input_pattern: str,
    gcs_output_prefix: str,
):
    """
    Deduplicates gzipped JSONL from GCS on 'link_target' using BigQuery,
    then shuffles the output and writes it back to GCS in multiple shards.

    Parameters:
    dataset_id (str): BigQuery dataset name
    table_id (str): BigQuery table name (will be created/overwritten)
    gcs_input_pattern (str): GCS URI pattern, e.g., "gs://bucket/my-data/*.jsonl.gz"
    gcs_output_prefix (str): GCS URI prefix for exported files, e.g., "gs://bucket/my-output/deduped_"
    """
    raw_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw_creds:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is not set")

    # 1) Create a BigQuery client
    creds_dict = json.loads(raw_creds)
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    project_id = credentials.project_id
    logger.info(f"Creating BigQuery client with project id {project_id}")
    client = bigquery.Client(credentials=credentials, project=project_id)

    # 2) Get or create dataset, ensuring correct location
    dataset_name = f"{project_id}.{dataset_id}"
    dataset_location = "us-central2"  # Required location
    logger.info(f"Checking for BigQuery dataset {dataset_name}...")

    try:
        dataset = client.get_dataset(dataset_name)  # Check existence first
        logger.info(f"Found existing dataset {dataset_name}.")
        if dataset.location != dataset_location:
            raise ValueError(
                f"Dataset {dataset_name} already exists in location {dataset.location}, "
                f"but script requires location {dataset_location}. "
                f"Please delete the existing dataset or use a different bq_dataset_id."
            )
        else:
            logger.info(f"Dataset already exists in the correct location: {dataset_location}.")

    except NotFound:
        logger.info(f"Dataset {dataset_name} not found. Creating in location {dataset_location}...")
        dataset_ref = bigquery.Dataset(dataset_name)
        dataset_ref.location = dataset_location
        try:
            client.create_dataset(dataset_ref, timeout=30)
            logger.info(f"Created dataset {dataset_id} in {dataset_location}.")
        except Exception as e:
            logger.error(f"Failed to create dataset {dataset_name}: {e}")
            raise

    logger.info(f"Using BigQuery dataset {dataset_name} in location {dataset_location}")

    # 3) Create an external table referencing the GCS files
    ext_table_id = f"{table_id}_ext_{uuid.uuid4().hex[:8]}"
    ext_table_ref = f"{project_id}.{dataset_id}.{ext_table_id}"
    schema = [
        bigquery.SchemaField("page_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("link_target", "STRING", mode="REQUIRED"),  # Used for deduplication
        bigquery.SchemaField("is_internal_link", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("in_main_content", "BOOLEAN", mode="NULLABLE"),
    ]

    logger.info(f"Creating external table {ext_table_ref} pointing to {gcs_input_pattern}...")
    external_config = bigquery.ExternalConfig("NEWLINE_DELIMITED_JSON")
    external_config.source_uris = [gcs_input_pattern]
    external_config.schema = schema
    external_config.compression = "GZIP"

    table = bigquery.Table(ext_table_ref)
    table.external_data_configuration = external_config

    try:
        client.delete_table(ext_table_ref, not_found_ok=True)
        table = client.create_table(table)
        logger.info(f"External table {ext_table_ref} created.")
    except Exception as e:
        logger.error(f"Error creating external table: {e}")
        raise

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    logger.info(f"Deduplicating data into {table_ref}...")

    dedup_query = f"""
    CREATE OR REPLACE TABLE `{table_ref}` AS
    WITH ranked AS (
        SELECT
            t.*,
            ROW_NUMBER() OVER (PARTITION BY link_target ORDER BY link_target) AS rn
        FROM `{ext_table_ref}` t
    )
    SELECT * EXCEPT(rn)
    FROM ranked
    WHERE rn = 1
    """

    # Ensure the query job runs in the correct location
    query_job = client.query(dedup_query, location=dataset_location)
    query_job.result()
    logger.info("Deduplication complete.")

    # 5) Shuffle rows by RAND(), this can be expensive.
    logger.info("Shuffling rows...")
    shuffled_table = f"{table_ref}_shuffled_{uuid.uuid4().hex}"
    shuffle_query = f"""
    CREATE OR REPLACE TABLE `{shuffled_table}` AS
    SELECT *
    FROM `{table_ref}`
    ORDER BY RAND()
    """
    # Ensure the shuffle query job runs in the correct location
    client.query(shuffle_query, location=dataset_location).result()
    logger.info("Shuffle complete.")

    # 6) Export deduplicated and shuffled table back to GCS in multiple shards
    #    BigQuery will automatically shard into multiple files if you use a wildcard (*).
    destination_uri = f"{gcs_output_prefix}*.jsonl.gz"
    logger.info(f"Exporting to {destination_uri}...")
    extract_job_config = bigquery.ExtractJobConfig(
        compression=bigquery.Compression.GZIP, destination_format=bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
    )
    extract_job = client.extract_table(shuffled_table, destination_uri, job_config=extract_job_config)
    extract_job.result()
    logger.info("Export complete!")

    # Clean up intermediate tables
    client.delete_table(ext_table_ref, not_found_ok=True)
    client.delete_table(table_ref, not_found_ok=True)
    client.delete_table(shuffled_table, not_found_ok=True)

    logger.info("Done.")


@draccus.wrap()
def deduplicate_and_shuffle_with_bq_driver(cfg: DeduplicateOutlinksConfig):
    deduplicate_and_shuffle_with_bq(
        gcs_input_pattern=cfg.gcs_input_pattern,
        gcs_output_prefix=cfg.gcs_output_prefix,
        dataset_id=cfg.bq_dataset_id,
        table_id=cfg.bq_table_id,
    )


if __name__ == "__main__":
    deduplicate_and_shuffle_with_bq_driver()
