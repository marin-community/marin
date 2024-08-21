# Processing Instruction Data for Marin

This README provides an overview of the Python script `ray_process.py` that demonstrates how to process instruction datasets using the Ray distributed computing framework. The script is designed to convert HTML content from JSONL or Parquet files into Markdown and HTML formats, and save the results as JSONL files in the Dolma Format.


Quick start command:
 ```bash
   ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python scripts/instruct/ray_process.py --input_dir gs://marin-us-central2/raw/instruct/ --output_dir gs://marin-us-central2/documents/marin_instructv1/ --input_type parquet
 ```
 
## Prerequisites

- Python 3.x
- Ray
- Required dependencies (listed in the script)

## Usage

1. Prepare your input dataset in either JSONL or Parquet format. The input files should contain HTML content that needs to be converted to Markdown and HTML.

2. Set up a Ray cluster or use a local Ray instance.

3. Update the `input_dir` and `output_dir` variables in the script to specify the input and output directories respectively. The input directory should contain the JSONL or Parquet files to be processed, and the output directory will be used to store the resulting Markdown and HTML JSONL files.

4. Run the script using the following command:

   ```bash
   ray job submit --address <ray_address> --working-dir . --no-wait -- python process_parquet_fw.py --input_dir <input_directory> --output_dir <output_directory> --input_type <jsonl|parquet>
   ```

   Replace `<ray_address>` with the address of your Ray cluster, `<input_directory>` with the path to your input directory, `<output_directory>` with the path to your desired output directory, and `<jsonl|parquet>` with the type of input files (`jsonl` for JSONL files or `parquet` for Parquet files).

   For example if the server is running the below will generate version 1.0 of the instruction
   dataset if it does not already exist:
   ```bash
   ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python process_parquet_fw.py --input_dir gs://marin-data/raw/instruct/ --output_dir gs://marin-data/processed/instruct/ --input_type parquet
   ```

5. The script will process the input files using Ray's distributed computing capabilities. It will convert the HTML content to Markdown and HTML formats and save the results as JSONL files in the specified output directory.

6. Monitor the progress of the job using the Ray dashboard or by checking the logs.

# How do I download the raw data?

Currently this needs to be done on a case by case basis according to the dataset since all the public instruct datasets have slightly different formats. For the tulumix checkout, the ```marin-data/raw/instruct``` bucket must exist apriori
```bash
# Original COMMAND
python scripts/copy_hf_dataset_to_gcs.py --dataset_name allenai/tulu-v2-sft-mixture --destination_path gs://marin-data/raw/instruct --urls_dir hf_dataset_transfer_bucket

# Updated to central-2b
python marin/processing/instruct/copy_hf_dataset_gcs.py --dataset_name allenai/tulu-v2-sft-mixture --destination_path gs://marin-us-central2/raw/instruct/ --urls_dir hf_dataset_transfer_bucket
```
## FAQ

### 1. What if the script stalls or fails?

If the script stalls or fails, you can check the Ray dashboard or logs to identify the issue. Common reasons for stalling or failure include:
- Insufficient resources allocated to the Ray cluster.
- Network connectivity issues.
- Errors in the input data or processing logic.

Make sure to address any identified issues and retry the job.

### 2. How can I customize the processing logic?

The main processing logic is implemented in the `html_to_md` function. You can modify this function to adapt the processing logic to your specific requirements. Be cautious when making changes to ensure the function remains idempotent and resumable.

### 3. Can I use this script with other file systems?

Yes, the script uses `fsspec` to handle file I/O, which supports various file systems, including local files and cloud storage like Google Cloud Storage (GCS). Make sure to provide the appropriate file paths based on your chosen file system.

### 4. What if I want to develop without using Ray first?
Then look at the `process.py` file for a simple example

### 5. What if I want to merge jsonl files locally to test how process.py is doing?
Then look at `merge_jsonl_local.py`

## Notes

- The script assumes the input files are in a specific format (JSONL or Parquet) and contain HTML content. Make sure your input files adhere to the expected format.
- The script uses custom modules (`marin.core.runtime` and `marin.web.convert`) for certain functionalities. Make sure these modules are available in your environment.
- The script is designed to be run as a Ray job using the `ray job submit` command. Ensure you have a running Ray cluster or a local Ray instance before executing the script.

For more details on how to use Ray and its features, refer to the Ray documentation: [https://docs.ray.io/](https://docs.ray.io/)