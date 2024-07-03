# how to download hf dataset to gcs

python scripts/copy_hf_dataset_to_gcs.py --dataset_name allenai/tulu-v2-sft-mixture --destination_path gs://marin-data/raw/instruct --urls_dir hf_dataset_transfer_bucket

# i think this is just the path to the data transfer bucket
https://storage.googleapis.com/hf_dataset_transfer_bucket/eloukas-edgar-corpus.tsv

# the above is broken the transfer job thing fails for some reason
# says i need to install some transfer_service_default agent
# gcp says i can't change permissions to public since bucket level access is uniform

# FIXED WITH PUBLIC BUCKET

# how to process html into markdown
the script from david doesn't work properly?

# below works now! now i can do local stuff
# do not include jsonl.gz
python scripts/instruct/process.py --input_dir gs://marin-data/raw/instruct/documents/

# attempt at getting ray to work (stalls)
# RAY STALLED FOR SOME REASON FIGURE OUT, MAKE SURE ISNTALL SERVER AND
# NO LARGE LOCAL FILES

# WORKS NOW WITH THIS COMMAND ALSO CREATES THE OUTPUT DIRECTORY:
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python scripts/instruct/ray_process.py --input_dir gs://marin-data/raw/instruct/documents/ --output_dir gs://marin-data/raw/instruct/instruct_proc/
# next steps: get the ray example working with parquet!
# errors
    response = func(input_file_path, output_file_path, *args, **kwargs)
  File "/tmp/ray/session_2024-06-22_09-21-13_381610_4354/runtime_resources/working_dir_files/_ray_pkg_a3369b6eb27fd4c7/scripts/instruct/ray_process.py", line 96, in html_to_md
    df = pd.read_parquet(input_file_path)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/pandas/io/parquet.py", line 667, in read_parquet
    return impl.read(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/pandas/io/parquet.py", line 274, in read
    pa_table = self.api.parquet.read_table(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/pyarrow/parquet/core.py", line 1762, in read_table
    dataset = ParquetDataset(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/pyarrow/parquet/core.py", line 1329, in __init__
    [fragment], schema=schema or fragment.physical_schema,
  File "pyarrow/_dataset.pyx", line 1431, in pyarrow._dataset.Fragment.physical_schema.__get__
  File "pyarrow/error.pxi", line 154, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 91, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Could not open Parquet input source 'marin-data/raw/instruct/instruct_pq/huggingface.co/datasets/allenai/tulu-v2-sft-mixture/resolve/6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752/data/train-00000-of-00003-99ee8754042a69f6.parquet': Parquet magic bytes not found in footer. Either the file is corrupted or this is not a parquet file.
[36m(html_to_md pid=85252, ip=10.130.1.145)[0m Output file already processed. Skipping gs://marin-data/raw/instruct/huggingface.co/datasets/allenai/tulu-v2-sft-mixture/resolve/6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752/data/train-00001-of-00003-278198836de5994c.parquet[32m [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(html_to_md pid=85352, ip=10.130.1.145)[0m Processing gs://marin-data/raw/instruct/instruct_pq/huggingface.co/datasets/allenai/tulu-v2-sft-mixture/resolve/6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752/data/train-00002-of-00003-a52de323599d586a.parquet to gs://marin-data/raw/instruct/instruct_pq/instruct_pq/huggingface.co/datasets/allenai/tulu-v2-sft-mixture/resolve/6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752/data/train-00002-of-00003-a52de323599d586a.parquet[32m [repeated 2x across cluster][0m

follow up with Abhi and David.