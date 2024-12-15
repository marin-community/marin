# Overview of the dataset

[google doc](https://docs.google.com/document/d/1yVYnhswWnyBYzj-nz6Od-9b2OQogoVtstDRsC6L9vjk/edit?usp=sharing)

# Example Run

```bash
python process_parquet_fw.py --parquet_file  gs://marin-data/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/
```
# Process
1. `process_dump.sh` will run for a single for a single dump of common crawl, and iterate through all the parquet files in
`gs://marin-data/raw/fineweb/fw-v1.0/DUMP/`. For each parquet file it will call `process_parquet_fw.py`
2. `process_parquet_fw.py` will split a fw parquet file into smaller parquet chunck such that each chunk only has entries from a
single WARC file. This smaller parquet chunck will be saved in `gs://marin-data/raw/fineweb/fw-v1.0/DUMP/<dump-idx>/<chunk-idx>.parquet`
4. Now for each of these smaller parquet chunck file, we will call `process_one_warc_file` using Ray.remote() using a single CPU.
`process_one_warc_file` downloads the WARC file, converts it to markdown and then saves three files:
   1. `gs://marin-data/raw/fineweb/fw-v1.0/DUMP/<dump-idx>/<chunk-idx>_processed_md.jsonl.gz : MD file in Dolma format
   2. `gs://marin-data/raw/fineweb/fw-v1.0/DUMP/<dump-idx>/<chunk-idx>_processed_html.jsonl.gz : HTML file in Dolma format
   3. `gs://marin-data/raw/fineweb/fw-v1.0/DUMP/<dump-idx>/<chunk-idx>_processed_md.jsonl.gz.SUCCESS : A success file to indicate that the processing is complete
# Making the system idompotent
1. We will run process_dump and process_parquet_fw on the main persistent node.
2. process_parquet_fw will check for .completed file to completely skip a parquet file
3. Each small parquet chunck also checks for .complete file to skip processing

# Important Functions inside `process_parquet_fw.py`
1. `process_fw_parquet` :  This function takes a single parquet file and splits it into smaller parquet chuncks. it takes about 8GB of memory to process about 2GB of parquet file.
2. `process_one_warc_file` : This function takes a single parquet chunck file, downloads the WARC file, converts it to markdown and saves the results to a new jsonl files. It takes about 1GB of memory.

# Time taken for processing

1. Current it takes about 2 hours to process one dump of fineweb data.
