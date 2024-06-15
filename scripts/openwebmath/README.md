# OpenWebMath

[[google doc](https://docs.google.com/document/d/1t4agH7fu51hZfRgrDRyDomiADhGAE_5Hh_IzO7kuXHs/edit)]

## Steps to process dataset
1. Download parquet files from HuggingFace and upload to GCS. From root dir, run:
```
python scripts/copy_hf_dataset_to_gcs.py --dataset_name open-web-math/open-web-math --destination_path gs://marin-data/raw/openwebmath --urls_dir gs://hf_dataset_transfer_bucket
```
This will create a folder in GCS with the parquet files. The folder will roughly look like:
```
gs://marin-data/raw/openwebmath/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8de2300f5e778f56261843dab89f230815/data/
```
2. Run the following commands to process the dataset. Since OWM is already (partly) converted to markdown, this command just filters non-markdown files and formats the data in Dolma format. First init Ray cluster:
```
ray dashboard infra/marin-cluster.yaml 
```
Then in a second terminal, run the processing script:
```
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python scripts/openwebmath/process_raw_owm.py --input_dir gs://marin-data/raw/openwebmath/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8de2300f5e778f56261843dab89f230815/data/
```