# Consolidation

## Overview
This repository contains the code for filtering out high-quality documents from a dataset, deduping those documents and combing the two to consolidate datasets.

### Serving Quality Filtering Models
To get started, run the following command in the root of the directory:
```bash
# Run DCLM Fasttext Model
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.inference --config marin/processing/classification/config/dclm-fasttext/dclm_fasttext.yaml

# Run Fineweb Edu Classifier
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.inference --config marin/processing/classification/config/fineweb-edu-classifier/fineweb_edu_classifier.yaml
```

Feel free to edit the config yaml to fit your needs:
- `input_path`: The directory containing the documents to be filtered. If you input a directory with multiple directories, the script will filter to run inference on each directory in parallel.
- `output_path`: The directory to save the filtered documents. We rebase the output directory's filepath to match that of the input directory.
- `model_name`: The name of the model on Huggingface to use. The model needs to be hosted on huggingface for now and it uses the same convention as Huggingface.
- `attribute_name`: The name of the attribute to use.
- `runtime`: The runtime environment and memory constraints to use. For example, DCLM fasttext models require downloading the fasttext package while Fineweb's edu classifier requires downloading Jax with TPU support as well as Huggingface.

The quickstart example is
```bash
ray job submit  --address http://127.0.0.1:8265
 --working-dir . --no-wait -- \
python -m marin.processing.classification.inference --config marin/processing/classification/config/quick_start.yaml
```

### Training Fasttext Classifiers
1. Create the dataset in fasttext format
```bash
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.fasttext.data.create_dataset --high-quality-files gs://{BUCKET}/path/to/high-quality.jsonl.gz --low-quality-files gs://{BUCKET}/path/to/low-quality.jsonl.gz --output-file gs://{BUCKET}/path/to/fasttext-file.txt.gz
```
2. Split the dataset into train and test dataset
```bash
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.fasttext.data.split_dataset --input-file gs://{BUCKET}/path/to/fasttext-file.txt.gz --train-file gs://{BUCKET}/path/to/fasttext-train.txt.gz --test-file gs://{BUCKET}/path/to/fasttext-test.txt.gz
```
3. Run the model training script
```bash
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.fasttext.model.train --input-file gs://{BUCKET}/path/to/fasttext-train.txt.gz --output-model-path gs://{BUCKET}/path/to/fasttext-model.bin
```

### Viewing Quality Filtering Annotations
Run the following command to start the annotations server:
```bash
python -m marin.processing.classification.eval.annotations_server --input-file gs://{BUCKET}/path/to/input.jsonl.gz --attributes-file gs://{BUCKET}/path/to/attributes.jsonl.gz
```


### Deduplication

See the dedupe.md file for more details; below is the quick start command

```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python -m marin.processing.classification.dedupe --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart/ --output_path gs://marin-us-central2/attributes/hello_world_fw/v1.0/quickstart_duplicates/
```

Or if you want to use the yaml quickstart file:
```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python -m marin.processing.classification.dedupe --config_path marin/processing/classification/config/quick_start_dedupe.yaml
```

To run decomination for MMLU on the quickstart data we will also use a yaml
```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python -m marin.processing.classification.dedupe --config_path marin/processing/classification/config/quickstart_decontaminate.yaml
```
### Consolidation Command
After the attribute folders have been generated, to filter the dataset based on the quality rules following the example above you can run the following quickstart. We currently only support yaml
and not command line args for this pipeline to allow support arbitrary classifiers for consolidation.

You can deduplicate files as follows

```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python -m marin.processing.classification.consolidate --config_path marin/processing/classification/config/quickstart_consolidate_dedupe.yaml
```
We now run quality filtering on the subsequent files like so

```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python -m marin.processing.classification.consolidate --config_path marin/processing/classification/config/quickstart_consolidate_fasttext.yaml
```
We can also run both consolidation operations  (or many more) all in parallel. For the quickstart the combined command for deduping and quality filtering is

```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python -m marin.processing.classification.consolidate --config_path marin/processing/classification/config/quickstart_consolidate.yaml
```

If you would like to test the decontamination demo then after generate the attributes for MMLU then run
```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python -m marin.processing.classification.consolidate --config_path marin/processing/classification/config/quickstart_consolidate_decontaminate.yaml
```
The yaml for the full consolidation is as follows:
```yaml
input_path: "gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart/"
output_path: "gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart_consolidate/"
max_tasks_in_flight: 1000

filters:
  - type: "dedupe"
    attribute_path: "gs://marin-us-central2/attributes/hello_world_fw/v1.0/quickstart_duplicates/"
    name: "duplicate_text"
  - type: "classify"
    attribute_path: "gs://marin-us-central2/attributes/hello_world_fw/v1.0/quickstart_olmo_fasttext/"
    name: "olmo-fasttext-quality"
    label: "__label__hq"
    threshold: 0.1
```

Currently we require the user specifiy the file format and the attribute to filter by

### Models supported:
- FastText models: `DCLM`, `DOLMA`
- BERT models: `Fineweb-edu`
