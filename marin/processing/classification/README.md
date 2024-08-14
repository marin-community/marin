# Quality Filtering

## Overview
This repository contains the code for filtering out high-quality documents from a dataset.

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
- `input_dir`: The directory containing the documents to be filtered. If you input a directory with multiple directories, the script will filter to run inference on each directory in parallel.
- `output_dir`: The directory to save the filtered documents. We rebase the output directory's filepath to match that of the input directory.
- `model_name`: The name of the model on Huggingface to use. The model needs to be hosted on huggingface for now and it uses the same convention as Huggingface.
- `attribute_name`: The name of the attribute to use. 
- `runtime`: The runtime environment and memory constraints to use. For example, DCLM fasttext models require downloading the fasttext package while Fineweb's edu classifier requires downloading Jax with TPU support as well as Huggingface.

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

Please note that deduplication assumes the input path is of the form gs://{$BUCKET_PATH}/documents/ and will write attributes to gs://{$BUCKET_PATH}/attributes/

```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python marin/processing/classification/dedupe.py --input_dir gs://marin-us-central2/scratch/documents/dummy_dedupe_data/                                   
```
### Consolidation Command
After the attribute folders have been generated, to filter the dataset based on the quality rules following the example above you can run
```bash
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.filter --input_dir gs://{BUCKET}/path/to/md/input.jsonl.gz --file_format md --attribute_name fineweb-edu-quality --threshold 3
```

Currently we require the user specifiy the file format and the attribute to filter by

### Models supported:
- FastText models: `DCLM`, `DOLMA`
- BERT models: `Fineweb-edu`