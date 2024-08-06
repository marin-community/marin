# Quality Filtering

## Overview
This repository contains the code for filtering out high-quality documents from a dataset.

### Quickstart
To get started, run the following command in the root of the directory:
```bash
# Run DCLM Fasttext Model
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.quality.inference --config marin/processing/quality/fasttext/dclm_fasttext.yaml

# Run Fineweb Edu Classifier
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.quality.inference --config marin/processing/quality/embedding/fineweb_edu_classifier.yaml
```

Feel free to edit the config yaml to fit your needs:
- `input_dir`: The directory containing the documents to be filtered. If you input a directory with multiple directories, the script will filter to run inference on each directory in parallel.
- `output_dir`: The directory to save the filtered documents. We rebase the output directory's filepath to match that of the input directory.
- `model_name`: The name of the model on Huggingface to use. The model needs to be hosted on huggingface for now and it uses the same convention as Huggingface.
- `attribute_name`: The name of the attribute to use. 
- `runtime`: The runtime environment and memory constraints to use. For example, DCLM fasttext models require downloading the fasttext package while Fineweb's edu classifier requires downloading Jax with TPU support as well as Huggingface.

### Models supported:
- FastText models: `DCLM`, `DOLMA`
- BERT models: `Fineweb-edu`