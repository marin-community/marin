#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

EXP="quickstart"

##Transform
ray job submit --working-dir . -- python scripts/hello_world_fw/process.py \
  --input_path gs://marin-us-central2/raw/hello_world_fw/8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/resolve/8fd6e8e/data/CC-MAIN-2024-10 \
  --output_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP/CC-MAIN-2024-10

##FastText classifier
ray job submit --working-dir . -- python scripts/fasttext/train_fasttext.py \
  --pos_doc_path gs://marin-us-central2/documents/marin_instructv1/v1_olmo_mix/text \
  --neg_doc_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP \
  --pos_sampling_rate 0.1 \
  --neg_sampling_rate 1.0 \
  --output_base_path gs://marin-us-central2 \
  --experiment $EXP \
  --training_args '{"lr": 0.01}'


## Use olmo classifier to annotate
ray job submit --working-dir . -- python -m marin.processing.classification.inference \
  --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP  \
   --output_path gs://marin-us-central2/attributes/hello_world_fw/v1.0/$EXP_olmo_fasttext \
     --model_name allenai/dolma-1_7-fasttext-quality-filter --model_type fasttext \
     --attribute_name "olmo-fasttext-quality"


## Dedup
ray job submit --working-dir . \
  -- python marin/processing/classification/dedupe.py \
  --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP/ \
  --output_path gs://marin-us-central2/attributes/hello_world_fw/v1.0/$EXP_duplicates/

#Annotate the documents
ray job submit --address http://127.0.0.1:8265 --working-dir . \
-- python -m marin.processing.classification.consolidate \
--input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP/ \
--output_path gs://marin-us-central2/documents/hello_world_fw/v1.0/${EXP}_consolidate/ \
--filters '[{    "type" : "dedupe" ,
"attribute_path": "gs://marin-us-central2/attributes/hello_world_fw/v1.0/'${EXP}'_duplicates/",
"name": "duplicate_text"},
    {"type": "classify",
    "attribute_path": "gs://marin-us-central2/attributes/hello_world_fw/v1.0/'${EXP}'_olmo_fasttext/",
    "name": "olmo-fasttext-quality",
    "label": "__label__hq",
    "threshold": 0.1}]'

#Tokenize
ray job submit --working-dir . \
-- python -m marin.processing.tokenize \
--input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/${EXP}_consolidate/CC-MAIN-2024-10/000_00000/*.jsonl.gz \
--cache_dir gs://marin-central2/tokenized/llama3/ \
--dataset_name $(whoami)-$EXP --tokenizer meta-llama/Meta-Llama-3.1-8B
