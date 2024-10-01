#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

EXP="quickstart_eval_0921-4"
#
#Transform
ray job submit --working-dir . -- python scripts/hello_world_fw/process.py \
  --input_path gs://marin-us-central2/raw/hello_world_fw/8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/resolve/8fd6e8e/data/CC-MAIN-2024-10 \
  --output_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP/CC-MAIN-2024-10

#FastText classifier
ray job submit --working-dir . -- python scripts/fasttext/train_fasttext.py \
  --input_doc_paths '[{    "input_doc_path" : "gs://marin-us-central2/documents/instruct/tulu_v2_mix/text" ,
    "label": "__label__hq",
    "sampling_rate": 0.1,
    "format": "dolma_formatted_jsonl"},
    {    "input_doc_path" : "gs://marin-us-central2/documents/hello_world_fw/v1.0/chris-test" ,
    "label": "__label__lq",
    "sampling_rate": 1.0,
    "format": "dolma_formatted_jsonl"}]' \
  --output_path gs://marin-us-central2/classifiers/$EXP \
  --fasttext_args '{"lr": 0.01, "thread": 4}'


### Use olmo classifier to annotate
ray job submit --working-dir . -- python -m marin.processing.classification.inference \
  --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP  \
  --output_path gs://marin-us-central2/attributes/hello_world_fw/v1.0/${EXP}_olmo_fasttext \
  --model_name allenai/dolma-1_7-fasttext-quality-filter --model_type fasttext \
  --attribute_name "olmo-fasttext-quality"


## Dedup
ray job submit --working-dir . -- python marin/processing/classification/dedupe.py \
  --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/$EXP/ \
  --output_path gs://marin-us-central2/attributes/hello_world_fw/v1.0/${EXP}_duplicates/

## Annotate the documents
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

# Tokenize
ray job submit --working-dir . \
  -- python -m marin.processing.tokenize \
  --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/${EXP}_consolidate/ \
  --cache_path gs://marin-us-central2/tokenized/llama3/ \
  --dataset_name hello_world_fw-${EXP} --tokenizer meta-llama/Meta-Llama-3.1-8B


export LC_CTYPE=C
export LANG=C

run_id=$(tr -dc 'a-zA-Z0-9' </dev/urandom | head -c 16)
python scripts/training/launch.py --experiment $EXP --base_config config/training/quickstart_run.yaml \
   --dataset_name hello_world_fw-${EXP} \
   --dataset_path "gs://marin-us-central2/documents/hello_world_fw/v1.0/'${EXP}_consolidate'/**/*.jsonl.gz"\
    --cache_dir gs://marin-us-central2/tokenized/llama3/ --zone us-central2-b --tpu_type v4-32 --run_id $run_id


# We assume that the model would be trained for 200 steps in 10 minutes

# Variables
bucket_path=gs://marin-us-central2/checkpoints/$EXP/$run_id/hf/$run_id/step-200/
timeout=600  # 10 minutes (600 seconds)
interval=30  # Check every 30 seconds
elapsed=0

# Loop to check if the path exists
while ((elapsed < timeout)); do
  if gsutil ls "$bucket_path" >/dev/null 2>&1; then
    echo "Path exists."
    break
  else
    echo "Waiting for path to be created..."
    sleep $interval
    elapsed=$((elapsed + interval))
  fi
done

if ((elapsed >= timeout)); then
  echo "Path not created within the 10-minute timeout."
  exit 1
fi

ray job submit --working-dir . -- python scripts/evaluation/run.py helm --model-name $run_id/step-200\
  --model-path gs://marin-us-central2/checkpoints/$EXP/$run_id/hf/$run_id/step-200/ \
  --evals mmlu --evaluation-path gs://marin-us-central2/evaluation/$EXP/$run_id/step-200/mmlu
