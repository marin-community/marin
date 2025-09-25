# Datashop Tutorial

Datashop is a tool for data filtering and evaluating the filtered data quality. This tutorial will walk through a basic example of using Datashop to filter a data pool for desired documents given a prompt and training a model on the filtered data.

## Prerequisites
TPUs are currently required to run this process for model inference. We perform this inference on TPU v6e-8s for serving and TPU v6e-128s for training for reference. Check the TPU Cluster yamls in `infra/` for reference. GPUs are not supported yet.


We use vLLM to run fast model inference to annotate each document given a rubric prompt. To install the docker image with vLLM and publish it to Google Cloud Artifact Registry, run the following commands:

```bash
# This command creates the vLLM docker image.
VLLM=1 make cluster_docker
```

Then, edit the DOCKER_TAGS in src/marin/ray/config.py file with the correct image commit id that was created in the previous step.

```bash
# Then run this command to update the cluster configs.
python infra/update-cluster-configs.py
```

Lastly, spin up the vLLM cluster with the following command.
```bash
ray up infra/marin-{region}-vllm.yaml
```

## Running a data filtering prompt
A common workload in data filtering is to use a rubric prompt that describes the desired documents and then filter a data pool for documents that match the rubric. This often requires multiple steps:

1. a large language model will first annotate a pool of documents to create a supervised training set

2. this training set is used to train a smaller model (e.g. BERT, Fasttext)

3. the smaller model is used to filter the data pool.

To run the datashop filtering process, there are five main attributes to know:

1. `annotator_model_name`: this is the model used to annotate the large data pool to create the supervised training set. We default to using Llama-3.3-70B-Instruct which is used commonly for this task (e.g. FineWebEdu, FineMath).

2. `pretraining_data_path`: this is the path to the large data pool that you would like to filter. We provide some defaults such as the first shard of the DCLM-Baseline pretraining dataset.

3. `annotator_data_path`: this is the path to a small set of documents that will be used for creating the supervised training set. We provide a default which is some randomly sampled documents from the pretraining data pool roughly totaling 500K documents.

4. `data_filter_prompt`: This is the prompt that will be used to annotate the documents from the annotator data path to develop the supervised training set.

5. `dataset_output_processor_config_kwargs`: These are keyword arguments passed into the dataset output processor, which can be helpful with processor initialization such as defining custom ways of parsing the final score from the LLM's generated text.


We now run through an example of how to run a datashop filtering prompt such as FineMath. We use this prompt the language model to assess whether a document satisfies useful mathematics data. See the full tutorial in `experiments/exp939_finemath.py`.

First, create a prompt that you would like to execute. For example, we can use the FineMath prompt (truncated for brevity, see the full prompt in `experiments/exp939_finemath.py`):
```python
FINEMATH_DATA_FILTER_PROMPT = """
Evaluate the following text extract for its potential usefulness for studying mathematics up to high school and early undergraduate levels. Use the following 5-point scoring system described below. Points are accumulated based on the satisfaction of
each criterion:

...

- Give a fifth point if the extract is outstanding in its educational value for teaching and studying mathematics in middle school
and high school. It should include very detailed and easy to follow explanations.
Question-answer formats (e.g., from educational websites or forums) are acceptable if they meet the criteria.
The text extract:
{example}
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: Final score: <total points>.
"""
```

We then import the pretraining and annotator data path from the datashop datasets module. If you want to use your own dataset, feel free to use that instead.
```python
from experiments.datashop.datashop_datasets import datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset
```

We then initialize the datashop runner with the appropriate parameters. Note that we use the Llama-3.1-8B-Instruct model as the annotator and we pass in a `processor_type` kwarg to the dataset output processor to signify that our outputs are a score between 0 and 5. We use the 8B model for this tutorial because it runs faster. If you have a different output format, you may need to write your custom processor in `marin/datashop/dataset_processor.py` and pass in the `processor_type` kwarg to that dataset output processor.
```python
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="finemath-replication",
        annotator_model_name="meta-llama/Llama-3.1-8B-Instruct",
        pretraining_data_path=datashop_dclm_pretraining_subset,
        annotator_data_path=datashop_dclm_annotation_subset,
        data_filter_prompt=FINEMATH_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
    )
)
```

We then run the datashop steps in the vLLM cluster to obtain the encoder model.

```python title="Expected runtime: 20 minutes"

datashop_runner.run_eval_cluster_steps()
```

Lastly, we run the datashop steps in the training cluster, which will filter data using the encoder model and then train a model on the filtered data.
```python
datashop_runner.run_all_steps()
```
