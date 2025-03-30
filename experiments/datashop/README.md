# DataShop Overview

DataShop is a framework to create a filtered dataset from a larger data pool. The idea is to utilize a large language model (e.g. Llama-3.3-70B-Instruct) to annotate examples in the data pool and then filter the data pool based on the annotations. We have implemented one way to automatically annotate data given the validation set of the desired benchmark is from the paper by Held et. al. [Optimizing Pretraining Data Mixtures with LLM-Estimated Utility](https://arxiv.org/pdf/2501.11747).

## Get Started
There are two ways to use DataShop:
1. [Filtering the data pool via a benchmark](#filtering-the-data-pool-via-a-benchmark): MEDU automatically generates a benchmark description prompt and then labels the data pool.
2. [Filtering the data pool via prompt](#filtering-the-data-pool-via-prompt): You can pass in a prompt that the large language model will use to label the data pool. Akin to techniques like FineWebEdu.

### Filtering the data pool via a benchmark
Follow the example in `experiments/exp923_medu_mmlu.py`. Simply pass in the corpus content you care about to the `corpus_content_paths` argument.
In the example, we use the MMLU dataset, which we take the filename of the sets of MMLU subjects that we care about as corpus content.
Corpus content can also be a list of strings or a single string that represents the entire corpus. We expect the filetype to be jsonl with any compression type.

To add a new dataset, simply download the new dataset using the `download_hf` function similar to `experiments/eval_datasets.py` and then pass in the corpus content's filenames to the `corpus_content_paths` argument. We expect the file to have a column like "text" that contains the text of the document that you are looking to target. In the config, you can specify the exact prompt column that you are targeting. You should inherit the `DatashopRunner` class with your class and pass in the corpus contents.

### Filtering the data pool via prompt
In the DatashopRunnerConfig, pass in the `user_data_filter_prompt` argument with the prompt that you want to use to label the data pool. You must include the `{example}` placeholder for where you want the example to be inserted.

## The stages in Datashop
Datashop is composed of 5 steps:
The first three steps (1-3) are to create the dataset and steps (4-5) are to train a model to evaluate the quality of the dataset.
1. Document labeling using a LLM (e.g. Llama-3.3-70B-Instruct) to score the documents
2. Training a smaller encoder model (e.g. Alibaba/gte-large) on the labeled documents
3. Filtering the pretraining data using the model trained in step (2)
4. Annealing the pretrained model on the filtered pretraining data from step (3)
5. Evaluating the performance of the annealed model to evaluate the quality of the filtered dataset. If the model is better than the control model, then we can conclude that the filtered dataset is of high quality. For example, if our filterd dataset targets mathematical reasoning data, then we would evaluate the annealed model on MATH and GSM8K to see if its accuracy is higher than the control model.

### Sharp edges
There are a few sharp edges to the current implementation:
1. The vLLM cluster is separated from the training cluster. Please run the `run_eval_cluster_steps` function in the vLLM cluster to get the encoder model then run the `run_all_steps` function in the training cluster to do the anneal.
