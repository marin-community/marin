# MEDU Overview

MEDU is a framework to reverse engineer the training data given a set of corpus content that the user cares about. It is based on the paper by Held et. al. [OPTIMIZING PRETRAINING DATA MIXTURES WITH LLM-ESTIMATED UTILITY](https://arxiv.org/pdf/2501.11747).

## Get Started (Adding a new dataset)
Follow the example in `medu_mmlu.py`. Simply pass in the corpus content you care about to the `corpus_content_paths` argument.
In the example, we use the MMLU dataset, which we take the filename of the sets of MMLU subjects that we care about as corpus content.
Corpus content can also be a list of strings or a single string that represents the entire corpus.

To add a new dataset, simply download the new dataset using the `download_hf` function in `medu_mmlu` and then pass in the corpus content's filenames to the `corpus_content_paths` argument. You should inherit the `MEDURunner` class with your class and pass in the corpus contents.

## The stages in MEDU
MEDU is composed of 4 stages:
1. Document labeling using a LLM (e.g. Llama-3.3-70B-Instruct)
2. Training a smaller encoder model (e.g. Alibaba/gte-large) on the labeled documents
3. Filtering the pretraining data using the model trained in stage (2)
4. Doing an anneal on a pretrained model using the filtered pretraining data from step (3)

### Sharp edges
There are a few sharp edges to the current implementation:
1. The vLLM cluster is separated from the training cluster. Please run the `run_eval_cluster_steps` function in the vLLM cluster to get the encoder model then run the `run_all_steps` function in the training cluster to do the anneal.
