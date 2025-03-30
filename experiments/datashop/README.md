# DataShop Overview

DataShop is a framework to create a filtered dataset from a larger data pool. The idea is to utilize a large language model (e.g. Llama-3.3-70B-Instruct) to annotate examples in the data pool and then filter the data pool based on the annotations.

## Get Started
There are two ways to use DataShop:
1. [Filtering the data pool via a dataset](#filtering-the-data-pool-via-a-dataset): MEDU automatically generates a dataset description prompt and then labels the data pool. We have implemented one way to automatically annotate data given a dataset from the paper by Held et. al. [Optimizing Pretraining Data Mixtures with LLM-Estimated Utility](https://arxiv.org/pdf/2501.11747).

2. [Filtering the data pool via prompt](#filtering-the-data-pool-via-prompt): You can pass in a prompt that the large language model will use to label the data pool. Akin to techniques like FineWebEdu.

### Filtering the data pool via a dataset
To add a new dataset, simply download the new dataset using the `download_hf` function similar to `experiments/eval_datasets.py`. For example, for MMLU, we download the dataset using the following:
```
mmlu_raw = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/cais/mmluhf",
)
```
Then, we pass in the corpus content's filenames to the `corpus_content_paths` argument. We expect the file to have a column like "text" that contains the text of the document that you are looking to target. In the config, you can specify the exact prompt column that you are targeting. In the case of MMLU, we want to target the "prompt" column. For example, in `experiments/exp923_medu_mmlu.py`, we define a way to create the corpus contents as follows:
```
def _create_corpus_contents(self):
        # Download the MMLU dataset
        corpus_contents = []
        for subject in self.config.subset_names:
            filepath = os.path.join(
                os.getenv("MARIN_PREFIX"), mmlu_subject_eval.name, "cais", f"mmlu-{subject}-dev-evaluation.jsonl.gz"
            )
            corpus_contents.append(CorpusContent(content=filepath, content_type="filepath", prompt_column="prompt"))

        return corpus_contents
```
Then, you should inherit the `DatashopRunner` class with your class and pass in the corpus contents. For example, in `experiments/exp923_medu_mmlu.py`, we pass in the corpus contents as follows:
```
class MMLUMeduPipeline(DatashopRunner):
    def __init__(self, config: MeduMMLUConfig):
        self.config = config
        self.corpus_contents = self._create_corpus_contents()
        super().__init__(
            DatashopRunnerConfig(
                experiment_name=self.config.experiment_name,
                annotator_model_name=self.config.annotator_model_name,
                pretraining_data_path=self.config.pretraining_data_path,
                annotator_data_path=self.config.annotator_data_path,
                corpus_content_paths=self.corpus_contents,
            )
        )
```

The corpus_content_paths argument is quite flexible. You can pass in a list of strings that represent the corpus contents or a list of files that represent the corpus contents.

### Filtering the data pool via prompt
In the DatashopRunnerConfig, pass in the `data_filter_prompt` argument with the prompt that you want to use to label the data pool. You must include the `{example}` placeholder for where you want the example to be inserted.

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
1. The vLLM cluster is separated from the training cluster. This is because vLLM uses a different libtpu version than levanter (used for training). We will need to either use a different inference engine or have a better way to isolate the dependencies for each step (e.g. uv). Please run the `run_eval_cluster_steps` function in the vLLM cluster to get the encoder model then run the `run_all_steps` function in the training cluster to do the anneal.
