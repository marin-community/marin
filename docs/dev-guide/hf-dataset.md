# How to download a dataset from HuggingFace to Marin

This guide will walk you through the process of downloading a datset from HuggingFace to Marin. We will
show how to download the deepseek-ai/DeepSeek-ProverBench dataset as an example.


When writing an ExecutorStep, we need three things:
1. The output path of the dataset
2. The huggingface dataset repo id.
3. The commit revision of the dataset.


For (1) we will place the dataset in the `raw` directory in the GCS bucket. The executor will automatically prefix the bucket with
`gs://{bucket_name}/` so we don't need to specify the bucket name in the output path. For (2), we need to find the dataset we want to download. In this case, we want the `deepseek-ai/DeepSeek-ProverBench` dataset on HuggingFace. For (3), we then need to find the commit hash of the dataset we want to download to fix the version of the dataset. At the time of this writing, the newest commit hash is `3b9f067`.

We then take these parameters and put them in the `default_download` function:
```python
from experiments.defaults import default_download
from marin.execution.executor import executor_main

deepseek_prover_bench = default_download("raw/deepseek-prover-bench", "deepseek-ai/DeepSeek-ProverBench", "3b9f067")

if __name__ == "__main__":
    executor_main(
        steps=[deepseek_prover_bench],
        description="Download the DeepSeek-ProverBench dataset",
    )
```

We put this in the file `experiments/tutorials/download_dataset.py`, and we can run this file with the following command:
```
python marin/run/ray_run.py --env_vars HF_TOKEN $HF_TOKEN -- python experiments/tutorials/download_dataset.py
```

After running the command, we can view the dataset in the GCS Bucket at `gs://$MARIN_PREFIX/raw/deepseek-prover-bench` or in the Marin Data Browser.
