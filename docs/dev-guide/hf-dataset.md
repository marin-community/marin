# How to download a dataset from HuggingFace to Marin

This guide will walk you through the process of downloading a datset from HuggingFace to Marin. We will
show how to download the deepseek-ai/DeepSeek-ProverBench dataset as an example.


When writing an ExecutorStep, we need three things:
1. The output path of the dataset
2. The function to download the dataset
3. The config for the download function such as the dataset id and revision


For (1) we will place the dataset in the `raw` directory in the GCS bucket. The executor will automatically prefix the bucket with
`gs://{bucket_name}/` so we don't need to specify the bucket name in the output path. For (2), we use the `download_hf` function from `operations.download.huggingface.download_hf` to download the dataset. For (3), we need to find the dataset we want to download. In this case, we want the `deepseek-ai/DeepSeek-ProverBench` dataset on HuggingFace. We then need to find the commit hash of the dataset we want to download to fix the version of the dataset. At the time of this writing, the newest commit hash is `3b9f067`.


Next we create an `ExecutorStep` with this information.

```python
from operations.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, executor_main

deepseek_prover_bench = (
    ExecutorStep(
        name="raw/deepseek-prover-bench",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="deepseek-ai/DeepSeek-ProverBench",
            revision="3b9f067",
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    )
    .with_output_path("raw/deepseek-prover-bench")
    .cd("3b9f067")
)

if __name__ == "__main__":
    executor_main(
        steps=[deepseek_prover_bench],
        description="Download the DeepSeek-ProverBench dataset",
    )
```

We simply add the information listed from (1), (2), and (3) to the `ExecutorStep`. Some additional notes - chaining the ExecutorStep
with `.with_output_path` and `.cd` is optional. It is useful for organizing the output path and making it easier to read. The Executor framework usually automatically generate a hash at the end of the output path, but if you want to override the output of the path, you can do so with `.with_output_path`. In this case, we override it to simply `raw/deepseek-prover-bench` to make it easier to read. The `cd` method is a way to navigate to a specific commit of the dataset that is downloaded.

We can run this with the following command:
```
RAY_ADDRESS=$RAY_ADDRESS python marin/run/ray_run.py --env_vars HF_TOKEN $HF_TOKEN -- python experiments/deepseek_prover_tutorial.py
```

After running the command, we can view the dataset in the GCS Bucket at `gs://$MARIN_PREFIX/raw/deepseek-prover-bench` or in the Marin Data Browser.
