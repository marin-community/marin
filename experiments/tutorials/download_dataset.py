from experiments.defaults import default_download
from marin.execution.executor import executor_main

deepseek_prover_bench = default_download("raw/deepseek-prover-bench", "deepseek-ai/DeepSeek-ProverBench", "3b9f067")

if __name__ == "__main__":
    executor_main(
        steps=[deepseek_prover_bench],
        description="Download the DeepSeek-ProverBench dataset",
    )
