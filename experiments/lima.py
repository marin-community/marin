"""
https://huggingface.co/datasets/GAIR/lima
"""

from marin.download import HfDownloadConfig, download_hf_gated_manual
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize.data_configs import TokenizerStep

lima = (
    ExecutorStep(
        name="raw/lima",
        fn=download_hf_gated_manual,
        config=HfDownloadConfig(
            hf_dataset_id=versioned("GAIR/lima"),
            revision=versioned("68958e9"),
            gcs_output_path=this_output_path(),
            hf_urls_glob=["*.jsonl"],
            wait_for_completion=True,
        ),
    )
    .with_output_path("raw/lima-68958e9")
    .cd("68958e9")
)


def lima_tokenized(
    *, base_path: str = "tokenized/", tokenizer: str = "stanford-crfm/marin-tokenizer", lima_raw: ExecutorStep = lima
) -> dict[str, TokenizerStep]:
    """Return steps to tokenize the LIMA validation set using both train and test splits."""
    # Avoid cyclic dependency
    from experiments.defaults import default_tokenize

    return {
        "lima": default_tokenize(
            name="lima",
            dataset=lima_raw.cd("*.jsonl"),  # load both train and test splits
            tokenizer=tokenizer,
            is_validation=True,
        )
    }


if __name__ == "__main__":
    executor_main(steps=[lima, *lima_tokenized().values()])
