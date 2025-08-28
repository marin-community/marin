"""Download the SmolLM Corpus subsets.

Pattern matches datasets with Hugging Face-exposed subsets like
``experiments/nemotron_cc.py``.
"""

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

SMOLLM_REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"

smollm_cosmopedia = ExecutorStep(
    name="raw/smollm_corpus/cosmopedia-v2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceTB/smollm-corpus",
        revision=versioned(SMOLLM_REVISION),
        hf_urls_glob=["cosmopedia-v2/*"],
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

smollm_fineweb_edu = ExecutorStep(
    name="raw/smollm_corpus/fineweb-edu-dedup",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceTB/smollm-corpus",
        revision=versioned(SMOLLM_REVISION),
        hf_urls_glob=["fineweb-edu-dedup/*"],
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

smollm_python_edu = ExecutorStep(
    name="raw/smollm_corpus/python-edu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceTB/smollm-corpus",
        revision=versioned(SMOLLM_REVISION),
        hf_urls_glob=["python-edu/*"],
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            smollm_cosmopedia,
            smollm_fineweb_edu,
            smollm_python_edu,
        ]
    )
