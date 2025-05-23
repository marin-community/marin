import os
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned, output_path_of # Added output_path_of
from marin.processing.tokenize import TokenizeConfig, tokenize # Ensure TokenizeConfig is here
from experiments.exp524_tokenizers import llama3_tokenizer
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf

# PILE_OF_LAW_BASE_PATH constant removed


def get_pile_of_law_download_step(
    base_output_path: str = "raw/"  # Base directory for raw data steps
) -> ExecutorStep[DownloadConfig]:
    dataset_id = "pile-of-law/pile-of-law"
    # The Pile of Law repo uses 'data/train.xyz.jsonl.xz' and 'data/validation.xyz.jsonl.xz'
    # The download_hf script will strip the 'datasets/pile-of-law/pile-of-law/' prefix,
    # effectively placing 'data/train...' and 'data/validation...' into the gcs_output_path.
    # So, the glob patterns should match what's inside the HF repo structure.
    # The tokenizer will later need to look for train*.jsonl.xz and validation*.jsonl.xz
    # *within* the directory structure created by download_hf.
    # If download_hf puts them into 'data/train*.jsonl.xz', then PILE_OF_LAW_BASE_PATH
    # for the tokenizer should be `os.path.join(output_path_of(download_step), "data")`.

    config = DownloadConfig(
        hf_dataset_id=dataset_id,
        revision="main",  # Assuming "main" branch is desired
        hf_urls_glob=[
            "data/train*.jsonl.xz",      # Glob for training files within the HF repo's 'data' directory
            "data/validation*.jsonl.xz"  # Glob for validation files within the HF repo's 'data' directory
        ],
        gcs_output_path=this_output_path(), # This will be the root for this download step's output
        # hf_repo_type_prefix is 'datasets' by default in DownloadConfig, which is correct.
    )

    step_name = os.path.join(base_output_path, "pile_of_law", "downloaded")

    return ExecutorStep(
        name=step_name,
        fn=download_hf,
        config=config,
        # fsspec and huggingface_hub are key dependencies for download_hf
        pip_dependency_groups=["fsspec", "huggingface_hub"],
    )


def get_pile_of_law_all_tokenized_step(
    download_step: ExecutorStep[DownloadConfig], # New first argument
    tokenizer: str = llama3_tokenizer
    # base_output_path is no longer needed here as step name is self-contained
) -> ExecutorStep[TokenizeConfig]:
    # The actual data files (train*.jsonl.xz, validation*.jsonl.xz) are expected
    # inside a 'data' subdirectory of the download_step's output.
    data_root_path = os.path.join(output_path_of(download_step), "data")

    config = TokenizeConfig(
        train_paths=[os.path.join(data_root_path, "train*.jsonl.xz")],
        validation_paths=[os.path.join(data_root_path, "validation*.jsonl.xz")],
        cache_path=this_output_path(), # This will be resolved by the ExecutorStep name
        tokenizer=versioned(tokenizer),
        # Add other TokenizeConfig parameters if necessary, e.g., cache_options
    )
    
    # The step name should be distinct, e.g., derived from the download step's name or a fixed one.
    # Let's use a fixed name for clarity for the tokenized output.
    step_name = "tokenized/pile_of_law_all_from_download" 
    
    return ExecutorStep(
        name=step_name, # Updated step name
        fn=tokenize,
        config=config,
        pip_dependency_groups=["sentencepiece"], # datasets is not strictly needed for TokenizeConfig with globs
        # The executor should handle the dependency on download_step automatically
        # when steps are correctly chained in executor_main.
    )

# Old function tokenize_pile_of_law_steps was here
# def tokenize_pile_of_law_steps(
# This is the end of the new get_pile_of_law_all_tokenized_step function
# The old tokenize_pile_of_law_steps function has been removed by the previous block.

if __name__ == "__main__":
    # 1. Define the download step
    download_step = get_pile_of_law_download_step()

    # 2. Define the tokenization step, depending on the download step
    tokenize_step = get_pile_of_law_all_tokenized_step(download_step=download_step)

    # 3. Run the executor with the steps
    # Listing both makes the dependency explicit for execution order if needed,
    # though the executor should resolve it via output_path_of.
    executor_main(steps=[download_step, tokenize_step])
