from marin.execution.executor import ExecutorStep, this_output_path
from operations.download.huggingface.download import DownloadConfig, download

download_fineweb = ExecutorStep(
    name="raw/fineweb",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb",
        revision="cd85054",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/fineweb",
)

fineweb_edu = ExecutorStep(
    name="raw/fineweb-edu",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/fineweb-edu-c2beb4",
).cd("3c452cb/huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/3c452cb")

slimpajama = ExecutorStep(
    name="raw/SlimPajama-627B",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="cerebras/SlimPajama-627B",
        revision="2d0accd",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/SlimPajama-627B-262830",
).cd("2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd")

slimpajama_6b = ExecutorStep(
    name="raw/SlimPajama-6B",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="DKYoon/SlimPajama-6B",
        revision="b5f90f4",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/SlimPajama-6B-be35b7",
).cd("b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4")

dolma = ExecutorStep(
    name="raw/dolma",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="allenai/dolma",
        revision="7f48140",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/dolma",
)

dclm_baseline = ExecutorStep(
    name="raw/dclm-baseline-1.0",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/dclm",
)
