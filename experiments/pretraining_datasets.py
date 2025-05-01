from experiments.defaults import default_download
from marin.execution.executor import ExecutorStep, this_output_path
from operations.download.huggingface.download import DownloadConfig, download
from operations.download.huggingface.download_gated_manual import download_and_upload_to_store
from operations.download.nemotron_cc.download_nemotron_cc import NemotronIngressConfig, download_nemotron_cc

fineweb = default_download(
    name="raw/fineweb",
    hf_dataset_id="HuggingFaceFW/fineweb",
    revision="cd85054",
    output_path="raw/fineweb",
    download_fn=download,
)

fineweb_edu = default_download(
    name="raw/fineweb-edu",
    hf_dataset_id="HuggingFaceFW/fineweb-edu",
    revision="3c452cb",
    output_path="raw/fineweb-edu-c2beb4",
    download_fn=download,
    cd="3c452cb/huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/3c452cb",
)

slimpajama = default_download(
    name="raw/SlimPajama-627B",
    hf_dataset_id="cerebras/SlimPajama-627B",
    revision="2d0accd",
    output_path="raw/SlimPajama-627B-262830",
    download_fn=download,
    cd="2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd",
)

slimpajama_6b = default_download(
    name="raw/SlimPajama-6B",
    hf_dataset_id="DKYoon/SlimPajama-6B",
    revision="b5f90f4",
    output_path="raw/SlimPajama-6B-be35b7",
    download_fn=download,
    cd="b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4",
)

dolma = default_download(
    name="raw/dolma",
    hf_dataset_id="allenai/dolma",
    revision="7f48140",
    output_path="raw/dolma",
    download_fn=download,
)


dclm_baseline_wrong = ExecutorStep(
    name="raw/dclm-baseline-1.0",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        timeout=24 * 60 * 60,
    ),
    override_output_path="raw/dclm_WRONG_20250211/",
)


dclm_baseline = ExecutorStep(
    name="raw/dclm-baseline-1.0",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        timeout=24 * 60 * 60,
    ),
    override_output_path="raw/dclm",
).cd("a3b142c")

the_stack_dedup = default_download(
    name="raw/the-stack-dedup",
    hf_dataset_id="bigcode/the-stack-dedup",
    revision="17cad72",
    output_path="raw/the-stack-dedup-4ba450",
    download_fn=download_and_upload_to_store,
    cd="17cad72",
)

proofpile_2 = default_download(
    name="raw/proof-pile-2",
    hf_dataset_id="EleutherAI/proof-pile-2",
    revision="901a927",
    output_path="raw/proof-pile-2-f1b1d8",
    download_fn=download,
    cd="901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927",
)

# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
starcoderdata = default_download(
    name="raw/starcoderdata",
    hf_dataset_id="bigcode/starcoderdata",
    revision="9fc30b5",
    output_path="raw/starcoderdata-720c8c",
)

dolmino = default_download(
    name="raw/dolmino-mix-1124",
    hf_dataset_id="allenai/dolmino-mix-1124",
    revision="bb54cab",
    output_path="raw/dolmino-mix-1124-157960",
    cd="bb54cab",
)

nemotron_cc = ExecutorStep(
    name="raw/nemotro-cc",
    fn=download_nemotron_cc,
    config=NemotronIngressConfig(
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["download_transform"],
)
