# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from experiments.defaults import default_download
from marin.download.huggingface.download_hf import DownloadConfig
from marin.download.huggingface.download_gated_manual import download_and_upload_to_store
from marin.download.huggingface.download_hf import download_hf
from marin.download.nemotron_cc.download_nemotron_cc import NemotronIngressConfig, download_nemotron_cc
from marin.execution.executor import ExecutorStep, this_output_path

fineweb = default_download(
    name="raw/fineweb",
    hf_dataset_id="HuggingFaceFW/fineweb",
    revision="cd85054",
    override_output_path="raw/fineweb",
)

fineweb_edu = default_download(
    name="raw/fineweb-edu",
    hf_dataset_id="HuggingFaceFW/fineweb-edu",
    revision="3c452cb",
    override_output_path="raw/fineweb-edu-c2beb4",
).cd("3c452cb/huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/3c452cb")

slimpajama = default_download(
    name="raw/SlimPajama-627B",
    hf_dataset_id="cerebras/SlimPajama-627B",
    revision="2d0accd",
    override_output_path="raw/SlimPajama-627B-262830",
).cd("2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd")

slimpajama_6b = default_download(
    name="raw/SlimPajama-6B",
    hf_dataset_id="DKYoon/SlimPajama-6B",
    revision="b5f90f4",
    override_output_path="raw/SlimPajama-6B-be35b7",
).cd("b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4")

dolma = default_download(
    name="raw/dolma",
    hf_dataset_id="allenai/dolma",
    revision="7f48140",
    override_output_path="raw/dolma",
)


dclm_baseline_wrong = default_download(
    name="raw/dclm-baseline-1.0",
    hf_dataset_id="mlfoundations/dclm-baseline-1.0",
    revision="a3b142c",
    override_output_path="raw/dclm_WRONG_20250211/",
    timeout=24 * 60 * 60,
)


dclm_baseline = default_download(
    name="raw/dclm-baseline-1.0",
    hf_dataset_id="mlfoundations/dclm-baseline-1.0",
    revision="a3b142c",
    override_output_path="raw/dclm",
    timeout=24 * 60 * 60,
).cd("a3b142c")


the_stack_dedup = ExecutorStep(
    name="raw/the-stack-dedup",
    fn=download_and_upload_to_store,
    config=DownloadConfig(
        hf_dataset_id="bigcode/the-stack-dedup",
        revision="17cad72",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/the-stack-dedup-4ba450",
).cd("17cad72")

proofpile_2 = default_download(
    name="raw/proof-pile-2",
    hf_dataset_id="EleutherAI/proof-pile-2",
    revision="901a927",
    override_output_path="raw/proof-pile-2-f1b1d8",
).cd("901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927")


the_pile_openwebtext2 = default_download(
    name="raw/the_pile_openwebtext2",
    hf_dataset_id="vietgpt/the_pile_openwebtext2",
    revision="1de27c6",
    override_output_path="raw/the_pile_openwebtext2",
).cd("1de27c6/huggingface.co/datasets/vietgpt/the_pile_openwebtext2/resolve/1de27c6")

# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
starcoderdata = ExecutorStep(
    name="raw/starcoderdata",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="bigcode/starcoderdata",
        revision="9fc30b5",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/starcoderdata-720c8c",
)

dolmino = (
    ExecutorStep(
        name="raw/dolmino-mix-1124",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="allenai/dolmino-mix-1124",
            revision="bb54cab",
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    )
    .with_output_path("raw/dolmino-mix-1124-157960")
    .cd("bb54cab")
)

nemotron_cc = ExecutorStep(
    name="raw/nemotro-cc",
    fn=download_nemotron_cc,
    config=NemotronIngressConfig(
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["download_transform"],
)
