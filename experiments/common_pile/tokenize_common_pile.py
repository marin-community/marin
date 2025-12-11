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

"""Tokenization and mixture configs for the Common Pile v0.1 dataset."""

from levanter.data.dataset import PermType

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config

# Common Pile v0.1 filtered dataset download steps
arxiv_abstracts_filtered = ExecutorStep(
    name="raw/common_pile/arxiv_abstracts_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/arxiv_abstracts_filtered",
        revision="f1d7a9a",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/arxiv_abstracts_filtered-f1d7a9a",
)

arxiv_papers_filtered = ExecutorStep(
    name="raw/common_pile/arxiv_papers_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/arxiv_papers_filtered",
        revision="033cf7f",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/arxiv_papers_filtered-033cf7f",
)

biodiversity_heritage_library_filtered = ExecutorStep(
    name="raw/common_pile/biodiversity_heritage_library_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/biodiversity_heritage_library_filtered",
        revision="0486ed6",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/biodiversity_heritage_library_filtered-0486ed6",
)

caselaw_access_project_filtered = ExecutorStep(
    name="raw/common_pile/caselaw_access_project_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/caselaw_access_project_filtered",
        revision="50e1961",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/caselaw_access_project_filtered-50e1961",
)

cccc_filtered = ExecutorStep(
    name="raw/common_pile/cccc_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/cccc_filtered",
        revision="03a3de5",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/cccc_filtered-03a3de5",
)

data_provenance_initiative_filtered = ExecutorStep(
    name="raw/common_pile/data_provenance_initiative_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/data_provenance_initiative_filtered",
        revision="8f5afcf",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/data_provenance_initiative_filtered-8f5afcf",
)

doab_filtered = ExecutorStep(
    name="raw/common_pile/doab_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/doab_filtered",
        revision="defb24c",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/doab_filtered-defb24c",
)

foodista_filtered = ExecutorStep(
    name="raw/common_pile/foodista_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/foodista_filtered",
        revision="bf2c7aa",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/foodista_filtered-bf2c7aa",
)

github_archive_filtered = ExecutorStep(
    name="raw/common_pile/github_archive_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/github_archive_filtered",
        revision="52282fe",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/github_archive_filtered-52282fe",
)

library_of_congress_filtered = ExecutorStep(
    name="raw/common_pile/library_of_congress_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/library_of_congress_filtered",
        revision="56725c7",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/library_of_congress_filtered-56725c7",
)

libretexts_filtered = ExecutorStep(
    name="raw/common_pile/libretexts_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/libretexts_filtered",
        revision="70388bc",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/libretexts_filtered-70388bc",
)

news_filtered = ExecutorStep(
    name="raw/common_pile/news_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/news_filtered",
        revision="59aaa8f",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/news_filtered-59aaa8f",
)

oercommons_filtered = ExecutorStep(
    name="raw/common_pile/oercommons_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/oercommons_filtered",
        revision="506b615",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/oercommons_filtered-506b615",
)

peS2o_filtered = ExecutorStep(
    name="raw/common_pile/peS2o_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/peS2o_filtered",
        revision="2977475",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/peS2o_filtered-2977475",
)

pre_1929_books_filtered = ExecutorStep(
    name="raw/common_pile/pre_1929_books_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/pre_1929_books_filtered",
        revision="23f9d96",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/pre_1929_books_filtered-23f9d96",
)

pressbooks_filtered = ExecutorStep(
    name="raw/common_pile/pressbooks_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/pressbooks_filtered",
        revision="1a1d3b5",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/pressbooks_filtered-1a1d3b5",
)

project_gutenberg_filtered = ExecutorStep(
    name="raw/common_pile/project_gutenberg_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/project_gutenberg_filtered",
        revision="3cdf687",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/project_gutenberg_filtered-3cdf687",
)

public_domain_review_filtered = ExecutorStep(
    name="raw/common_pile/public_domain_review_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/public_domain_review_filtered",
        revision="efc7f21",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/public_domain_review_filtered-efc7f21",
)

pubmed_filtered = ExecutorStep(
    name="raw/common_pile/pubmed_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/pubmed_filtered",
        revision="c156f05",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/pubmed_filtered-c156f05",
)

python_enhancement_proposals_filtered = ExecutorStep(
    name="raw/common_pile/python_enhancement_proposals_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/python_enhancement_proposals_filtered",
        revision="5821709",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/python_enhancement_proposals_filtered-5821709",
)

regulations_filtered = ExecutorStep(
    name="raw/common_pile/regulations_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/regulations_filtered",
        revision="3327364",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/regulations_filtered-3327364",
)

stackexchange_filtered = ExecutorStep(
    name="raw/common_pile/stackexchange_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/stackexchange_filtered",
        revision="c0ac737",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/stackexchange_filtered-c0ac737",
)

stackv2_edu_filtered = ExecutorStep(
    name="raw/common_pile/stackv2_edu_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/stackv2_edu_filtered",
        revision="c354dbe",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/stackv2_edu_filtered-c354dbe",
)

stackv2_html_filtered = ExecutorStep(
    name="raw/common_pile/stackv2_html_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/stackv2_html_filtered",
        revision="92c9fa8",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/stackv2_html_filtered-92c9fa8",
)

stackv2 = ExecutorStep(
    name="raw/common_pile/stackv2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/stackv2",
        revision="d0e3266",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/stackv2-d0e3266",
)

ubuntu_irc_filtered = ExecutorStep(
    name="raw/common_pile/ubuntu_irc_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/ubuntu_irc_filtered",
        revision="84f88c9",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/ubuntu_irc_filtered-84f88c9",
)

uk_hansard_filtered = ExecutorStep(
    name="raw/common_pile/uk_hansard_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/uk_hansard_filtered",
        revision="c88adc4",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/uk_hansard_filtered-c88adc4",
)

usgpo_filtered = ExecutorStep(
    name="raw/common_pile/usgpo_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/usgpo_filtered",
        revision="b150cc2",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/usgpo_filtered-b150cc2",
)

uspto_filtered = ExecutorStep(
    name="raw/common_pile/uspto_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/uspto_filtered",
        revision="13894c5",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/uspto_filtered-13894c5",
)

wikimedia_filtered = ExecutorStep(
    name="raw/common_pile/wikimedia_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/wikimedia_filtered",
        revision="0641bb8",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/wikimedia_filtered-0641bb8",
)

wikiteam_filtered = ExecutorStep(
    name="raw/common_pile/wikiteam_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/wikiteam_filtered",
        revision="f4ed055",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/wikiteam_filtered-f4ed055",
)

youtube_filtered = ExecutorStep(
    name="raw/common_pile/youtube_filtered",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="common-pile/youtube_filtered",
        revision="dff8c8a",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/common_pile/youtube_filtered-dff8c8a",
)

# Map dataset names to their corresponding raw download steps
COMMON_PILE_DATASETS: dict[str, TokenizerStep] = {
    "arxiv_abstracts": arxiv_abstracts_filtered,
    "arxiv_papers": arxiv_papers_filtered,
    "biodiversity_heritage_library": biodiversity_heritage_library_filtered,
    "caselaw_access_project": caselaw_access_project_filtered,
    "cccc": cccc_filtered,
    "data_provenance_initiative": data_provenance_initiative_filtered,
    "doab": doab_filtered,
    "foodista": foodista_filtered,
    "github_archive": github_archive_filtered,
    "library_of_congress": library_of_congress_filtered,
    "libretexts": libretexts_filtered,
    "news": news_filtered,
    "oercommons": oercommons_filtered,
    "peS2o": peS2o_filtered,
    "pre_1929_books": pre_1929_books_filtered,
    "pressbooks": pressbooks_filtered,
    "project_gutenberg": project_gutenberg_filtered,
    "public_domain_review": public_domain_review_filtered,
    "pubmed": pubmed_filtered,
    "python_enhancement_proposals": python_enhancement_proposals_filtered,
    "regulations": regulations_filtered,
    "stackexchange": stackexchange_filtered,
    "stackv2_edu": stackv2_edu_filtered,
    "stackv2_html": stackv2_html_filtered,
    "ubuntu_irc": ubuntu_irc_filtered,
    "uk_hansard": uk_hansard_filtered,
    "usgpo": usgpo_filtered,
    "uspto": uspto_filtered,
    "wikimedia": wikimedia_filtered,
    "wikiteam": wikiteam_filtered,
    "youtube": youtube_filtered,
}

# Effective token counts for the main training stage (in teratokens)
# Weights pulled from https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset under Main stage
COMMA_MAIN_MIXTURE_WEIGHTS = {
    "common_pile/arxiv_abstracts": 0.00342,
    "common_pile/arxiv_papers": 0.036,
    "common_pile/biodiversity_heritage_library": 0.00245,
    "common_pile/caselaw_access_project": 0.0197,
    "common_pile/cccc": 0.0912,
    "common_pile/data_provenance_initiative": 0.00552,
    "common_pile/doab": 0.018,
    "common_pile/foodista": 0.00015,
    "common_pile/github_archive": 0.066,
    "common_pile/library_of_congress": 0.00237,
    "common_pile/libretexts": 0.00056,
    "common_pile/news": 0.00038,
    "common_pile/oercommons": 0.00007,
    "common_pile/peS2o": 0.2598,
    "common_pile/pre_1929_books": 0.0124,
    "common_pile/pressbooks": 0.00084,
    "common_pile/project_gutenberg": 0.0057,
    "common_pile/public_domain_review": 0.00001,
    "common_pile/pubmed": 0.0366,
    "common_pile/python_enhancement_proposals": 0.00002,
    "common_pile/regulations": 0.0084,
    "common_pile/stackexchange": 0.1434,
    "common_pile/stackv2_edu": 0.1356,
    "common_pile/stackv2_html": 0.0024,
    "common_pile/ubuntu_irc": 0.0114,
    "common_pile/uk_hansard": 0.0138,
    "common_pile/usgpo": 0.0022,
    "common_pile/uspto": 0.03935,
    "common_pile/wikimedia": 0.0948,
    "common_pile/wikiteam": 0.0172,
    "common_pile/youtube": 0.0047,
}

# Effective token counts for the cooldown stage (in teratokens)
# Weights pulled from https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset under Cooldown stage
COMMA_COOLDOWN_MIXTURE_WEIGHTS = {
    "common_pile/arxiv_papers": 0.003,
    "common_pile/cccc": 0.00456,
    "common_pile/data_provenance_initiative": 0.00184,
    "common_pile/doab": 0.006,
    "common_pile/foodista": 0.00005,
    "common_pile/libretexts": 0.00019,
    "common_pile/news": 0.00013,
    "common_pile/oercommons": 0.00002,
    "common_pile/peS2o": 0.00433,
    "common_pile/pressbooks": 0.00028,
    "common_pile/public_domain_review": 0.0,
    "common_pile/python_enhancement_proposals": 0.00001,
    "common_pile/stackexchange": 0.00597,
    "common_pile/stackv2_edu": 0.00678,
    "common_pile/wikimedia": 0.00632,
}


def common_pile_tokenized(*, tokenizer: str = llama3_tokenizer) -> dict[str, TokenizerStep]:
    """Return tokenization steps for the Common Pile filtered datasets."""
    tokenized: dict[str, TokenizerStep] = {}
    for dataset, step in COMMON_PILE_DATASETS.items():
        tokenized[f"common_pile/{dataset}"] = default_tokenize(
            name=f"common_pile/{dataset}",
            dataset=step,
            tokenizer=tokenizer,
        )
    return tokenized


def comma_main_mixture(*, tokenizer: str = llama3_tokenizer, permutation_type: PermType = "feistel"):
    """LmMixtureDatasetConfig for the main training stage."""
    tokenized = common_pile_tokenized(tokenizer=tokenizer)
    components = {f"common_pile/{dataset}": tokenized[f"common_pile/{dataset}"] for dataset in COMMON_PILE_DATASETS}
    return lm_mixture_data_config(
        components=components,
        weights=COMMA_MAIN_MIXTURE_WEIGHTS,
        permutation_type=permutation_type,
    )


def comma_cooldown_mixture(*, tokenizer: str = llama3_tokenizer, permutation_type="feistel"):
    """LmMixtureDatasetConfig for the cooldown stage."""
    tokenized = common_pile_tokenized(tokenizer=tokenizer)
    components = {f"common_pile/{dataset}": tokenized[f"common_pile/{dataset}"] for dataset in COMMON_PILE_DATASETS}
    return lm_mixture_data_config(
        components=components,
        weights=COMMA_COOLDOWN_MIXTURE_WEIGHTS,
        permutation_type=permutation_type,
    )


if __name__ == "__main__":
    steps = list(common_pile_tokenized().values())
    executor_main(steps=steps)
