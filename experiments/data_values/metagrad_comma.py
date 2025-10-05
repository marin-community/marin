from experiments.data_values.train import DatagradsConfig, datagrads_train_step
from marin.execution.executor import ExecutorMainConfig, executor_main

force_run_failed = True
"""Set to True to force previously failed steps to run again."""

mixture_cache_dir = "gs://data-values-us-central-1/data_cache/common_pile_CPT_mix_20_wiki_warmup"
mixture_block_size = 2000

mixture_config_train_ds_names = [
    "arxiv_abstracts",
    "arxiv_papers",
    "biodiversity_heritage_library",
    "caselaw_access_project",
    "data_provenance_initiative",
    "doab",
    "foodista",
    "github_archive",
    "library_of_congress",
    "libretexts",
    "news",
    "oercommons",
    "peS2o",
    "pre_1929_books",
    "pressbooks",
    "project_gutenberg",
    "public_domain_review",
    "stack_exchange",
]

mixture_configs = {
    "arxiv_abstracts": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/arxiv_abstracts_filtered-f1d7a9a/arxiv-abstracts-dolma-{0000..0001}.json.gz",
        ]
    },
    "arxiv_papers": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/arxiv_papers_filtered-033cf7f/arxiv-papers-{0000..0007}.json.gz",
        ]
    },
    "biodiversity_heritage_library": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/biodiversity_heritage_library_filtered-0486ed6/biodiversity-heritage-library-{0000..0009}.json.gz",
        ]
    },
    "caselaw_access_project": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/caselaw_access_project_filtered-50e1961/CAP-Dolma-{0000..0009}.json.gz",
        ]
    },
    "data_provenance_initiative": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/data_provenance_initiative_filtered-8f5afcf/dpi-common-pile-{0000..0002}.json.gz",
        ]
    },
    "doab": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/doab_filtered-defb24c/doab-{0000..0004}.json.gz",
        ]
    },
    "foodista": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/foodista_filtered-bf2c7aa/foodista-dolma-{0000..0000}.json.gz",
        ]
    },
    "github_archive": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/github_archive_filtered-52282fe/gharchive-dolma-{0000..0009}.json.gz",
        ]
    },
    "library_of_congress": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/library_of_congress_filtered-56725c7/loc_books_dolma-{0000..0009}.json.gz",
        ]
    },
    "libretexts": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/libretexts_filtered-70388bc/libretexts-{0000..0000}.json.gz",
        ]
    },
    "news": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/news_filtered-59aaa8f/news-dolma-{0000..0000}.json.gz",
        ]
    },
    "oercommons": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/oercommons_filtered-506b615/oercommons-{0000..0000}.json.gz",
        ]
    },
    "peS2o": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/peS2o_filtered-2977475/peS2o-{0000..0009}.json.gz",
        ]
    },
    "pre_1929_books": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/pre_1929_books_filtered-23f9d96/public_library_1929_dolma-{0000..0009}.json.gz",
        ]
    },
    "pressbooks": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/pressbooks_filtered-1a1d3b5/pressbooks-{0000..0000}.json.gz",
        ]
    },
    "project_gutenberg": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/project_gutenberg_filtered-3cdf687/project_gutenberg-dolma-{0000..0009}.json.gz",
        ]
    },
    "public_domain_review": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/public_domain_review_filtered-efc7f21/public-domain-review-{0000..0000}.json.gz",
        ]
    },
    # "pubmed": {
    #     "train_urls": [
    #         "gs://marin-us-central1/raw/common_pile/pubmed_filtered-c156f05/licensed_pubmed-{0000..0009}.json.gz",
    #     ]
    # },
    "stack_exchange": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/stackexchange_filtered-c0ac737/stackexchange-dolma-{0000..0009}.json.gz",
        ]
    },
    "stackv2_edu": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/stackv2_edu_filtered-c354dbe/stack-edu-{0000..0009}.json.gz",
        ]
    },
    "youtube": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/youtube_filtered-dff8c8a/youtube-commons-{0000..0008}.json.gz",
        ]
    },
    "wikimedia": {
        "train_urls": [
            "gs://marin-us-central1/raw/common_pile/wikimedia_filtered-0641bb8/wikimedia-{0000..0034}.json.gz",
        ]
    },
    "paloma/dolma_100_subreddits": {
        "validation_urls": [
            "gs://levanter-data/paloma/dolma_100_subreddits/val/val*.jsonl.gz",
        ]
    },
}

mixture_train_weights_stage_1 = {
    "wikimedia": 1.0,
    #"paloma/dolma_100_programing_languages": 0.0,
    "paloma/dolma_100_subreddits": 0.0,
}

mixture_train_weights_stage_2 = {
    "arxiv_abstracts": 1.0,
    "arxiv_papers": 1.0,
    "biodiversity_heritage_library": 1.0,
    "caselaw_access_project": 1.0,
    "data_provenance_initiative": 1.0,
    "doab": 1.0,
    "foodista": 1.0,
    "github_archive": 1.0,
    "library_of_congress": 1.0,
    "libretexts": 1.0,
    "news": 1.0,
    "oercommons": 1.0,
    "peS2o": 1.0,
    "pre_1929_books": 1.0,
    "pressbooks": 1.0,
    "project_gutenberg": 1.0,
    "public_domain_review": 1.0,
    # "pubmed": 1.0,
    "stack_exchange": 1.0,
    "stackv2_edu": 1.0,
    "youtube": 1.0,
    #"paloma/dolma_100_programing_languages": 0.0,
    "paloma/dolma_100_subreddits": 0.0,
}

mixture_train_weights = [
    (0, mixture_train_weights_stage_1),
    (3000, mixture_train_weights_stage_2),
]

#zero_loss_datasets = ["arxiv_abstracts"]

eval_task_spec = [{"task": "arc_easy", "num_fewshot": 0}]
eval_max_eval_length = 128
eval_max_examples = 100

initialize_from_checkpoint_path = (
    "gs://marin-us-central1/checkpoints/data_grads/smoothness-v4-bs1024-s24000-lr1.0e-03-wd0.10-gc1.0-er1.00e-08/checkpoints/step-23999"
)


def zero_out_single_dataset(mixture_train_weights, dataset_name):
    weights_ = mixture_train_weights.copy()
    weights_[dataset_name] = 0.0
    return weights_


train_steps = [
    datagrads_train_step(
        DatagradsConfig(
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            train_batch_size=bs,
            num_train_steps=num_train_steps,
            schedule_steps=num_train_steps,
            epsilon_root=epsilon_root,
            metagrad_segment_size=metagrad_segment_size,
            wandb_project_name="levanter",
            # Optional: override a few defaults baked from the prior YAML
            gcs_bucket="data-values-us-central-1",
            tpu_type="v5p-8",
            use_comma_main_mixture=False,
            lr_schedule="cosine", # cosine
            exp_name="metagrad-comma->paloma_subreddit-v0",
            mixture_configs=mixture_configs,
            mixture_train_weights=mixture_train_weights,
            mixture_cache_dir=mixture_cache_dir,
            mixture_block_size=mixture_block_size,
            #hf_dataset_id="dlwh/wikitext_103_detokenized",
            #zero_loss_datasets=zero_loss_datasets,
            train_only=False,
            #initialize_from_checkpoint_path=initialize_from_checkpoint_path,
            #eval_task_spec=eval_task_spec,
            #eval_max_eval_length=eval_max_eval_length,
            #eval_max_examples=eval_max_examples,
            #eval_harness_steps=10000,
        )
    )
    for num_train_steps in [24_000]
    for max_grad_norm in [1.0] #, 10., None]
    for lr in [4e-4]
    for bs in [1024]
    for weight_decay in [0.1] #, 0.2] #, 0.2]
    for epsilon_root in [1e-8] #, 1e-8, 1e-7]
    for metagrad_segment_size in [100]
]


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    for batch in chunked(train_steps, 20):
        # The draccus wrapper around executor_main allows us to pass in a config
        # object with our desired settings.
        executor_main(ExecutorMainConfig(force_run_failed=force_run_failed), batch)


"""
if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Metagrads test grid",
    )
"""