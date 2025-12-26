from experiments.metagrads.train_config import DatagradsConfig, datagrads_train_step
from experiments.defaults import default_tokenize
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from levanter.data.text import LMMixtureDatasetConfig, TextLmDatasetFormat
import hashlib

force_run_failed = True
"""Set to True to force previously failed steps to run again."""

GCS_BUCKET = "marin-us-central1"
EXPERIMENT_TAG = "debug_Nov27" #"debug_Nov16" #"metagrad_comma_v3"

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
    "peS2o",
    "pre_1929_books",
    "pressbooks",
    "project_gutenberg",
    "public_domain_review",
    "stack_exchange",
    "stackv2_edu",
    "youtube",
    "wikimedia",
    'null'
]

EVAL_TARGET = "paloma_c4_en"

EVAL_DATASET_NAMES = [
    "paloma_wikitext_103",
    "paloma_c4_en",
    "paloma_falcon-refinedweb",
    "paloma_m2d2_s2orc_unsplit",
    "paloma_m2d2_wikipedia_unsplit",
    "paloma_dolma_100_programing_languages",
    "paloma_dolma_100_subreddits",
    "paloma_4chan_meta_sep",
]

EVAL_DATASETS = {
    "paloma_wikitext_103": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/wikitext_103/val/val.jsonl.gz",
        ]
    },
    "paloma_c4_en": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/c4_en/val/val*.jsonl.gz",
        ]
    },
    "paloma_falcon-refinedweb": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/falcon-refinedweb/val/val*.jsonl.gz",
        ]
    },
    "paloma_m2d2_s2orc_unsplit": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/m2d2_s2orc_unsplit/val/val*.jsonl.gz",
        ]
    },
    "paloma_m2d2_wikipedia_unsplit": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/m2d2_wikipedia_unsplit/val/val*.jsonl.gz",
        ]
    },
    "paloma_dolma_100_programing_languages": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/dolma_100_programing_languages/val/val*.jsonl.gz",
        ]
    },
    "paloma_dolma_100_subreddits": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/dolma_100_subreddits/val/val*.jsonl.gz",
        ]
    },
    "paloma_4chan_meta_sep": {
        "format": TextLmDatasetFormat(),
        "validation_urls": [
            f"gs://{GCS_BUCKET}/raw/paloma-fc6827/65cd6fc/4chan_meta_sep/val/val*.jsonl.gz",
        ]
    },
}



mixture_configs = {
    "arxiv_abstracts": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/arxiv_abstracts_filtered-f1d7a9a/arxiv-abstracts-dolma-{{0000..0001}}.json.gz",
        ]
    },
    "arxiv_papers": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/arxiv_papers_filtered-033cf7f/arxiv-papers-{{0000..0007}}.json.gz",
        ]
    },
    "biodiversity_heritage_library": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/biodiversity_heritage_library_filtered-0486ed6/biodiversity-heritage-library-{{0000..0009}}.json.gz",
        ]
    },
    "caselaw_access_project": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/caselaw_access_project_filtered-50e1961/CAP-Dolma-{{0000..0009}}.json.gz",
        ]
    },
    "data_provenance_initiative": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/data_provenance_initiative_filtered-8f5afcf/dpi-common-pile-{{0000..0002}}.json.gz",
        ]
    },
    "doab": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/doab_filtered-defb24c/doab-{{0000..0004}}.json.gz",
        ]
    },
    "foodista": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/foodista_filtered-bf2c7aa/foodista-dolma-{{0000..0000}}.json.gz",
        ]
    },
    "github_archive": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/github_archive_filtered-52282fe/gharchive-dolma-{{0000..0009}}.json.gz",
        ]
    },
    "library_of_congress": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/library_of_congress_filtered-56725c7/loc_books_dolma-{{0000..0009}}.json.gz",
        ]
    },
    "libretexts": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/libretexts_filtered-70388bc/libretexts-{{0000..0000}}.json.gz",
        ]
    },
    "news": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/news_filtered-59aaa8f/news-dolma-{{0000..0000}}.json.gz",
        ]
    },
    "oercommons": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/oercommons_filtered-506b615/oercommons-{{0000..0000}}.json.gz",
        ]
    },
    "peS2o": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/peS2o_filtered-2977475/peS2o-{{0000..0009}}.json.gz",
        ]
    },
    "pre_1929_books": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/pre_1929_books_filtered-23f9d96/public_library_1929_dolma-{{0000..0009}}.json.gz",
        ]
    },
    "pressbooks": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/pressbooks_filtered-1a1d3b5/pressbooks-{{0000..0000}}.json.gz",
        ]
    },
    "project_gutenberg": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/project_gutenberg_filtered-3cdf687/project_gutenberg-dolma-{{0000..0009}}.json.gz",
        ]
    },
    "public_domain_review": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/public_domain_review_filtered-efc7f21/public-domain-review-{{0000..0000}}.json.gz",
        ]
    },
    # "pubmed": {
    #     "format": "text",
    #     "train_urls": [
    #         f"gs://{GCS_BUCKET}/raw/common_pile/pubmed_filtered-c156f05/licensed_pubmed-{{0000..0009}}.json.gz",
    #     ]
    # },
    "stack_exchange": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/stackexchange_filtered-c0ac737/stackexchange-dolma-{{0000..0009}}.json.gz",
        ]
    },
    "stackv2_edu": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/stackv2_edu_filtered-c354dbe/stack-edu-{{0000..0009}}.json.gz",
        ]
    },
    "youtube": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/youtube_filtered-dff8c8a/youtube-commons-{{0000..0008}}.json.gz",
        ]
    },
    "wikimedia": {
        "format": TextLmDatasetFormat(),
        "train_urls": [
            f"gs://{GCS_BUCKET}/raw/common_pile/wikimedia_filtered-0641bb8/wikimedia-{{0000..0009}}.json.gz",
        ]
    },
}


eval_task_spec = [{"task": "arc_easy", "num_fewshot": 0}]
eval_max_eval_length = 128
eval_max_examples = 100


def create_tokenization_steps(eval_target: str) -> dict[str, ExecutorStep]:
    """Create executor tokenization steps for train mixture and the eval target.

    We convert raw URLs into Marin tokenization steps so that tokenization is
    cached and reusable across experiments.
    """
    tokenized_steps: dict[str, ExecutorStep] = {}

    def compute_step_suffix(dataset_pattern: str, tokenizer: str = "gpt2") -> str:
        base = f"{EXPERIMENT_TAG}|{tokenizer}|{dataset_pattern}"
        return hashlib.sha256(base.encode()).hexdigest()[:6]

    # Training mixture tokenization steps
    for ds_name in mixture_config_train_ds_names:
        cfg = mixture_configs.get(ds_name)
        # Skip entries that do not have train URLs (e.g., placeholders)
        if not cfg or "train_urls" not in cfg:
            continue
        train_url_pattern = cfg["train_urls"][0]
        fmt = cfg.get("format", TextLmDatasetFormat())
        suffix = compute_step_suffix(train_url_pattern)
        tokenized_steps[f"common_pile/{ds_name}-{suffix}"] = default_tokenize(
            name=f"common_pile/{ds_name}-{suffix}",
            dataset=train_url_pattern,
            tokenizer="gpt2",
            format=fmt,
        )

    # Validation tokenization step for the selected eval target
    eval_cfg = EVAL_DATASETS[eval_target]
    eval_url = eval_cfg["validation_urls"][0]
    eval_suffix = compute_step_suffix(eval_url)
    tokenized_steps[f"val/{eval_target}-{eval_suffix}"] = default_tokenize(
        name=f"{eval_target}-{eval_suffix}",
        dataset=eval_url,
        tokenizer="gpt2",
        format=eval_cfg.get("format", TextLmDatasetFormat()),
        is_validation=True,
    )

    return tokenized_steps


def get_training_data_config(eval_target: str) -> LMMixtureDatasetConfig:
    """Create LMMixtureDatasetConfig with uniform train weights throughout.

    All training sources receive equal weight. The eval target is included with
    zero weight so it is available for validation but excluded from training.
    """
    tokenized_steps = create_tokenization_steps(eval_target)

    # Convert tokenization steps into mixture components
    configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=True)
        for name, step in tokenized_steps.items()
    }

    # Separate train vs eval step names
    train_step_names = [n for n in tokenized_steps.keys() if n.startswith("common_pile/")]
    eval_step_candidates = [n for n in tokenized_steps.keys() if n.startswith(f"val/{eval_target}")]
    eval_step_name = eval_step_candidates[0] if len(eval_step_candidates) > 0 else f"val/{eval_target}"

    # Uniform weights across all train sources; eval stays 0.0
    uniform_weights = {name: 1.0 for name in train_step_names}
    uniform_weights[eval_step_name] = 0.0

    train_weights = [
        (0, uniform_weights),
    ]

    return LMMixtureDatasetConfig(
        configs=configs,
        train_weights=train_weights,
        tokenizer="gpt2",
        cache_dir=None,
        shuffle=True,
    )


def zero_out_single_dataset(mixture_train_weights, dataset_name):
    weights_ = mixture_train_weights.copy()
    weights_[dataset_name] = 0.0
    return weights_


CHECKPOINT_PATHS = {
    '0': f"gs://marin-us-central1/checkpoints/data_grads/debug_metagrad-comma->paloma_dolma_100_subreddits-8k-Nov14-v0-bs1024-s8000-lr1.0e-04-wd0.10-gc2.0-er/checkpoints/step-5499",
    '2': f"gs://marin-us-central1/checkpoints/data_grads/debug_metagrad-comma->paloma_dolma_100_subreddits-8k-Nov14-v2-bs1024-s8000-lr1.0e-04-wd0.10-gc2.0-er/checkpoints/step-5499"
}


train_steps = [
    datagrads_train_step(
        DatagradsConfig(
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            train_batch_size=bs,
            num_train_steps=num_train_steps,
            #schedule_steps=schedule_steps,
            epsilon_root=epsilon_root,
            #metagrad_segment_size=metagrad_segment_size,
            wandb_project_name="levanter",
            gcs_bucket=GCS_BUCKET,
            tpu_type="v5p-8",
            lr_schedule="linear",
            #exp_name=f"metagrad-comma->{eval_target}-DEBUG-ND-{trial}",
            #exp_name=f"debug_metagrad-comma->{eval_target}-8k-Nov15-v{trial}-resume-{seed}",
            #exp_name=f"debug_metagrad-comma->{eval_target}-8k-Nov15-noresume-{seed}",
            #exp_name=f"debug_metagrad-comma->{eval_target}-8k-Nov15-noresume-bad-tag-new-text-recache-{seed}",
            #exp_name=f"debug_metagrad-comma->{eval_target}-8k-Nov29-bisect-ef34-fix-{seed}",
            exp_name=f"metagrad_Dec26-rebased-firstpass-v2",
            data_config=get_training_data_config(eval_target),
            #train_only=True,
            #train_only=True,
            #initialize_from=CHECKPOINT_PATHS[str(trial)],
            # initialize_from_checkpoint_path=initialize_from_checkpoint_path,
            # eval_task_spec=eval_task_spec,
            # eval_max_eval_length=eval_max_eval_length,
            # eval_max_examples=eval_max_examples,
            # eval_harness_steps=10000,
            fsdp_axis='embed',
            per_device_parallelism=128,
            per_device_eval_parallelism=128,
            seed=0,
            data_seed=0,
            save_input_ids=True,
        )
    )
    for seed in [1]
    #for trial in [0, 2]
    #for eval_target in EVAL_DATASET_NAMES #\
    for eval_target in ['paloma_dolma_100_subreddits']
    for num_train_steps in [20]
    for schedule_steps in [12_000]
    for max_grad_norm in [2.0]
    for lr in [1e-4]
    for bs in [1024]
    for weight_decay in [0.1]
    for epsilon_root in [1e-8]
    for metagrad_segment_size in [2]
]#


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    # Optionally, run tokenization steps first to pre-populate caches
    # tokenization_steps = list(create_tokenization_steps(EVAL_TARGET).values())
    # executor_main(ExecutorMainConfig(force_run_failed=force_run_failed), tokenization_steps)

    for batch in chunked(train_steps, 20):
        executor_main(ExecutorMainConfig(force_run_failed=force_run_failed), batch)


"""
if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Metagrads test grid",
    )
"""