from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned

############################################################
# Download the pretraining data
from operations.download.huggingface.download import DownloadConfig, download

raw_download_step = ExecutorStep(
    name="raw/hello_world_fw-pliang",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="skaramcheti/hello_world_fw",
        revision="8fd6e8e",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)
raw_data = output_path_of(
    raw_download_step, "8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/resolve/8fd6e8e/data/CC-MAIN-2024-10"
)

############################################################
# Transform HTML to text

from marin.schemas.web.convert import TrafilaturaConfig  # noqa
from scripts.hello_world_fw.process import FineWebConfig, transform  # noqa

transform_trafilatura_step = ExecutorStep(
    name="documents/hello_world_fw-pliang-trafilatura",
    fn=transform,
    config=FineWebConfig(
        input_path=raw_data,
        output_path=this_output_path(),
        extract_method=versioned("trafilatura"),
        config=TrafilaturaConfig(
            favor_precision=versioned(False),
            favor_recall=versioned(True),
            include_comments=versioned(False),
            deduplicate=versioned(False),
        ),
    ),
)

transform_resiliparse_step = ExecutorStep(
    name="documents/hello_world_fw-pliang-resiliparse",
    fn=transform,
    config=FineWebConfig(
        input_path=raw_data,
        output_path=this_output_path(),
        extract_method=versioned("resiliparse"),
    ),
)

transform_readability_step = ExecutorStep(
    name="documents/hello_world_fw-pliang-readability",
    fn=transform,
    config=FineWebConfig(
        input_path=raw_data,
        output_path=this_output_path(),
        extract_method=versioned("readability"),
    ),
)

############################################################
# Download good data

download_sft_step = ExecutorStep(
    name="raw/tulu-v2-sft-mixture",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture",
        revision="6248b17",
        gcs_output_path=this_output_path(),
    ),
)

# TODO: convert the data into Dolma format
# For now, just use the one that's already given
sft_data = "gs://marin-us-central2/documents/instruct/v1_olmo_mix/text"

############################################################
# Train quality classifier

from scripts.fasttext.train_fasttext import TrainFasttextClassifierConfig, train  # noqa

train_quality_step = ExecutorStep(
    name="classifiers/hello_world_fw-pliang",
    fn=train,
    config=TrainFasttextClassifierConfig(
        pos_doc_path=sft_data,
        neg_doc_path=output_path_of(transform_trafilatura_step),
        output_path=this_output_path(),
        pos_sampling_rate=0.1,
        neg_sampling_rate=1.0,
        fasttext_args={"lr": 0.1},
    ),
)

############################################################
# Run inference with quality classifier

from marin.processing.classification.inference import InferenceConfig, run_inference  # noqa

inference_quality_step = ExecutorStep(
    name="attributes/hello_world_fw-pliang",
    fn=run_inference,
    config=InferenceConfig(
        input_path=output_path_of(transform_trafilatura_step),
        output_path=this_output_path(),
        model_name="allenai/dolma-1_7-fasttext-quality-filter",
        model_type="fasttext",
        attribute_name="olmo-fasttext-quality",
    ),
)

############################################################
# Deduplicate

from marin.processing.classification.dedupe import DedupeConfig, dedupe  # noqa

dedupe_step = ExecutorStep(
    name="attributes/hello_world_fw-pliang-dedupe",
    fn=dedupe,
    config=DedupeConfig(
        input_path=output_path_of(transform_trafilatura_step),
        output_path=this_output_path(),
    ),
)

############################################################
# Consolidate

from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate  # noqa

consolidate_step = ExecutorStep(
    name="documents/hello_world_fw-pliang-consolidate",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=output_path_of(transform_trafilatura_step),
        output_path=this_output_path(),
        filters=[
            FilterConfig(
                type=versioned("classify"),
                attribute_path=output_path_of(inference_quality_step),
                name=versioned("olmo-fasttext-quality"),
                label="__label__hq",
                threshold=versioned(0.1),
            ),
            FilterConfig(
                type=versioned("remove_spans"),
                attribute_path=output_path_of(dedupe_step),
                name=versioned("duplicate_text"),
            ),
        ],
    ),
)

############################################################
# Tokenize

from marin.processing.tokenize import TokenizeConfig, tokenize  # noqa

tokenize_step = ExecutorStep(
    name="tokenized/llama3/hello_world_fw-pliang",
    fn=tokenize,
    config=TokenizeConfig(
        input_path=output_path_of(consolidate_step),
        cache_path=this_output_path(),
        dataset_name="hello_world_fw-pliang",  # Does this have to be unique?
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
    ),
)

############################################################
# Training

# TODO: wait for Ray version

############################################################
# Evaluate

# TODO: wait for draccus version

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            transform_trafilatura_step,
            transform_resiliparse_step,  # Not used
            transform_readability_step,  # Not used
            # train_quality_step,  # Not used  (TODO: fails right now)
            tokenize_step,
        ]
    )
