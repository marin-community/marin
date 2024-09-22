import draccus
import marin
import scripts
from marin.execution.executor import ExecutorStep, get_input, get_output, versioned, executor_main

############################################################
# Download the pretraining data

from operations.download.huggingface.download import download, DownloadConfig

raw_download_step = ExecutorStep(name="raw/hello_world_fw-pliang", fn=download, config=DownloadConfig(
    hf_dataset_id="skaramcheti/hello_world_fw",
    revision="8fd6e8e",
    gcs_output_path=get_output(),
))
raw_data = get_input(raw_download_step, "8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/resolve/8fd6e8e/data/CC-MAIN-2024-10")

############################################################
# Transform HTML to text

from scripts.hello_world_fw.process import transform, FineWebConfig
from marin.schemas.web.convert import TrafilaturaConfig

transform_trafilatura_step = ExecutorStep(name="documents/hello_world_fw-pliang-trafilatura", fn=transform, config=FineWebConfig(
    input_path=raw_data,
    output_path=get_output(),
    extract_method=versioned("trafilatura"),
    config=TrafilaturaConfig(
        favor_precision=versioned(False),
        favor_recall=versioned(True),
        include_comments=versioned(False),
        deduplicate=versioned(False),
    ),
))

transform_resiliparse_step = ExecutorStep(name="documents/hello_world_fw-pliang-resiliparse", fn=transform, config=FineWebConfig(
    input_path=raw_data,
    output_path=get_output(),
    extract_method=versioned("resiliparse"),
))

transform_readability_step = ExecutorStep(name="documents/hello_world_fw-pliang-readability", fn=transform, config=FineWebConfig(
    input_path=raw_data,
    output_path=get_output(),
    extract_method=versioned("readability"),
))

############################################################
# Download good data

download_sft_step = ExecutorStep(name="raw/tulu-v2-sft-mixture", fn=download, config=DownloadConfig(
    hf_dataset_id="allenai/tulu-v2-sft-mixture",
    revision="6248b17",
    gcs_output_path=get_output(),
))

# TODO: convert the data into Dolma format
# For now, just use the one that's already given
sft_data = "gs://marin-us-central2/documents/instruct/v1_olmo_mix/text",

############################################################
# Train quality classifier

from scripts.fasttext.train_fasttext import train, TrainFasttextClassifierConfig

train_quality_step = ExecutorStep(name="classifiers/hello_world_fw-pliang", fn=train, config=TrainFasttextClassifierConfig(
    pos_doc_path=sft_data,
    neg_doc_path=get_input(transform_trafilatura_step),
    output_path=get_output(),
    pos_sampling_rate=0.1,
    neg_sampling_rate=1.0,
    fasttext_args={"lr": 0.1},
))

############################################################
# Apply quality classifier

from marin.processing.classification.inference import main_ray, InferenceConfig

annotate_quality_step = ExecutorStep(name="attributes/hello_world_fw-pliang", fn=main_ray, config=InferenceConfig(
    input_path=raw_data,
    model_name="gs://marin-us-central2/classifiers/dclm-replication/fasttext_model.bin",
    model_type="fasttext",
    attribute_name="quickstart-fasttext-quality",
))

############################################################

if __name__ == "__main__":
    executor_main(steps=[
        raw_download_step,
        #transform_trafilatura_step,
        #transform_resiliparse_step,
        #transform_readability_step,
        train_quality_step,
        #annotate_quality_step,
    ])

#
# ## Dedup, see the dependency on transform_ref
# from marin.processing.classification.dedupe import DedupeConfig  # noqa
#
# config = DedupeConfig(
#     input_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXPERIMENT}",
#     output_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXPERIMENT}_duplicates",
# )
# dedup_ref = execute.remote(marin.processing.classification.dedupe.main_ray, config, depends_on=[transform_ref])
#
# # Consolidate all the results
# from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig  # noqa
#
# config = ConsolidateConfig(
#     input_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXPERIMENT}",
#     output_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXPERIMENT}_consolidate",
#     filters=[
#         FilterConfig(
#             type="dedupe",
#             attribute_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXPERIMENT}_duplicates/",
#             name="duplicate_text",
#         ),
#         FilterConfig(
#             type="classify",
#             attribute_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXPERIMENT}_olmo_fasttext/",
#             name="olmo-fasttext-quality",
#             label="__label__hq",
#             threshold=0.1,
#         ),
#     ],
# )
#
# consolidate_ref = execute.remote(
#     marin.processing.classification.consolidate.main_ray, config, depends_on=[transform_ref, dedup_ref]
# )
#
#
# # Tokenize
# from marin.processing.tokenize import TokenizeConfig  # noqa
#
# config = TokenizeConfig(
#     input_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXPERIMENT}_consolidate/",
#     cache_path="gs://marin-us-central2/tokenized/llama3/",
#     dataset_name=f"{DATASET}-{EXPERIMENT}",
#     tokenizer="meta-llama/Meta-Llama-3.1-8B",
# )
# tokenize_ref = execute.remote(marin.processing.tokenize.main_ray, config, depends_on=[consolidate_ref])
#
# # Train
# from scripts.training.launch import LaunchConfig  # noqa
#
# config = LaunchConfig(
#     experiment=EXPERIMENT,
#     base_config="config/training/quickstart_run.yaml",
#     dataset_name=f"{DATASET}-{EXPERIMENT}",
#     dataset_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXPERIMENT}_consolidate/**/*.jsonl.gz",
#     cache_dir="gs://marin-us-central2/tokenized/llama3/",
#     zone="us-central2-b",
#     tpu_type="v4-32",
# )
#
# train_ref = execute.remote(scripts.training.launch.main, config, depends_on=[tokenize_ref])
#
# ray.get(train_ref)
