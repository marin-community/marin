import logging
import ray

import marin
import scripts
from experiments.utils import Executor

logger = logging.getLogger("ray")




# General config
RAWDATAPATH = (
    "gs://marin-us-central2/raw/hello_world_fw/8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/"
    "resolve/8fd6e8e/data"
)
EXPERIMENT = "quickstart"
DATASET = "hello_world_fw"

executor = Executor(region="us-central2", experiment_prefix=EXPERIMENT, **{"dataset" : DATASET})
# Transform
from scripts.hello_world_fw.process import FineWebConfig  # noqa

config = FineWebConfig(
    input_path=RAWDATAPATH
)
transform_ref = executor.add(scripts.hello_world_fw.process.main_ray, config=config)

# FastText classifier
from scripts.fasttext.train_fasttext import MainConfig  # noqa

config = MainConfig(
    pos_doc_path="gs://marin-us-central2/documents/instruct/v1_olmo_mix/text",
    neg_doc_path=transform_ref,
    pos_sampling_rate=0.1,
    neg_sampling_rate=1.0,
    training_args={"lr": 0.1},
)
fasttext_ref = executor.add(scripts.fasttext.train_fasttext.main_ray, config=config)

## Use olmo classifier to annotate, Note the dependency on transform only
from marin.processing.classification.inference import InferenceConfig  # noqa

config = InferenceConfig(
    input_path=transform_ref,
    model_name="allenai/dolma-1_7-fasttext-quality-filter",
    model_type="fasttext",
    attribute_name="olmo-fasttext-quality",
)
annotate_ref = executor.add(marin.processing.classification.inference.main_ray, config = config)

## Use quickstart classifier to annotate, Note the dependency on fasttext_ref
config = InferenceConfig(
    input_path=transform_ref,
    model_name=fasttext_ref,
    model_type="fasttext",
    attribute_name="quickstart-fasttext-quality",
)
annotate_ref_2 = executor.add(marin.processing.classification.inference.main_ray, config=config)
# Getting annotate_ref_2 as it will not be used later no and not in DAG
executor.run(annotate_ref_2)
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
