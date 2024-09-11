import ray

import marin
import scripts


@ray.remote
def utils(remote_func, depends_on, *args, **kwargs):
    for ref in depends_on:
        ray.get(ray.get(ref))
    print(f"Executing {remote_func} with {args} and {kwargs}")

    return remote_func.remote(*args, **kwargs)


# Create Node objects

ray.init()

# General config
RAWDATAPATH = (
    "gs://marin-us-central2/raw/hello_world_fw/8fd6e8e/huggingface.co/datasets/skaramcheti/hello_world_fw/"
    "resolve/8fd6e8e/data"
)
EXP = "quickstart_single_script"
DATASET = "hello_world_fw"

# Transform
from scripts.hello_world_fw.process import FineWebConfig

config = FineWebConfig(input_path=RAWDATAPATH, output_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXP}")
transform_ref = utils.remote(scripts.hello_world_fw.process.main_ray, [], config)

# FastText classifier
from scripts.fasttext.train_fasttext import MainConfig

config = MainConfig(
    output_base_path="gs://marin-us-central2",
    experiment=EXP,
    pos_doc_path="gs://marin-us-central2/documents/marin_instructv1/v1_olmo_mix/text",
    neg_doc_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXP}",
    pos_sampling_rate=0.1,
    neg_sampling_rate=1.0,
    training_args={"lr": 0.1},
)
fasttext_ref = utils.remote(scripts.fasttext.train_fasttext.main_ray, [transform_ref], config)

## Use olmo classifier to annotate, Note the dependency on transform only
from marin.processing.classification.inference import InferenceConfig

config = InferenceConfig(
    input_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXP}",
    output_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXP}_olmo_fasttext",
    model_name="allenai/dolma-1_7-fasttext-quality-filter",
    model_type="fasttext",
    attribute_name="olmo-fasttext-quality",
)
annotate_ref = utils.remote(marin.processing.classification.inference.main_ray, [transform_ref], config)

## Use olmo classifier to annotate, Note the dependency on fasttext_ref
config = InferenceConfig(
    input_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXP}",
    output_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXP}_{EXP}_fasttext",
    model_name=f"gs://marin-us-central2/classifiers/{EXP}/",
    model_type="fasttext",
    attribute_name="quickstart-fasttext-quality",
)
annotate_ref_2 = utils.remote(marin.processing.classification.inference.main_ray, [fasttext_ref], config)
# Getting annotate_ref_2 as it will not be used later no and not in DAG
ray.get(ray.get(annotate_ref_2))

## Dedup, see the dependency on transform_ref
from marin.processing.classification.dedupe import DedupeConfig

config = DedupeConfig(
    input_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXP}",
    output_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXP}_duplicates",
)
dedup_ref = utils.remote(marin.processing.classification.dedupe.main_ray, [transform_ref], config)

# Consolidate all the results
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig

config = ConsolidateConfig(
    input_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXP}",
    output_path=f"gs://marin-us-central2/documents/{DATASET}/v1.0/{EXP}_consolidate",
    filters=[
        FilterConfig(
            type="dedupe",
            attribute_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXP}_duplicates/",
            name="duplicate_text",
        ),
        FilterConfig(
            type="classify",
            attribute_path=f"gs://marin-us-central2/attributes/{DATASET}/v1.0/{EXP}_olmo_fasttext/",
            name="olmo-fasttext-quality",
            label="__label__hq",
            threshold=0.1,
        ),
    ],
)

consolidate_ref = utils.remote(marin.processing.classification.consolidate.main_ray, [dedup_ref, annotate_ref], config)
ray.get(ray.get(consolidate_ref))
