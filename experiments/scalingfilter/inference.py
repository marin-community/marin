"""Script to reproduce the ScalingFilter Paper

TODO: remove monkey patch with requirements.txt in the ExecutorStep.
"""

import os

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, TaskConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.custom.custom_attribute import CustomAttributeConfig, create_custom_attribute
from marin.processing.classification.inference import run_inference


def perplexity_inference_step(small_model_name, large_model_name, data_path, data_path_name, attribute_name, filetype):
    """Create inference steps for perplexity models and quality factor calculation."""
    small_model_basename = os.path.basename(small_model_name)
    large_model_basename = os.path.basename(large_model_name)

    small_model_attribute_name = f"{small_model_basename}-perplexity-seq-len-512"
    large_model_attribute_name = f"{large_model_basename}-perplexity-seq-len-512"
    small_model_inference_step = ExecutorStep(
        name=f"attributes/quality_filtering/{small_model_basename}-perplexity/{data_path_name}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=data_path,
            output_path=this_output_path(),
            model_name=small_model_name,
            model_type="perplexity",
            attribute_name=small_model_attribute_name,
            runtime=RuntimeConfig(
                memory_limit_gb=12,
                resources={"TPU": 1},
            ),
            task=TaskConfig(max_in_flight=500),
            filetype=filetype,
            batch_size=60,
            classifier_kwargs={"max_length": 512},
        ),
        pip_dependency_groups=["experiments/scalingfilter/requirements.txt"],
    )

    large_model_inference_step = ExecutorStep(
        name=f"attributes/quality_filtering/{large_model_basename}-perplexity/{data_path_name}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=data_path,
            output_path=this_output_path(),
            model_name=large_model_name,
            model_type="perplexity",
            attribute_name=large_model_attribute_name,
            runtime=RuntimeConfig(
                memory_limit_gb=12,
                resources={"TPU": 1},
            ),
            task=TaskConfig(max_in_flight=500),
            filetype=filetype,
            batch_size=60,
            classifier_kwargs={"max_length": 512},
        ),
        pip_dependency_groups=["experiments/scalingfilter/requirements.txt"],
    )

    def perplexity_quality_factor(doc, attrs) -> dict:
        small_model_perplexity = attrs[0]["attributes"][small_model_attribute_name]
        large_model_perplexity = attrs[1]["attributes"][large_model_attribute_name]

        return {attribute_name: small_model_perplexity / large_model_perplexity}

    quality_factor_step = ExecutorStep(
        name=f"attributes/quality_filtering/{attribute_name}/{data_path_name}",
        fn=create_custom_attribute,
        config=CustomAttributeConfig(
            input_doc_path=data_path,
            output_attr_path=this_output_path(),
            label_func=perplexity_quality_factor,
            filetype=filetype,
            input_attr_paths=[output_path_of(small_model_inference_step), output_path_of(large_model_inference_step)],
        ),
    )

    return small_model_inference_step, large_model_inference_step, quality_factor_step


attribute_name = "scalingfilter-quality-factor"
dclm_data_path = "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/\
    resolve/a3b142c/global-shard_01_of_10/local-shard_3_of_10"
dclm_small_model_inference_step, dclm_large_model_inference_step, dclm_quality_factor_step = perplexity_inference_step(
    small_model_name="/opt/gcsfuse_mount/perplexity-models/llama-200m-local-shard-2",
    large_model_name="/opt/gcsfuse_mount/perplexity-models/llama-500m-local-shard-2",
    data_path=dclm_data_path,
    data_path_name="dclm-global-shard-01-of-10-local-shard_3_of_10",
    attribute_name=attribute_name,
    filetype="jsonl.zst",
)

fineweb_data_path = "gs://marin-us-central2/documents/scaling-filter-fineweb-subset"
fineweb_small_model_inference_step, fineweb_large_model_inference_step, fineweb_quality_factor_step = (
    perplexity_inference_step(
        small_model_name="/opt/gcsfuse_mount/perplexity-models/llama-200m-local-shard-2",
        large_model_name="/opt/gcsfuse_mount/perplexity-models/llama-500m-local-shard-2",
        data_path=fineweb_data_path,
        data_path_name="fineweb-subset",
        attribute_name=attribute_name,
        filetype="jsonl.gz",
    )
)


top_50_scalingfilter_dclm = ExecutorStep(
    name="documents/quality_filtering/scaling-filter-quality-factor/dclm-global-shard-01-of-10-local-shard_3_of_10",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=dclm_data_path,
        output_path=this_output_path(),
        filters=[
            FilterConfig(
                type="classify",
                attribute_path=output_path_of(dclm_quality_factor_step),
                name="scalingfilter-quality-factor",
                keep_fraction=0.50,
            ),
        ],
        ray_memory_limit_gb=12,
        filetype="jsonl.zst",
    ),
    pip_dependency_groups=["ddsketch"],
)

top_50_scalingfilter_dclm_tokenized = default_tokenize(
    name="tokenized/quality_filtering/scaling-filter-quality-factor/dclm-global-shard-01-of-10-local-shard_3_of_10",
    dataset=top_50_scalingfilter_dclm,
    tokenizer=llama3_tokenizer,
)


def default_quality_filter_consolidate_and_tokenize(
    experiment_name, model_type, model_name, attribute_name, requires_tpu, input_data_path
):
    """Create a pipeline to filter and tokenize data based on quality metrics."""
    if requires_tpu:
        resources = {"TPU": 1}
    else:
        resources = {}

    model_type_to_label = {
        "fineweb": "score",
        "perplexity": None,
        "fasttext": "__label__hq",
    }

    quality_filter_inference = ExecutorStep(
        name=f"attributes/quality_filtering/{experiment_name}/dclm-global-shard-01-of-10-local-shard_3_of_10",
        fn=run_inference,
        config=InferenceConfig(
            input_path=input_data_path,
            output_path=this_output_path(),
            model_name=model_name,
            model_type=model_type,
            attribute_name=attribute_name,
            runtime=RuntimeConfig(
                memory_limit_gb=12,
                resources=resources,
            ),
            task=TaskConfig(max_in_flight=500),
            filetype="jsonl.zst",
            batch_size=128,
            classifier_kwargs={"max_length": 512},
        ),
        pip_dependency_groups=["experiments/scalingfilter/requirements.txt"],
    )

    top_50_dclm = ExecutorStep(
        name="documents/quality_filtering/fineweb-edu/dclm-global-shard-01-of-10-local-shard_3_of_10",
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=input_data_path,
            output_path=this_output_path(),
            filters=[
                FilterConfig(
                    type="classify",
                    attribute_path=output_path_of(quality_filter_inference),
                    name=attribute_name,
                    label=model_type_to_label[model_type],
                    keep_fraction=0.50,
                ),
            ],
            ray_memory_limit_gb=12,
            filetype="jsonl.zst",
        ),
        pip_dependency_groups=["ddsketch"],
    )

    top_50_tokenized = default_tokenize(
        name=f"tokenized/quality_filtering/{experiment_name}/dclm-global-shard-01-of-10-local-shard_3_of_10",
        dataset=top_50_dclm,
        tokenizer=llama3_tokenizer,
    )

    return top_50_tokenized


top50_fineweb_edu_dclm_tokenized = default_quality_filter_consolidate_and_tokenize(
    experiment_name="fineweb-edu",
    model_type="fineweb",
    model_name="HuggingFaceFW/fineweb-edu-classifier",
    attribute_name="fineweb-edu",
    requires_tpu=True,
    input_data_path=dclm_data_path,
)

top50_fasttext_hq_dclm_tokenized = default_quality_filter_consolidate_and_tokenize(
    experiment_name="dclm-fasttext",
    model_type="fasttext",
    model_name="mlfoundations/fasttext-oh-eli5",
    attribute_name="dclm-fasttext-quality",
    requires_tpu=False,
    input_data_path=dclm_data_path,
)

if __name__ == "__main__":
    # executor_main([inference_step])
    # executor_main([llama_200m_inference_step,
    # llama_500m_inference_step,
    # quality_factor_step,
    # top_50_scalingfilter_dclm,
    # top_50_scalingfilter_dclm_tokenized])
    # executor_main([fineweb_edu_inference])
    # executor_main([top50_fasttext_hq_dclm_tokenized])s
    # executor_main([fineweb_small_model_inference_step])
    executor_main([fineweb_large_model_inference_step])
