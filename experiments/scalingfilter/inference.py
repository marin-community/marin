from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, TaskConfig
from marin.processing.classification.inference import run_inference

input_data_path = "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10/local-shard_3_of_10"
MODEL_BASENAME = "llama-200m-local-shard-2"
inference_step = ExecutorStep(
    name=f"attributes/quality_filtering/{MODEL_BASENAME}-perplexity/dclm-global-shard-01-of-10-local-shard_3_of_10",
    fn=run_inference,
    config=InferenceConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/perplexity-models/llama-200m-local-shard-2/hf/step-11999",
        model_type="perplexity",
        attribute_name=f"{MODEL_BASENAME}-perplexity-seq-len-512",
        runtime=RuntimeConfig(
            memory_limit_gb=12,
            resources={"TPU": 1},
        ),
        task=TaskConfig(max_in_flight=500),
        filetype="jsonl.zst",
        batch_size=60,
        classifier_kwargs={"max_length": 512},
    ),
    pip_dependency_groups=["experiments/scalingfilter/requirements.txt"],
)

# Did inference over the entire global shard accidentally so directory will be named a bit confusingly.
llama_500m_inference_step = ExecutorStep(
    name="attributes/quality_filtering/llama-500m-perplexity/dclm-global-shard-01-of-10/local-shard_0_of_10",
    fn=run_inference,
    config=InferenceConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/perplexity-models/llama-500m",
        model_type="perplexity",
        attribute_name="llama-500m-perplexity-seq-len-512",
        runtime=RuntimeConfig(
            memory_limit_gb=12,
            resources={"TPU": 1},
        ),
        task=TaskConfig(max_in_flight=500),
        filetype="jsonl.zst",
        batch_size=24,
        classifier_kwargs={"max_length": 512},
    ),
    pip_dependency_groups=["fasttext", "datasets", "filelock"],
    override_output_path="attributes/quality_filtering/llama-500m-perplexity/dclm-global-shard-01-of-10/local-shard_0_of_10",
)

# consolidate_step = ExecutorStep(
#     name="documents/quality_filtering/dclm-global-shard-01-of-10-medu-economics-3plus",
#     fn=consolidate,
#     config=ConsolidateConfig(
#         input_path=input_data_path,
#         output_path=this_output_path(),
#         filters=[
#             FilterConfig(
#                 type="classify",
#                 attribute_path=output_path_of(inference_step),
#                 name="economic-bert",
#                 label="int_score",
#                 threshold=3,
#             ),
#         ],
#         ray_memory_limit_gb=12,
#         filetype="jsonl.zst",
#     ),
#     pip_dependency_groups=["ddsketch"],
# )

if __name__ == "__main__":
    executor_main([inference_step])
    # executor_main([llama_500m_inference_step])
