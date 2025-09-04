from experiments.datashop.datashop_datasets import dclm_baseline_global_shard_1_local_shard_1
from marin.download.filesystem.transfer import TransferConfig, transfer_files
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.chunk_utils import ChunkingConfig, chunk_with_config, ChunkStrategy
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig
from marin.processing.classification.inference import run_inference

# Total files in this shard is 279. The total dataset is roughly 40B tokens. We want to sample
# around 20 files first which equates to roughly 7B tokens.
synthetic_dclm_annotation_subset = ExecutorStep(
    name="documents/datashop-datasets/synthetic-dclm-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline_global_shard_1_local_shard_1,
        output_path=this_output_path(),
        num_random_files=20,
    ),
)
dclm_data_chunked = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-1024",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=synthetic_dclm_annotation_subset,
        output_path=this_output_path(),
        filetype="jsonl.zst",
        chunk_strategy=ChunkStrategy.CHAR,
        chunk_size=1024,
    ),
)

generation_kwargs = {
    "max_tokens": 1024,
    "temperature": 0.7,
}
engine_kwargs = {
    "tensor_parallel_size": 1,
    "enforce_eager": False,
    "max_model_len": 8192,
}

WRAP_QA_REPHRASE_PROMPT = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.
USER: Convert the following paragraph into a conversational format with
multiple tags of "Question:" followed by "Answer:":

<example>
{example}
</example>

Just return the converted paragraph. Do not include any other text.
"""

# synthetic_dclm_data_chunked = default_synthetic_data_generation(
#     # datashop_runner.filtered_documents,
#     # output_path_of(datashop_runner.filtered_documents),
#     dclm_data_chunked,
#     "documents/synthetic-dclm-subset-chunk-qa-1024",
#     "/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
#     WRAP_QA_REPHRASE_PROMPT,
#     "jsonl.zst",
#     "text",
#     checkpoint_id_column={"metadata": "WARC-Record-ID"},
#     engine_kwargs=engine_kwargs,
#     generation_kwargs=generation_kwargs,
#     resource_config=SINGLE_TPU_V5p_8,
#     chunk_strategy=ChunkStrategy.CHAR,
#     chunk_size=1024,
# )


synthetic_dclm_wrapqa = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-qa-1024-wrapqa",
    fn=run_inference,
    config=InferenceConfig(
        input_path=dclm_data_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU-v6e-8-head": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

WRAP_MED_PROMPT = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.
USER: For the following paragraph give me a diverse paraphrase of the same
in high quality English language as in sentences on Wikipedia:

<example>
{example}
</example>

Just return the paraphrased paragraph. Do not include any other text.
"""

synthetic_dclm_wrapmed = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-qa-1024-wrapmed",
    fn=run_inference,
    config=InferenceConfig(
        input_path=dclm_data_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="wrap_med_paraphrase",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_MED_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)


if __name__ == "__main__":
    executor_main([synthetic_dclm_wrapmed])
