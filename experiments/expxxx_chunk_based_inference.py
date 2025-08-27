import copy

from experiments.datashop.default_configs import default_engine_kwargs, default_generation_kwargs
from experiments.datashop.defaults import default_synthetic_data_generation
from experiments.defaults import default_tokenize
from experiments.evals.resource_configs import TPU_V6E_8_STRICT_PACK
from experiments.exp1361_datashop_medical import datashop_runner
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.chunk_utils import ChunkStrategy, chunk_with_config, ChunkingConfig
from marin.processing.classification.dedupe import DedupeConfig, NGramConfig, dedupe

qa_rephrase_prompt = """
A chat between a curious patient and an educated doctor.
The doctor gives helpful, detailed, and polite answers to the questions.
Convert the following paragraph into a conversational format with
multiple tags of "Question:" followed by "Answer:".

{example}
"""

engine_kwargs = copy.deepcopy(default_engine_kwargs)
engine_kwargs["tensor_parallel_size"] = 8
generation_kwargs = copy.deepcopy(default_generation_kwargs)
generation_kwargs["max_tokens"] = 1024

# datashop_medical_qa_sampled = ExecutorStep(
#     name="documents/datashop-datasets/datashop-medical-qa-sampled",
#     fn=create_dataset,
#     config=CreateDatasetConfig(
#         input_doc_path=datashop_runner.filtered_documents,
#         output_dataset_path=this_output_path(),
#         max_sample_size=100_000,
#         filetype="jsonl.zst",
#         merge_dataset_shards=False,
#         columns_to_keep=["text", "metadata"],
#     ),
# )

medical_data_chunked = ExecutorStep(
    name="documents/datashop-medical-qa-whole-chunk-1024",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=datashop_runner.filtered_documents,
        output_path=this_output_path(),
        filetype="jsonl.zst",
        chunk_strategy=ChunkStrategy.CHAR,
        chunk_size=1024,
    ),
)

synthetic_medical_data_chunked = default_synthetic_data_generation(
    # datashop_runner.filtered_documents,
    # output_path_of(datashop_runner.filtered_documents),
    medical_data_chunked,
    # datashop_medical_qa_sampled,
    "documents/datashop-medical-qa-whole-chunk-1024",
    # "meta-llama/Llama-3.2-8B-Instruct",
    "/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
    qa_rephrase_prompt,
    "jsonl.zst",
    "text",
    checkpoint_id_column={"metadata": "WARC-Record-ID"},
    engine_kwargs=engine_kwargs,
    generation_kwargs=generation_kwargs,
    resource_config=TPU_V6E_8_STRICT_PACK,
    chunk_strategy=ChunkStrategy.CHAR,
    chunk_size=1024,
)

synthetic_medical_data_dedupe_attributes = ExecutorStep(
    name="attributes/datashop-medical-qa-whole-dedupe",
    fn=dedupe,
    config=DedupeConfig(
        input_path=synthetic_medical_data_chunked,
        output_path=this_output_path(),
        attribute_name="duplicate_text",
        min_length=0,
        min_words=0,
        estimated_doc_count=2_000_000_000,
        false_positive_rate=0.001,
        ngram=NGramConfig(
            ngram_length=8,
            stride=0,
            overlap_threshold=0.7,
        ),
    ),
)

synthetic_medical_data_tokenized = default_tokenize(
    "datashop-medical-qa-whole",
    synthetic_medical_data_chunked,
    llama3_tokenizer,
)

# total tokens was around 3B, let's epoch around 10B tokens
# so 10B * 1/ 0.3 = ~33B tokens
# anneal_model = default_anneal(
#     "datshop-medical-qa",
#     AnnealConfig(
#         dataset_config=lm_mixture_data_config(
#             {
#                 "synthetic_medical_data": synthetic_medical_data_tokenized,
#                 "dclm": dclm_components_llama3["dclm_baseline"],
#             },
#             {
#                 "synthetic_medical_data": 0.3,
#                 "dclm": 0.7,
#             },
#         ),
#         num_anneal_training_tokens=33_000_000_000,
#         resources=TpuPodConfig(tpu_type="v6e-128", slice_count=2),
#     ),
# )

# datashop_medical_synthetic_exported = upload_dir_to_hf(
#     "gs://marin-us-east1/documents/datashop-medical-qa-whole-fe2400",
#     "marin-community/datashop-medical-qa",
#     "dataset",
# )

if __name__ == "__main__":
    executor_main([synthetic_medical_data_dedupe_attributes])
