from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from operations.transform.format.parquet2jsonl import JsonlConversionConfig, convert_dataset_parquet2jsonl

lavita_pubmed_to_jsonl = ExecutorStep(
    name="documents/lavita-pubmed-jsonl",
    fn=convert_dataset_parquet2jsonl,
    config=JsonlConversionConfig(
        input_path="gs://marin-us-east1/documents/lavita_pubmed-d01d16",
        output_path=this_output_path(),
    ),
)

lavita_all_processed_to_jsonl = ExecutorStep(
    name="documents/lavita-allprocessed-jsonl",
    fn=convert_dataset_parquet2jsonl,
    config=JsonlConversionConfig(
        input_path="gs://marin-us-east1/documents/lavita_allprocessed-12cc53",
        output_path=this_output_path(),
    ),
)

lavita_medmcqa_to_jsonl = ExecutorStep(
    name="documents/lavita-medmcqa-jsonl",
    fn=convert_dataset_parquet2jsonl,
    config=JsonlConversionConfig(
        input_path="gs://marin-us-east1/documents/lavita_medmcqa-72b3d9",
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    # Run only the StarCoder Parquet→JSONL conversion
    executor_main(
        steps=[lavita_pubmed_to_jsonl, lavita_all_processed_to_jsonl, lavita_medmcqa_to_jsonl],
        description="Convert StarCoderData Parquet shards to JSONL.zst",
    )
