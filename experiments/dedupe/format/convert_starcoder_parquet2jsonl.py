from experiments.pretraining_datasets import starcoderdata
from operations.transform.format.parquet2jsonl import JsonlConversionConfig
from operations.transform.format.parquet2jsonl import convert_dataset_parquet2jsonl
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path

# Wrap the generic converter around the starcoderdata step
starcoder_to_jsonl = ExecutorStep(
    name="documents/starcoderdata",
    fn=convert_dataset_parquet2jsonl,
    config=JsonlConversionConfig(
        input_path=output_path_of(starcoderdata),
        output_path=this_output_path(),
        rename_key_from="content",
        rename_key_to="text",
    ),
    override_output_path="documents/starcoderdata-720c8c",
)

if __name__ == "__main__":
    # Run only the StarCoder Parquet→JSONL conversion
    executor_main(
        steps=[starcoder_to_jsonl],
        description="Convert StarCoderData Parquet shards to JSONL.zst",
    ) 