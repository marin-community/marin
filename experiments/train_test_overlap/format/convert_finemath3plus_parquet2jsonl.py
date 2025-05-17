import logging

from experiments.midtraining_datasets import finemath_3_plus
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from operations.transform.format.parquet2jsonl import JsonlConversionConfig, convert_dataset_parquet2jsonl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Wrap the generic converter around the Finemath-3plus Parquet shards
finemath3plus_to_jsonl = ExecutorStep(
    name="documents/finemath-3plus",
    fn=convert_dataset_parquet2jsonl,
    config=JsonlConversionConfig(
        input_path=finemath_3_plus,
        output_path=this_output_path(),
    ),
    override_output_path="documents/finemath-3plus-8f233cf",
)

if __name__ == "__main__":
    # Run only the Finemath-3plus Parquet→JSONL conversion
    executor_main(
        steps=[finemath3plus_to_jsonl],
        description="Convert Finemath-3plus Parquet shards to JSONL.zst",
    )
