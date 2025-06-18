import logging

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the dedupe step that consumes the converted MMLU JSONL
starcoder_dedupe = ExecutorStep(
    name="train_test_overlap/dolma/debug",
    fn=dedupe,
    config=DedupeConfig(
        input_path="gs://marin-us-central2/decontamination/",
        # input_path="gs://marin-us-central2/documents/starcoderdata-720c8c/9fc30b5/ada/",
        output_path=this_output_path(),
        attribute_name="mmlu_overlap",
        false_positive_rate=0.0001,
        ngram=NGramConfig(
            ngram_length=[10, 15],
            overlap_threshold=1e-6,
            stride=0,
        ),
        processes=16,
        mode=DedupMode.TRAIN_TEST_OVERLAP,
        decontaminate_source="gs://marin-us-central2/documents/finemath-3plus-8f233cf/train-00000-of-00128.jsonl.zst",
        # decontaminate_path="gs://marin-us-central2/dolma/mmlu_dev_set/",
    ),
)

if __name__ == "__main__":
    # First convert StarCoder Parquet shards to JSONL, then run dedupe
    executor_main(
        steps=[starcoder_dedupe],
        description="Detect n-gram dedupe overlap",
    )
