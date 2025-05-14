import logging

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.classification.dedupe import DedupeConfig, NGramConfig, dedupe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the dedupe step that consumes the converted StarCoder JSONL
starcoder_dedupe = ExecutorStep(
    name="train_test_overlap/decontaminate_starcoder_againstmmlu_ngram500",
    fn=dedupe,
    config=DedupeConfig(
        input_path="gs://marin-us-central2/dolma/mmlu_dev_set/",
        # input_path="gs://marin-us-central2/documents/starcoderdata-720c8c/9fc30b5/ada/",
        output_path=this_output_path(),
        attribute_name="starcoder_overlap",
        false_positive_rate=0.00001,
        ngram=NGramConfig(
            ngram_length=13,
            overlap_threshold=0.9,
            stride=0,
        ),
        processes=4,
        decontaminate=True,
        decontaminate_path="gs://marin-us-central2/documents/starcoderdata-720c8c/9fc30b5/ada/",
        # decontaminate_path="gs://marin-us-central2/dolma/mmlu_dev_set/",
    ),
)

if __name__ == "__main__":
    # First convert StarCoder Parquet shards to JSONL, then run dedupe
    executor_main(
        steps=[starcoder_dedupe],
        description="Detect n-gram dedupe overlap",
    )
