from experiments.llama import llama3_tokenizer
from marin.execution.executor import versioned
from marin.processing.tokenize.tokenize import TokenizeConfig

TokenizeConfig(
    train_paths=versioned(
        [
            "gs://marin-us-east5/documents/quality_filtering/dclm-global-shard-01-of-10-medu-economics-3plus-bbb96b/global-shard_01_of_10/**/*.jsonl.zst"
        ]
    ),
    validation_paths=[],
    cache_path="gs://asdasd",
    tokenizer=llama3_tokenizer,
)
