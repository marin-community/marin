from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, old_tokenize

tokenize_neox_step = ExecutorStep(
    name="scratch/dlwh/tokenized/neox/dclm_base",
    fn=old_tokenize,
    config=TokenizeConfig(
        input_path=versioned("gs://marin-us-central2/raw/dclm/v2024-07-09-baseline-dedup/"),
        cache_path=this_output_path(),
        dataset_name=versioned("dclm/v2024-07-09-baseline-dedup"),
        tokenizer=versioned("EleutherAI/gpt-neox-20b"),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[tokenize_neox_step])
