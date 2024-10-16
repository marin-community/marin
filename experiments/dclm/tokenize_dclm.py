from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize


def tokenize_dclm(tokenizer="meta-llama/Meta-Llama-3.1-8B") -> ExecutorStep:
    return ExecutorStep(
        name="tokenized/dclm-baseline-dedup",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=versioned("gs://marin-us-central2/raw/dclm/v2024-07-09-baseline-dedup/"),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )


tokenize_dclm_llama3_step = tokenize_dclm()

if __name__ == "__main__":
    executor_main(steps=[tokenize_dclm_llama3_step])
