import llama
import pretraining_datasets

from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize


def tokenize_dclm(tokenizer=llama.llama3_tokenizer) -> ExecutorStep:
    return ExecutorStep(
        name="tokenized/dclm-baseline-dedup",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=output_path_of(pretraining_datasets.dclm_baseline),
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )


tokenize_dclm_llama3_step = tokenize_dclm()

if __name__ == "__main__":
    executor_main(steps=[tokenize_dclm_llama3_step])
