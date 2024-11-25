import experiments.pretraining_datasets
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize


def tokenize_dclm(tokenizer=llama3_tokenizer) -> ExecutorStep:
    return ExecutorStep(
        name="tokenized/dclm-baseline-dedup",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=output_path_of(experiments.pretraining_datasets.dclm_baseline),
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )


dclm_tokenized_llama3 = tokenize_dclm(llama3_tokenizer)

if __name__ == "__main__":
    executor_main(steps=[dclm_tokenized_llama3])
