from experiments.cooldown_quality import default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.exp822_stackexchange_markdownify import stackexchange_text_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

stackexchange_tokenized = default_tokenize(
    "stackexchange-markdownified",
    stackexchange_text_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

stackexchange_cooldown_ablation = default_quality_ablation(stackexchange_tokenized)

if __name__ == "__main__":
    executor_main(
        steps=[
            stackexchange_tokenized,
            stackexchange_cooldown_ablation,
        ]
    )

