from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.exp575_wikipedia_markdownify import wikipedia_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

markdownified_wiki_tokenized = default_tokenize(
    "wikipedia-markdownified",
    wikipedia_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

cooldown_config = QualityAblationConfig(
    tpu_type="v4-128",
)

wikipedia_cooldown_ablation = default_quality_ablation(
    markdownified_wiki_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            markdownified_wiki_tokenized,
            wikipedia_cooldown_ablation,
        ]
    )

