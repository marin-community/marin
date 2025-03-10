from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.exp579_ar5iv_markdownify import ar5iv_no_problem_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

markdownified_arxiv_tokenized = default_tokenize(
    "ar5iv-no-problem-markdownified",
    ar5iv_no_problem_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

cooldown_config = QualityAblationConfig(
    tpu_type="v4-128",
)

ar5iv_cooldown_ablation = default_quality_ablation(
    markdownified_arxiv_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            markdownified_arxiv_tokenized,
            ar5iv_cooldown_ablation,
        ]
    )

