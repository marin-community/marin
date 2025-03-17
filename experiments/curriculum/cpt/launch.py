from itertools import chain

from marin.execution.executor import executor_main

from experiments.curriculum.cpt.cpt_launch import full_cpt_varying_mixture

base_steps = {
    "finemath": 200,
    "open-web-math": 2000,
    "pubmed": 100,
}

if __name__ == "__main__":
    stage_pairs = [
        full_cpt_varying_mixture(
            data1_name=data1_name,
            data2_name="dclm",
            total_data1_portion=total_data1_portion,
            duration_frac_stage2=1.0,  # Single stage training
            data1_frac_alloc_stage2=1.0,
            schedule_type="cosine",
            model_name="meta-llama/Meta-Llama-3.1-8B",
            num_train_steps=int(base_steps[data1_name] * num_data1_repetitions / total_data1_portion),
            learning_rate=5e-6,
            num_eval=20,
            num_lm_eval_harness=4,
            num_data1_repetitions=num_data1_repetitions,
            batch_size=128,
            additional_tags=[f"{data1_name}-rarity-sweep-epochs-owm"],
            min_lr_ratio=0.0,
            version_tag=f"",
            warmup_steps=0.05,
        )
        for total_data1_portion in [1.0, 0.95, 0.75]
        for data1_name in ["open-web-math"]
        for num_data1_repetitions in [5]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Train on finemath with varying rarity",
    )