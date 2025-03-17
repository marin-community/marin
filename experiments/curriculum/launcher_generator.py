file_to_generate = """
from marin.execution.executor import executor_main
from itertools import chain
from experiments.curriculum.exp702_targeted_curriculum import full_training_varying_mixture

if __name__ == "__main__":
    stage_pairs = [
        full_training_varying_mixture(
            data1_name="flan",
            data2_name="c4",
            total_data1_portion=0.005,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=1.0,
            schedule_type="linear",
            cooldown_frac=0.05,
            model_size=model_size,
            num_train_steps=3000,
            learning_rate={learning_rate},
            additional_tags=["flan-c4-eu-model-scaling"],
            version_tag="{tag}"
        )
        for model_size in ["{model_size}"] # "150m", "300m", "600m", "1_9b"
        for duration_frac_stage2 in [{duration_frac_stage2}] # 0.1, 0.05, 0.02, 0.01
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description=f"Test training with varying mixtures",
    )
"""

learning_rate_dict = {
    "150m": 3e-3,
    "300m": 3e-3,
    "600m": 1e-4,
    "600m_0.003": 3e-3,
    "1_9b": 3e-4,
    "8b_1024_0.001": 1e-3,
    "8b_1024": 3e-4,
    "8b_1024_0.0001": 1e-4,
}

# clear launcher_scripts
# import os
# os.system("rm -rf experiments/curriculum/launcher_scripts/*")

command = ""

for model_size_for_dict in ["600m", "600m_0.003"]:
    if model_size_for_dict == "600m_0.003":
        model_size = "600m"
    elif model_size_for_dict == "8b_1024_0.0001" or model_size_for_dict == "8b_1024_0.001":
        model_size = "8b_1024"
    else:
        model_size = model_size_for_dict
    for duration_frac_stage2 in [0.4]:
        lr = learning_rate_dict[model_size_for_dict]
        file_name = f"exp702_launcher_{model_size}_{duration_frac_stage2}_{lr}.py"
        with open(f"experiments/curriculum/launcher_scripts/{file_name}", "w") as f:
            print(f"Writing to file: experiments/curriculum/launcher_scripts/{file_name}")
            f.write(file_to_generate.format(learning_rate=lr, model_size=model_size, duration_frac_stage2=duration_frac_stage2, tag=f"-lr{lr}" if lr != 3e-3 else ""))

            command += f"python marin/run/ray_run.py -- python experiments/curriculum/launcher_scripts/{file_name} --force_run_failed True ; "
print("command:")
print(command)