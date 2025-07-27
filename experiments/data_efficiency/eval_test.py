from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step, data_efficiency_eval_model, data_efficiency_eval_ensemble
from marin.execution.executor import executor_main
from experiments.data_efficiency.ensemble_members import ensemble_members_train_steps_dict

eval_steps = []

max_runs = 5

for key in ensemble_members_train_steps_dict:
    if key[-1] == 0:
        ensemble_members = []
        for seed in range(max_runs):
            key_copy = key[:-1] + (seed,)
            ensemble_members.append(ensemble_members_train_steps_dict[key_copy])
        
        for seed_count in range(1, max_runs + 1):
            eval_steps.append(data_efficiency_eval_ensemble(ensemble_members[:seed_count], key="varying-hparams-experiment"))

if __name__ == "__main__":
    executor_main(
        steps=eval_steps,
        description="Data scaling baseline",
    )