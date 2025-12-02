from experiments.data_efficiency.train import data_efficiency_eval_ensemble
from marin.execution.executor import executor_main
# from experiments.data_efficiency.ensemble_dclm_200m import ensemble_members_train_steps_dict
from experiments.data_efficiency.wrap_ensemble_dclm_200m import ensemble_members_train_steps_dict

eval_steps = []

for key in ensemble_members_train_steps_dict:
    if key[-1] == 0:
        ensemble_members = []
        for seed in range(5):
            key_copy = key[:-1] + (seed,)
            ensemble_members.append(ensemble_members_train_steps_dict[key_copy])
        
        for seed_count in range(1, 6):
            eval_steps.append(data_efficiency_eval_ensemble(ensemble_members[:seed_count]))

if __name__ == "__main__":
    executor_main(
        steps=eval_steps,
        description="Ensemble with weight decay eval",
    )
