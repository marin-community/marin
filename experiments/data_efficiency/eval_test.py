from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step, data_efficiency_eval_model, data_efficiency_eval_ensemble
from marin.execution.executor import executor_main
from experiments.data_efficiency.ensemble_members import ensemble_members_train_steps_dict

eval_steps = []

# for base_train_step in unique_base_train_steps:
#     for seed_count in range(1, 11):
#         ensemble_member_train_steps = [
#             ensemble_members_train_steps_dict[(base_train_step, seed)]
#             for seed in list(range(seed_count))
#         ]
#         eval_steps.append(data_efficiency_eval_ensemble(ensemble_member_train_steps))

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
        description="Data scaling baseline",
    )