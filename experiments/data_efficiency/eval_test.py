from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step, data_efficiency_eval_model, data_efficiency_eval_ensemble
from marin.execution.executor import executor_main
from experiments.data_efficiency.ensemble_members import ensemble_members_train_steps

eval_steps = [
    # data_efficiency_eval_model(ensemble_members_train_steps[0]),
    # data_efficiency_eval_ensemble(ensemble_members_train_steps[:1]),
    data_efficiency_eval_ensemble(ensemble_members_train_steps[:4]),
    # data_efficiency_eval_ensemble(ensemble_members_train_steps[:5]),
    # data_efficiency_eval_ensemble(ensemble_members_train_steps[:10])
    # data_efficiency_eval_ensemble([ensemble_members_train_steps[0], ensemble_members_train_steps[0], ensemble_members_train_steps[0]])
]

if __name__ == "__main__":
    executor_main(
        steps=eval_steps,
        description="Data scaling baseline",
    )