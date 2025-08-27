from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main
from experiments.data_efficiency.convex_certificate_scaling import valid_epochs, valid_weight_decays, extract_neighbors


# for now, do not vary LR
def get_ensemble_bounding_box(base_train_steps, epochs, lr, weight_decay, model_name):
    lower_epochs, upper_epochs = extract_neighbors(epochs, valid_epochs)
    lower_weight_decay, upper_weight_decay = extract_neighbors(weight_decay, valid_weight_decays)

    return [
        (base_train_steps, epochs, lr, weight_decay, model_name),
        (base_train_steps, lower_epochs, lr, weight_decay, model_name),
        (base_train_steps, upper_epochs, lr, weight_decay, model_name),
        (base_train_steps, epochs, lr, lower_weight_decay, model_name),
        (base_train_steps, epochs, lr, upper_weight_decay, model_name),
    ]


ensemble_members_train_steps_dict_list = [
    {
        (base_train_steps, epochs, lr, weight_decay, model_name, seed): data_efficiency_train_step(
            DataEfficiencyConfig(
                train_seed=seed,
                data_seed=seed,
                data_name="dclm",
                epochs=epochs,
                base_train_steps=base_train_steps,
                train_batch_size=64,
                lr_schedule="cosine",
                lr=lr,
                weight_decay=weight_decay,
                wandb_project_name="suhas-data-efficiency",
                model_name=model_name,
                nametag=f"-seed{seed}",
                tpu_type="v4-128",
            )
        )
        for base_train_steps, epochs, lr, weight_decay, model_name in (
            get_ensemble_bounding_box(*candidate_hparams) if use_bounding_box else [candidate_hparams]
        )
    }
    for candidate_hparams, use_bounding_box in [
        # # Vanilla
        ((800, 16, 3e-3, 0.8, "150m4k"), False),
        ((800, 16, 3e-3, 1.6, "300m4k"), False),
        ((800, 8, 1e-3, 3.2, "600m4k"), False),
        ((800, 8, 1e-3, 3.2, "1_4b4k"), False),
        # # Ensembling guess
        # ((800, 32, 3e-3, 0.4, "150m4k"), True),
        # ((800, 32, 3e-3, 0.8, "300m4k"), True),
        # ((800, 16, 1e-3, 1.6, "600m4k"), True),
        # ((800, 16, 1e-3, 1.6, "1_4b4k"), True),
        # # Vanilla
        # ((1600, 32, 3e-3, 0.8, "150m4k"), False),
        # ((1600, 16, 3e-3, 0.8, "300m4k"), False),
        # ((1600, 8, 1e-3, 1.6, "600m4k"), False),
        # ((1600, 8, 1e-3, 3.2, "1_4b4k"), False),
        # # Ensembling guess
        # ((1600, 64, 3e-3, 0.4, "150m4k"), False),
        # ((1600, 32, 3e-3, 0.4, "300m4k"), False),
        # ((1600, 16, 1e-3, 0.8, "600m4k"), False),
        # ((1600, 16, 1e-3, 1.6, "1_4b4k"), False),
        # Vanilla
        # ((3200, 64, 3e-3, 0.4, "150m4k"), False),
        # ((3200, 16, 3e-3, 0.4, "300m4k"), False),
        # ((3200, 16, 3e-3, 0.4, "600m4k"), False),
        # ((3200, 8, 1e-3, 1.6, "1_4b4k"), False),
        # Ensembling guess
        # ((3200, 128, 3e-3, 0.2, "150m4k"), False),
        # ((3200, 32, 3e-3, 0.2, "300m4k"), False),
        # ((3200, 32, 3e-3, 0.2, "600m4k"), False),
        # ((3200, 16, 1e-3, 0.8, "1_4b4k"), False),
        # Vanilla
        # ((6400, 64, 3e-3, 0.1, "150m4k"), False),
        # ((6400, 32, 1e-3, 0.4, "300m4k"), False),
        # ((6400, 16, 1e-3, 0.8, "600m4k"), False),
        # ((6400, 8, 1e-3, 0.8, "1_4b4k"), False),
        # Ensembling guess
        # ((6400, 128, 3e-3, 0.1, "150m4k"), False),
        # ((6400, 64, 1e-3, 0.2, "300m4k"), False),
        # ((6400, 32, 1e-3, 0.4, "600m4k"), False),
        # ((6400, 16, 1e-3, 0.4, "1_4b4k"), False),
    ]
    for seed in list(range(5))
]

ensemble_members_train_steps_dict = {}
for train_steps_dict in ensemble_members_train_steps_dict_list:
    for key, value in train_steps_dict.items():
        assert key not in ensemble_members_train_steps_dict
        ensemble_members_train_steps_dict[key] = value

ensemble_members_train_steps = list(ensemble_members_train_steps_dict.values())

if __name__ == "__main__":
    executor_main(
        steps=ensemble_members_train_steps,
        description="Data scaling baseline",
    )
