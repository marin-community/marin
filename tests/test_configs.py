from test_utils import check_load_config, parameterize_with_configs


@parameterize_with_configs("training/*.yaml")
def test_training_configs(config_file):
    """
    Validate all the training configs (config/training/*.yaml).
    """
    from marin.training.training import TrainLmOnPodConfig

    config_class = TrainLmOnPodConfig
    check_load_config(config_class, config_file)
