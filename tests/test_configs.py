import os

from test_utils import check_load_config, parameterize_with_configs

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@parameterize_with_configs("training/*.yaml")
def test_training_configs(config_file):
    """
    Validate all the training configs (config/training/*.yaml).
    """
    from marin.training.training import TrainLmOnPodConfig

    config_class = TrainLmOnPodConfig
    check_load_config(config_class, config_file)


@parameterize_with_configs(pattern="train_bert.yaml", config_path=PROJECT_DIR)
def test_bert_configs(config_file):
    """
    Validate all the bert configs (config/bert/*.yaml).
    """
    from scripts.bert.train_bert import TrainBertClassifierConfig

    config_class = TrainBertClassifierConfig
    check_load_config(config_class, config_file)


@parameterize_with_configs(pattern="train_fasttext.yaml", config_path=PROJECT_DIR)
def test_fasttext_configs(config_file):
    """
    Validate all the fasttext configs (config/fasttext/*.yaml).
    """
    from scripts.fasttext.train_fasttext import TrainFasttextClassifierConfig

    config_class = TrainFasttextClassifierConfig
    check_load_config(config_class, config_file)


@parameterize_with_configs(pattern="*dedupe.yaml", config_path=PROJECT_DIR)
def test_dedupe_configs(config_file):
    """
    Validate all the dedupe configs (config/dedupe/*.yaml).
    """
    from marin.processing.classification.dedupe import DedupeConfig

    config_class = DedupeConfig
    check_load_config(config_class, config_file)
