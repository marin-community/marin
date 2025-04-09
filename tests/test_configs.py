import os

from .test_utils import check_load_config, parameterize_with_configs

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@parameterize_with_configs(pattern="train_fasttext.yaml", config_path=PROJECT_DIR)
def test_fasttext_configs(config_file):
    """
    Validate all the fasttext configs (config/fasttext/*.yaml).
    """
    from marin.processing.classification.fasttext.train_fasttext import TrainFasttextClassifierConfig

    config_class = TrainFasttextClassifierConfig
    check_load_config(config_class, config_file)


@parameterize_with_configs(pattern="*quickstart_dedupe.yaml", config_path=PROJECT_DIR)
def test_dedupe_configs(config_file):
    """
    Validate all the dedupe configs.
    """
    from marin.processing.classification.dedupe import DedupeConfig

    config_class = DedupeConfig
    check_load_config(config_class, config_file)
