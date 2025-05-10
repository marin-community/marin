import logging

from experiments.exp164_quality_classifiers import (
    dclm_eli5_100k_oh_100k_rw_200k,
    dclm_eli5_100k_oh_100k_rw_200k_seed_1,
    dclm_eli5_100k_oh_100k_rw_200k_seed_2,
    dclm_eli5_200k_rw_200k,
    teknium_oh_200k_rw_200k,
)
from experiments.quality_classifier_experiment_utils import ExperimentConfig, create_steps
from marin.execution.executor import (
    executor_main,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_experiment_configs() -> list[ExperimentConfig]:
    marin_eli5_100k_oh_100k_rw_200k_config = ExperimentConfig(
        experiment_name="eli5-100k-oh-100k-rw-200k",
        quality_classifier_model_path=dclm_eli5_100k_oh_100k_rw_200k,
    )

    marin_eli5_100k_oh_100k_rw_200k_seed_1_config = ExperimentConfig(
        experiment_name="eli5-100k-oh-100k-rw-200k-seed-1",
        quality_classifier_model_path=dclm_eli5_100k_oh_100k_rw_200k_seed_1,
    )

    marin_eli5_100k_oh_100k_rw_200k_seed_2_config = ExperimentConfig(
        experiment_name="eli5-100k-oh-100k-rw-200k-seed-2",
        quality_classifier_model_path=dclm_eli5_100k_oh_100k_rw_200k_seed_2,
    )

    marin_eli5_200k_rw_200k_config = ExperimentConfig(
        experiment_name="eli5-200k-rw-200k",
        quality_classifier_model_path=dclm_eli5_200k_rw_200k,
    )

    marin_oh_200k_rw_200k_config = ExperimentConfig(
        experiment_name="oh-200k-rw-200k",
        quality_classifier_model_path=teknium_oh_200k_rw_200k,
    )

    original_dclm_quality_classifier_config = ExperimentConfig(
        experiment_name="original-dclm-quality-classifier",
        quality_classifier_model_path="mlfoundations/fasttext-oh-eli5",
    )

    return [
        marin_eli5_100k_oh_100k_rw_200k_config,
        marin_eli5_100k_oh_100k_rw_200k_seed_1_config,
        marin_eli5_100k_oh_100k_rw_200k_seed_2_config,
        marin_eli5_200k_rw_200k_config,
        marin_oh_200k_rw_200k_config,
        original_dclm_quality_classifier_config,
    ]


def main():
    steps = []
    for experiment_config in create_experiment_configs():
        steps.extend(create_steps(experiment_config))
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
