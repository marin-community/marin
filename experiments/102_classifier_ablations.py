import logging
from dataclasses import dataclass

import draccus

from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import InferenceConfig, run_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USER = "chris"


@dataclass
class ExperimentConfig:
    experiment_name: str = "eli5-100k-oh-100k-rw-200k"
    input_data_path: str = "gs://marin-data/processed/fineweb/fw-v1.0/text_fw/CC-MAIN-2020-10/"
    inference_threshold: float = 0.8
    quality_classifier_model_path: str = (
        "gs://marin-us-central2/classifiers/dclm_eli5_100k_oh_100k_rw_200k-4a11ec/model.bin"
    )


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    inference_step = ExecutorStep(
        name=f"attributes/{config.experiment_name}-{USER}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=config.input_data_path,
            output_path=this_output_path(),
            model_name=config.quality_classifier_model_path,
            model_type="fasttext",
            attribute_name=f"{config.experiment_name}-quality",
        ),
    )

    consolidate_step = ExecutorStep(
        name=f"documents/{config.experiment_name}-{USER}",
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=config.input_data_path,
            output_path=this_output_path(),
            filters=[
                FilterConfig(
                    type=versioned("classify"),
                    attribute_path=output_path_of(inference_step),
                    name=versioned(f"{config.experiment_name}-quality"),
                    label="__label__hq",
                    threshold=versioned(config.inference_threshold),
                ),
            ],
        ),
    )

    return [inference_step, consolidate_step]


@draccus.wrap()
def main(config: ExperimentConfig):
    steps = create_steps(config)
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
