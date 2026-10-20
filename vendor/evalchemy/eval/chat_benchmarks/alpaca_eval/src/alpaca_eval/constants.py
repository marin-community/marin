import ast
import os
from functools import partial
from pathlib import Path

import datasets

CURRENT_DIR = Path(__file__).parent.absolute()
DATASETS_TOKEN = os.environ.get("DATASETS_TOKEN")
DATASETS_FORCE_DOWNLOAD = os.environ.get("DATASETS_FORCE_DOWNLOAD", False)
DEFAULT_CACHE_DIR = None
IS_ALPACA_EVAL_2 = ast.literal_eval(os.environ.get("IS_ALPACA_EVAL_2", "True"))
ANNOTATOR_CONFIG_AE1 = "alpaca_eval_gpt4"
ANNOTATOR_CONFIG_AE2 = "weighted_alpaca_eval_gpt4_turbo"
DEFAULT_ANNOTATOR_CONFIG = ANNOTATOR_CONFIG_AE2 if IS_ALPACA_EVAL_2 else ANNOTATOR_CONFIG_AE1
EVALUATORS_CONFIG_DIR = CURRENT_DIR / "evaluators_configs"


def get_alpaca_eval_data(dataset="alpaca_eval_gpt4_baseline"):
    dataset = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        dataset,
        cache_dir=DEFAULT_CACHE_DIR,
        token=DATASETS_TOKEN,
        download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
        trust_remote_code=True,
    )["eval"]
    return dataset



ALPACAEVAL_REFERENCE_OUTPUTS_2 = get_alpaca_eval_data
ALPACAEVAL_REFERENCE_OUTPUTS_1 = partial(get_alpaca_eval_data, dataset="alpaca_eval")
ALPACAEVAL_REFERENCE_OUTPUTS = ALPACAEVAL_REFERENCE_OUTPUTS_2 if IS_ALPACA_EVAL_2 else ALPACAEVAL_REFERENCE_OUTPUTS_1
