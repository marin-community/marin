import ast
import getpass
import os
from functools import partial
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

CURRENT_DIR = Path(__file__).parent.absolute()
BASE_DIR = Path(__file__).parents[2].absolute()

### API specific ###
API_MAX_CONCURRENCY = int(os.environ.get("API_MAX_CONCURRENCY", 5))

OPENAI_MAX_CONCURRENCY = int(os.environ.get("OPENAI_MAX_CONCURRENCY", 5))
OPENAI_CLIENT_CONFIG_PATH = os.environ.get("OPENAI_CLIENT_CONFIG_PATH", BASE_DIR / "client_configs/openai_configs.yaml")
# the following is for backward compatibility, the recommended way is to use OPENAI_CLIENT_CONFIG_PATH
OPENAI_API_KEYS = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", None))
if isinstance(OPENAI_API_KEYS, str):
    OPENAI_API_KEYS = OPENAI_API_KEYS.split(",")
OPENAI_ORGANIZATION_IDS = os.environ.get("OPENAI_ORGANIZATION_IDS", None)
if isinstance(OPENAI_ORGANIZATION_IDS, str):
    OPENAI_ORGANIZATION_IDS = OPENAI_ORGANIZATION_IDS.split(",")
#

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
ANTHROPIC_MAX_CONCURRENCY = int(os.environ.get("ANTHROPIC_MAX_CONCURRENCY", 4))

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", None)

DATASETS_TOKEN = os.environ.get("DATASETS_TOKEN", None)
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
DATASETS_FORCE_DOWNLOAD = os.environ.get("DATASETS_FORCE_DOWNLOAD", False)
########################

IS_ALPACA_EVAL_2 = ast.literal_eval(os.environ.get("IS_ALPACA_EVAL_2", "True"))
ANNOTATOR_CONFIG_AE1 = "alpaca_eval_gpt4"
ANNOTATOR_CONFIG_AE2 = "weighted_alpaca_eval_gpt4_turbo"
DEFAULT_ANNOTATOR_CONFIG = ANNOTATOR_CONFIG_AE2 if IS_ALPACA_EVAL_2 else ANNOTATOR_CONFIG_AE1
DEFAULT_CACHE_DIR = None
EVALUATORS_CONFIG_DIR = CURRENT_DIR / "evaluators_configs"
MODELS_CONFIG_DIR = CURRENT_DIR / "models_configs"


MINIMAL_EVALUATORS = (
    ANNOTATOR_CONFIG_AE2,
    ANNOTATOR_CONFIG_AE1,
    "aviary_gpt4",
    "gpt4",
    "claude",
    "text_davinci_003",
    "chatgpt",
    "lmsys_gpt4",
    "humans",
    "alpaca_farm_greedy_gpt4",
)

VERIFIED_EVALUATORS = tuple(
    list(MINIMAL_EVALUATORS)
    + [
        "claude_ranking",
        "improved_aviary_gpt4",
        "improved_lmsys_gpt4",
        "lmsys_gpt4",
        "cohere",
        "alpaca_farm",
        "alpaca_farm_greedy_gpt4",
        "guanaco_33b",
        "longest",
    ]
)

# order matters i => i+1 when filtering
ORDERED_LEADERBOARD_MODES = ["minimal", "verified", "community", "dev"]


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


def ALPACAEVAL_INSTRUCTION_PARAMETERS():
    out = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="instruction_difficulty.csv",
        repo_type="dataset",
        force_download=DATASETS_FORCE_DOWNLOAD,
        cache_dir=DEFAULT_CACHE_DIR,
        token=DATASETS_TOKEN,
    )
    pd.read_csv(out, index_col=0).squeeze()
    return df


def ALPACAFARM_GOLD_CROSSANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_crossannotations",
        cache_dir=DEFAULT_CACHE_DIR,
        token=DATASETS_TOKEN,
        download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
        trust_remote_code=True,
    )["validation"].to_pandas()

    # turkers took around 9 min for 15 examples in AlpacaFarm
    df["time_per_example"] = 9.2 * 60 / 15
    df["price_per_example"] = 0.3  # price we paid for each example
    return df


def ALPACAFARM_GOLD_ANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_annotations",
        cache_dir=DEFAULT_CACHE_DIR,
        token=DATASETS_TOKEN,
        download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
        trust_remote_code=True,
    )["validation"].to_pandas()

    # turkers took around 9 min for 15 examples in AlpacaFarm
    df["time_per_example"] = 9.2 * 60 / 15
    df["price_per_example"] = 0.3  # price we paid for each example
    return df


ALPACAEVAL_2_LEADERBOARD_PATHS = CURRENT_DIR / f"leaderboards/data_AlpacaEval_2"
ALPACAEVAL_1_LEADERBOARD_PATHS = CURRENT_DIR / f"leaderboards/data_AlpacaEval"
ALPACAEVAL_LEADERBOARD_PATHS = ALPACAEVAL_2_LEADERBOARD_PATHS if IS_ALPACA_EVAL_2 else ALPACAEVAL_1_LEADERBOARD_PATHS


PRECOMPUTED_LEADERBOARDS = {
    (str(ALPACAEVAL_REFERENCE_OUTPUTS_1), "claude"): ALPACAEVAL_1_LEADERBOARD_PATHS / "claude_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS_1), ANNOTATOR_CONFIG_AE1): ALPACAEVAL_1_LEADERBOARD_PATHS
    / f"{ANNOTATOR_CONFIG_AE1}_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS_1), "chatgpt_fn"): ALPACAEVAL_1_LEADERBOARD_PATHS / "chatgpt_fn_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS_2), ANNOTATOR_CONFIG_AE2): ALPACAEVAL_2_LEADERBOARD_PATHS
    / f"{ANNOTATOR_CONFIG_AE2}_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS_2), "weighted_alpaca_eval_gpt4_turbo"): ALPACAEVAL_2_LEADERBOARD_PATHS
    / f"weighted_alpaca_eval_gpt4_turbo_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS_2), "mistral-large-2402_ranking"): ALPACAEVAL_2_LEADERBOARD_PATHS
    / f"mistral-large-2402_ranking_leaderboard.csv",
    (str(ALPACAEVAL_REFERENCE_OUTPUTS_2), "claude_3_opus_ranking"): ALPACAEVAL_2_LEADERBOARD_PATHS
    / f"claude_3_opus_ranking_leaderboard.csv",
    # (str(ALPACAEVAL_REFERENCE_OUTPUTS_2), "gpt-3.5-turbo-1106_ranking"): ALPACAEVAL_2_LEADERBOARD_PATHS
    # / f"gpt-3.5-turbo-1106_ranking_leaderboard.csv",
    # (str(ALPACAEVAL_REFERENCE_OUTPUTS_2), "alpaca_eval_cot_gpt4_turbo_fn"): ALPACAEVAL_2_LEADERBOARD_PATHS
    # / f"alpaca_eval_cot_gpt4_turbo_fn_leaderboard.csv",
}

HUMAN_ANNOTATED_MODELS_TO_KEEP = (
    "GPT-4 300 characters",
    "GPT-4",
    "AlpacaFarm PPO sim (step 40)",
    "ChatGPT",
    "ChatGPT 300 characters",
    "AlpacaFarm best-of-16 human",
    "AlpacaFarm PPO sim (gpt4 greedy, step 30)",
    "Davinci003",
    "AlpacaFarm ExpIter human (n=128)",
    "AlpacaFarm SFT 10K",
    "AlpacaFarm PPO human (10k, step 40)",
    "Alpaca 7B",
    "AlpacaFarm FeedMe human",
    "Davinci001",
    "LLaMA 7B",
)

EVALUATORS_LEADERBOARD_COLS_TO_PRIORITIZE = [
    "Human agreement",
    "Price [$/1000 examples]",
    "Time [seconds/1000 examples]",
    "Spearman corr.",
    "Pearson corr.",
    "Bias",
    "Variance",
    "Proba. prefer longer",
    "Proba. prefer lists",
    "Proba. prefer 1",
]

MINIMAL_MODELS_FOR_NEW_LEADERBOARD = [
    "gpt4_turbo",
    "gpt4",
    "tulu-2-dpo-70b",
    "Yi-34B-Chat",
    "llama-2-70b-chat-hf",
    "claude-2.1",
    "cohere",
    "chatgpt",
    "gemini-pro",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    "vicuna-33b-v1.3",
    "alpaca-7b",
]


EVALUATORS_LEADERBOARD_COLS_TO_PRINT = EVALUATORS_LEADERBOARD_COLS_TO_PRIORITIZE[:8]

CURRENT_USER = getpass.getuser()
if CURRENT_USER in ["yanndubs"]:
    DEFAULT_CACHE_DIR = "/juice5/scr5/nlp/crfm/human-feedback/cache"


def ALPACAFARM_ALL_OUTPUTS():
    if IS_ALPACA_EVAL_2:
        return [f"results/{m}/model_outputs.json" for m in MINIMAL_MODELS_FOR_NEW_LEADERBOARD]
    else:
        return datasets.load_dataset(
            "tatsu-lab/alpaca_eval",
            "alpaca_eval_all_outputs",
            cache_dir=DEFAULT_CACHE_DIR,
            token=DATASETS_TOKEN,
            download_mode="force_redownload" if DATASETS_FORCE_DOWNLOAD else None,
            trust_remote_code=True,
        )["eval"]
