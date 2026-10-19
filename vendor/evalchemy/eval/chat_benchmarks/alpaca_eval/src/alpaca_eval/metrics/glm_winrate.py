import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import sklearn
from huggingface_hub import hf_hub_download
from patsy import build_design_matrices, dmatrix
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, StratifiedKFold

from alpaca_eval import constants, types, utils

from .winrate import get_winrate

__all__ = ["get_length_controlled_winrate"]

GLM_INFO = {
    "length_controlled_v1": {
        "formula": "np.tanh(std_delta_len) + instruction_difficulty + not_gamed_baseline.astype(float) - 1",
        "regularize_to_baseline_lambda": 0.2,
        "kwargs": {"n_splits": 5},
    },
    "length_controlled_noreg": {
        "formula": "np.tanh(std_delta_len) + instruction_difficulty + not_gamed_baseline.astype(float) - 1",
        "regularize_to_baseline_lambda": None,
        "kwargs": {"n_splits": 5},
    },
    "length_controlled_minimal": {
        "formula": "np.tanh(std_delta_len) + not_gamed_baseline.astype(float) - 1",
        "regularize_to_baseline_lambda": None,
        "kwargs": {"n_splits": 5},
    },
}
DFLT_WEIGHT_PATH = (
    Path(__file__).parent
    / "weights/weighted_alpaca_eval_gpt4_turbo/length_controlled_v1/baseline_gpt4_1106_preview.csv"
)


def get_length_controlled_winrate(
    annotations: Union[pd.DataFrame, Sequence[dict]],
    glm_name="length_controlled_v1",
    save_weights_dir: Optional[Union[str, Path]] = "auto",
    baseline: Optional[str] = None,
    is_add_glm_preference_inplace: bool = True,
    is_warn_extreme_changes: bool = True,
    glm_info=None,
    df_gamed: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference, and also predict the length
    controlled winrate using a GLM.

    Parameters
    ----------
    annotations : pd.DataFrame or Sequence of dict
        The annotations to compute the winrate from.

    glm_name : str, optional
        The name of the GLM to use.

    save_weights_dir : Path, optional
        The directory to save the weights of the GLM. If None, the weights are not saved. If "auto", we save the weights
        weights / annotator / glm_name. Can only be "auto" if there's a unique annotator.

    baseline : str, optional
        The name of the baseline model to compare to. If None, we use the default for that annotation (i.e. output_2).

    is_add_glm_preference_inplace : bool, optional
        Whether to add the GLM preference to the annotations inplace. Only possible if annotations is a DataFrame.

    is_warn_extreme_changes : bool, optional
        Warn if the length controlled win rate is very different from the raw one.

    glm_info : dict, optional
        The information to use for the GLM. If None, we use the default for that glm_name.
    """
    glm_info = glm_info or GLM_INFO[glm_name]

    metrics = get_winrate(annotations)  # get the non-length controlled winrate
    df = utils.convert_to_dataframe(annotations)

    if save_weights_dir == "auto":
        assert len(df["annotator"].unique()) == 1
        save_weights_dir = Path(__file__).parent / "weights" / df["annotator"].unique()[0]

    assert len(df["generator_2"].unique()) == 1
    model_name = list(df["generator_2"].unique())[0]
    baseline_name = list(df["generator_1"].unique())[0]
    is_baseline = model_name == baseline_name

    if not is_baseline:
        df_XY_train, df_X_test, sample_weight = _get_featurized_data(
            df,
            formula=glm_info["formula"],
            regularize_to_baseline_lambda=glm_info["regularize_to_baseline_lambda"],
            df_gamed=load_gamed_data() if df_gamed is None else df_gamed,
        )
        filter_df = df_XY_train["preference"].notna()
        df_XY_train = df_XY_train[filter_df]
        if sample_weight is not None:
            sample_weight = sample_weight[filter_df]

        model = fit_LogisticRegressionCV(
            df_XY_train, "preference", is_ytrue_proba=True, sample_weight=sample_weight, **glm_info["kwargs"]
        )
        predicted_preferences = model.predict_proba(df_X_test)[:, 1]
        weights = dict(zip(df_X_test.columns, model.coef_[0]))
    else:
        weights = {c.strip(): 0 for c in glm_info["formula"].split("-")[0].split("+")}
        predicted_preferences = (df["preference"] * 0) + 0.5  # by construction

    if is_add_glm_preference_inplace and isinstance(annotations, pd.DataFrame):
        annotations["glm_preference"] = predicted_preferences

    metrics["length_controlled_winrate"] = predicted_preferences.mean() * 100
    metrics["lc_standard_error"] = pd.Series(predicted_preferences).sem() * 100

    if save_weights_dir is not None:
        save_weights_dir = Path(save_weights_dir) / glm_name
        save_weights_dir.mkdir(exist_ok=True, parents=True)
        weights_path = save_weights_dir / f"baseline_{baseline_name}.csv"
        if weights_path.exists():
            saved_weights = pd.read_csv(weights_path, index_col=0)
            new_weights = pd.DataFrame(weights, index=[model_name])
            saved_weights = pd.concat([saved_weights, new_weights], axis=0)
        else:
            saved_weights = pd.DataFrame(weights, index=[model_name])
        saved_weights = saved_weights[~saved_weights.index.duplicated(keep="last")]
        saved_weights.to_csv(weights_path, float_format="%.16f")

    if baseline is not None:
        assert save_weights_dir is not None
        metrics["length_controlled_winrate"] = predict_winrate(
            model=model_name,
            baseline=baseline,
            weights=weights_path,
            glm_name=glm_name,
        )

    if is_warn_extreme_changes and get_is_extreme_changes(metrics["win_rate"], metrics["length_controlled_winrate"]):
        logging.warning(
            f"Length controlled win rate is very different from the raw one: {metrics['length_controlled_winrate']:.1f}"
            f"% vs {metrics['win_rate']:.1f}%. This might be a sign of failure of the GLM."
        )

    return metrics


# helper functions specific to the glm
def predict_winrate(
    model: str,
    baseline: str,
    weights: types.AnyLoadableDF = DFLT_WEIGHT_PATH,
    glm_name="length_controlled_v1",
) -> float:
    """Predict the length corrected winrate of a model compared to a baseline using a GLM.

    Parameters
    ----------
    model : str
        Model name to predict the winrate for.

    baseline : str
        Baseline model name.

    weights: DataFrame or Path, optional
        Dataframe (or path to load one) containing the weights of the GLM to use for predictions.

    glm_name : str, optional
        The name of the GLM to use.
    """
    assert glm_name == "length_controlled_v1"
    instruction_difficulty = _get_instructions_difficulty()

    weights = utils.load_or_convert_to_dataframe(weights)
    delta_weights = weights.loc[model] - weights.loc[baseline]
    p = _logistic(
        delta_weights["not_gamed_baseline.astype(float)"]
        + delta_weights["instruction_difficulty"] * instruction_difficulty
    )
    return p.mean()


def get_is_extreme_changes(prev_winrate, new_winrate, abs_diff=10, rel_diff=4, min_warn=True, max_warn=True):
    """Whether the win-rate changed by more than abs_diff or rel_diff.  E.g. if  abs_diff=7, rel_diff=4 and old win
    rate is 20, this will return true if the new win rate is <10 (i.e. min(20-20/4, 20-10)) or > 40 (i.e. 20+(100-20)/2)
    Or if the old win rate is 50 and we predict <37.5 or >62.5.
    """
    too_small = new_winrate < min(prev_winrate - (prev_winrate / rel_diff), prev_winrate - abs_diff)
    too_large = new_winrate > max(prev_winrate + ((100 - prev_winrate) / rel_diff), prev_winrate + abs_diff)
    return (too_small and min_warn) or (too_large and max_warn)


def _logistic(x):
    """Logistic function."""
    return np.exp(-np.logaddexp(0, -x))


def _get_featurized_data(
    df_annotations: pd.DataFrame, formula: str, regularize_to_baseline_lambda: Optional[float],
    df_gamed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """Featurizes annotations using R-style formula and returns the design matrix for the train and test set.

    Parameters
    ----------
    df_annotations : pd.DataFrame
        The input dataframe, should have columns "preference", "output_1", "output_2", "index", "generator_1",
        "generator_2".

    formula : str
        The R-style formula to use for the GLM. See patsy.

    regularize_to_baseline_lambda : float, optional
        Strength of the regularizations to the baseline parameters. If None, no regularization is applied.
    """
    instruction_difficulty = df_gamed.drop_duplicates("index")["instruction_difficulty"]

    # 2. add features necessary for the glm
    df = df_annotations.reset_index()
    len_1 = df["output_1"].str.len()
    len_2 = df["output_2"].str.len()
    std_delta_len = len_1 - len_2
    df = df[["preference", "index"]].copy()
    df["std_delta_len"] = std_delta_len / std_delta_len.std()
    df["preference"] = df["preference"].astype(float).replace({0.0: 1.5}) - 1  # easier to work with in [0,1]
    df["instruction_difficulty"] = df["index"].transform(lambda g: instruction_difficulty[g])
    df["not_gamed_baseline"] = True

    # 3. make the design matrix for the model you would like to predict for, i.e., if there was no length difference
    df_test = df[["instruction_difficulty", "not_gamed_baseline"]].copy()
    df_test["std_delta_len"] = 0

    if regularize_to_baseline_lambda:
        df_gamed_and_m = pd.concat([df_gamed, df], axis=0)
        df_XY_train, df_X_test = make_dmatrix_for_model(df_gamed_and_m, df_test, formula=formula)

        # divided by 2 because there are two gamed baselines.
        sample_weight = (df_gamed_and_m["not_gamed_baseline"]).astype(float) + (
            regularize_to_baseline_lambda * (~df_gamed_and_m["not_gamed_baseline"])
        ).astype(float) / 2
    else:
        sample_weight = None
        df_XY_train, df_X_test = make_dmatrix_for_model(df, df_test, formula=formula)

    return df_XY_train, df_X_test, sample_weight


def load_gamed_data() -> pd.DataFrame:
    """Load instruction difficulty and reference-model annotations for length control."""
    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="df_gamed.csv",
        repo_type="dataset",
        token=constants.DATASETS_TOKEN,
        force_download=constants.DATASETS_FORCE_DOWNLOAD,
        cache_dir=constants.DEFAULT_CACHE_DIR,
    )
    return pd.read_csv(path).drop(columns=["model"])


def make_dmatrix_for_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame, formula: str, col_y_true="preference"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the design matrix based on a patsy formula.

    Parameters
    ----------
    df_train : pd.DataFrame
        The dataframe to train on. Should contain features and the true labels.

    df_test : pd.DataFrame
        The dataframe to predict on. Should contain the same features as df_train.

    formula : str
        The R-style formula to use for the GLM. See patsy.

    col_y_true : str, optional
        The name of the column containing the true labels.
    """
    df_XY_train = dmatrix(formula, df_train, return_type="dataframe")
    df_X_test = build_design_matrices([df_XY_train.design_info], df_test, return_type="dataframe")[0]
    df_XY_train[col_y_true] = df_train[col_y_true]  # adds the label
    return df_XY_train, df_X_test


def logloss(y_true, y_pred, sample_weight=None):
    """Compute the logloss of the predictions, potentially weighted."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    all_logloss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    if sample_weight is not None:
        # weight as desired
        all_logloss = all_logloss * sample_weight
    return -np.mean(all_logloss)


def logloss_continuous(y_true, y_pred, true_prob, true_sample_weight=None):
    """Computes the log loss with continuous targets for sklearn. Need to hack it as sklearn expects int targets."""
    # y_true is binary and sample_weight says the proba for that class. Let's revert that transform
    y_true = np.where(y_true == 1, true_prob, 1 - true_prob)
    return logloss(y_true, y_pred, sample_weight=true_sample_weight)


def fit_LogisticRegressionCV(data, col_y_true, is_ytrue_proba=True, n_splits=5, C=100, sample_weight=None, **kwargs):
    """Fits LogisticRegressionCV with optionally y_true being probabilities rather than the labels."""
    sklearn.set_config(enable_metadata_routing=True)
    dflt_kwargs = dict(random_state=123, dual=False, penalty="l1", solver="liblinear", n_jobs=None, fit_intercept=False)
    dflt_kwargs.update(kwargs)
    if not is_ytrue_proba:
        if n_splits > 0:
            cv = StratifiedKFold(n_splits=n_splits)
            scorer = make_scorer(sk_log_loss, greater_is_better=False, needs_proba=True)
            if sample_weight is None:
                model = LogisticRegressionCV(cv=cv, scorer=scorer, **dflt_kwargs)
            else:
                scorer = scorer.set_score_request(sample_weight=True)
                model = LogisticRegressionCV(cv=cv, scorer=scorer, **dflt_kwargs)

        else:
            model = LogisticRegression(C=C, **dflt_kwargs)

        model.fit(data.drop(columns=[col_y_true]), (data[col_y_true]).round().astype(int), sample_weight=sample_weight)

    else:  # use log loss without assuming that labels are discrete
        # duplicate the df, once with label 0 and once with label 1
        data = data.reset_index(drop=True).reset_index(drop=False, names=["group"])
        data_1 = data.copy()
        data_1["y"] = 1
        data_0 = data.copy()
        data_0[col_y_true] = 1 - data_0[col_y_true]
        data_0["y"] = 0
        data_dup = pd.concat([data_1, data_0], axis=0).reset_index(drop=True)
        true_prob = data_dup[col_y_true]

        if sample_weight is None:
            true_sample_weight = None
            sample_weight = true_prob
        else:
            true_sample_weight = np.concatenate([sample_weight, sample_weight], axis=0)  # actual sample weight
            # multiply the true probabilities by the actual sample_weight you want
            sample_weight = true_prob * true_sample_weight

        if n_splits > 0:
            cv = GroupKFold(n_splits=n_splits)
            scorer = make_scorer(
                logloss_continuous, response_method="predict_proba", greater_is_better=False
            ).set_score_request(true_sample_weight=True, true_prob=True)
            model = LogisticRegressionCV(cv=cv, scoring=scorer, **dflt_kwargs)
            fit_kwargs = dict(groups=data_dup["group"], true_sample_weight=true_sample_weight, true_prob=true_prob)
        else:
            model = LogisticRegression(C=C, **dflt_kwargs)
            fit_kwargs = dict()

        model.set_fit_request(sample_weight=True)
        model.fit(
            X=data_dup.drop(columns=[col_y_true, "y", "group"]),
            y=data_dup["y"],
            sample_weight=sample_weight,
            **fit_kwargs,
        )
    return model


def _get_instructions_difficulty():
    out = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="instruction_difficulty.csv",
        repo_type="dataset",
        token=constants.DATASETS_TOKEN,
        force_download=constants.DATASETS_FORCE_DOWNLOAD,
        cache_dir=constants.DEFAULT_CACHE_DIR,
    )
    return pd.read_csv(out, index_col=0).squeeze()


def _predicted_winrate_matrix(
    models,
    weights: types.AnyLoadableDF = DFLT_WEIGHT_PATH,
):
    instruction_difficulty = _get_instructions_difficulty()

    weights = utils.load_or_convert_to_dataframe(weights)

    winrate_matrix = dict()
    for b in models:
        winrate_matrix[b] = dict()
        for m in models:
            delta_weights = weights.loc[m] - weights.loc[b]
            winrate_matrix[b][m] = _logistic(
                delta_weights["not_gamed_baseline.astype(float)"]
                + delta_weights["instruction_difficulty"] * instruction_difficulty
            ).mean()
    return pd.DataFrame(winrate_matrix) * 100
