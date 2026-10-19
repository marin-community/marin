"""
Main module for analyzing an evaluation benchmark (annotator and data).
"""

import logging
from itertools import combinations
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import constants, utils
from .metrics import SCORING_RULES
from .types import AnyData, AnyPath


class Analyzer:
    """Helper class to compare and understand annotations from different annotators.

    Parameters
    ----------
    gold_crossannotations : path or data or callable
        The cross annotations from the gold annotators. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain all of `keys` and `preference` keys.

    gold_annotations : path or data or callable, optional
        The annotations from the gold annotators. Same format as `gold_crossannotations`. If None we use the first
        annotation from `gold_crossannotations`.

    keys : tuple
        Keys to use to compare the annotations.

    n_annotators : int
        Minimum number of annotators for treating as gold annotation.

    scoring_rule : {"zero_one", "absolute", "squared"}, optional
        Scoring rule to use for computing the agreement. "zero_one" is the classification error, which was used in the
        first version of AlpacaEval but only makes sense for discrete predictions. "absolute" is the mean absolute error
        (MAE) and "squared" is the mean squared error (MSE). Both MAE and MSE are equivalent to zero_one for discrete
        predictions as we are performing binary classification. However, they allow for continuous predictions. We
        recommend "absolute" which is more interpretable (0.5 gets half the error) and keep the same "bias" and
        "variance" definitions as in the discrete case with zero_one loss. Note that to make the generalization correct
        we use an integer median by sampling from the two possible medians.

    annotator_kwargs : dict
        Arguments that will be passed to all annotators being analyzed.
    """

    def __init__(
        self,
        gold_crossannotations: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_GOLD_CROSSANNOTATIONS,
        gold_annotations: Optional[Union[AnyPath, AnyData, Callable]] = constants.ALPACAFARM_GOLD_ANNOTATIONS,
        keys=("instruction", "output_1", "output_2"),
        n_annotators: Optional[int] = 4,
        seed: Optional[int] = 0,
        scoring_rule: str = "absolute",
        **annotator_kwargs,
    ):
        self.keys = list(keys)
        self.n_annotators = n_annotators
        self.annotator_kwargs = annotator_kwargs
        self.scoring_rule = SCORING_RULES[scoring_rule]()
        self.annotation_key = "preference"

        df_gold_crossannotations = utils.load_or_convert_to_dataframe(gold_crossannotations)
        # adding a random index to differentiate between the n_annotators
        self.df_gold_crossannotations = self._select_n_annotations(
            df_gold_crossannotations, n_annotators=self.n_annotators
        )

        if gold_annotations is None:
            self.df_gold_annotations = self.df_gold_crossannotations.query("annotator_index == 0")
        else:
            self.df_gold_annotations = utils.load_or_convert_to_dataframe(gold_annotations)

        self.all_df_annotations = dict()
        self.seed = seed

    def agreement_of_annotations(
        self,
        annotations_1: Union[pd.DataFrame, str],
        annotations_2: Optional[Union[pd.DataFrame, str]] = "gold_crossannotations",
        n_majority_vote_1: Optional[int] = 1,
        n_majority_vote_2: Optional[int] = None,
        is_same_annotator: Optional[bool] = None,
    ) -> pd.Series:
        """Compare (cross)annotations from two annotators.

        Notes
        -----
        - if you want to compute the agreement of 1 annotation vs the rest (eg to estimate the variance) then use
        n_majority_vote_1=1 and n_majority_vote_2=None and annotations_2=None.
        - if you want to measure the agreement of N annotators between two different annotators (eg to estimate the bias
        use n_majority_vote_1=N and n_majority_vote_2=N.

        Parameters
        ----------
        annotations_1 : pd.DataFrame or "gold_crossannotations" or "gold_annotations"
            First annotations. If "gold_crossannotations" or "gold_annotations" we use the corresponding gold
            annotations. If there are more than one annotation per example (ie index > 0) then we either use majority
            vote (if n_majority_vote_1 == n_annotators) or take an expectation over possible annotators.

        annotations_2 : pd.DataFrame or "gold_crossannotations" or "gold_annotations"
            First annotations. If "gold_crossannotations" or "gold_annotations" we use the corresponding gold
            annotations. If None we use the same as `annotations_1`. If there are more than one annotation per example
            (ie index > 0) then we either use majority vote (if n_majority_vote_1 == n_annotators) or take an
            expectation over possible annotators.

        n_majority_vote_1 : int, optional
            If more than 1 we will use the majority vote of annotations_1. If None we use the maximum possible.
            It can only be None if both annotations_1 and annotations_2 are different.

        n_majority_vote_2 : int, optional
            If more than 1 we will use the majority vote of annotations_2. If None we use the maximum possible, which
            is all annotations if both annotations are the same, or the complement of annotations_1 if they are
            different.

        is_same_annotator : bool, optional
            Whether both annotations_1 and annotations_2 are the same or a subset of each other => you should not
            compare the same indices as this will bias the agreement. If None we will check if they are the same.

        Examples
        --------
        >>> analyzer = Analyzer(n_annotators=4)
        >>> df_crossannotations = analyzer.df_gold_crossannotations.head(8).copy()
        >>> df_crossannotations["preference"] = [1] * 4 + [2,2,2,1]
        >>> analyzer.agreement_of_annotations(df_crossannotations, annotations_2=None,
        ...                                   n_majority_vote_1=1,  n_majority_vote_2=1)
        score             0.750000
        error             0.250000
        sem_samples       0.250000
        counts            2.000000
        sem_annotators    0.075378
        dtype: float64
        >>> # score above is 3/4 because for the first 3 comparison you get 2 * 100% and 1 * 50%. I.e. you get 50%
        >>> # when the second index is 3.  And for the last comparison the first index is always 3 so you get 3*50%
        >>> analyzer.agreement_of_annotations(df_crossannotations, annotations_2=None,
        ...                                   n_majority_vote_1=1,  n_majority_vote_2=3)
        score             0.875
        error             0.125
        sem_samples       0.125
        counts            2.000
        sem_annotators    0.125
        dtype: float64
        >>> # above you are doing 4 comparison of 1 vs 3. As you are doing majority vote of 3 you get 100% for 3 out
        >>> # of 4 comparisons and 50% for the last one. So you get 3*100% + 1*50% = 87.5%
        """
        annotations_1 = self._get_annotations(annotations_1)

        if annotations_2 is None:
            annotations_2 = annotations_1

        annotations_2 = self._get_annotations(annotations_2)
        if is_same_annotator is None:
            is_same_annotator = annotations_2.equals(annotations_1)

        if is_same_annotator and n_majority_vote_1 is None:
            raise ValueError("n_majority_vote_1 cannot be None if annotations_1 and annotations_2 are the same")

        annotations_1 = self._select_n_annotations(annotations_1, n_annotators=n_majority_vote_1, is_rm_less_than=False)
        max_majority_vote_1 = annotations_1["n_annotated"].max()
        n_majority_vote_1 = n_majority_vote_1 or max_majority_vote_1
        if n_majority_vote_1 > max_majority_vote_1:
            raise ValueError(
                f"n_majority_vote_1={n_majority_vote_1} is larger than the maximum possible " f"({max_majority_vote_1})"
            )

        if is_same_annotator:
            logging.info("You are comparing twice the same annotators.")
            # the maximum number of votes you should compare is the complement given that it's the same data
            n_majority_vote_2 = n_majority_vote_2 or (max_majority_vote_1 - n_majority_vote_1)
            assert (n_majority_vote_2 <= max_majority_vote_1) and (n_majority_vote_1 <= max_majority_vote_1)

        annotations_2 = self._select_n_annotations(annotations_2, n_annotators=n_majority_vote_2, is_rm_less_than=False)
        max_majority_vote_2 = annotations_2["n_annotated"].max()
        n_majority_vote_2 = n_majority_vote_2 or max_majority_vote_2
        if n_majority_vote_2 > max_majority_vote_2:
            raise ValueError(
                f"n_majority_vote_2={n_majority_vote_2} is larger than the maximum possible " f"({max_majority_vote_2})"
            )

        results = dict()
        for idcs_1 in combinations(range(max_majority_vote_1), n_majority_vote_1):
            for idcs_2 in combinations(range(max_majority_vote_2), n_majority_vote_2):
                is_overlapping_idcs = len(set(idcs_1).intersection(idcs_2)) > 0
                if is_same_annotator:
                    if is_overlapping_idcs:
                        continue  # skipping overlapping indices because biased
                    elif (idcs_2, idcs_1) in results.keys():
                        # not skipping for unbiased but no need to compute twice
                        results[(idcs_1, idcs_2)] = results[(idcs_2, idcs_1)]
                        continue

                results[(idcs_1, idcs_2)] = self._score_of_single_annotations(
                    df_annotations_1=self._get_bayes_estimator(annotations_1, idcs_1),
                    df_annotations_2=self._get_bayes_estimator(annotations_2, idcs_2),
                )

        logging.info(
            f"n_majority_vote_1={n_majority_vote_1}, n_majority_vote_2={n_majority_vote_2}. "
            f"Compared results of indices: {list(results.keys())}"
        )

        # maybe better to use from_dict(results, orient='index')
        sem_annotators = pd.DataFrame(results).T["score"].sem()
        results = sum(results.values()) / len(results.values())
        results["sem_annotators"] = sem_annotators

        return results

    def estimate_bias(self, annotations: pd.DataFrame) -> float:
        """(over)Estimates the bias of the annotations by computing the agreement error between the majority vote of
        the annotations and the gold annotations.

        Parameters
        ----------
        annotations: pd.DataFrame
            Annotations to estimate the bias of. For better results, it should have multiple annotations per example.
        """
        assert annotations["index"].nunique() > 1

        # all vs all of gold annotations
        agreement = self.agreement_of_annotations(
            annotations,
            annotations_2="gold_crossannotations",
            n_majority_vote_1=None,
            n_majority_vote_2=None,
        )
        return agreement["error"]

    def estimate_variance(self, annotations: Union[pd.DataFrame, str]) -> float:
        """(over)Estimates the variance of the annotations by computing the 1 vs all agreement error.

        Parameters
        ----------
        annotations: pd.DataFrame
            Annotations to estimate the variance of. For better results, it should have multiple annotations per
            example.
        """
        # 1 vs rest
        agreement = self.agreement_of_annotations(
            annotations, annotations_2=None, n_majority_vote_1=1, n_majority_vote_2=None
        )
        return agreement["error"]

    def estimate_correlations(
        self,
        annotations_1: pd.DataFrame,
        annotations_2: Union[pd.DataFrame, str] = "gold_crossannotations",
        groupby: Sequence[str] = ("generator",),
    ) -> dict[str, float]:
        """Estimate the correlations between different methods.

        Parameters
        ----------
        annotations_1: pd.DataFrame
            Annotations to estimate the correlations of. For better results, it should have multiple annotations per
            example.

        annotations_2: pd.DataFrame or "gold_crossannotations" or "gold_annotations"
            Annotations to compare rankings with. If "gold_crossannotations" or "gold_annotations" we use the
            corresponding gold annotations.

        groupby: list[str], optional
            Columns to groupby for computing the leaderboard.

        Returns
        -------
        correlations: pd.DataFrame
            Correlations between different methods.
        """

        annotations_1 = self._get_annotations(annotations_1)
        is_add_generator = annotations_2 == "gold_crossannotations"
        annotations_2 = self._get_annotations(annotations_2)

        if "generator" not in annotations_2.columns and is_add_generator:
            # TODO clean: following is because we don't save generator in HF crossannotation dataset => reconstructs it.
            # takes only eval set for the leaderboard
            merge_kwargs = dict(right=self.df_gold_annotations[self.keys + ["generator"]], on=self.keys)
            annotations_2 = annotations_2.query("datasplit == 'eval'").merge(**merge_kwargs)
            annotations_1 = annotations_1.query("datasplit == 'eval'").merge(**merge_kwargs)
            n_per_generator = annotations_2.groupby("generator").size()
            n_annotated = 100  # we annotated around 140 per generator, there are other generators due to same outputs
            selected_generators = list(n_per_generator[n_per_generator > n_annotated].index)
            annotations_2 = annotations_2[annotations_2.generator.isin(selected_generators)]
            annotations_1 = annotations_1[annotations_1.generator.isin(selected_generators)]

        # 1. get the leaderboard for each annotations, where we groupby the keys and compute the score
        groupby = list(groupby)
        leaderboard_1 = (
            annotations_1.groupby(groupby)[self.annotation_key]
            .aggregate(self.scoring_rule.generalized_win_rate)
            .rename("win_rate_1")
        )
        leaderboard_2 = (
            annotations_2.groupby(groupby)[self.annotation_key]
            .aggregate(self.scoring_rule.generalized_win_rate)
            .rename("win_rate_2")
        )

        # 2. get the correlations between both leaderboards
        df = pd.merge(
            leaderboard_1,
            leaderboard_2,
            left_index=True,
            right_index=True,
        )
        try:
            s = spearmanr(df["win_rate_2"], df["win_rate_1"]).statistic
            r = pearsonr(df["win_rate_2"], df["win_rate_1"]).statistic
        except ValueError:
            logging.warning(
                (
                    "Could not compute correlations. This issue may be due to a lack of different "
                    f"values of the column data is grouped by (using {groupby} column of dataset). "
                    f"The computation failed for the following dataframe:\n{df}"
                ),
            )
            s = np.nan
            r = np.nan

        return dict(spearman=s, pearson=r)

    def get_length_biases(
        self, annotations: Union[pd.DataFrame, str], significant_delta_length: int = 30
    ) -> dict[str, float]:
        """Estimate the biases for longer sentences."""
        try:
            df = annotations.drop_duplicates(subset=self.keys).copy()
            df["best_output"] = np.where(
                df[self.annotation_key].between(1, 1.5, inclusive="left"), df.output_1, df.output_2
            )
            df["worse_output"] = np.where(
                df[self.annotation_key].between(1.5, 2, inclusive="right"), df.output_1, df.output_2
            )

            # Step 1: Create new columns indicating the length of `best_output` and `worse_output`
            df["best_output_length"] = df["best_output"].apply(len)
            df["worse_output_length"] = df["worse_output"].apply(len)
            # Step 2: Create a new column indicating whether one output is (significantly) longer than the other
            df["one_is_longer"] = (
                df["best_output_length"] - df["worse_output_length"]
            ).abs() > significant_delta_length
            df["is_prefer_longer"] = df["best_output_length"] > df["worse_output_length"]
            # Step 3: Count the number of times you prefer the longer output
            prefer_longer = df[df["one_is_longer"] & df["is_prefer_longer"]].shape[0]
            # Step 4: Count the total number of instances when one output is longer than the other
            total_one_is_longer = df[df["one_is_longer"]].shape[0]
            # Step 5: Calculate the probability of preferring the longer output
            probability_prefer_longer = prefer_longer / total_one_is_longer

            percentage_longer = (
                (df["best_output_length"] - df["worse_output_length"]) / df["worse_output_length"]
            ).mean()

        except Exception as e:
            logging.warning(f"Could not compute length biases: {e}")
            probability_prefer_longer = np.nan
            percentage_longer = np.nan

        return dict(
            probability_prefer_longer=probability_prefer_longer,
            percentage_longer=percentage_longer,
        )

    def get_list_biases(self, annotations: Union[pd.DataFrame, str]) -> dict[str, float]:
        """Estimate the biases for sentences with lists."""
        try:
            df = annotations.drop_duplicates(subset=self.keys).copy()
            df["best_output"] = np.where(
                df[self.annotation_key].between(1, 1.5, inclusive="left"), df.output_1, df.output_2
            )
            df["worse_output"] = np.where(
                df[self.annotation_key].between(1.5, 2, inclusive="right"), df.output_1, df.output_2
            )

            # Step 1: Create new columns indicating whether `best_output` and `worse_output` contain lists
            df["is_best_list"] = df["best_output"].apply(utils.contains_list)
            df["is_worse_list"] = df["worse_output"].apply(utils.contains_list)
            # Step 2: Create a new column indicating whether either `best_output` or `worse_output` has a list but
            # not both
            df["either_list"] = df["is_best_list"] ^ df["is_worse_list"]
            # Step 3: Count the number of times you prefer `best_output` when either `best_output` or `worse_output` has
            # a list but not both
            prefer_best_either_list = df[(df["either_list"]) & df["is_best_list"]].shape[0]
            # Step 4: Count number of instances when either `best_output` or `worse_output` has a list but not both
            total_either_list = df[df["either_list"]].shape[0]
            # Step 5: Calculate the probability
            probability_prefer_list = prefer_best_either_list / total_either_list

            percentage_list = (df["is_best_list"].mean() - df["is_worse_list"].mean()) / df["is_worse_list"].mean()
        except Exception as e:
            logging.warning(f"Could not compute list biases: {e}")
            probability_prefer_list = np.nan
            percentage_list = np.nan

        return dict(
            probability_prefer_list=probability_prefer_list,
            percentage_list=percentage_list,
        )

    def _select_n_annotations(self, df, n_annotators=None, is_rm_less_than: bool = True):
        """Gets examples with at least n annotations. Adds `index` and `n_annotated` columns."""
        if "n_annotated" in df.columns:
            df = df.drop(columns="n_annotated")

        df["index"] = df.groupby(self.keys)[self.annotation_key].cumcount()

        if is_rm_less_than:
            # remove samples that have more than n_annotators
            df = df[df["index"] < n_annotators]

        # select examples that have at least n_annotators
        counts = df.groupby(self.keys)[self.annotation_key].count()
        counts.name = "n_annotated"
        n_annotators = n_annotators or counts.min()
        counts = counts[counts >= n_annotators].reset_index()
        df_selected = df.merge(counts, on=self.keys)

        return df_selected.copy()

    def _get_annotations(self, annotations: Union[pd.DataFrame, str]):
        if isinstance(annotations, str):
            if annotations == "gold_crossannotations":
                annotations = self.df_gold_crossannotations
            elif annotations == "gold_annotations":
                annotations = self.df_gold_annotations
            else:
                raise ValueError(f"Unknown annotations: {annotations}")

        return annotations

    def _get_bayes_estimator(self, annotations, idcs):
        annotations = annotations[annotations["index"].isin(idcs)]
        return annotations.groupby(self.keys)[self.annotation_key].aggregate(self.scoring_rule.bayes_estimator)

    def _score_of_single_annotations(
        self,
        df_annotations_1: pd.DataFrame,
        df_annotations_2: pd.DataFrame,
    ):
        merged = pd.merge(df_annotations_1, df_annotations_2, on=self.keys, suffixes=("_1", "_2"))
        out = pd.Series(
            dict(
                score=self.scoring_rule.score(
                    prediction=merged[f"{self.annotation_key}_1"], target=merged[f"{self.annotation_key}_2"]
                ),
                error=self.scoring_rule.error(
                    prediction=merged[f"{self.annotation_key}_1"], target=merged[f"{self.annotation_key}_2"]
                ),
                sem_samples=self.scoring_rule.sem(
                    prediction=merged[f"{self.annotation_key}_1"], target=merged[f"{self.annotation_key}_2"]
                ),
                counts=len(merged),
            )
        )
        return out


def get_crossannotations(
    analyzer,
    Annotator,
    max_instances: Optional[int] = None,
    is_single_annotator: bool = False,
    is_keep_gold_preference: bool = True,
    **kwargs,
):
    """Get cross annotations by `Annotator` corresponding to `analyzer.df_gold_crossannotations`."""
    n_crossannotations = 1 if is_single_annotator else analyzer.n_annotators
    all_annotations = []
    for seed in range(n_crossannotations):
        annotator = Annotator(seed=seed, **kwargs)

        df_gold_crossannotations = analyzer.df_gold_crossannotations.query(f"index == {seed}").copy()
        if max_instances is not None:
            df_gold_crossannotations = df_gold_crossannotations.head(max_instances)

        if is_keep_gold_preference:
            df_gold_crossannotations = df_gold_crossannotations.rename(columns={"preference": "gold_preference"})
            annotator.other_keys_to_keep += ["gold_preference"]

        annotations = annotator.annotate_pairs(df_gold_crossannotations)
        df_annotations = utils.load_or_convert_to_dataframe(annotations)

        df_annotations["index"] = seed
        all_annotations.append(df_annotations)
    df = pd.concat(all_annotations, axis=0)
    df["n_annotated"] = n_crossannotations
    return df


def get_annotations(analyzer, Annotator, max_instances: Optional[int] = None, **kwargs):
    """Get annotations by `Annotator` corresponding to `analyzer.df_gold_annotations`."""
    annotator = Annotator(**kwargs)
    df_gold_annotations = analyzer.df_gold_annotations
    if max_instances is not None:
        df_gold_annotations = df_gold_annotations.head(max_instances)
    annotations = annotator.annotate_pairs(df_gold_annotations)
    df_annotations = utils.load_or_convert_to_dataframe(annotations)
    return df_annotations


def get_metrics_evaluator(analyzer, df_crossannotations, evaluator_name=None):
    """Gets the metrics for an annotator given its crossannotations."""

    all_metrics = dict()
    all_metrics["Human agreement"] = (
        analyzer.agreement_of_annotations(annotations_1=df_crossannotations, n_majority_vote_1=1)["score"] * 100
    )

    all_metrics["Price [$/1000 examples]"] = df_crossannotations["price_per_example"].mean() * 1000
    all_metrics["Time [seconds/1000 examples]"] = df_crossannotations["time_per_example"].mean() * 1000

    correlations = analyzer.estimate_correlations(df_crossannotations)
    all_metrics["Spearman corr."] = correlations["spearman"]
    all_metrics["Pearson corr."] = correlations["pearson"]

    if evaluator_name == "humans":
        all_metrics["Bias"] = 0
        all_metrics["Variance"] = analyzer.estimate_variance(df_crossannotations) * 100
    else:
        try:
            all_metrics["Bias"] = analyzer.estimate_bias(df_crossannotations) * 100
        except:
            all_metrics["Bias"] = np.nan

        try:
            all_metrics["Variance"] = analyzer.estimate_variance(df_crossannotations) * 100
        except:
            all_metrics["Variance"] = np.nan

    all_metrics["Proba. prefer longer"] = analyzer.get_length_biases(df_crossannotations)["probability_prefer_longer"]
    all_metrics["Proba. prefer lists"] = analyzer.get_list_biases(df_crossannotations)["probability_prefer_list"]
    all_metrics["Proba. prefer 1"] = 2 - df_crossannotations["preference"].mean()
    all_metrics["# parsed"] = len(df_crossannotations.preference.dropna())
    return all_metrics


###############################


def _get_longest_predictor(df_annotations):
    """TUrn the current predictions as the predictions from an annotator that always picks the longest output."""
    curr = df_annotations.copy()
    curr["annotator"] = "longest"
    curr["preference"] = np.where(curr.output_1.str.len() > curr.output_2.str.len(), 1, 2)
    curr["time_per_example"] = 0
    curr["price_per_example"] = 0
    return curr
