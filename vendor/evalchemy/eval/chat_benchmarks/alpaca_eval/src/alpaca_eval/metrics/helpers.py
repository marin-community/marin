import abc
import logging
from dataclasses import dataclass
from numbers import Number

import numpy as np
import numpy.typing as npt
import pandas as pd

from alpaca_eval.utils import validate_alpacaeval_preference


@dataclass
class BaseScoringRule(abc.ABC):
    """Base class for a scoring rule."""

    @abc.abstractmethod
    def _score(self, prediction: pd.Series, target: pd.Series) -> float:
        pass

    @abc.abstractmethod
    def _bayes_estimator(self, predictions: pd.Series) -> Number:
        pass

    def _idcs_draws(self, predictions: pd.Series) -> pd.Series:
        return (predictions == 1.5) | (predictions == 0)

    def score(self, prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
        """Score a prediction. Higher is better."""
        prediction, target = self.preprocess(prediction, target)
        return self._score(prediction, target)

    def error(self, prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
        """Compute the error of the prediction. Lower is better"""
        return 1 - self.score(prediction, target)

    def sem(self, prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
        """Compute the standard error of the error."""
        return pd.Series([self.score([p], [t]) for p, t in zip(prediction, target)]).sem()

    def generalized_win_rate(self, predictions: npt.ArrayLike) -> float:
        """Compute the generalized win rate of the prediction."""
        return self.describe_head2head(predictions)["win_rate"]

    def describe_head2head(self, predictions: npt.ArrayLike) -> dict[str, float]:
        """Compute the generalized win rate of the prediction."""
        predictions = self.preprocess_predictions(predictions)
        n_draws = self._idcs_draws(predictions).sum()

        # makes it easier to work with
        predictions = predictions.astype(float).replace({0.0: 1.5})

        is_preference = predictions.apply(validate_alpacaeval_preference, is_allow_nan=False)
        n_not_pair = sum(~is_preference)
        if n_not_pair > 0:
            logging.info(f"drop {n_not_pair} outputs that are not preferences")

        predictions = predictions[is_preference] - 1

        n_wins = (predictions > 0.5).sum()
        n_wins_base = (predictions < 0.5).sum()
        n_total = len(predictions)

        return dict(
            win_rate=predictions.mean() * 100,
            standard_error=predictions.sem() * 100,
            n_wins=n_wins,
            n_wins_base=n_wins_base,
            n_draws=n_draws,
            # note that n_draws will happen more often for weighted win rate because you can get 1.5 somewhat often due
            # to float precision
            n_total=n_total,
        )

    def bayes_estimator(self, predictions: npt.ArrayLike) -> Number:
        """Compute the bayes estimator of the predictions."""
        predictions = self.preprocess_predictions(predictions)
        return self._bayes_estimator(predictions)

    def preprocess(self, predictions: npt.ArrayLike, targets: npt.ArrayLike) -> tuple[pd.Series, pd.Series]:
        """Validate the predictions and targets."""
        predictions = self.preprocess_predictions(predictions)
        targets = self.preprocess_targets(targets)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"predictions and targets should have the same shape. Got {predictions.shape} and {targets.shape}"
            )

        return predictions, targets

    def preprocess_predictions(self, predictions: npt.ArrayLike) -> pd.Series:
        """Validate the predictions."""
        # backward compatibility for 0 -> 1.5
        return pd.Series(predictions).replace({0: 1.5}).astype(float)

    def preprocess_targets(self, targets: npt.ArrayLike) -> pd.Series:
        """Validate the targets."""
        # backward compatibility for 0 -> 1.5
        return pd.Series(targets).replace({0: 1.5}).astype(float)


class ZeroOneScoringRule(BaseScoringRule):
    """Scoring rule for binary predictions."""

    def _score(self, prediction, target):
        # accuracy
        return (target == prediction).mean()

    def _bayes_estimator(self, predictions):
        # mode
        return _random_mode(predictions)

    def generalized_win_rate(self, predictions):
        descriptions = self.describe_head2head(predictions)
        assert (
            descriptions["win_rate"]
            == (descriptions["n_wins"] + descriptions["n_draws"] / 2) / descriptions["n_total"] * 100
        ), "generalized_win_rate should be equal to the win_rate for binary predictions"
        return descriptions["win_rate"]

    def preprocess_predictions(self, predictions: npt.ArrayLike) -> pd.Series:
        return pd.Series(predictions).replace({1.5: 0}).round().astype(pd.Int64Dtype())

    def preprocess_targets(self, targets: npt.ArrayLike) -> pd.Series:
        return pd.Series(targets).replace({1.5: 0}).round().astype(pd.Int64Dtype())


class AbsoluteScoringRule(BaseScoringRule):
    """Absolute loss scoring rule (i.e. MAE)."""

    def _score(self, prediction, target):
        # 1 - MAE
        return 1 - (target - prediction).abs().mean()

    def _bayes_estimator(self, predictions):
        # if all the values are 0.0, 1.0, 2.0, nan, or 1.5 then for backward compatibility we return the random mode
        # note that this doesn't change the expected value of the estimator, but increases the variance. The new version
        # is thus better
        if pd.Series(predictions.unique()).isin([0.0, 1.0, 2.0, np.nan, 1.5]).all():
            return _random_mode(predictions)

        return predictions.median()


class SquaredScoringRule(BaseScoringRule):
    """Squared loss scoring rule (i.e. MSE)."""

    def _score(self, prediction, target):
        # 1 - MSE
        return 1 - ((target - prediction) ** 2).mean()

    def _bayes_estimator(self, predictions):
        return predictions.mean()


SCORING_RULES = {
    "zero_one": ZeroOneScoringRule,
    "absolute": AbsoluteScoringRule,
    "squared": SquaredScoringRule,
}


def _random_mode(s, available_modes=None, favorite_mode=None, seed=123, is_dropna=True):
    """Take the mode of a series, but if there are multiple modes, randomly sample one
    (with potential restriction to `available_modes` or favoring a specific mode `favorite_mode`).

    Example
    -------
    >>> import pandas as pd
    >>> from alpaca_eval.metrics.helpers import _random_mode
    >>> _random_mode(pd.Series([1.0,2.0,1.0]))
    1.0
    >>> _random_mode(pd.Series([1.0,2.0])) in [1.0, 2.0]
    True
    >>> _random_mode(pd.Series([1.0,2.0,-1.0]), favorite_mode=-1.0)
    -1.0
    >>> _random_mode(pd.Series([1.0,2.0,2.0,-1.0]), favorite_mode=-1.0)
    2.0
    """
    out = pd.Series.mode(s)
    if is_dropna:
        out = out.dropna()

    if len(out) > 1:
        if favorite_mode is not None and favorite_mode in out.values:
            return favorite_mode
        if available_modes is not None:
            out = out[out.isin(available_modes)]
        out = out.sample(1, random_state=seed)

    if len(out) == 0:
        return np.nan

    return out.item()
