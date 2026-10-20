from typing import Sequence, Union

import pandas as pd

from alpaca_eval import utils

from .helpers import AbsoluteScoringRule, ZeroOneScoringRule

__all__ = ["get_winrate", "pairwise_to_winrate"]


def get_winrate(annotations: Union[pd.DataFrame, Sequence]) -> dict[str, float]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference.
    This assumes that the preference is encoded as 0 or 1.5 for draw, 1 for base win, 2 when the model to compare wins.
    """
    annotations = utils.convert_to_dataframe(annotations)
    preferences = annotations["preference"]
    out = AbsoluteScoringRule().describe_head2head(preferences)
    out["discrete_win_rate"] = ZeroOneScoringRule().describe_head2head(preferences)["win_rate"]
    return out


# backward compatibility
def pairwise_to_winrate(preferences: Union[pd.DataFrame, Sequence]) -> dict[str, float]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference.
    This assumes that the preference is encoded as 0 or 1.5 for draw, 1 for base win, 2 when the model to compare wins.
    """
    return get_winrate(annotations=[dict(preference=p) for p in preferences])
