"""
Helper classes for processing the data. Each of those should have a function preprocess and postprocess, which will
respectively be called in SingleAnnotator._preprocess and SingleAnnotator._postprocess in reverse order.
"""

import abc
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import utils

__all__ = [
    "RandomSwitchTwoColumnsProcessor",
    "PaddingForBatchesProcessor",
]


class BaseProcessor(abc.ABC):
    """Base class for a processor."""

    # additional input and output keys that should be kept in the annotator
    other_input_keys_to_keep = []
    other_output_keys_to_keep = []

    def __init__(
        self,
        seed: int = 123,
        annotation_column: str = "annotation",
        completion_column: str = "raw_completion",
    ):
        self.seed = seed
        self.annotation_column = annotation_column
        self.completion_column = completion_column

    @abc.abstractmethod
    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Process the annotation dataframe before annotations."""
        pass

    @abc.abstractmethod
    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Process the annotation dataframe after annotations."""
        pass


class RandomSwitchTwoColumnsProcessor(BaseProcessor):
    r"""Randomly switch the order of two columns.

    Parameters
    ----------
    two_columns_to_switch : Sequence[str]
        The two columns to switch.

    fn_replace_if_switch : Optional[Callable[[pd.DataFrame], pd.DataFrame]], optional
        Function to apply to the dataframe formed of the rows with a switch. By default, does nothing.

    fn_replace_if_unswitch : Optional[Callable[[pd.DataFrame], pd.DataFrame]], optional
        Function to apply to the dataframe formed of the rows without a switch. By default, applies the same as
        `fn_replace_if_switch`.

    random_seed_columns : Optional[Sequence[str]], optional
        The columns to use to seed the random choice of switching or not. If None, will use `columns_to_switch`.

    kwargs :
        Additional arguments to pass to `BaseProcessor`. E.g. seed.

    Examples
    --------
    >>> df = pd.DataFrame([dict(instruction='2+2', output_1='10', output_2='4', preference=2),
    ...                    dict(instruction='2+3', output_1='5', output_2='7', preference=1)])
    >>> processor = RandomSwitchTwoColumnsProcessor(two_columns_to_switch=['output_1', 'output_2'],
    ...                                             fn_replace_if_switch = lambda x: x.replace({"preference":{1: 2, 2: 1}}))
    >>> processor.preprocess(df)
        instruction output_1 output_2  preference is_switch_output_1_output_2
    0         2+2         4       10           1                         True
    1         2+3         5        7           1                        False
    >>> (processor.postprocess(processor.preprocess(df)) == df).all(axis=None)
    True
    """

    def __init__(
        self,
        two_columns_to_switch: Sequence[str],
        fn_replace_if_switch=None,
        fn_replace_if_unswitch=None,
        random_seed_columns: Optional[Sequence[str]] = None,
        _switch_column: Optional[str] = None,
        **kwargs,
    ):
        self.two_columns_to_switch = list(set(two_columns_to_switch))
        if len(self.two_columns_to_switch) != 2:
            raise ValueError(
                f"two_columns_to_switch should have exactly two different columns but {two_columns_to_switch}"
            )
        self.fn_replace_if_switch = fn_replace_if_switch or (lambda x: x)
        # by default we assume that it's an involutive function
        self.fn_replace_if_unswitch = fn_replace_if_unswitch or self.fn_replace_if_switch

        # `switch_column` used for backward compatibility
        if _switch_column is None:
            _switch_column = "_".join(["is_switch"] + list(two_columns_to_switch))
        self._switch_column = _switch_column

        if random_seed_columns is None:
            random_seed_columns = two_columns_to_switch
        self.random_seed_columns = sorted(list(random_seed_columns))

        super().__init__(**kwargs)

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """When preprocessing, we select the rows to switch and perform the switch."""
        df_to_annotate = df_to_annotate.copy()

        # randomize order of output_1, output_2 base on inputs
        df_to_annotate[self._switch_column] = df_to_annotate.apply(
            # we add "_switch_column" at the beginning to not use the same seed for all tasks
            lambda x: utils.random_seeded_choice(
                seed=self._switch_column + "".join(x[self.random_seed_columns]) + str(self.seed),
                choices=[False, True],
            ),
            axis=1,
        )
        return self._switch_or_unswitch(df_to_annotate, is_switch=True)

    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """When postprocessing, we undo the switch and remove the switch column."""
        df_annotated = df_annotated.copy()
        df_annotated = self._switch_or_unswitch(df_annotated, is_switch=False)
        df_annotated = df_annotated.drop(columns=[self._switch_column])
        return df_annotated

    @property
    def col1(self):
        return self.two_columns_to_switch[0]

    @property
    def col2(self):
        return self.two_columns_to_switch[1]

    def _switch_or_unswitch(self, df: pd.DataFrame, is_switch: bool) -> pd.DataFrame:
        """Applies the switch to the dataframe. If `is_switch=False` will undo the switch."""

        # switching two columns is an involution => no need to use is_switch here
        col1_values = df[self.col1].copy()
        col2_values = df[self.col2].copy()
        is_switch_arr = df[self._switch_column]
        df[self.col2] = np.where(is_switch_arr, col1_values, col2_values)
        df[self.col1] = np.where(is_switch_arr, col2_values, col1_values)

        if is_switch:
            df.loc[is_switch_arr, :] = self.fn_replace_if_switch(df.loc[is_switch_arr, :])
        else:
            df.loc[is_switch_arr, :] = self.fn_replace_if_unswitch(df.loc[is_switch_arr, :])

        return df


class PaddingForBatchesProcessor(BaseProcessor):
    r"""Pad the dataframe to have a number of examples divisible by `batch_size`.

    Parameters
    ----------
    batch_size : int
        Number of examples to batch in a single prompt.

    padding_example : dict
        Padding example to use if len(df) not divisible by batch_size.

    kwargs :
        Additional arguments to pass to `BaseProcessor`. E.g. seed.

    Examples
    --------
    >>> df = pd.DataFrame({"instruction": ["solve", "write", "other 1"],
    ...                    "input": ["1+1", "'abc'", ""]})
    >>> processor = PaddingForBatchesProcessor(batch_size=2, padding_example=dict(instruction="pad", input="pad_in"))
    >>> processor.preprocess(df)
        instruction   input  is_padding
    0         solve     1+1       False
    1         write   'abc'       False
    2       other 1               False
    3           pad  pad_in        True
    >>> (processor.postprocess(processor.preprocess(df)) == df).all(axis=None)
    True
    """

    def __init__(self, batch_size, padding_example: dict, **kwargs):
        self.batch_size = batch_size
        self.padding_example = padding_example
        super().__init__(**kwargs)

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        # padding if you don't have enough examples
        n_to_pad = (self.batch_size - len(df_to_annotate)) % self.batch_size
        padding = pd.DataFrame([self.padding_example] * n_to_pad)
        padding["is_padding"] = True
        df_out = pd.concat([df_to_annotate, padding], axis=0, ignore_index=True)
        df_out["is_padding"] = df_out["is_padding"].fillna(False)
        return df_out

    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        return df_annotated[~df_annotated["is_padding"].astype(bool)].drop(columns=["is_padding"]).copy()
