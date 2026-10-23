from typing import Callable, Optional, Sequence, Union

import pandas as pd

from .. import utils
from .base import SingleAnnotator

__all__ = ["SinglePairwiseAnnotator"]


class SinglePairwiseAnnotator(SingleAnnotator):
    __doc__ = (
        SingleAnnotator.__doc__.replace(
            "A helper class for a single auto annotator.",
            "A helper class for a single pairwise auto annotator.",
        )
        + """
    is_randomize_output_order : bool
        Whether to randomize output_1, output_2 when formatting.
        
    random_seed_keys : str
        The column to use to seed the randomization of output_1, output_2.
    """
    )

    def __init__(
        self,
        *args,
        annotation_column: str = "preference",
        random_seed_column: Sequence[str] = ("instruction",),
        processors_to_kwargs: Optional[dict[str, dict]] = None,
        is_randomize_output_order: bool = True,
        fn_completion_parser: Optional[Union[Callable, str]] = "ranking_parser",
        **kwargs,
    ):
        processors_to_kwargs = processors_to_kwargs or {}
        self.is_randomize_output_order = is_randomize_output_order
        if is_randomize_output_order:

            def _fn_replace_if_switch(df: pd.DataFrame) -> pd.DataFrame:
                # applies to annotation_column (preference) 3-x => 2 becomes 1, 1 becomes 2 and everything in between if flaot
                if df.empty or self.annotation_column not in df.columns:
                    return df
                df = df.copy()
                df[self.annotation_column] = df[self.annotation_column].apply(lambda x: 3 - x)
                return df

            # swith output columns by default
            processors_to_kwargs["RandomSwitchTwoColumnsProcessor"] = dict(
                two_columns_to_switch=["output_1", "output_2"],
                fn_replace_if_switch=_fn_replace_if_switch,
                random_seed_columns=random_seed_column,
                _switch_column="is_switched_outputs",  # backward compatibility
            )

        super().__init__(
            *args,
            annotation_column=annotation_column,
            processors_to_kwargs=processors_to_kwargs,
            fn_completion_parser=fn_completion_parser,
            **kwargs,
        )
        self.random_seed_column = list(random_seed_column)

    def _postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        df_annotated = super()._postprocess(df_annotated)

        all_values = df_annotated[self.annotation_column]
        all_values = all_values[~all_values.isna()]
        assert all_values.apply(utils.validate_alpacaeval_preference, is_allow_nan=True).all()

        return df_annotated
