import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from .. import completion_parsers, constants, processors, types, utils

__all__ = ["SingleAnnotator"]


class SingleAnnotator:
    """A helper class for a single auto annotator.

    Parameters
    ----------
    prompt_template : str or path
        A prompt template that will be given to `fn_prompter` or path to those prompts. Path is relative to
        `evaluators_configs/`

    fn_completion_parser : callable or str
        Function that maps (parses) the completion to a list of annotations. If a string, it should be a function in
        `completion_parsers.py` to use for parsing the completions into annotations. For each completion, the number of
        annotations (lenght of list) should be equal to the batch_size if not we set all the annotations in that batch
        to NaN.

    completion_parser_kwargs : dict
        Kwargs for fn_completion_parser.

    fn_completions : callable, optional
        Explicit completion callback for native annotation parity checks.

    completions_kwargs : dict
        kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

    is_shuffle : bool
        Whether to shuffle the order of the examples before making the prompt. Useful if batch_size > 1.

    seed : int
        Seed for randomization.

    batch_size : int
        Number of examples that will be added in a single prompt.

    base_dir : Path, optional
        Path to the directory containing the annotators configs. I.e. annotators_config will be relative
        to this directory.

    annotation_column : str, optional
        Name of the annotation column in the output dataframe.

    completion_column : str, optional
        Name of the raw completion column in the output dataframe. If None will not store the raw completions. Note that
        raw_completion will not be modified by the postprocessors. E.g. if we switch the columns output_1 and output_2
        in the prompt then the raw completion will show the switched order, which makes interpretation harder. This
        should nevertheless not be an issue when using reapply_parsing because of seeding.

    processors_to_kwargs : Sequence[dict(str, dict)], optional
        A dictionary of BaseProcessor objects to apply for preprocessing the  dataframe before making the prompts and
        prostprocessing after anntoations. The key should be the names of the BaseProcessor objectsto use in
        `processors.py` the values are the kwargs for the constructor of the Processor. Order matters.

    is_add_default_processors : bool, optional
        Whether to add the default processors to the list of processors.

    completion_key : str, optional
        Key of the output of `fn_completions` to use for parsing the completions into annotations.

    packages_for_which_to_show_version : Sequence[str], optional
        List of packages for which to show the version in the metadata of the completions.
    """

    def __init__(
        self,
        prompt_template: types.AnyPath,
        fn_completion_parser: Optional[Union[Callable, str]] = None,
        completion_parser_kwargs: Optional[dict[str, Any]] = None,
        fn_completions: Optional[Callable] = None,
        completions_kwargs: Optional[dict[str, Any]] = None,
        is_shuffle: bool = True,
        seed: Optional[int] = 123,
        batch_size: int = 1,
        base_dir: types.AnyPath = constants.EVALUATORS_CONFIG_DIR,
        annotation_column: str = "annotation",
        completion_column: Optional[str] = "raw_completion",
        processors_to_kwargs: Optional[dict[str, dict]] = None,
        is_add_default_processors: bool = True,
        completion_key: str = "completions",
        packages_for_which_to_show_version: Optional[Sequence[str]] = ("alpaca_eval",),
        prfx_to_completion_cols: Optional[str] = "{annotation_column}_",
        # The following two keys are only for the documentation
        pretty_name: Optional[str] = None,
        link: Optional[str] = None,
    ):
        self.base_dir = Path(base_dir)
        self.prompt_template = self._get_prompt_template(prompt_template)

        if fn_completion_parser is None:
            fn_completion_parser = lambda x: [x]
        elif isinstance(fn_completion_parser, str):
            fn_completion_parser = self._search_fn_completion_parser(fn_completion_parser)
        completion_parser_kwargs = completion_parser_kwargs or {}
        self.fn_completion_parser = partial(fn_completion_parser, **completion_parser_kwargs)

        self.fn_completions = fn_completions
        self.completions_kwargs = completions_kwargs or {}
        self.seed = seed
        self.is_shuffle = is_shuffle
        self.batch_size = batch_size
        self.annotation_column = annotation_column
        self.completion_column = completion_column
        self.packages_for_which_to_show_version = packages_for_which_to_show_version
        if prfx_to_completion_cols is None:
            prfx_to_completion_cols = ""
        self.prfx_to_completion_cols = prfx_to_completion_cols.format(annotation_column=annotation_column)

        self.is_add_default_processors = is_add_default_processors
        self.processors = []
        self.completion_key = completion_key
        processors_to_kwargs = processors_to_kwargs or {}
        if (
            batch_size > 1
            and self.is_add_default_processors
            and "PaddingForBatchesProcessor" not in processors_to_kwargs
        ):
            processors_to_kwargs["PaddingForBatchesProcessor"] = {
                "batch_size": batch_size,
                "padding_example": utils.DUMMY_EXAMPLE,
            }
        for processor, processor_kwargs in processors_to_kwargs.items():
            processor_kwargs["seed"] = self.seed
            processor_kwargs["annotation_column"] = self.annotation_column
            processor_kwargs["completion_column"] = self.completion_column
            Processor = self._search_processor(processor)
            self.processors += [Processor(**processor_kwargs)]

    ### Public methods ###
    def __call__(self, df_to_annotate: pd.DataFrame, **decoding_kwargs) -> pd.DataFrame:
        """Annotates the given examples.

        Parameters
        ----------
        df_to_annotate : pd.DataFrame
            Examples to annotate

        decoding_kwargs :
            Additional arguments to pass to `fn_completions`.
        """
        df_to_annotate = df_to_annotate.copy()  # avoid in place modifications

        if df_to_annotate.empty:
            df_to_annotate[self.annotation_column] = []
            return df_to_annotate

        df_to_annotate = self._preprocess(df_to_annotate)

        # the following only reapplies the parsing in case you already stored the raw completions. requires batch_size=1
        if self.completion_column in df_to_annotate.columns and self.batch_size == 1:
            # keep only the rows that have not been annotated yet
            main_df_to_annotate = df_to_annotate
            idx_not_completed = df_to_annotate[self.completion_column].isna()
            df_to_annotate = df_to_annotate[idx_not_completed].copy()

        if not df_to_annotate.empty:
            # prompts and completions here will not be the same length as the dataframe due to batching
            prompts, df_to_annotate = self._make_prompts(df_to_annotate)
            if self.fn_completions is None:
                raise ValueError("Annotation execution requires an explicit completion callback")
            completions = self.fn_completions(prompts=prompts, **self.completions_kwargs, **decoding_kwargs)
            self._add_metadata_to_completions_(completions)
            completions = {
                f"{self.prfx_to_completion_cols}{k}" if k != self.completion_key else k: v
                for k, v in completions.items()
            }

            for k, v in completions.items():
                if k != self.completion_key:
                    if self.batch_size != 1 and (len(df_to_annotate) == len(v) * self.batch_size):
                        v = [el for el in v for _ in range(self.batch_size)]
                    df_to_annotate[k] = v
                    if "per_example" in k:
                        df_to_annotate[k] = df_to_annotate[k] / self.batch_size

        # the following is only needed if you want to only reapply the parsing
        if self.completion_column in df_to_annotate.columns:
            if not df_to_annotate.empty:
                df_to_annotate[self.completion_column] = completions[self.completion_key]  # only works for bs 1
            main_df_to_annotate[idx_not_completed] = df_to_annotate  # puts back all the new completions
            df_to_annotate = main_df_to_annotate
            completions_to_parse = df_to_annotate[self.completion_column]
        else:
            completions_to_parse = completions[self.completion_key]

        # note: reparsing only works if you use the same completion_key
        annotations_to_save, completions_to_save = self._parse_completions(completions=completions_to_parse)
        df_to_annotate[self.annotation_column] = annotations_to_save
        if self.completion_column is not None:
            df_to_annotate[self.completion_column] = completions_to_save

        df_annotated = self._postprocess(df_to_annotate)

        return df_annotated

    ######################

    ### Private methods ###
    def _search_fn_completion_parser(self, name: str) -> Callable:
        """Search for a completion parser by name."""
        return utils.get_module_attribute(completion_parsers, name)

    def _search_processor(self, name: Union[str, Type["processors.BaseProcessor"]]) -> Type["processors.BaseProcessor"]:
        """Search for a Processor class by name."""
        if isinstance(name, str):
            return utils.get_module_attribute(processors, name)
        else:
            assert issubclass(name, processors.BaseProcessor)
            return name

    def _get_prompt_template(self, prompt_template: types.AnyPath):
        return utils.read_or_return(prompt_template, relative_to=self.base_dir)

    def _make_prompts(
        self, df_to_annotate: pd.DataFrame, prompt_template: Optional[str] = None
    ) -> tuple[list[str], pd.DataFrame]:
        """Make all the prompts for the given examples.

        Parameters
        ----------
        df_to_annotate : pd.DataFrame
            Examples to annotate

        prompt_template : str
            Template to use for the prompt. If None, use the one from the constructor.

        Returns
        -------
        prompts : list[str]
            Formatted prompts for the given examples.

        df_to_annotate : pd.DataFrame
            Examples to annotate in the same order as prompts.
        """
        if prompt_template is None:
            prompt_template = self.prompt_template
        return utils.make_prompts(df=df_to_annotate, template=prompt_template, batch_size=self.batch_size)

    def _add_metadata_to_completions_(self, completions: dict[str, Any]):
        """Add metadata to the completions."""
        completions["date"] = datetime.now().isoformat()
        if self.packages_for_which_to_show_version is not None:
            completions["version"] = utils.get_multi_package_version(self.packages_for_which_to_show_version)

    def _preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the examples before annotating. In particular, takes care of all the randomization."""

        for processor in self.processors:
            df_to_annotate = processor.preprocess(df_to_annotate)

        if self.is_shuffle:
            df_to_annotate = df_to_annotate.sample(frac=1, random_state=self.seed)

        return df_to_annotate

    def _parse_completions(self, completions: list[str]) -> tuple[list[Any], list[Any]]:
        """Converts the completions into annotations."""
        all_annotations = []
        all_completions = []
        for completion in completions:
            try:
                batch_annotations = self.fn_completion_parser(completion)
                batch_annotations = list(batch_annotations)

                if len(batch_annotations) != self.batch_size:
                    logging.warning(
                        f"Found {len(batch_annotations)} annotations in:'''\n{completion}\n''' but expected"
                        f" {self.batch_size}. We are setting all annotations to None."
                    )
                    batch_annotations = [None] * self.batch_size

            except Exception as e:
                logging.exception(f"Error while parsing completion: '''\n{completion}\n'''")
                batch_annotations = [None] * self.batch_size

            all_annotations += batch_annotations

            all_completions += [completion] * self.batch_size
        return all_annotations, all_completions

    def _postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the annotated examples."""

        arr_is_na = df_annotated[self.annotation_column].isna()
        if arr_is_na.any():
            logging.warning(
                f"{arr_is_na.sum().item()} samples had no auto annotation. We are filtering them for now. "
                f"If you are using chain of thought it might be that max_tokens limit is too low. "
            )
            df_annotated = df_annotated[~arr_is_na]

        for processor in self.processors[::-1]:  # postprocess in reverted order => no interactions between processors
            df_annotated = processor.postprocess(df_annotated)

        return df_annotated

    #######################
