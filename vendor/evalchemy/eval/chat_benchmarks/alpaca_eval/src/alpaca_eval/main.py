import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import fire
import pandas as pd

from . import analyze, annotators, constants, decoders, metrics, utils
from .types import AnyData, AnyLoadableDF, AnyPath

CUR_DIR = Path(__file__).parent

__all__ = ["evaluate", "evaluate_from_model", "analyze_evaluators", "make_leaderboard"]


def evaluate(
    model_outputs: Optional[AnyLoadableDF] = None,
    reference_outputs: AnyLoadableDF = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
    annotators_config: AnyPath = constants.DEFAULT_ANNOTATOR_CONFIG,
    name: Optional[str] = None,
    output_path: Optional[Union[AnyPath, str]] = "auto",
    precomputed_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
    is_overwrite_leaderboard: bool = False,
    leaderboard_mode_to_print: Optional[Union[str, Sequence[str]]] = "minimal",
    current_leaderboard_mode: str = "community",
    is_return_instead_of_print: bool = False,
    fn_metric: Union[str, callable] = "get_length_controlled_winrate" if constants.IS_ALPACA_EVAL_2 else "get_winrate",
    metric_kwargs: Optional[dict[str, Any]] = None,
    is_recompute_metrics_only: bool = False,
    sort_by: str = "length_controlled_winrate" if constants.IS_ALPACA_EVAL_2 else "win_rate",
    is_cache_leaderboard: Optional[bool] = None,
    max_instances: Optional[int] = None,
    annotation_kwargs: Optional[dict[str, Any]] = None,
    Annotator=annotators.PairwiseAnnotator,
    **annotator_kwargs,
):
    """Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

    Parameters
    ----------
    model_outputs : path or data or dict
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`. If None, we just print the leaderboard.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are a
        specific set of Davinci 003 outputs on the AlpacaEval set:
        https://huggingface.co/datasets/tatsu-lab/alpaca_eval.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file. For details see the docstring of
        `PairwiseAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard. If None we check if `generator is in model_outputs` if not
        we use "Current model".

    output_path : path, optional
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
        If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

    precomputed_leaderboard : path or data, optional
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
        column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
        in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if the model is already in it.

    leaderboard_mode_to_print : {"minimal", "verified", "community", None} or list, optional
        The mode of the leaderboard to use. Only used if the precomputed leaderboard has a column `mode`, in which case
        it will filter the leaderboard by this mode. If None keeps all. If a list, will print all the models in the
        list.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard for the current method.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    fn_metric : str or callable, optional
        The function or function name in `metrics` that will be used to convert preference to metrics. The function
        should take a sequence of dict annotations. Each dict has a preference key (1.5 for draw, 1 for base win,
        2 when the model to compare wins) and return a dictionary of metrics and the key by which to sort the
        leaderboard. Common choices: `get_winrate`, `get_length_controlled_winrate`, `get_length_controlled_elo`.

    metric_kwargs : dict, optional
        Additional arguments to pass to `fn_metric`.

    is_recompute_metrics_only : bool, optional
        Whether to recompute the metrics. Useful if all you want to recompute the metrics without reannotating.

    sort_by : str, optional
        The key by which to sort the leaderboard.

    is_cache_leaderboard : bool, optional
        Whether to save the result leaderboard to `precomputed_leaderboard`. If None we save only if max_instances
        not None. A preferred way of adding models to the leaderboard is to set `precomputed_leaderboard` to the
        previously saved leaderboard at `<output_path>/leaderboard.csv`.

    max_instances : int, optional
        The maximum number of instances to annotate. Useful for testing.

    annotation_kwargs : dict, optional
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

    Annotator : class, optional
        The annotator class to use.

    annotator_kwargs :
        Additional arguments to pass to `PairwiseAnnotator`.
    """
    if (
        isinstance(current_leaderboard_mode, str)
        and current_leaderboard_mode not in constants.ORDERED_LEADERBOARD_MODES
    ):
        raise ValueError(f"current_leaderboard_mode should be one of {constants.ORDERED_LEADERBOARD_MODES}")

    annotation_kwargs = annotation_kwargs or dict()

    leaderboard, precomputed_leaderboard = utils.get_precomputed_leaderboard(
        precomputed_leaderboard, reference_outputs, annotators_config
    )
    annotations = None

    arg_model_outputs = model_outputs
    if model_outputs is not None:
        model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
        reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)
        name = utils.get_generator_name(name, model_outputs)

        if (name not in leaderboard) or is_overwrite_leaderboard or is_recompute_metrics_only:
            logging.info(f"Evaluating the {name} outputs.")

            if not is_recompute_metrics_only:
                leaderboard[name] = {}
                if max_instances is not None:
                    # first we shuffle both outputs with a fix seed => more representative
                    if len(model_outputs) != len(reference_outputs):
                        logging.warning(
                            "model_outputs and reference_outputs have different lengths, so we cannot shuffle before taking the first max_instances."
                        )
                    else:
                        seed = 123
                        model_outputs = model_outputs.sample(frac=1, random_state=seed)
                        reference_outputs = reference_outputs.sample(frac=1, random_state=seed)

                    model_outputs = model_outputs[:max_instances]
                    reference_outputs = reference_outputs[:max_instances]

                annotator = Annotator(annotators_config=annotators_config, **annotator_kwargs)
                annotations = annotator.annotate_head2head(
                    outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs
                )

                leaderboard[name]["mode"] = current_leaderboard_mode
                leaderboard[name]["avg_length"] = int(model_outputs["output"].str.len().mean())

            else:
                # load previously computed annotations so that we can recompute metrics
                assert output_path is not None and name in leaderboard
                output_path = utils.get_output_path(
                    output_path, arg_model_outputs, name, annotators_config=annotators_config
                )
                annotations = pd.read_json(output_path / "annotations.json")

            # Note: I'm using _ to make clear that we may change the annotations in-place. This is bad practice
            # but gives much more control for saving annotations with desired metrics. E.g. that's how we save
            # "glm_preference" in the annotations
            # TODO: change this and use classes
            if isinstance(fn_metric, str):
                fn_metric_ = getattr(metrics, fn_metric)
            else:
                fn_metric_ = fn_metric

            leaderboard[name].update(fn_metric_(annotations, **(metric_kwargs or {})))

        else:
            logging.info(f"Skipping evaluation of {name} as it is already in the precomputed leaderboard.")

    output_path = utils.get_output_path(output_path, arg_model_outputs, name, annotators_config=annotators_config)

    df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(by=sort_by, ascending=False)
    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), ["win_rate", "standard_error"])
    ]

    if output_path is not None:
        logging.info(f"Saving all results to {output_path}")
        df_leaderboard.to_csv(output_path / "leaderboard.csv")
        if annotations is not None:
            utils.convert_to_dataframe(annotations).to_json(
                output_path / "annotations.json", orient="records", indent=2
            )

    if is_cache_leaderboard is None:
        is_cache_leaderboard = max_instances is None

    if is_cache_leaderboard:
        if isinstance(precomputed_leaderboard, AnyPath):
            logging.info(f"Saving result to the precomputed leaderboard at {precomputed_leaderboard}")
            df_leaderboard.to_csv(precomputed_leaderboard)
        else:
            logging.info(
                f"Not saving the result to the cached leaderboard because precomputed_leaderboard is not a "
                f"path but {type(precomputed_leaderboard)}."
            )

    if is_return_instead_of_print:
        return df_leaderboard, annotations
    else:
        utils.print_leaderboard(
            df_leaderboard,
            leaderboard_mode_to_print,
            current_name=name,
            cols_to_print=[sort_by, "win_rate", "standard_error", "n_total", "avg_length"],
        )


def evaluate_from_model(
    model_configs: Union[AnyPath, dict],
    reference_model_configs: Optional[Union[AnyPath, dict]] = None,
    evaluation_dataset: AnyLoadableDF = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
    annotators_config: AnyPath = constants.DEFAULT_ANNOTATOR_CONFIG,
    output_path: AnyPath = "auto",
    max_instances: int = None,
    is_strip_output: bool = True,
    is_load_outputs: bool = True,
    chunksize: int = 64,
    **kwargs,
):
    """Evaluate a model from HuggingFace or an API provider. This is a wrapper around `evaluate` which includes
    generating from
    a desired model.

    Parameters
    ----------
    model_configs : path or dict
        A dictionary or path (relative to `models_configs`) to a yaml file containing the configuration of the model to
        decode from. If a directory,we search for 'configs.yaml' in it. The keys in the first dictionary should be the
        generator's name, and the value should be a dictionary of the generator's configuration which should have the
        following keys:
        - prompt_template (str): a prompt template or path to one. Each template should contain placeholders for
        keys in the data dictionary, typically {instruction} and {output}.
        - fn_completions (str): function in `alpaca_farm.decoders` for completions. Needs to accept as first argument
            `prompts` which is a list of string.
        - completions_kwargs (dict): kwargs for fn_completions. E.g. model_name, max_tokens, temperature...

    reference_model_configs : path or dict, optional
        Same as in `model_configs` but for the reference model. If None, we use the default Davinci003 outputs.

    evaluation_dataset : path or callable, optional
        Path to the evaluation dataset or a function that returns a dataframe. If None, we use the default evaluation

    annotators_config : path or dict, optional
        Path to the annotators configuration or a dictionary. If None, we use the default annotators configuration.

    output_path : path, optional
        Path to save the generations, annotations and leaderboard. If auto saves at `results/<model_name>`

    max_instances : int, optional
        Maximum number of instances to generate and evaluate. If None, we evaluate all instances.

    is_strip_output : bool, optional
        Whether to strip trailing and leading whitespaces from the outputs.

    is_load_outputs : bool, optional
        Whether to try to load outputs from the output path. If True and outputs exist we only generate outputs for
        instructions that don't have outputs yet.

    chunksize : int, optional
        Number of instances to generate before saving. If None, we save after all generations.

    kwargs:
        Other kwargs to `evaluate`
    """
    df_dataset = utils.load_or_convert_to_dataframe(evaluation_dataset)

    if chunksize is not None and not is_load_outputs:
        logging.info("`is_load_outputs` has to be true to use chunksize. Setting it to True.")
        is_load_outputs = True

    if chunksize is not None and max_instances is not None:
        logging.info("cannot use `chunksize` with max_instances. Setting `chunksize` to None.")
        chunksize = None

    base_dir = Path(kwargs.get("base_dir", constants.MODELS_CONFIG_DIR))
    model_configs = utils.load_configs(model_configs, relative_to=base_dir)
    if reference_model_configs is not None:
        reference_model_configs = utils.load_configs(reference_model_configs, relative_to=base_dir)

    if output_path == "auto":
        output_path = Path("results") / list(model_configs.keys())[0]
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

    def get_completions(configs, df: pd.DataFrame, old_output_path: Optional[Path] = None):
        columns_to_keep = ["dataset", "instruction", "output", "generator"]
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        curr_outputs = df[columns_to_keep].copy()
        is_loading_old_outputs = old_output_path is not None and old_output_path.exists()
        assert len(configs) == 1
        generator = list(configs.keys())[0]
        configs = list(configs.values())[0]

        if is_loading_old_outputs:
            logging.info(f"Loading outputs from {old_output_path}")
            old_outputs = utils.load_or_convert_to_dataframe(old_output_path)
            # select only rows in curr_outputs that have "instruction" that are not in old_outputs
            idx_found_old_outputs = curr_outputs["instruction"].isin(old_outputs["instruction"])
            curr_outputs = curr_outputs[~idx_found_old_outputs]
            assert (old_outputs["generator"] == generator).all()
            logging.info(f"Found {len(old_outputs)}. Only generating {len(curr_outputs)} .")

        if max_instances is not None:
            curr_outputs = curr_outputs.iloc[:max_instances]

        if len(curr_outputs) > 0:
            prompts, _ = utils.make_prompts(
                curr_outputs,
                template=utils.read_or_return(configs["prompt_template"], relative_to=base_dir),
            )
            fn_completions = decoders.get_fn_completions(configs["fn_completions"])
            completions = fn_completions(prompts=prompts, **configs["completions_kwargs"])["completions"]
            if is_strip_output:
                completions = [c.strip() for c in completions]
            curr_outputs["output"] = completions
            curr_outputs["generator"] = generator

        if is_loading_old_outputs:
            curr_outputs = pd.concat([old_outputs, curr_outputs], axis=0)

        return curr_outputs

    for df_chunk in utils.dataframe_chunk_generator(
        df_dataset, chunksize=chunksize, tqdm_desc="Chunking for generation"
    ):
        if is_load_outputs and output_path is not None:
            model_outputs = get_completions(
                model_configs, df=df_chunk, old_output_path=output_path / "model_outputs.json"
            )
        else:
            model_outputs = get_completions(model_configs, df=df_chunk)

        if reference_model_configs is None:
            if "output" not in df_chunk.columns:
                raise ValueError("evaluation_dataset should have a column 'output' containing references outputs")
            reference_outputs = df_dataset.copy()
        else:
            reference_outputs = get_completions(
                reference_model_configs,
                df=df_chunk,
                old_output_path=output_path / "reference_outputs.json",
            )

        if output_path is not None:
            model_outputs.to_json(output_path / "model_outputs.json", orient="records", indent=2)
            reference_outputs.to_json(output_path / "reference_outputs.json", orient="records", indent=2)

    if reference_model_configs is None:
        # using a default reference outputs => uses the right leaderboard
        if evaluation_dataset in [constants.ALPACAEVAL_REFERENCE_OUTPUTS]:
            reference_outputs = evaluation_dataset

    return evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        annotators_config=annotators_config,
        output_path=output_path,
        max_instances=max_instances,
        **kwargs,
    )


def make_leaderboard(
    leaderboard_path: Optional[AnyPath] = None,
    annotators_config: AnyPath = constants.DEFAULT_ANNOTATOR_CONFIG,
    all_model_outputs: AnyLoadableDF = constants.ALPACAFARM_ALL_OUTPUTS,
    reference_outputs: AnyLoadableDF = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
    fn_add_to_leaderboard: Callable = "evaluate",
    leaderboard_mode: str = "verified",
    is_return_instead_of_print: bool = False,
    **kwargs,
):
    """Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.

    Parameters
    ----------
    leaderboard_path : path
        The path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will
        append

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    all_model_outputs : path or data or callable, optional
        The outputs of all models to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv potentially with globbing) or a function to generate
        those. If the path contains a globbing pattern, we will read all files matching the pattern and concatenate
        them. Each dictionary (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by
        default `instruction` and `output` with optional `input`. It should also contain a column `generator` with the
        name of the current model. Could also be a list of the above, in which case the output is the concatenation.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `all_model_outputs` but without needing `generator`. By
        default, the reference outputs are the 003 outputs on AlpacaEval set.

    fn_add_to_leaderboard : callable or str, optional
        The function to use to add a model to the leaderboard. If a string, it should be the name of a function in
        `main.py`. The function should take the arguments: `model_outputs`, `annotators_config`, `name`,
        `precomputed_leaderboard`, `is_return_instead_of_print`, `reference_outputs`.

    leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to save all new entries with.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    kwargs :
        Additional arguments to pass to `fn_add_to_leaderboard`.
    """
    if isinstance(fn_add_to_leaderboard, str):
        fn_add_to_leaderboard = globals()[fn_add_to_leaderboard]

    if leaderboard_path is None:
        assert isinstance(annotators_config, str) and "/" not in annotators_config, (
            "If `leaderboard_path` is None, `annotators_config` should be a string with the name of the annotator "
            "configuration."
        )
        leaderboard_path = Path(constants.ALPACAEVAL_LEADERBOARD_PATHS) / f"{annotators_config}_leaderboard.csv"

    Path(leaderboard_path).parent.mkdir(exist_ok=True, parents=True)
    all_model_outputs = utils.load_or_convert_to_dataframe(all_model_outputs)
    if "generator" not in all_model_outputs.columns:
        raise ValueError(f"all_model_outputs should have a column 'generator' with the name of the model.")

    all_annotations = []
    for model in all_model_outputs["generator"].unique():
        model_outputs = all_model_outputs[all_model_outputs["generator"] == model]
        df_leaderboard, annotations = fn_add_to_leaderboard(
            model_outputs=model_outputs,
            reference_outputs=reference_outputs,
            annotators_config=annotators_config,
            name=model,
            precomputed_leaderboard=leaderboard_path,
            is_return_instead_of_print=True,
            current_leaderboard_mode=leaderboard_mode,
            **kwargs,
        )
        if annotations is not None:
            all_annotations += annotations
        df_leaderboard.to_csv(leaderboard_path)

    leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
    df_leaderboard = pd.DataFrame(leaderboard)

    if is_return_instead_of_print:
        return df_leaderboard, all_annotations
    else:
        utils.print_leaderboard(
            df_leaderboard, leaderboard_mode=None, cols_to_print=["win_rate", "standard_error", "n_total"]
        )


def analyze_evaluators(
    annotators_config: Optional[AnyPath] = constants.DEFAULT_ANNOTATOR_CONFIG,
    Annotator=annotators.PairwiseAnnotator,
    analyzer_kwargs: Optional[dict] = None,
    precomputed_leaderboard: Optional[Union[AnyPath, AnyData]] = CUR_DIR
    / "leaderboards/evaluators/evaluators_leaderboard.csv",
    is_save_leaderboard: bool = False,
    is_return_instead_of_print: bool = False,
    is_overwrite_leaderboard: bool = False,
    max_instances: Optional[int] = None,
    is_single_annotator: bool = False,
    leaderboard_mode_to_print: str = "minimal",
    current_leaderboard_mode: str = "minimal",
    output_path: Optional[Union[AnyPath, str]] = "auto",
    **annotator_kwargs,
):
    """Analyze an evaluator and populates the evaluators leaderboard (agreement with human, speed, price,...).

    Parameters
    ----------
    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    Annotator : class, optional
        The annotator class to use.

    analyzer_kwargs : dict, optional
        Additional arguments to pass to the analyzer.

    precomputed_leaderboard : path or data, optional
        The precomputed (meta)leaderboard of annotators or a path to it (json, csv, or tsv).

    is_save_leaderboard : bool, optional
        Whether to save the leaderboard (ie analyzed results).

    is_return_instead_of_print : bool, optional
        Whether to return the leaderboard (ie analyzed results). If True, it will not print the results.

    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if it already exists.

    max_instances : int, optional
        The maximum number of instances to analyze.

    is_single_annotator : bool, optional
        Whether to analyze a single annotator. If True, will not be able to estimate the annotator's bias.

    leaderboard_mode_to_print : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to print.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to save all new entries with.

    output_path : path, optional
        Path to save the leaderboard and annotataions. If None, we don't save.

    annotator_kwargs :
        Additional arguments to pass to `Annotator`.
    """
    leaderboard = dict()
    if precomputed_leaderboard is not None:
        try:
            leaderboard = utils.load_or_convert_to_dataframe(precomputed_leaderboard).to_dict(orient="index")
        except FileNotFoundError:
            logging.warning(
                f"Could not find precomputed leaderboard at {precomputed_leaderboard}. Starting from " f"scratch."
            )

    analyzer_kwargs = analyzer_kwargs or {}

    all_crossannotations = dict()
    key = None
    if annotators_config is not None:
        key = annotators_config.replace("/", "_").replace("_configs.yaml", "")
        if key not in leaderboard or is_overwrite_leaderboard:
            analyzer = analyze.Analyzer(**analyzer_kwargs)

            if key == "humans":
                df_crossannotations = analyzer.df_gold_crossannotations
            elif key == "longest":
                df_crossannotations = analyze._get_longest_predictor(analyzer.df_gold_crossannotations)
            else:
                annotator_kwargs = annotator_kwargs or {}
                df_crossannotations = analyze.get_crossannotations(
                    analyzer=analyzer,
                    Annotator=Annotator,
                    max_instances=max_instances,
                    annotators_config=annotators_config,
                    is_single_annotator=is_single_annotator,
                    **annotator_kwargs,
                )

            leaderboard[key] = analyze.get_metrics_evaluator(analyzer, df_crossannotations, evaluator_name=key)
            leaderboard[key]["mode"] = current_leaderboard_mode
            all_crossannotations[key] = df_crossannotations

    df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(
        by="Human agreement", ascending=False
    )

    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), constants.EVALUATORS_LEADERBOARD_COLS_TO_PRIORITIZE)
    ]

    if is_save_leaderboard:
        df_leaderboard.to_csv(precomputed_leaderboard)

    if key is not None and output_path is not None:
        output_path = utils.get_output_path(output_path, annotators_config, key, dflt_dir="results_evaluators")
        if isinstance(annotators_config, str) and "/" not in annotators_config:
            output_path = Path(output_path) / annotators_config
            output_path.mkdir(exist_ok=True, parents=True)
        logging.info(f"Saving all results to {output_path}")
        df_leaderboard.to_csv(output_path / f"leaderboard.csv")
        for annotator_name, df_crossannotations in all_crossannotations.items():
            annotations_name = f"annotation.json"
            df_crossannotations.to_json(output_path / annotations_name, orient="records", indent=2)

    if is_return_instead_of_print:
        return df_leaderboard, all_crossannotations
    else:
        utils.print_leaderboard(
            df_leaderboard, leaderboard_mode_to_print, cols_to_print=constants.EVALUATORS_LEADERBOARD_COLS_TO_PRINT
        )


ALL_FUNCTIONS = {
    "evaluate": evaluate,
    "evaluate_from_model": evaluate_from_model,
    "make_leaderboard": make_leaderboard,
    "analyze_evaluators": analyze_evaluators,
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(evaluate)


if __name__ == "__main__":
    fire.Fire(ALL_FUNCTIONS)
