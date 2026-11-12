"""AlpacaEval prompt processors and parsers, without its execution driver."""

from functools import lru_cache

from eval.contracts.eval_spec import Grade, GraderSpec, JudgePlan, ModelRequest, RequestKind


@lru_cache
def annotation_config(name):
    from alpaca_eval import constants, utils

    configs = utils.load_configs(name, relative_to=constants.EVALUATORS_CONFIG_DIR)
    if len(configs) != 1:
        raise ValueError("an AlpacaEval source must select one annotator")
    annotator, config = next(iter(configs.items()))
    return annotator, config


def annotator_for(name):
    from alpaca_eval.annotators import SinglePairwiseAnnotator

    _, configuration = annotation_config(name)
    config = dict(configuration)
    config["batch_size"] = 1
    # The native corpus annotator supplies zero instead of SingleAnnotator's 123.
    config["seed"] = 0
    return SinglePairwiseAnnotator(**config)


def annotation_grade(name, annotation, responses):
    """Retain native corpus rows and distinguish missing preferences from scores."""
    annotation = {**annotation, "annotator": annotation_config(name)[0]}
    preference = annotation["preference"]
    metrics = {"judge_failed": float(preference is None)}
    if preference is not None:
        metrics["preference"] = float(preference)
    return Grade(metrics, {"alpaca_annotation": annotation, "judge_raw": [str(response) for response in responses]})


def alpaca(spec, responses):
    import pandas as pd
    from alpaca_eval import utils

    name = spec.grader.reference["config"]
    annotation = {
        "index": spec.ordinal,
        "instruction": spec.doc["instruction"], "output_1": spec.target, "output_2": str(responses[-1]),
        "generator_1": spec.grader.reference["baseline"], "generator_2": "candidate",
    }
    if annotation["output_1"] == annotation["output_2"]:
        return annotation_grade(name, {**annotation, "preference": 1.5}, ())
    annotator = annotator_for(name)
    frame = pd.DataFrame([annotation])
    frame = annotator._preprocess(frame)
    prompts, frame = annotator._make_prompts(frame)
    settings = dict(annotator.completions_kwargs)
    settings["model"] = settings.pop("model_name")
    requests = tuple(ModelRequest(RequestKind.GENERATE, utils.prompt_to_chatml(prompt), settings) for prompt in prompts)
    return JudgePlan(requests, GraderSpec("alpaca_preference", {
        "config": name, "frame": frame.to_dict("records"), "annotation": annotation,
    }))


def alpaca_preference(spec, responses):
    import pandas as pd

    name = spec.grader.reference["config"]
    annotator = annotator_for(name)
    completions = [response.response.raw_choice if annotator.completion_key == "completions_all" else str(response)
                   for response in responses]
    preferences, _ = annotator._parse_completions(completions)
    if pd.isna(preferences[0]):
        # Native postprocessing drops missing preferences. Retain the original,
        # unswitched row so the corpus reducer can exclude it and report coverage.
        return annotation_grade(name, {**spec.grader.reference["annotation"], "preference": None}, responses)
    frame = pd.DataFrame(spec.grader.reference["frame"])
    frame["preference"] = preferences
    frame = annotator._postprocess(frame)
    return annotation_grade(name, frame.to_dict("records")[0], responses)
