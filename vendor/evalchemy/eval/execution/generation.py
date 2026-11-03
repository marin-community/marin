"""Merge generation intent using the caller's explicit settings last."""

from evalchemy_config.limits import MAX_OUTPUT_ALIASES, resolve_limit


def canonical_generation_kwargs(parameters):
    """Normalize external adapter spellings before merging source/run intent."""
    parameters = dict(parameters)
    aliases = (*MAX_OUTPUT_ALIASES, "max_completion_tokens")
    maximum = resolve_limit("max_tokens", [(key, parameters.pop(key)) for key in aliases if key in parameters])
    if maximum is not None:
        parameters["max_gen_toks"] = maximum
    return parameters


def generation_parameters(defaults, overrides):
    parameters = {**defaults, **overrides}
    if "temperature" in overrides and "do_sample" not in overrides:
        parameters["do_sample"] = float(overrides["temperature"]) > 0
    return parameters
