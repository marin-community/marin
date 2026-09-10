# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron instruction-following IFEval constraint tasks.

``tests/verifier_data.json`` carries ``instruction_id_list`` (IFEval instruction ids such as
``keywords:letter_frequency``) and a parallel ``kwargs`` list of per-instruction parameters. The
old grader re-implemented the IFEval checks inline; the tool's own registry
(``tasktrove_verify.modes.ifeval_constraints.CONSTRAINTS``) already covers every id this source
uses, so the task maps straight onto ``IfevalSpec.constraints``.
"""

from tasktrove_verify.modes.ifeval_constraints import CONSTRAINTS
from tasktrove_verify.spec import Constraint, IfevalSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import metadata, verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles


def convert_nemotron_ifeval(task: TaskFiles) -> ConvertedTask | Rejected:
    """IFEval constraint checks: ``{"instruction_id_list": [...], "kwargs": [...]}``."""
    data = verifier_data(task)
    instruction_ids = data.get("instruction_id_list", [])
    if not isinstance(instruction_ids, list) or not instruction_ids:
        return Rejected(ConvertStatus.NULL_GRADER, "empty instruction_id_list")
    kwargs_list = data.get("kwargs", [])
    if not isinstance(kwargs_list, list):
        return Rejected(ConvertStatus.NULL_GRADER, f"kwargs is not a list: {type(kwargs_list).__name__}")
    kwargs_list = kwargs_list + [{}] * (len(instruction_ids) - len(kwargs_list))
    if not all(isinstance(name, str) for name in instruction_ids):
        return Rejected(ConvertStatus.NULL_GRADER, "instruction_id_list has a non-string entry")
    unsupported = sorted({name for name in instruction_ids if name not in CONSTRAINTS})
    if unsupported:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"constraints the mode does not implement: {unsupported}")
    constraints = tuple(
        Constraint(name=name, params=dict(params) if isinstance(params, dict) else {})
        for name, params in zip(instruction_ids, kwargs_list, strict=True)
    )
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=IfevalSpec(constraints=constraints),
        dockerfile=task.text(DOCKERFILE),
        tags=("instruction-following", "ifeval", "nemotron"),
        metadata=metadata(task),
    )


CONVERTER = Converter(
    name="nemotron_ifeval",
    keys=(ConverterKey("instruction-following", frozenset({"tests/test.sh", "tests/verifier.py"})),),
    convert=convert_nemotron_ifeval,
)
