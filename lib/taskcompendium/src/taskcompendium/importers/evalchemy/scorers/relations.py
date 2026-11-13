# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""References made only of relations without free symbols, shared by the boxed-math and OlympiadBench scorers.

``math_verify.verify`` matches such a reference to any equation that solves to no value, such as ``x = x``, and
to other relations without free symbols, so the scorers require the reference's own relations instead.
"""

from sympy.core.relational import Relational


def symbol_free_relation(parsed: list) -> bool:
    """Whether a math-verify parse holds only relations, none of them with a free symbol."""
    values = [value for value in parsed if not isinstance(value, str)]
    return bool(values) and all(isinstance(value, Relational) and not value.free_symbols for value in values)


def same_relations(prediction: list, reference: list) -> bool:
    """Whether the prediction holds the reference's relations, each identical or reversed (`a < b` as `b > a`)."""
    values = [value for value in prediction if not isinstance(value, str)]
    relations = [value for value in reference if not isinstance(value, str)]
    return len(values) == len(relations) and all(
        value in (relation, relation.reversed) for value, relation in zip(values, relations, strict=True)
    )
