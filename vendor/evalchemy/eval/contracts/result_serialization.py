"""JSON conversion for Evalchemy result documents."""

from __future__ import annotations

import math
import sys
from typing import Any

from lm_eval.utils import handle_non_serializable

_BIT_CAP = 15_000


def handle_non_serializable_extended(value: Any) -> Any:
    """Serialize lm-eval values without converting huge SymPy numbers to decimal."""
    try:
        from sympy import Integer, Rational

        digit_limit = sys.get_int_max_str_digits()

        if isinstance(value, Integer):
            bits = value.p.bit_length()
            digits = int(bits * math.log10(2)) + 1
            if bits > _BIT_CAP or (digit_limit and digits > digit_limit):
                return f"<Integer ~{digits} digits>"
            return str(int(value))

        if isinstance(value, Rational):
            num_bits = value.p.bit_length()
            den_bits = value.q.bit_length()
            num_digits = int(num_bits * math.log10(2)) + 1
            den_digits = int(den_bits * math.log10(2)) + 1
            if (num_bits > _BIT_CAP or den_bits > _BIT_CAP) or (
                digit_limit and (num_digits > digit_limit or den_digits > digit_limit)
            ):
                return f"<Rational {num_digits}/{den_digits} digits>"
            return str(value)
    except ModuleNotFoundError:
        pass

    return handle_non_serializable(value)
