# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grade_kit: strict, stdlib-only building blocks for a code grader (grade.py).

Ship it next to grade.py as the private grader file ``grade_kit.py``. ``python3 /tests/grade.py`` puts
``/tests`` first on ``sys.path``, so ``from grade_kit import ...`` works in the verifier machine with no path
setup. Or paste the functions you use into grade.py. It needs nothing beyond CPython 3.12's standard library.

Every reader and validator here is strict by default and raises ``GradeError`` on anything the contract does
not allow: no case folding, no stripping, no coercion of strings to numbers, no fallback channel. Leniency is
an explicit argument, and the instruction must state the same choice word for word. ``Checks`` turns a
``GradeError`` inside a check into a failed check with its reason, so a malformed answer scores low and the
script still writes its verdict. ``run`` writes a verdict for every candidate.

Paths default to the verifier machine's (``/app/answer.txt``, ``/tests``, ``/logs/verifier/reward.json``).
Setting ``GRADE_KIT_ROOT`` to a directory prefixes them, which is how ``grade_local.py`` runs a grader in your
sandbox. Never set it in a task.
"""

import json
import math
import os
import re
import sys
import traceback
from collections import Counter
from datetime import date
from decimal import Decimal
from fractions import Fraction

VERSION = "d3-v2"


class GradeError(ValueError):
    """The candidate violates the answer contract at ``path`` (a dotted location such as ``rows[2].id``)."""

    def __init__(self, path, reason):
        super().__init__(f"{path}: {reason}" if path else reason)
        self.path = path
        self.reason = reason


# ------------------------------------------------------------------------------------------- paths


def _root():
    return os.environ.get("GRADE_KIT_ROOT", "")


def machine_path(path):
    """``path`` (absolute, as the verifier machine sees it) under ``GRADE_KIT_ROOT`` when that is set."""
    if not path.startswith("/"):
        raise ValueError(f"machine paths are absolute: {path!r}")
    return _root() + path


ANSWER_PATH = "/app/answer.txt"
TESTS_DIR = "/tests"
VERDICT_PATH = "/logs/verifier/reward.json"


def read_reply():
    """The solver's final reply as text; "" when there is none. One leading BOM is dropped, nothing else."""
    try:
        with open(machine_path(ANSWER_PATH), encoding="utf-8-sig") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""


def read_answer_file(path):
    """The text of the answer file the solver wrote at ``path`` (an absolute machine path such as
    ``/workspace/answer.json``). A missing file is a GradeError: the answer channel is this file, and the
    reply is never consulted in its place."""
    try:
        with open(machine_path(path), encoding="utf-8-sig") as handle:
            return handle.read()
    except FileNotFoundError:
        raise GradeError(path, "the answer file was not written") from None
    except UnicodeDecodeError as error:
        raise GradeError(path, f"the answer file is not UTF-8 text: {error}") from None


def read_private_json(name):
    """A private data file under /tests (for example ``config.json``), parsed. Raises on any problem: the
    private files are yours, so a failure here is a grader defect, not a candidate's."""
    with open(machine_path(f"{TESTS_DIR}/{name}"), encoding="utf-8") as handle:
        return json.load(handle, parse_float=Decimal)


# ------------------------------------------------------------------------------------------- JSON

EXACT = "exact"
"""The whole reply (surrounding whitespace aside) is one JSON value."""
FENCED = "fenced"
"""Exactly one Markdown code fence (``` or ```json) holding one JSON value, or the bare value; nothing else."""
LAST = "last"
"""Prose may come first; the reply ends with one JSON object, and no earlier object carries the answer keys."""
FRAMINGS = (EXACT, FENCED, LAST)

_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\Z", re.DOTALL)


def _pairs(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise GradeError("", f"duplicate key {key!r}")
        seen[key] = value
    return seen


def _constant(name):
    raise GradeError("", f"{name} is not a JSON number")


def _decoder():
    return json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant, parse_float=Decimal)


def _loads(text):
    try:
        value, end = _decoder().raw_decode(text)
    except json.JSONDecodeError as error:
        raise GradeError("", f"not valid JSON: {error.msg} at line {error.lineno} column {error.colno}") from None
    if text[end:].strip():
        raise GradeError("", f"text follows the JSON value: {text[end:end + 40]!r}")
    return value


def parse_json(text, framing=EXACT, required_keys=()):
    """The one JSON value of a reply, read strictly.

    Always: duplicate keys, NaN and Infinity are refused; floats come back as ``Decimal`` so the digits the
    solver wrote survive (``sig_figs``, ``decimals``); one leading BOM and surrounding whitespace are allowed
    (JSON itself refuses any other invisible character between tokens). ``framing`` is one of ``EXACT``, ``FENCED``,
    ``LAST``: choose it from the contract and say the same thing in the instruction. ``required_keys`` (for
    ``LAST``) are the top-level keys that make an object an answer.
    """
    if framing not in FRAMINGS:
        raise ValueError(f"framing is one of {FRAMINGS}, got {framing!r}")
    text = text.removeprefix("\ufeff").strip()
    if not text:
        raise GradeError("", "the reply is empty")
    if framing == FENCED:
        fence = _FENCE.match(text)
        if fence:
            text = fence.group(1).strip()
        elif text.startswith("```"):
            raise GradeError("", "a code fence that is not exactly one ``` or ```json block around the answer")
    if framing in (EXACT, FENCED):
        return _loads(text)
    return _last_object(text, tuple(required_keys))


def _last_object(text, required_keys):
    if not required_keys:
        raise ValueError("framing LAST needs required_keys, the keys that make an object an answer")
    decoder = _decoder()
    found = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except GradeError:
            # A decodable-looking object with a duplicate key or NaN still counts as an answer object.
            found.append((start, None, len(text)))
            index = start + 1
            continue
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict) and all(key in value for key in required_keys):
            found.append((start, value, end))
        index = end if isinstance(value, dict) else start + 1
    if not found:
        raise GradeError("", f"no JSON object with the keys {list(required_keys)}")
    if len(found) > 1:
        raise GradeError("", f"{len(found)} answer objects in the reply; give exactly one")
    start, value, end = found[0]
    if value is None:
        raise GradeError("", "the answer object repeats a key or holds NaN/Infinity")
    if text[end:].strip():
        raise GradeError("", "text follows the answer object")
    return value


# ------------------------------------------------------------------------------------------- values


def _where(path):
    return path or "answer"


def as_object(value, required, optional=(), path=""):
    """``value`` as a dict with exactly the keys ``required`` plus any of ``optional``; nothing else."""
    if not isinstance(value, dict):
        raise GradeError(_where(path), f"expected an object, got {type(value).__name__}")
    keys = set(value)
    missing = sorted(set(required) - keys)
    extra = sorted(keys - set(required) - set(optional))
    if missing:
        raise GradeError(_where(path), f"missing keys {missing}")
    if extra:
        raise GradeError(_where(path), f"unexpected keys {extra}")
    return value


def as_list(value, length=None, min_length=None, max_length=None, path=""):
    """``value`` as a list, with its length checked. Items are never dropped: validate each one yourself."""
    if not isinstance(value, list):
        raise GradeError(_where(path), f"expected a list, got {type(value).__name__}")
    if length is not None and len(value) != length:
        raise GradeError(_where(path), f"expected {length} items, got {len(value)}")
    if min_length is not None and len(value) < min_length:
        raise GradeError(_where(path), f"expected at least {min_length} items, got {len(value)}")
    if max_length is not None and len(value) > max_length:
        raise GradeError(_where(path), f"expected at most {max_length} items, got {len(value)}")
    return value


def as_str(value, path="", nonempty=True):
    if not isinstance(value, str):
        raise GradeError(_where(path), f"expected a string, got {type(value).__name__}")
    if nonempty and not value.strip():
        raise GradeError(_where(path), "expected a non-empty string")
    return value


def as_bool(value, path=""):
    """A JSON true or false; never "yes", 1 or "true"."""
    if value is not True and value is not False:
        raise GradeError(_where(path), f"expected true or false, got {value!r}")
    return value


def as_int(value, path="", integral_float=False):
    """A JSON integer, never a bool or a string. ``integral_float=True`` also takes 6.0 (say so in the
    contract: "a JSON number")."""
    if isinstance(value, bool):
        raise GradeError(_where(path), "expected an integer, got a boolean")
    if isinstance(value, int):
        return value
    if integral_float and isinstance(value, Decimal | float) and value == int(value):
        return int(value)
    raise GradeError(_where(path), f"expected an integer, got {value!r}")


def as_number(value, path=""):
    """A finite JSON number (int, or Decimal as ``parse_json`` returns floats), never a bool or a string."""
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise GradeError(_where(path), f"expected a number, got {value!r}")
    if isinstance(value, float | Decimal) and not math.isfinite(value):
        raise GradeError(_where(path), f"expected a finite number, got {value!r}")
    return value


def as_label(value, allowed, path=""):
    """One of ``allowed``, exactly as written: no stripping, case folding or aliases."""
    if not isinstance(value, str) or value not in allowed:
        raise GradeError(_where(path), f"expected one of {sorted(allowed)}, got {value!r}")
    return value


def as_alias(value, aliases, path=""):
    """The canonical label for ``value`` from an explicit ``aliases`` map (every accepted spelling is a key,
    the canonical label is its value). Only the spellings you list are accepted."""
    if not isinstance(value, str) or value not in aliases:
        raise GradeError(_where(path), f"expected one of {sorted(aliases)}, got {value!r}")
    return aliases[value]


_HHMM = re.compile(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", re.ASCII)
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)", re.ASCII)
_RATIONAL = re.compile(r"(-?(?:0|[1-9][0-9]*))/([1-9][0-9]*)", re.ASCII)


def as_hhmm(value, path=""):
    """A 24-hour "HH:MM" string, returned as minutes after midnight."""
    if not isinstance(value, str) or not _HHMM.fullmatch(value):
        raise GradeError(_where(path), f"expected HH:MM, got {value!r}")
    return int(value[:2]) * 60 + int(value[3:])


def as_iso_date(value, path=""):
    """A "YYYY-MM-DD" date string (``fromisoformat`` alone would also take "YYYYMMDD")."""
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise GradeError(_where(path), f"expected YYYY-MM-DD, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise GradeError(_where(path), str(error)) from None


def as_rational(value, path="", reduced=True):
    """An exact value written as an integer string "-3" or a fraction string "7/12" (ASCII digits, no "+",
    no spaces, positive denominator; reduced unless ``reduced=False``), as a ``Fraction``."""
    if not isinstance(value, str):
        raise GradeError(_where(path), f"expected an integer or fraction string, got {value!r}")
    if _INTEGER.fullmatch(value):
        return Fraction(int(value))
    match = _RATIONAL.fullmatch(value)
    if not match:
        raise GradeError(_where(path), f"expected an integer or a fraction like 7/12, got {value!r}")
    numerator, denominator = int(match.group(1)), int(match.group(2))
    if reduced and (denominator == 1 or math.gcd(numerator, denominator) != 1):
        raise GradeError(_where(path), f"{value!r} is not in lowest terms")
    return Fraction(numerator, denominator)


# ------------------------------------------------------------------------------------------- collections


def unique_by(items, key, path=""):
    """``{key(item): item}``, refusing a repeated key instead of letting a later item overwrite an earlier one."""
    index = {}
    for position, item in enumerate(items):
        value = key(item)
        if value in index:
            raise GradeError(f"{_where(path)}[{position}]", f"repeats {value!r}")
        index[value] = item
    return index


def same_items(got, expected, ordered, path=""):
    """Whether ``got`` holds exactly the items of ``expected`` (in order when ``ordered``), duplicates counted:
    never compare lists through ``set``, which hides a repeated or a missing duplicate."""
    as_list(got, path=path)
    if ordered:
        return list(got) == list(expected)
    try:
        return Counter(got) == Counter(expected)
    except TypeError:
        raise GradeError(_where(path), "items must be strings or numbers") from None


def within_universe(items, universe, path=""):
    """Refuse any item outside ``universe`` (ids, nodes, accounts the task defines)."""
    unknown = [item for item in items if item not in universe]
    if unknown:
        raise GradeError(_where(path), f"not in the task: {unknown}")
    return items


# ------------------------------------------------------------------------------------------- numbers


def _decimal(value):
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    raise TypeError(f"{value!r} is not a number parse_json produced (int or Decimal)")


def sig_figs(value):
    """Significant figures of the number as the solver wrote it (``parse_json`` keeps its digits).

    Trailing zeros of an integer are ambiguous; they count as significant (write 1.20e3 to say three).
    """
    digits = _decimal(value).as_tuple().digits
    stripped = "".join(map(str, digits)).lstrip("0")
    return max(len(stripped), 1)


def decimals(value):
    """Digits after the decimal point, as written."""
    exponent = _decimal(value).as_tuple().exponent
    return max(-exponent, 0) if isinstance(exponent, int) else 0


def _leading_exponent(target):
    """The exponent ``e`` of the leading digit of a nonzero ``target``: ``10**e <= |target| < 10**(e + 1)``."""
    size = abs(Fraction(target))
    exponent = math.floor(math.log10(size.numerator) - math.log10(size.denominator))
    while Fraction(10) ** exponent > size:
        exponent -= 1
    while Fraction(10) ** (exponent + 1) <= size:
        exponent += 1
    return exponent


def close_to(value, truth, *, rel=None, abs_=None, sig=None, places=None, path=""):
    """Whether a reported number matches ``truth`` within the precision the contract states.

    Every tolerance comes from ``truth`` and the stated precision, never from the digits the answer wrote, so
    extra digits are never penalised: the full float a plain ``json.dump`` writes (``2.3333333333333335`` for
    7/3) and a value padded with trailing zeros pass whenever the value itself is close enough. Give at least
    one of:

    - ``sig``: the contract asks for ``sig`` significant figures. The value must lie within half a unit in the
      ``sig``-th significant digit of the truth, ``|value - truth| <= 0.5 * 10**(e - sig + 1)`` where
      ``10**e <= |truth| < 10**(e + 1)``: ``sig=5`` against 7/3 allows 0.00005. An expected 0 has no
      significant figures; state an absolute tolerance instead.
    - ``places``: the contract asks for ``places`` decimal places, ``|value - truth| <= 0.5 * 10**-places``.
    - ``rel`` (relative tolerance, ``|value - truth| <= rel * |truth|``) and ``abs_`` (absolute tolerance,
      ``|value - truth| <= abs_``): the explicit alternatives, for a contract that states a band, and for a
      consistency check, whose window you propagate from its inputs' stated precision (not their written
      digits).

    Combine them when the contract states several; every one given must hold. A contract that also requires
    the digits to be written out (exactly two decimals shown, at least four significant figures shown) states
    that as its own rule: check it separately with ``decimals`` or ``sig_figs``. ``value`` is a JSON number as
    ``parse_json`` returns it (int or Decimal) or a float; ``truth`` may be an int, Fraction, Decimal or float.
    """
    if rel is None and abs_ is None and sig is None and places is None:
        raise ValueError("close_to needs a stated precision: rel, abs_, sig or places")
    if sig is not None and sig < 1:
        raise ValueError(f"sig counts significant figures, at least 1: got {sig}")
    number = as_number(value, path)
    target = Fraction(truth)
    error = abs(Fraction(number) - target)
    if rel is not None and error > Fraction(rel) * abs(target):
        return False
    if abs_ is not None and error > Fraction(abs_):
        return False
    if sig is not None:
        if target == 0:
            raise ValueError("an expected 0 has no significant figures: state an absolute tolerance (abs_)")
        if error > Fraction(1, 2) * Fraction(10) ** (_leading_exponent(target) - sig + 1):
            return False
    if places is not None and error > Fraction(1, 2) * Fraction(10) ** -places:
        return False
    return True


def meets(value, op, limit):
    """``value op limit`` exactly as the contract states it (op one of <, <=, >, >=, ==); no slack."""
    left, right = Fraction(value), Fraction(limit)
    return {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right, "==": left == right}[op]


# ------------------------------------------------------------------------------------------- verdict


class Checks:
    """The verdict's per-check results.

    ``check(name, weight, test)`` runs ``test()``; a ``GradeError`` it raises is a failed check with its
    reason (the candidate broke the contract). Any other exception propagates, because it is a defect of the
    grader and must not be scored as the candidate's.
    """

    def __init__(self):
        self.results = []

    def add(self, name, passed, weight=1.0, note=None):
        if any(result["name"] == name for result in self.results):
            raise ValueError(f"check {name!r} recorded twice")
        if not isinstance(passed, bool):
            raise TypeError(f"check {name!r}: passed must be a bool, got {passed!r}")
        result = {"name": name, "passed": passed, "weight": float(weight)}
        if note is not None:
            result["note"] = str(note)
        self.results.append(result)
        return passed

    def check(self, name, weight, test):
        try:
            outcome = test()
        except GradeError as error:
            return self.add(name, False, weight, str(error))
        return self.add(name, bool(outcome), weight)

    @property
    def all_passed(self):
        return bool(self.results) and all(result["passed"] for result in self.results)

    def weighted_share(self):
        total = sum(result["weight"] for result in self.results)
        if total <= 0:
            raise ValueError("the checks carry no weight")
        return sum(result["weight"] for result in self.results if result["passed"]) / total

    def reward(self, partial_cap=None):
        """1.0 when every check passed. Otherwise the weighted share of passed checks, scaled into
        [0, ``partial_cap``] when one is given: with ``partial_cap`` at or below the task's rejection ceiling,
        no answer that fails a check can earn more than the ceiling, however the weights fall. Without it, a
        wrong answer can collect up to ``worst_single_miss()``; check that against the ceiling."""
        if self.all_passed:
            return 1.0
        share = self.weighted_share()
        return share if partial_cap is None else partial_cap * share

    def worst_single_miss(self):
        """The reward of an answer that fails only the lightest check, under the plain weighted share."""
        total = sum(result["weight"] for result in self.results)
        lightest = min(result["weight"] for result in self.results)
        return (total - lightest) / total


def write_verdict(reward, checks, **detail):
    if not 0.0 <= reward <= 1.0:
        raise ValueError(f"reward {reward} lies outside [0, 1]")
    path = machine_path(VERDICT_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"reward": float(reward), "detail": {"checks": checks.results, **detail}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, default=str)
    return payload


def run(grade):
    """Call ``grade() -> (reward, checks)`` and write its verdict; always exit 0 with a verdict.

    A ``GradeError`` escaping ``grade`` (the reply does not parse, the answer file is missing) is a verdict of
    0 with a failed ``answer_contract`` check carrying the reason. Any other exception is also a verdict of 0,
    with a failed ``grader_error`` check carrying the traceback, so you see it in ``run_grade_candidate`` and
    in validation: fix the grader, because a correct answer that hits the same line scores 0 too.
    """
    try:
        reward, checks = grade()
        return write_verdict(reward, checks)
    except GradeError as error:
        checks = Checks()
        checks.add("answer_contract", False, 1.0, str(error))
        return write_verdict(0.0, checks)
    except Exception:
        checks = Checks()
        checks.add("grader_error", False, 1.0, traceback.format_exc(limit=8))
        print(traceback.format_exc(), file=sys.stderr)
        return write_verdict(0.0, checks)
