# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle and NoP probes for the self-contained Micro210 build gate.

For every prepared row the gate grades a reference answer and a constant no-progress
answer using only that row's TaskSpec contents in an isolated environment. This module
constructs both answers from the spec: each script-graded benchmark derives its reference
from the embedded document (target letter, boxed answer, reference assertion, canonical
solution, ...), and the NoP answer is one constant string per benchmark. The IFBench
reference is constructed inside the isolated container by IFBENCH_ORACLE_PROGRAM, which
verifies candidates against the row's own vendored checkers before the gate grades it.
"""

import base64
import json
from dataclasses import dataclass

NOP_ANSWER = "I don't know."
# MMLU-Pro's fallback extractor scores a bare "I" as answer letter I, so the constant
# NoP for that benchmark must carry no standalone A--J letter.
MMLU_NOP_ANSWER = "The answer is not known."
GRADER_LIB_DIRNAME = "grader_lib"
CONTINUOUS_METRIC_EPSILON = 0.5

HARBOR_BENCHMARKS = frozenset(
    {"bixbench-pi", "tb2-recovery", "ot-tblite-recovery", "ds-1000-local", "bfclparity-pi", "swebench-verified"}
)


@dataclass(frozen=True)
class RowProbes:
    """The two probe answers for one row and how their rewards must come out."""

    oracle: str | None
    """None when the oracle is constructed inside the isolated container (IFBench)."""
    nop: str
    judge_required: bool = False


def _files(spec: dict) -> dict[str, bytes]:
    return {
        resource["path"]: base64.b64decode(resource["source"]["content_base64"])
        for resource in spec["resources"]["verifier"]
    }


def _config(spec: dict) -> dict:
    return json.loads(_files(spec)["config.json"])


def _reference(spec: dict) -> dict:
    return _config(spec)["spec"]["grader"].get("reference") or {}


def _box(target: str) -> str:
    return f"\\boxed{{{target}}}"


def truthfulqa_oracle(spec: dict) -> str:
    """The letter of the recorded target choice, which the shuffled labels accept."""
    target = int(_config(spec)["spec"]["target"])
    return _box(chr(65 + target))


def gsm8k_oracle(spec: dict) -> str:
    """The document's own reference answer, ending in its ``#### N`` marker."""
    return str(_config(spec)["spec"]["doc"]["answer"])


def triviaqa_oracle(spec: dict) -> str:
    """The contracted ``Answer: <alias>`` form of the first reference alias."""
    doc = _config(spec)["spec"]["doc"]
    aliases = doc["answer"]["aliases"]
    if not aliases:
        raise ValueError("TriviaQA reference has no aliases")
    return f"Answer: {aliases[0]}"


def letter_oracle(spec: dict) -> str:
    return f"Answer: {_config(spec)['spec']['target']}"


def mmlu_oracle(spec: dict) -> str:
    return f"The answer is ({_config(spec)['spec']['target']})."


def boxed_oracle(spec: dict) -> str:
    return _box(str(_config(spec)["spec"]["target"]))


def olympiadbench_oracle(spec: dict) -> str:
    answers = _reference(spec)["answers"]
    if not answers:
        raise ValueError("OlympiadBench reference carries no answers")
    return _box(str(answers[0]))


def cruxeval_oracle(spec: dict) -> str:
    """The reference assertion in the prompt's own format."""
    doc = _config(spec)["spec"]["doc"]
    return f"assert f({doc['input']}) == {doc['output']}"


def nupa_oracle(spec: dict) -> str:
    return str(_config(spec)["spec"]["target"])


def mrcr_oracle(spec: dict) -> str:
    return str(_config(spec)["spec"]["target"])


def humanevalplus_oracle(spec: dict) -> str:
    problem = _reference(spec)["problem"]
    return f"```python\n{problem['canonical_solution']}\n```"


def mbppplus_oracle(spec: dict) -> str:
    code = _reference(spec)["problem"]["code"]
    return f"```python\n{code}\n```"


IFBENCH_ORACLE_PROGRAM = """
import ast
import importlib
import json
import re
import sys

sys.path.insert(0, "/tests/grader_lib")


def _capital_order():
    source = open("/tests/grader_lib/eval/chat_benchmarks/IFBench/instructions.py").read()
    match = re.search(r"class EuropeanCapitalsSortChecker.*?order = \\[(.*?)\\]", source, re.DOTALL)
    if match is None or "Kyiv" not in match.group(1):
        raise SystemExit("vendored checker order list unusable")
    return ast.literal_eval(f"[{match.group(1)}]")


def _single_candidates(name, kwargs):
    if name == "custom:european_capitals_sort":
        return [", ".join(_capital_order())]
    if name == "words:words_position":
        keyword = kwargs.get("keyword") or "window"
        return [f"Go {keyword} ahead and {keyword} now !"]
    if name == "format:list":
        sep = kwargs.get("sep") or "..."
        return [f"{sep} first item\\n{sep} second item\\n{sep} third item"]
    if name == "words:consonants":
        return ["Strong strikes bring bright strength."]
    if name == "words:odd_even_syllables":
        return ["Sun daily cat upon dog idea fish"]
    if name == "format:line_indent":
        return ["first\\n  second\\n    third"]
    if name == "ratio:sentence_words":
        return ["cat dog ant.\\nbat hen owl.\\nfox pig ram."]
    if name == "count:unique_word_count":
        count = int(kwargs.get("N") or 0)
        return [" ".join(f"word{index}" for index in range(count))]
    if name == "format:quotes":
        return ['"One \\'two "three" four\\' five"']
    if name == "words:palindrome":
        return ["level civic kayak madam radar refer rotor stats tenet sagas"]
    if name == "format:no_bullets_bullets":
        return ["First sentence stands alone here.\\nSecond sentence stands alone here.\\n* one\\n* two"]
    raise SystemExit(f"no oracle template for {name}")


_COMBOS = {
    ("words:consonants", "words:odd_even_syllables"): ["strong candle spring purple bright"],
    ("format:line_indent", "ratio:sentence_words"): ["cat dog ant.\\n  bat hen owl.\\n    fox pig ram."],
}


def _kwargs_for(doc, index):
    doc_kwargs = doc.get("kwargs") or []
    kwargs = doc_kwargs[index] if index < len(doc_kwargs) else {}
    return {key: value for key, value in (kwargs or {}).items() if value is not None}


def main():
    registry = importlib.import_module("eval.chat_benchmarks.IFBench.instructions_registry")
    doc = json.loads(open("/tests/config.json").read())["spec"]["doc"]
    names = tuple(doc["instruction_id_list"])
    checkers = []
    for index, name in enumerate(names):
        checker = registry.INSTRUCTION_DICT[name](name)
        checker.build_description(**_kwargs_for(doc, index))
        checkers.append(checker)
    candidates = list(_COMBOS.get(names) or [])
    if len(names) == 1:
        candidates += _single_candidates(names[0], _kwargs_for(doc, 0))
    for candidate in candidates:
        if all(checker.check_following(candidate) for checker in checkers):
            print(candidate)
            return
    raise SystemExit(f"no oracle candidate satisfied {names}")


main()
"""


def row_probes(benchmark: str, spec_json: str) -> RowProbes:
    """Build the oracle and NoP answers for one prepared row from its TaskSpec contents."""
    spec = json.loads(spec_json)
    if benchmark in HARBOR_BENCHMARKS:
        return RowProbes(oracle="", nop="", judge_required=benchmark == "bixbench-pi")
    builders = {
        "truthfulqa": truthfulqa_oracle,
        "gsm8k-0shot": gsm8k_oracle,
        "triviaqa": triviaqa_oracle,
        "gpqa-diamond": letter_oracle,
        "mmlu-pro": mmlu_oracle,
        "aime24": boxed_oracle,
        "math500": boxed_oracle,
        "olympiadbench": olympiadbench_oracle,
        "cruxeval": cruxeval_oracle,
        "NUPA": nupa_oracle,
        "mrcr": mrcr_oracle,
        "humanevalplus": humanevalplus_oracle,
        "mbppplus": mbppplus_oracle,
    }
    if benchmark not in builders and benchmark != "ifbench":
        raise ValueError(f"No oracle builder for benchmark {benchmark!r}")
    nop = MMLU_NOP_ANSWER if benchmark == "mmlu-pro" else NOP_ANSWER
    oracle = None if benchmark == "ifbench" else builders[benchmark](spec)
    return RowProbes(oracle=oracle, nop=nop)


def expectations(benchmark: str) -> tuple[float, float | None]:
    """Return (oracle minimum, NoP maximum or None for exact zero) for one benchmark."""
    if benchmark == "mrcr":
        return 1.0, CONTINUOUS_METRIC_EPSILON
    return 1.0, 0.0
