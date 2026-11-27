# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a structured-output reply with the official StructEval non-renderable evaluator.

Ports TIGER-AI-Lab/StructEval's ``eval_engine``/``render_engine.render_utils`` semantics for the
non-renderable formats (JSON/YAML/CSV/TOML/XML): the official extraction priority (markdown fence
over ``<|BEGIN_CODE|>`` markers over the whole reply, after the official preprocessing), the
official per-format parse, the official path grammar (``csv::`` header tokens, backtick segments,
``[N]`` indices, ``*`` wildcards on lists only, dict keys with the ``@attr`` fallback), and the
official non-renderable final score ``0.2 * parse_ok + 0.8 * path_fraction``.

``config.json`` carries the row's ``output_type`` and ``raw_output_metric`` paths; the candidate's
reply is at ``/app/answer.txt``. The reward prints to stdout; harness defects exit nonzero.
"""

import codecs
import csv
import io
import json
import re
import sys
from pathlib import Path

import toml
import xmltodict
import yaml

from experiments.post_training.task_curation.datasets.structeval_paths import path_exists

BEGIN_END = re.compile(r"<\|BEGIN_CODE\|\>[ \t]*\n?(?P<payload1>.*?)(?:<\|END_CODE\|\>|$)", re.DOTALL | re.IGNORECASE)


def fence_pattern(output_type: str) -> re.Pattern:
    return re.compile(
        rf"```(?:{re.escape(output_type)}|[^\n]*)[ \t]*\n(?P<payload2>.*?)(?:```|$)",
        re.DOTALL | re.IGNORECASE,
    )


def extract_code(text: str, output_type: str) -> str:
    """The official ``extract_code_and_save`` preprocessing and extraction priority."""
    try:
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("<think>\n\n</think>\n\n", "")
        text = codecs.decode(text, "unicode_escape")
    except Exception:
        pass  # The official keeps the original string when the escape decode fails.
    fence = fence_pattern(output_type)
    # Interpolating .pattern drops the compiled flags; the official always searches DOTALL|IGNORECASE.
    match = re.search(f"(?:{BEGIN_END.pattern})|(?:{fence.pattern})", text, re.DOTALL | re.IGNORECASE)
    if match:
        code = (match.group("payload1") or match.group("payload2")).strip()
    else:
        code = text.strip()
    # The official re-runs the fence search and lets a fenced block override the marker payload.
    fenced = fence.search(text)
    if fenced:
        code = fenced.group("payload2").strip()
    return code


def load_structure(code: str, output_type: str):
    """The official ``load_file_structure`` parse per format, in memory.

    Returns ``(structure, parse_ok)``. The truthiness quirks are verbatim-official: a ``DictReader``
    object is always truthy, while an empty JSON/YAML/TOML document parses falsy.
    """
    try:
        if output_type == "json":
            result = json.loads(code)
        elif output_type == "yaml":
            result = yaml.safe_load(code)
        elif output_type == "toml":
            result = toml.loads(code)
        elif output_type == "xml":
            result = xmltodict.parse(code)
        elif output_type == "csv":
            reader = csv.DictReader(io.StringIO(code))
            result = {"csv_headers": reader.fieldnames, "csv_rows": list(reader)}
        else:
            raise ValueError(f"Unsupported StructEval output type: {output_type}")
    except Exception:
        return None, 0
    return result, 1 if result else 0


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    answer = Path("/app/answer.txt").read_text()
    output_type = str(config["output_type"]).lower()
    metrics = config["raw_output_metric"]
    assert isinstance(metrics, list) and metrics, "the shipped metric list must be nonempty"
    code = extract_code(answer, output_type)
    structure, parse_ok = load_structure(code, output_type)
    matched = sum(1 for metric in metrics if parse_ok and path_exists(structure, metric))
    reward = round(0.2 * parse_ok + 0.8 * (matched / len(metrics)), 2)
    print(reward)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # harness defects are infrastructure errors, not rewards
        print(f"structeval grader harness defect: {error}", file=sys.stderr)
        raise SystemExit(1) from error
