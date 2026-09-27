# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect the final MT-MBPP tests, with their validation outcome, into ``tests/validated/<language>.jsonl``.

Every (language, task) keeps its latest translation and the sandbox verdict for exactly that test. ``valid`` tests are
the ones graded; an excluded test records why: ``reference_wrong`` (the repair round found that the o4-mini reference
disagrees with MBPP's expected outputs), ``reference_fails`` (the reference does not pass the test) or ``stub_passes``
(the test does not detect a wrong answer). Also writes the card text for the dataset's tests section.

usage: uv run --offline --no-sync python -m experiments.domain_phase_mix.mt_mbpp_exec.collect_tests --project DIR
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.domain_phase_mix.mt_mbpp_exec.sandbox import IMAGE
from experiments.domain_phase_mix.mt_mbpp_exec.translate_tests import FIRST_PASS_EFFORT, MAX_TOKENS, MODEL, REPAIR_EFFORT
from experiments.domain_phase_mix.mt_mbpp_exec.validate_tests import latest_translations, test_key

CARD = """Each test is MBPP's Python test cases (same inputs, same expected outputs) written for the target language
against the disclosed signature: `imports` go above the solution and `main`, which exits non-zero on the first failed
case, goes after it. Python uses MBPP's own asserts. The other languages were translated by DeepSeek (`{model}`,
thinking effort `{first}`, output cap {cap} tokens; 11 translations that hit the cap were redone with a 65,536-token
cap). Every test ran in a network-free sandbox (`{image}`, one toolchain per language) twice: with the o4-mini
reference solution, which must pass, and with a stub whose tested function returns a fixed wrong value, which must
fail. The 424 tests that failed this check went back to DeepSeek once at effort `{repair}` with the sandbox output;
the model was told not to adapt a test to a reference that disagrees with MBPP's expected outputs and to flag such
references instead.

`valid` marks the {valid} tests that pass the check ({per_language}); only those documents are scored. Excluded tests
keep their reason: {reasons}.

The sandbox is somewhat lenient: TypeScript is transpiled without type checking, C# gets a console project's implicit
usings, and a few widely used libraries beyond the standard libraries are installed because reference solutions use
them (Haskell regex-tdfa, split, vector and containers packages; Rust regex and num crates; bc, gawk, jq and rev for
bash; PHP mbstring). Script languages must print a completion marker after the tests, so a solution that exits early
cannot pass.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    tests_dir = args.project / "tests"
    latest = latest_translations(tests_dir / "translations.jsonl")
    wrong = set()
    for row in map(json.loads, (tests_dir / "translations.jsonl").read_text().splitlines()):
        if row.get("response", {}).get("reference_wrong"):
            wrong.add((row["language"], row["task_id"]))
    verdicts = {
        (r["language"], r["task_id"], r["test_sha256"]): r
        for r in map(json.loads, (tests_dir / "validation.jsonl").read_text().splitlines())
    }
    out = tests_dir / "validated"
    out.mkdir(exist_ok=True)
    reasons, per_language = Counter(), Counter()
    rows_by_language = {}
    for (language, task_id), t in sorted(latest.items()):
        verdict = verdicts[(language, task_id, test_key(t["test"]))]
        reason = None
        if not verdict["valid"]:
            if (language, task_id) in wrong:
                reason = "reference_wrong"
            elif not verdict["reference_passed"]:
                reason = "reference_fails"
            else:
                reason = "stub_passes"
            reasons[reason] += 1
        else:
            per_language[language] += 1
        source = t["source"]
        rows_by_language.setdefault(language, []).append(
            {
                "language": language,
                "task_id": task_id,
                "valid": verdict["valid"],
                "exclusion": reason,
                "imports": t["test"]["imports"],
                "main": t["test"]["main"],
                "stub": t["test"]["stub"],
                "translator": source.get("model"),
                "effort": source.get("effort", FIRST_PASS_EFFORT if source.get("model") else None),
                "repaired": bool(source.get("repair")),
                "prompt_sha256": source.get("prompt_sha256"),
                "reference_run": {k: verdict["reference"][k] for k in ("passed", "phase", "exit_code", "timeout")},
                "stub_run": {k: verdict["stub"][k] for k in ("passed", "phase", "exit_code", "timeout")},
            }
        )
    for language, rows in rows_by_language.items():
        with (out / f"{language}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    card = CARD.format(
        model=MODEL,
        first=FIRST_PASS_EFFORT,
        cap=f"{MAX_TOKENS:,}",
        repair=REPAIR_EFFORT,
        image=IMAGE,
        valid=f"{sum(per_language.values()):,} of {sum(len(v) for v in rows_by_language.values()):,}",
        per_language=", ".join(f"{k} {v}" for k, v in sorted(per_language.items())),
        reasons=", ".join(f"{k} {v}" for k, v in reasons.most_common()),
    )
    (out / "README.md").write_text(card)
    print(f"valid {sum(per_language.values())}; excluded {dict(reasons)}")


if __name__ == "__main__":
    main()
