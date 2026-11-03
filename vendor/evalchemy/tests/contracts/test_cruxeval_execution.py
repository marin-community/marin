import json
import subprocess
import sys


def test_cruxeval_grades_without_fork(tmp_path):
    script = tmp_path / "grade_without_fork.py"
    script.write_text(
        """
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from eval.chat_benchmarks.CruxEval.execution import check_correctness


def reject_fork(event, _args):
    if event == "os.fork":
        raise RuntimeError("os.fork is unsafe")


if __name__ == "__main__":
    sys.addaudithook(reject_fork)
    cases = [
        ("passes", {"test_code": "assert 2 + 2 == 4"}, "python"),
        ("fails", {"test_code": "assert 2 + 2 == 5"}, "python"),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda args: check_correctness(*args), cases))
    print(json.dumps(outcomes))
"""
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    outcomes = json.loads(completed.stdout)

    assert [outcome["passed"] for outcome in outcomes] == [True, False]
    assert [outcome["result"] for outcome in outcomes] == ["passed", "failed: AssertionError"]
