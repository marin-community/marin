# TaskTrove converters

[`registry.py`](registry.py) maps a source family and archive file shape to one converter. A converter reads `TaskFiles` and returns the typed [`ConvertedTask`](converted_task.py) or `Rejected` result. The result preserves the task instruction, VerifyIT mode, private files, optional oracle solution, and adjusted source Dockerfile. Unrecognized or unsound variants are rejected by their converter.

Modules such as [`python_unit_tests.py`](python_unit_tests.py), [`agent_calendar.py`](agent_calendar.py), [`judge_rubric.py`](judge_rubric.py), and [`stdio_cases.py`](stdio_cases.py) handle distinct archived task layouts. The [TaskTrove release README](../README.md) explains the conversion stages and how to add a converter. The task curation experiment adapts these results in its [TaskTrove declarations](../../task_curation/datasets/README.md).
