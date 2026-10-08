# Vendored ARC scorer

One NVARC scorer grades the TaskTrove ARC tasks and the Nemotron Ultra NVARC rows. A task ships these files to `/tests` at the same
relative paths, together with the package markers, `aime/utils.py` and `answer_extraction.py` from
[`../../nemotron_ultra/scorers/`](../../nemotron_ultra/scorers/VENDORED.md), which `nvarc.py` imports.

| File | Upstream | Note |
| --- | --- | --- |
| `skyrl_gym/envs/nemotron_ultra/nvarc.py` | [`nemotron_ultra/nvarc.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/nvarc.py) at MarinSkyRL `d8b6e8c163def3660e9d3072c1c174226a1709fa` | Unchanged. |
| `skyrl_gym/envs/nemotron_ultra/sandbox.py` | [`nemotron_ultra/sandbox.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/sandbox.py) at the same revision | Unchanged. `nvarc.py` imports its output limit and client type; importing it needs `requests`. |
| `local_sandbox.py` | none | Ours. `LocalSandbox.execute` replaces the NeMo Skills sandbox server: it runs the program with `python -c` under a timeout and returns the server's response fields. |

`arc_grade.py` passes a `LocalSandbox(user=65534)` as the `sandbox` argument of
`nvarc.grade_inductive_arc`, so neither upstream file needs an edit or a monkeypatch. It first makes
`/tests/config.json` readable by root alone, so the candidate program, running as uid 65534 in the
temporary directory, cannot read the expected output. The program imports `numpy` in the grader's
interpreter.

License: both upstream files keep NVIDIA's `SPDX-License-Identifier: Apache-2.0` headers. The MarinSkyRL
repository is Apache-2.0.
