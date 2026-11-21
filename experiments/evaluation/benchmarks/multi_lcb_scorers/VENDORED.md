# Vendored Multi-LCB modules

Byte-for-byte copies from [Multi-LCB/Multi-LCB](https://github.com/Multi-LCB/Multi-LCB/tree/d80be9f24268e9e5c6e1463e26c9f05417fe06a0)
at `d80be9f24268e9e5c6e1463e26c9f05417fe06a0`, at their upstream paths. None is changed.

| File | Used by | Role |
| --- | --- | --- |
| `prompt_enhancer/unified_converter.py` | `multi_lcb.py` | Converts LeetCode functional tests and prompt examples to stdin/stdout (`ConverterFactory`, format `v2`). |
| `prompt_enhancer/example_parser.py` | `multi_lcb.py` | Parses the examples in a LeetCode problem statement for the prompt. |
| `lcb_runner/evaluation/testing_plang.py` | `multi_lcb_grade.py`, shipped to `/tests` | Compiles and runs a program in its language against every test (`eval_plang_code`). |
| `lcb_runner/utils/extraction_utils.py` | `multi_lcb_grade.py`, shipped to `/tests` | Truncates reasoning and extracts the first fenced code block. |

`multi_lcb.py` reproduces the prompt template of `lcb_runner/prompts/code_generation.py`, the example
formatting of `prompt_enhancer/enhancer.py` and the functional-test parsing of
`lcb_runner/benchmarks/code_generation.py` instead of vendoring them, because those modules import the
rest of upstream's runner. A comparison against those upstream modules, run on all 497 problems in
Python, C++ and Java, found identical prompts and tests.

`testing_plang.py` sets `RLIMIT_AS`, which macOS rejects, so its grading tests run on Linux only. After each
program it runs `pkill`, which the worker image lacks; `multi_lcb_grade.py` supplies a stand-in on `PATH`.

License: MIT, in [`LICENSE`](LICENSE).
