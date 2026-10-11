# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source tasks rejected after manual inspection of their public and grading contracts."""

SOURCE_DEFECTS: dict[tuple[str, str], str] = {
    (
        "DCAgent__exp_rpt_stack-pytest-v2",
        "stack-pytest-0249",
    ): "AlignGraph instruction exposes fixed test answers and leaves the rotation convention ambiguous",
    (
        "DCAgent__exp_rpt_stack-pytest-v2",
        "stack-pytest-0380",
    ): "fake Selenium tests accept empty country data and do not check zone details or back navigation",
    (
        "DCAgent__exp_rpt_stack-pytest-v2",
        "stack-pytest-0474",
    ): (
        "StreamFile tests require absent mitmproxy test helpers and use an invalid pytest.raises " "argument"
    ),
    (
        "DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
        "task_1492",
    ): (
        "oracle passes reverse-numeric inputs to comm, whose lexicographic comparison produces the wrong "
        "file difference"
    ),
    (
        "DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
        "task_3002",
    ): (
        "captured character dump fixes the oracle's chosen text although the instruction permits any " "small text file"
    ),
    (
        "DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
        "task_9748",
    ): "shipped solve script does not write command_capture.txt",
    (
        "DCAgent__code-contests-noblock",
        "code_contests-4395",
    ): "constructive problem has multiple valid outputs but no special judge",
    (
        "DCAgent__code-contests-noblock",
        "code_contests-9694",
    ): "constructive problem has interchangeable witnesses but no special judge",
    (
        "DCAgent__exp_rpt_e2egit-v2",
        "e2egit-0289",
    ): "inventory tests impose an unstated underflow policy and omit required price checks",
    (
        "DCAgent__exp_rpt_e2egit-v2",
        "e2egit-0328",
    ): (
        "factorial instruction places student tests in the private verifier directory and grading omits "
        "required recursion and test coverage"
    ),
    (
        "DCAgent__exp_rpt_e2egit-v2",
        "e2egit-0359",
    ): (
        "calculator instruction omits the imported module path and grading omits required arithmetic and "
        "documentation checks"
    ),
    (
        "DCAgent__exp_rpt_e2egit-v2",
        "e2egit-0433",
    ): "instruction requires JavaScript or TypeScript but tests import Python and assume isolated state",
    (
        "DCAgent__exp_rpt_pymethods2test-large-v2",
        "pymethods2test-2067",
    ): "tests require a recurrence that the instruction does not define",
    (
        "DCAgent__swe_rebench_v2_patched_oracle-v2",
        "pybamm-team__pybamm-809",
    ): "trusted tests do not exercise the requested porosity change",
    (
        "laion__exp_rpt_taco-v2",
        "taco-4675",
    ): "bundled expected output disagrees with the bundled golden solution",
    (
        "laion__nemotron-gym-competitive-coding-v2",
        "comp-coding-dd2a13b32896.tar.gz",
    ): "problem permits any valid witness but has no special judge",
    (
        "laion__nemotron-gym-knowledge-mcqa-v2",
        "Nemotron-RL-knowledge-mcqa-f7e357b86af7.tar.gz",
    ): "clinical vignette does not supply enough information for its gold option",
    (
        "laion__nemotron-gym-math-stack-overflow-v3",
        "Nemotron-RL-math-stack_overflow-75a68efb194b.tar.gz",
    ): "compound-interest question asks for an amount without supplying a principal",
    (
        "laion__nemotron-gym-structured-outputs-v4",
        "if-structured-v2-ae3d7b6f2860.tar.gz",
    ): "required schema fields are absent from the source document",
    (
        "laion__wizardlm-orca-v4",
        "wizardlm_orca-9163",
    ): "undated legal question requests facts that are not fixed by the prompt",
}
