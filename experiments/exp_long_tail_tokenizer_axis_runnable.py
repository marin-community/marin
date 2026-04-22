# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the first-pass tokenizer-axis long-tail diagnostics for issue #5079.

See https://github.com/marin-community/marin/issues/5079.
"""

from experiments.evals.long_tail_tokenizer_axis import default_tokenizer_axis_step
from marin.execution.executor import executor_main

TOKENIZER_AXIS = default_tokenizer_axis_step(
    name="long-tail-tokenizer-axis-runnable-first-pass",
    max_docs_per_slice=512,
    max_doc_bytes=32_768,
    baseline_tokenizer_name="llama3_1_8b",
    include_planned_raw_slices=True,
    include_o200k_base=False,
    include_byte_reference=False,
)


if __name__ == "__main__":
    executor_main(
        [TOKENIZER_AXIS],
        description=(
            "Tokenizer-axis diagnostics on runnable long-tail slices with planned symbolic slices recorded as "
            "non-runnable placeholders."
        ),
    )
