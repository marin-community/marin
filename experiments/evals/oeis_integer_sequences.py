# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OEIS integer-sequence raw validation slices for perplexity-gap evals."""

from __future__ import annotations

from marin.datakit.download.oeis import (
    DEFAULT_MAX_SEQUENCES,
    DEFAULT_RECORDS_PER_DOC,
    oeis_eval_step,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import executor_main
from marin.execution.step_spec import StepSpec

OEIS_INTEGER_SEQUENCES_ISSUE = 5770
OEIS_INTEGER_SEQUENCES_KEY = "oeis/integer_sequences"
OEIS_INTEGER_SEQUENCES_GLOB = "oeis_integer_sequences-*.parquet"

oeis_integer_sequences_eval = oeis_eval_step()


def oeis_integer_sequence_raw_validation_sets(
    *,
    oeis_raw: StepSpec | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Materialize OEIS as an eval-only raw-text dataset."""

    raw_step = oeis_integer_sequences_eval if oeis_raw is None else oeis_raw
    return {
        OEIS_INTEGER_SEQUENCES_KEY: raw_text_dataset(
            raw_step.as_executor_step().cd(OEIS_INTEGER_SEQUENCES_GLOB),
            tags=(
                "oeis",
                "integer_sequences",
                f"issue:{OEIS_INTEGER_SEQUENCES_ISSUE}",
                f"max_sequences:{DEFAULT_MAX_SEQUENCES}",
                f"records_per_doc:{DEFAULT_RECORDS_PER_DOC}",
            ),
        )
    }


if __name__ == "__main__":
    executor_main(steps=[oeis_integer_sequences_eval.as_executor_step()])
