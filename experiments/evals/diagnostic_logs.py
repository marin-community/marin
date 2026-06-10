# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw perplexity-gap provider for capped public diagnostic-log slices."""

from marin.datakit.download.diagnostic_logs import (
    DEFAULT_GHALOGS_MAX_MEMBERS,
    DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    DEFAULT_LOGHUB_MAX_FILES,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset

from experiments.exp5094_public_diagnostic_logs import diagnostic_log_extract_steps


def diagnostic_log_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return capped public diagnostic-log slices for perplexity-gap scoring."""

    ghalogs_step, logchunks_step, loghub_step = diagnostic_log_extract_steps(
        source_path=None,
        max_ghalogs_members=DEFAULT_GHALOGS_MAX_MEMBERS,
        max_logchunks_examples=DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
        max_loghub_files=DEFAULT_LOGHUB_MAX_FILES,
    )
    return {
        "diagnostic_logs/ghalogs_issue_5093_holdout": raw_text_dataset(
            ghalogs_step.as_input_name() / "issue_5093_holdout/*.jsonl",
            tags=("diagnostic_logs", "issue:5094", "source:ghalogs", "split:issue_5093_holdout"),
        ),
        "diagnostic_logs/logchunks_eval": raw_text_dataset(
            logchunks_step.as_input_name() / "eval_only/logchunks/*.jsonl",
            tags=("diagnostic_logs", "issue:5094", "source:logchunks", "split:eval_only"),
        ),
        "diagnostic_logs/loghub_eval": raw_text_dataset(
            loghub_step.as_input_name() / "eval_only/loghub/*.jsonl",
            tags=("diagnostic_logs", "issue:5094", "source:loghub", "split:eval_only"),
        ),
    }
