# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the locuslab Safety Pretraining sources with the Marin tokenizer.

Filters :func:`marin.datakit.sources.all_sources` to the ``safety_pt/*`` keys
and builds one ``tokenize`` :class:`StepSpec` per subset that depends on the
source's normalized output. ``StepRunner`` walks back through the chain — so
any missing download or normalize step is materialized first, then the
tokenize step writes a Levanter cache to
``tokenized/<tokenizer>/<source_name>``.
"""

import logging

from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.artifact import Artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

TOKENIZER = "marin-community/marin-tokenizer"
SAFETY_PT_PREFIX = "safety_pt/"


def _tokenize_step(source: DatakitSource) -> StepSpec:
    normalized = source.normalized
    return StepSpec(
        name=f"tokenized/{TOKENIZER.replace('/', '--')}/{source.name}",
        deps=[normalized],
        hash_attrs={"tokenizer": TOKENIZER},
        fn=lambda output_path, n=normalized: tokenize(
            TokenizeConfig(
                train_paths=[Artifact.from_path(n, NormalizedData).main_output_dir],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=TOKENIZER,
            )
        ),
    )


def main() -> None:
    sources = [s for name, s in all_sources().items() if name.startswith(SAFETY_PT_PREFIX)]
    if not sources:
        raise RuntimeError(f"No sources matched prefix {SAFETY_PT_PREFIX!r}")
    terminals = [_tokenize_step(s) for s in sources]
    logger.info("Tokenizing %d safety_pt sources with %s", len(sources), TOKENIZER)
    StepRunner().run(terminals)
    logger.info("All %d sources tokenized", len(sources))


if __name__ == "__main__":
    configure_logging()
    main()
