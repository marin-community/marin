# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run question grounding against the hybrid BM25+dense provider."""

from experiments.post_training.task_curation.datasets.openresearcher.hybrid_grounding_pipeline import main

if __name__ == "__main__":
    main()
