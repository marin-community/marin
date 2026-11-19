# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small CPU retrieval image built through the maintained environment builder."""

import argparse
import logging
import os
from pathlib import Path

from marin.execution.lazy import run
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation.datasets.openresearcher.index import JAVA_ARCHIVE, JAVA_SHA256
from experiments.post_training.task_curation.environment import Archive, Environment
from experiments.post_training.task_curation.images.build import environment_artifact

IMPLEMENTATION_REVISION = "036cceab31e06844720488fd29e504d479777f05"
IMPLEMENTATION_ARCHIVE = f"https://github.com/marin-community/marin/archive/{IMPLEMENTATION_REVISION}.tar.gz"
IMPLEMENTATION_SHA256 = "461b10db82d9d021123c9f5da4c8790a5e89a1dd55166cc9aff87041289df236"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
IMAGE_REPOSITORY = "ghcr.io/marin-community/rl-data-openresearcher"
CPU_PACKAGE_INDEX = "https://download.pytorch.org/whl/cpu"
ACTOR_RUNTIME = Environment(
    requirements=("pyserini==1.2.0", "openai==1.97.0", "transformers==4.53.2", "torch==2.7.0+cpu"),
    files={
        "/opt/java21": Archive(JAVA_ARCHIVE, JAVA_SHA256, "jdk-21.0.9+10-jre"),
        **{
            f"/opt/offline-search/{package}": Archive(
                IMPLEMENTATION_ARCHIVE,
                IMPLEMENTATION_SHA256,
                f"marin-{IMPLEMENTATION_REVISION}/experiments/post_training/task_curation/datasets/{package}",
            )
            for package in ("browsecomp", "openresearcher")
        },
    },
    env={
        "JAVA_HOME": "/opt/java21",
        "JDK_JAVA_OPTIONS": "-Xmx768m",
        "PYTHONPATH": "/opt/offline-search",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "UV_EXTRA_INDEX_URL": CPU_PACKAGE_INDEX,
        "UV_INDEX_STRATEGY": "unsafe-best-match",
    },
    dockerfile=Path(__file__).with_name("RuntimeDockerfile"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("UV_EXTRA_INDEX_URL", "UV_INDEX_STRATEGY"):
        if os.environ.get(name) != ACTOR_RUNTIME.env[name]:
            raise ValueError(f"The hash-lock resolver requires the declared {name}")
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    result = run(environment_artifact(ACTOR_RUNTIME, IMAGE_REPOSITORY))[0]
    args.output.write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
