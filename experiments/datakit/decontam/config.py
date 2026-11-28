# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared decontamination policy for the reference pipeline and testbed (``marin.datakit.decon``).

The models, the cutoff and the key settings come from the measurements in
``experiments/datakit/decontam/encoder/README.md``.
"""

from fray.types import ResourceConfig
from marin.datakit.decon import ScoreWorkers
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.decontam.encoder_models import EncoderModels

# The encoder classifiers (iteration 16, seeds 0 and 1) under the storage prefix; each region that runs the
# pipeline needs a copy, as it needs the item table.
MODELS_RELATIVE = "datakit/decontam/models"
MODEL_NAMES = ("encoder-it16-s0", "encoder-it16-s1")
MODELS_VERSION = "it16-s0-s1"
# A record is removed when the mean score of its best pair reaches this: the lowest cutoff whose removal check
# (set 26) passes the floor of 0.75 weighted precision with a 5% bound of 0.65, confirmed on fresh removals
# (set 27: 0.852 [0.818]).
CUTOFF = 0.6
# Keys are counted over this many reference shards of every source (seed fixed), as for the validated keys.
REFERENCE_SHARDS_PER_SOURCE = 2
REFERENCE_SEED = 11
FLAGGED_SAMPLE_SIZE = 8
# Production scores on H100s, about 50 GPU-hours per pass over the whole catalog. The workers sync the gpu extra
# for the CUDA build of torch, since a driver launched with the cpu extra has the CPU build. One worker per
# million pairs of a source is about 40 minutes of an H100 at the measured 425 pairs per second, so a worker's
# start (the gpu sync and 2.4 GB of models) is a small part of its time.
GPU_SCORE_WORKERS = ScoreWorkers(
    resources=ResourceConfig.with_gpu("H100", count=1, cpu=8, ram="64g", disk="100g"),
    max_workers=30,
    pairs_per_worker=1_000_000,
    extras=("gpu",),
)
# The GCP cluster that runs the ferries has no GPUs; there the workers score on CPU in the driver's environment.
# CPU throughput is not measured.
CPU_SCORE_WORKERS = ScoreWorkers(
    resources=ResourceConfig(cpu=8, ram="16g", disk="20g"), max_workers=16, pairs_per_worker=10_000, extras=()
)


def decon_models() -> EncoderModels:
    """The decontamination models under the current storage prefix."""
    paths = tuple(prefix_join(marin_prefix(), f"{MODELS_RELATIVE}/{name}") for name in MODEL_NAMES)
    return EncoderModels(paths=paths, version=MODELS_VERSION)
