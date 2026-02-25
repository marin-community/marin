# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
from haliax import Axis
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from levanter.data import AsyncDataset, DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text import GrugLmExample
from levanter.data.text import LmDataConfig
from levanter.schedule import BatchSchedule


def build_train_dataset(
    data_config: LmDataConfig,
    *,
    max_seq_len: int,
    batch_schedule: BatchSchedule,
    key: PRNGKeyArray,
) -> MixtureDataset[GrugLmExample]:
    Pos = Axis("position", max_seq_len)
    mix_key, shuffle_key = jax.random.split(key)
    weights = data_config.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    initial_batch_size = batch_schedule.batch_size_at_step(0)
    datasets = data_config.train_sets(Pos, key=shuffle_key, initial_batch_size=initial_batch_size)
    return MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
    batch_pspec: P = P(("data",)),
    allow_nondivisible_batch_size: bool | None = None,
) -> DataLoader[GrugLmExample]:
    if len(batch_pspec) != 1:
        raise ValueError(f"batch_pspec must describe a single logical batch axis, got {batch_pspec}")

    axis_resource = batch_pspec[0]
    if axis_resource is not None and not isinstance(axis_resource, (str, tuple)):
        raise ValueError(f"batch_pspec must map to mesh axis names, got {batch_pspec}")

    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": axis_resource},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=bool(allow_nondivisible_batch_size),
    )
