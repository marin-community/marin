# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Mapping

from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data import AsyncDataset, ConcatDataset
from levanter.data.text import GrugLmExample, LmDataConfig, NamedLmDataset
from levanter.models.lm_model import LmExample
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config, tokenize
from levanter.schedule import BatchSchedule

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.lean4_slt import tokenized_lean4_slt
from experiments.pretraining_datasets.lean_workbook import tokenized_lean_workbook


@dataclasses.dataclass(frozen=True)
class ConcatLmDataConfig(LmDataConfig):
    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        initial_batch_size = batch_schedule.batch_size_at_step(0)
        datasets = self.train_sets(Pos, initial_batch_size=initial_batch_size, key=key)
        return NamedLmDataset(ConcatDataset(datasets), Pos)

    def train_grug_sets(
        self,
        *,
        seq_len: int,
        initial_batch_size: int | None = None,
        key: PRNGKeyArray,
    ) -> Mapping[str, AsyncDataset[GrugLmExample]]:
        datasets = self.train_sets(
            self._position_axis(seq_len),
            initial_batch_size=initial_batch_size,
            key=key,
        )
        return {"lean": ConcatDataset(datasets)}


numina_math_lean = ExecutorStep(
    name="raw/NuminaMath-LEAN",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="AI-MO/NuminaMath-LEAN",
        revision="51fa67f",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/NuminaMath-LEAN-51fa67f",
)

tokenized_numina_math_lean = ExecutorStep(
    name="tokenized/NuminaMath-LEAN",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[numina_math_lean],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
)


lean_tokenized_llama3 = {
    "lean4_slt": tokenized_lean4_slt,
    "lean_workbook": tokenized_lean_workbook,
    "numina_math_lean": tokenized_numina_math_lean,
}


def make_lean_concat_config(*, shuffle: bool | int = False) -> ConcatLmDataConfig:
    base = lm_mixture_data_config(
        lean_tokenized_llama3,
        {name: 1.0 for name in lean_tokenized_llama3},
        shuffle=shuffle,
    )
    return ConcatLmDataConfig(**base.__dict__)


lean_concat_config_llama3 = make_lean_concat_config()
