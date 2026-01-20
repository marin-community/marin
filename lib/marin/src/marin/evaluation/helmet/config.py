# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from fray.cluster import ResourceConfig

HelmetEvalName: TypeAlias = Literal["recall", "rag", "longqa", "summ", "icl", "rerank", "cite"]

HELMET_EVALS_FULL: tuple[HelmetEvalName, ...] = ("recall", "rag", "longqa", "summ", "icl", "rerank", "cite")


@dataclass(frozen=True)
class HelmetConfig:
    """Configuration for running HELMET via Marin's Executor framework."""

    use_chat_template: bool
    """Whether to run HELMET in chat-template mode (required; no default)."""

    helmet_repo_url: str = "https://github.com/princeton-nlp/HELMET.git"
    helmet_repo_sha: str | None = None
    """Git SHA to check out; `None` resolves the current `main` when building the pipeline."""

    helmet_data_repo_id: str = "princeton-nlp/HELMET"
    helmet_data_revision: str = "main"
    helmet_data_sha: str | None = None
    """HF dataset commit SHA; `None` resolves `helmet_data_revision` via HF API when building the pipeline."""

    resource_config: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v4-8"))

    evals: tuple[HelmetEvalName, ...] = HELMET_EVALS_FULL

    evals_per_instance: int | Literal["all"] = 1
    """How many HELMET configs to run per TPU job."""

    vllm_serve_args: tuple[str, ...] = ()
    """Additional CLI args appended to `vllm serve` (e.g. `--max-model-len 131072`)."""

    eval_py_args: tuple[str, ...] = ()
    """
    Additional CLI args appended to HELMET's `eval.py` invocation.

    This can be used to override values coming from the HELMET config YAML (e.g. to run
    a single dataset/test-file as a smoke test):

        ("--datasets", "msmarco_rerank_psg", "--test_files", "data/msmarco/test_reranking_data_k10_dep3.jsonl", ...)
    """

    seed: int = 42
    tag: str = "v1"

    data_scratch_dir: str = "/opt/gcsfuse_mount/helmet-data"
    """
    Local scratch directory (backed by gcsfuse) where extracted HELMET data is stored.
    The actual directory used will be suffixed with the resolved dataset SHA.
    """

    data_output_root: str = "gcsfuse_mount/helmet-data"
    """
    Executor-visible output root (relative to the Executor prefix) where extracted HELMET data is stored.
    The actual directory used will be suffixed with the resolved dataset SHA.
    """

    def require_use_chat_template(self) -> bool:
        if self.use_chat_template is None:
            raise ValueError("HelmetConfig.use_chat_template must be set (True/False).")
        return self.use_chat_template
