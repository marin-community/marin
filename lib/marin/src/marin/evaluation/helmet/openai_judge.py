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

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Literal

import fsspec

from marin.utils import fsspec_copy_path_into_dir


def _local_path_for_gcsfuse_mount(path: str, *, local_mount_root: str = "/opt/gcsfuse") -> str:
    marker = "gcsfuse_mount/"
    if marker not in path:
        return path
    relative = path.split(marker, 1)[1].lstrip("/")
    return os.path.join(local_mount_root, relative)


def _list_files(dir_path: str) -> list[str]:
    fs, root = fsspec.core.url_to_fs(dir_path)
    return [fs.unstrip_protocol(p) for p in fs.find(root)]


def _write_outputs_to_executor_output(local_dir: str, output_path: str) -> None:
    fs_out, out_root = fsspec.core.url_to_fs(output_path)
    fs_out.makedirs(out_root, exist_ok=True)
    for entry in os.listdir(local_dir):
        if not entry.endswith("-gpt4eval_o.json"):
            continue
        src = os.path.join(local_dir, entry)
        dst = os.path.join(out_root, entry)
        fs_out.put(src, dst)


def _select_eval_inputs(eval_output_paths: list[str], *, tag: str, kind: Literal["longqa", "summ"]) -> list[str]:
    patterns: list[re.Pattern[str]]
    if kind == "longqa":
        patterns = [re.compile(rf"^narrativeqa_.*_{re.escape(tag)}_.*\\.json$")]
    else:
        patterns = [
            re.compile(rf"^multi_lexsum_.*_{re.escape(tag)}_.*\\.json$"),
            re.compile(rf"^infbench_sum_.*_{re.escape(tag)}_.*\\.json$"),
        ]

    matches: list[str] = []
    for root in eval_output_paths:
        for p in _list_files(root):
            base = os.path.basename(p)
            if any(pat.match(base) for pat in patterns):
                matches.append(p)
    return sorted(matches)


@dataclass(frozen=True)
class HelmetOpenAiJudgeConfig:
    kind: Literal["longqa", "summ"]

    helmet_repo_url: str
    helmet_repo_sha: str
    helmet_data_output_path: str

    model_name: str
    tag: str

    eval_output_paths: list[str]
    output_path: str

    shard_idx: int
    num_shards: int


def judge_with_helmet_scripts(config: HelmetOpenAiJudgeConfig) -> None:
    """Run HELMET's upstream OpenAI judging scripts.

    This intentionally shells out to HELMET's own `scripts/eval_gpt4_{longqa,summ}.py`
    to avoid reimplementing prompts/scoring and to keep parity with upstream.
    """
    eval_inputs = _select_eval_inputs(config.eval_output_paths, tag=config.tag, kind=config.kind)
    eval_inputs = eval_inputs[config.shard_idx :: config.num_shards]
    if not eval_inputs:
        return

    with tempfile.TemporaryDirectory(prefix="helmet_judge_") as tmpdir:
        repo_dir = os.path.join(tmpdir, "HELMET")
        subprocess.run(["git", "clone", config.helmet_repo_url, repo_dir], check=True)
        subprocess.run(["git", "checkout", config.helmet_repo_sha], check=True, cwd=repo_dir)

        # Symlink data (scripts read e.g. data/multi_lexsum/multi_lexsum_val.jsonl)
        data_root = _local_path_for_gcsfuse_mount(config.helmet_data_output_path)
        data_src = os.path.join(data_root, "data")
        data_dst = os.path.join(repo_dir, "data")
        if os.path.lexists(data_dst):
            if os.path.islink(data_dst) or os.path.isfile(data_dst):
                os.remove(data_dst)
            else:
                shutil.rmtree(data_dst)
        os.symlink(data_src, data_dst)

        # HELMET scripts glob under `output/<model_name>/...`
        output_model_dir = os.path.join(repo_dir, "output", config.model_name)
        os.makedirs(output_model_dir, exist_ok=True)

        for src in eval_inputs:
            fsspec_copy_path_into_dir(src_path=src, dst_path=output_model_dir)

        script = "eval_gpt4_longqa.py" if config.kind == "longqa" else "eval_gpt4_summ.py"
        subprocess.run(
            [
                sys.executable,
                os.path.join("scripts", script),
                "--num_shards",
                "1",
                "--shard_idx",
                "0",
                "--model_to_check",
                config.model_name,
                "--tag",
                config.tag,
            ],
            check=True,
            cwd=repo_dir,
        )

        _write_outputs_to_executor_output(output_model_dir, config.output_path)
