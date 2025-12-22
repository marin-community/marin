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

import subprocess


def resolve_git_sha(repo_url: str, ref: str = "refs/heads/main") -> str:
    """Resolve `ref` in `repo_url` to a full commit SHA via `git ls-remote`."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url, ref],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to resolve git ref {ref} for {repo_url}") from e

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        raise RuntimeError(f"No ls-remote output for {repo_url} {ref}")
    sha = line.split()[0]
    if len(sha) < 7:
        raise RuntimeError(f"Unexpected git SHA output for {repo_url} {ref}: {line}")
    return sha


def resolve_hf_dataset_sha(repo_id: str, revision: str = "main") -> str:
    """Resolve a HF dataset repo revision to the underlying commit SHA via HF API."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"HF repo_info returned no sha for dataset {repo_id}@{revision}")
    return sha
