# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the `ModelLauncher` protocol.

Brings up a vLLM OpenAI-compatible server for a `ModelDeployment`, stages HF
tokenizer files locally when the model lives in object storage, and yields a
`RunningModel` whose `tokenizer_ref` is resolvable by lm-eval's `local-completions`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

from fsspec.implementations.local import LocalFileSystem
from rigging.filesystem import open_url, url_to_fs

from marin.inference.model_launcher import ModelDeployment, OpenAIEndpoint, RunningModel
from marin.inference.vllm_server import VllmEnvironment

# HF tokenizer files that need to be present locally for lm-eval's
# `local-completions` model when the served model is in an object store.
_TOKENIZER_FILENAMES: tuple[str, ...] = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "config.json",
)


class VllmLauncher:
    """Launches a vLLM-backed OpenAI server from a `ModelDeployment`."""

    def __init__(self, *, extra_args: list[str] | None = None) -> None:
        self._extra_args = extra_args

    @contextmanager
    def launch(self, deployment: ModelDeployment) -> Iterator[RunningModel]:
        if deployment.path is None:
            raise ValueError(
                "VllmLauncher requires ModelDeployment.path to be set; "
                "an external OpenAI-style API should construct RunningModel directly instead."
            )
        with VllmEnvironment(
            path=deployment.path,
            engine_kwargs=dict(deployment.engine_kwargs),
            extra_args=self._extra_args,
        ) as env:
            if env.model_id is None:
                raise RuntimeError("vLLM server did not report a model id.")
            staged_cm = (
                _stage_remote_tokenizer_dir(env.model_name_or_path)
                if _is_remote(env.model_name_or_path)
                else nullcontext(None)
            )
            with staged_cm as staged_tokenizer_dir:
                tokenizer_ref = _resolve_tokenizer_ref(deployment, env, staged_tokenizer_dir)
                yield RunningModel(
                    endpoint=OpenAIEndpoint(url=env.server_url, model=env.model_id),
                    tokenizer_ref=tokenizer_ref,
                )


def _is_remote(path: str) -> bool:
    fs, _ = url_to_fs(path)
    return not isinstance(fs, LocalFileSystem)


@contextmanager
def _stage_remote_tokenizer_dir(remote_dir: str) -> Iterator[str | None]:
    """Download HF tokenizer files from `remote_dir` to a tmp dir for lm-eval.

    Yields the staged directory path, or None if no tokenizer files were found.
    The directory is cleaned up on exit.
    """
    with tempfile.TemporaryDirectory(prefix="marin-tokenizer-") as local_dir:
        copied_any = False
        for filename in _TOKENIZER_FILENAMES:
            remote_path = f"{remote_dir.rstrip('/')}/{filename}"
            if not _is_remote(remote_path):
                continue
            fs, fs_path = url_to_fs(remote_path)
            if not fs.exists(fs_path):
                continue
            local_path = os.path.join(local_dir, filename)
            with open_url(remote_path, "rb") as src:
                data = src.read()
            with open(local_path, "wb") as dst:
                dst.write(data)
            copied_any = True
        if not copied_any:
            yield None
            return
        yield local_dir


def _resolve_tokenizer_ref(
    deployment: ModelDeployment,
    env: VllmEnvironment,
    staged_tokenizer_dir: str | None,
) -> str:
    """Pick the tokenizer that clients like lm-eval should load.

    Priority: explicit override > locally-staged dir (for object-store models) >
    the server-reported model id (which for HF repos is itself a valid tokenizer).
    """
    if deployment.tokenizer_override is not None:
        return deployment.tokenizer_override
    if staged_tokenizer_dir is not None:
        return staged_tokenizer_dir
    if env.model_id is None:
        raise RuntimeError("vLLM server did not report a model id.")
    if _is_remote(env.model_name_or_path):
        # Served path is an object-store URI and no tokenizer was stageable —
        # lm-eval can't load a tokenizer from the model id. Point at the override.
        raise ValueError(
            "lm-eval's `local-completions` model requires a Hugging Face tokenizer name/path, "
            f"but the served model id is a remote object-store URI: {env.model_id!r}, and no "
            f"tokenizer files were found under {env.model_name_or_path!r}. "
            "Set `ModelDeployment.tokenizer_override` to an HF tokenizer id (e.g. "
            "'meta-llama/Llama-3.1-8B-Instruct') or a local tokenizer path."
        )
    return env.model_id
