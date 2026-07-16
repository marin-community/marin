"""TPU-specific serve-flag helpers.

Extracted from OT-Agent ``hpc/local_runner_utils.py`` (the
``_drop_tpu_unsupported_serve_flags`` and ``_add_tpu_serve_default_flags``
functions), renamed to public.
"""

from __future__ import annotations
from typing import List


def drop_tpu_unsupported_serve_flags(cli_args: List[str]) -> List[str]:
    """Strip serve flags the tpu-inference api_server rejects, from a vLLM CLI arg list.

    ``--swap-space`` (CPU KV offload) is a GPU-only vLLM concept. The
    tpu-inference api_server has no such argument and exits with
    ``error: unrecognized arguments: --swap-space``, tearing down the whole serve.
    """
    out: List[str] = []
    i = 0
    n = len(cli_args)
    while i < n:
        tok = cli_args[i]
        if tok == "--swap-space":
            if i + 1 < n and not str(cli_args[i + 1]).startswith("--"):
                i += 2
            else:
                i += 1
            continue
        if isinstance(tok, str) and tok.startswith("--swap-space="):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def add_tpu_serve_default_flags(cli_args: List[str], default_flags: List[str]) -> List[str]:
    """Append cluster-level default TPU-serve flags that aren't already present.

    A default flag is only added when neither it, its ``=value`` form, nor its
    ``--no-<flag>`` negation already appears in ``cli_args``.
    """
    if not default_flags:
        return list(cli_args)

    def _stem(tok: str) -> str:
        name = str(tok).lstrip("-").split("=", 1)[0]
        if name.startswith("no-"):
            name = name[3:]
        return name

    present = {_stem(tok) for tok in cli_args if isinstance(tok, str) and tok.startswith("--")}
    out = list(cli_args)
    for flag in default_flags:
        if _stem(flag) not in present:
            out.append(flag)
    return out
