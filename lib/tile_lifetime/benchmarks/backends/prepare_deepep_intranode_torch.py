# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a pinned DeepEP worktree for an intranode-only Torch build."""

import argparse
import subprocess
from pathlib import Path

DEEPEP_COMMIT = "7febc6e25660af0f54d95dd781ecdcd62265ecca"


def _replace_once(path: Path, old: str, new: str) -> None:
    contents = path.read_text()
    if contents.count(old) != 1:
        raise ValueError(f"expected exactly one metadata block in {path}")
    path.write_text(contents.replace(old, new))


def _prepare_worktree(source_root: Path, output_root: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != DEEPEP_COMMIT:
        raise ValueError(f"DeepEP checkout is {revision}; expected {DEEPEP_COMMIT}")
    if output_root.exists():
        raise FileExistsError(f"output worktree already exists: {output_root}")
    subprocess.run(
        ["git", "-C", str(source_root), "worktree", "add", "--detach", str(output_root), DEEPEP_COMMIT],
        check=True,
    )

    _replace_once(
        output_root / "deep_ep" / "__init__.py",
        "from .hybrid_ep_buffer import HybridEPBuffer\n\n",
        "\n",
    )
    _replace_once(
        output_root / "deep_ep" / "__init__.py",
        "from hybrid_ep_cpp import HybridEpConfigInstance\n",
        "",
    )
    _replace_once(
        output_root / "setup.py",
        """    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
""",
        """    disable_nvshmem = os.getenv('DEEPEP_DISABLE_NVSHMEM', '0') == '1'
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if disable_nvshmem:
        nvshmem_dir = None
    elif nvshmem_dir is None:
""",
    )
    _replace_once(
        output_root / "setup.py",
        """    include_dirs = ['csrc/']
    library_dirs = []
""",
        """    include_dirs = ['csrc/']
    cuda_cccl_include = os.getenv('CUDA_CCCL_INCLUDE')
    if cuda_cccl_include:
        include_dirs.append(cuda_cccl_include)
    library_dirs = []
""",
    )
    _replace_once(
        output_root / "setup.py",
        """        ext_modules=[
            get_extension_deep_ep_cpp(),
            get_extension_hybrid_ep_cpp()
        ],
""",
        """        ext_modules=(
            [get_extension_deep_ep_cpp()]
            if os.getenv('DEEPEP_BUILD_INTRANODE_ONLY', '0') == '1'
            else [get_extension_deep_ep_cpp(), get_extension_hybrid_ep_cpp()]
        ),
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _prepare_worktree(args.source_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
