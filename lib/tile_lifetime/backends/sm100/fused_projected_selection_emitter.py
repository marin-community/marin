# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a direct SM100 Contract/maximum-Fold physical instantiation."""

from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPLATE_SHA256 = {
    "fmha_sm100_inst.jinja": "7181b94bae8be3631824343e8ff68cbc001d38d1b929eebe9958928f968a7423",
    "fmha_sm100_variant_run.cu.jinja": "3e8af59327dcbe3bf354a17b7445c027b9aa04b664a97ff4dad4cb52e8c4a9cc",
    "fmha_sm100_params.h": "a776dc84103e87931672d903a2b5737b163182d0a1ea7d07c9d3d105e8b34782",
    "tvm_ffi_utils.h": "e77134f7fc39637bcdd490e6371ee140014dbadf9a670627b673c9737a7e9b86",
    "gmem_bounds_check.h": "c6a174cd5d392bca154c17a119da4bd2ff1116371a20609b38f4eea27fa43597",
}

GENERATED_VARIANT_NAME = "shuttle_projected_contract_maximum_fold_bf16_m256n128k128"
GENERATED_KERNEL_NAME = "shuttle_projected_contract_maximum_fold_kernel"
GENERATED_ENTRY_NAME = f"run_{GENERATED_VARIANT_NAME}"


@dataclass(frozen=True)
class DirectProjectedSelectionSources:
    """Generated direct physical instantiation and static audit."""

    instantiation_source: str
    binding_source: str
    source_sha256: dict[str, str]
    retained_physical_tokens: tuple[str, ...]
    forbidden_tokens: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.forbidden_tokens


def render_direct_projected_selection_sources(msa_root: Path) -> DirectProjectedSelectionSources:
    """Render the low-level score Contract/Fold template from pinned sources."""
    csrc = msa_root / "python" / "fmha_sm100" / "csrc"
    source_hashes = _verify_templates(csrc)
    return render_direct_projected_selection_template_text(
        (csrc / "fmha_sm100_inst.jinja").read_text(),
        (csrc / "fmha_sm100_variant_run.cu.jinja").read_text(),
        source_sha256=source_hashes,
    )


def render_direct_projected_selection_template_text(
    instantiation_template: str,
    binding_template: str,
    *,
    source_sha256: dict[str, str],
) -> DirectProjectedSelectionSources:
    """Instantiate a generic score-Contract/maximum-Fold physical template."""
    jinja2 = importlib.import_module("jinja2")
    parameters = {
        "variant_name": GENERATED_VARIANT_NAME,
        "func_name": GENERATED_KERNEL_NAME,
        "dtype_in": "nv_bfloat16",
        "cutlass_dtype_out": "cutlass::bfloat16_t",
        "tile_q": "_256",
        "tile_kv": "_128",
        "thread_shape": "_2, _1, _1",
        "is_split_kv": "false",
        "single_wg": "false",
        "page_size": 128,
        "pack_factor": 1,
        "sparse_mode": "OnlyScore",
    }
    instantiation = jinja2.Template(instantiation_template).render(**parameters)
    binding = jinja2.Template(binding_template).render(**parameters)
    instantiation = instantiation.replace(
        f"Auto-generated FMHA variant: {GENERATED_VARIANT_NAME}",
        "Generated Shuttle Contract plus maximum-Fold physical instantiation",
    )
    binding = binding.replace(
        f"Auto-generated per-variant FMHA run binding: {GENERATED_VARIANT_NAME}",
        "Generated Shuttle Contract plus maximum-Fold FFI binding",
    ).replace(
        f"FMHA variant {GENERATED_VARIANT_NAME} failed",
        "Shuttle projected Contract/maximum-Fold kernel failed",
    )
    combined = f"{instantiation}\n{binding}"
    forbidden_candidates = (
        "get_fmha_variant",
        "fmha_sm100(",
        "sparse_atten_func",
        "sparse_fmha(",
    )
    forbidden = tuple(token for token in forbidden_candidates if token in combined)
    required_candidates = (
        "run_fmha_fwd<",
        "SparseAttnMode::OnlyScore",
        "CausalMask",
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC",
    )
    retained = tuple(token for token in required_candidates if token in combined)
    if set(retained) != set(required_candidates):
        missing = sorted(set(required_candidates) - set(retained))
        raise ValueError(f"generated Contract/Fold instantiation lost physical tokens {missing}")
    return DirectProjectedSelectionSources(
        instantiation_source=instantiation,
        binding_source=binding,
        source_sha256=source_sha256,
        retained_physical_tokens=retained,
        forbidden_tokens=forbidden,
    )


def compile_direct_projected_selection(
    msa_root: Path,
    *,
    cache_root: Path | None = None,
) -> tuple[Any, DirectProjectedSelectionSources]:
    """Compile and load the generated physical entry without a variant manager."""
    sources = render_direct_projected_selection_sources(msa_root)
    if not sources.clean:
        raise ValueError(f"generated physical source retains forbidden calls {sources.forbidden_tokens}")
    source_digest = hashlib.sha256(f"{sources.instantiation_source}\n{sources.binding_source}".encode()).hexdigest()[:16]
    if cache_root is None:
        cache_root = Path.home() / ".cache" / "shuttle" / "projected_selection"
    cache_dir = cache_root / source_digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    instantiation_path = cache_dir / "shuttle_projected_contract_fold_inst.cu"
    binding_path = cache_dir / "shuttle_projected_contract_fold_binding.cu"
    shared_object = cache_dir / "shuttle_projected_contract_fold.so"
    _write_if_changed(instantiation_path, sources.instantiation_source)
    _write_if_changed(binding_path, sources.binding_source)

    csrc = msa_root / "python" / "fmha_sm100" / "csrc"
    for name in ("fmha_sm100_params.h", "tvm_ffi_utils.h", "gmem_bounds_check.h"):
        destination = cache_dir / name
        if not destination.exists() or destination.read_bytes() != (csrc / name).read_bytes():
            shutil.copy2(csrc / name, destination)

    python_root = str((msa_root / "python").resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    jit = importlib.import_module("fmha_sm100.jit")
    nvcc = Path(jit._get_cuda_home()) / "bin" / "nvcc"
    flags = jit._get_nvcc_flags(cache_dir)
    instantiation_object = cache_dir / "instantiation.o"
    binding_object = cache_dir / "binding.o"
    ninja_source = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda

build {instantiation_object}: nvcc_compile {instantiation_path}
build {binding_object}: nvcc_compile {binding_path}
build {shared_object}: nvcc_link {instantiation_object} {binding_object}
"""
    _write_if_changed(cache_dir / "build.ninja", ninja_source)
    if not shared_object.exists():
        result = subprocess.run(
            ["ninja", "-j1"],
            cwd=cache_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "direct projected Contract/Fold compilation failed:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    tvm_ffi = importlib.import_module("tvm_ffi")
    module = tvm_ffi.load_module(str(shared_object))
    return getattr(module, GENERATED_ENTRY_NAME), sources


def _verify_templates(csrc: Path) -> dict[str, str]:
    observed = {}
    for name, expected in TEMPLATE_SHA256.items():
        digest = hashlib.sha256((csrc / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"pinned MSA source mismatch for {name}: {digest} != {expected}")
        observed[name] = digest
    return observed


def _write_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
