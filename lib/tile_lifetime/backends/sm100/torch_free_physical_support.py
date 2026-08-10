# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the Torch-free subset of QuACK used by generic SM100 primitives."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

QUACK_DISTRIBUTION = "quack-kernels"
QUACK_VERSION = "0.2.10"
SAFE_QUACK_MODULES = ("activation", "copy_utils", "layout_utils")
MSA_ALIGNMENT_FUNCTIONS = ("assume_strides_aligned", "assume_tensor_aligned")


@dataclass(frozen=True)
class TorchFreePhysicalSupport:
    """Pinned source identity for low-level physical helper modules."""

    distribution: str
    version: str
    source_root: Path
    msa_source_root: Path
    source_sha256: tuple[tuple[str, str], ...]
    loaded_modules: tuple[str, ...]


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _load_source_module(name: str, source_path: Path) -> ModuleType:
    source = source_path.read_text()
    parsed = ast.parse(source)
    imports_torch = any(
        (isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "torch" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch")
        for node in ast.walk(parsed)
    )
    if imports_torch:
        raise ValueError(f"physical support module {source_path} imports Torch")
    spec = importlib.util.spec_from_file_location(name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create a module specification for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _params_base_module(source_path: Path) -> tuple[ModuleType, str]:
    """Extract QuACK's generic CuTe parameter carrier without its Torch adapters."""
    source = source_path.read_text()
    parsed = ast.parse(source)
    classes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "ParamsBase"]
    if len(classes) != 1:
        raise ValueError(f"expected exactly one ParamsBase definition in {source_path}")
    class_source = ast.unparse(classes[0])
    extracted = "\n".join(
        (
            "from dataclasses import dataclass, fields",
            "import cutlass",
            "from cutlass.cutlass_dsl import NumericMeta",
            "StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))",
            class_source,
            "",
        )
    )
    name = "quack.cute_dsl_utils"
    module = ModuleType(name)
    module.__file__ = str(source_path)
    module.__package__ = "quack"
    sys.modules[name] = module
    exec(compile(extracted, f"<{name}:ParamsBase>", "exec"), module.__dict__)
    return module, _source_hash(extracted)


def _msa_alignment_module(source_path: Path) -> tuple[ModuleType, str]:
    """Extract generic CuTe alignment assumptions without MSA's Torch adapters."""
    source = source_path.read_text()
    parsed = ast.parse(source)
    functions = {
        node.name: node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in MSA_ALIGNMENT_FUNCTIONS
    }
    if tuple(functions) != MSA_ALIGNMENT_FUNCTIONS:
        raise ValueError(f"missing generic alignment functions in {source_path}: {tuple(functions)}")
    extracted = "\n".join(("import cutlass.cute as cute", *(ast.unparse(functions[name]) for name in functions), ""))
    name = "src.common.cute_dsl_utils"
    module = ModuleType(name)
    module.__file__ = str(source_path)
    module.__package__ = "src.common"
    sys.modules[name] = module
    exec(compile(extracted, f"<{name}:alignment>", "exec"), module.__dict__)
    return module, _source_hash(extracted)


def install_torch_free_physical_support(msa_source_root: Path) -> TorchFreePhysicalSupport:
    """Install only generic QuACK CuTe helpers without importing its package.

    The upstream ``quack`` package initializer and MSA's general runtime helper
    import Torch-facing wrappers. The extracted SM100 skeleton needs only three
    architecture-generic QuACK modules, ``ParamsBase``, and two MSA alignment
    functions. Loading those definitions directly keeps the JAX runtime free of
    Torch while retaining their pinned low-level implementation.
    """
    if "torch" in sys.modules:
        raise RuntimeError("Torch was loaded before physical support installation")
    distribution = importlib.metadata.distribution(QUACK_DISTRIBUTION)
    if distribution.version != QUACK_VERSION:
        raise RuntimeError(f"expected {QUACK_DISTRIBUTION}=={QUACK_VERSION}, found {distribution.version}")
    source_root = Path(distribution.locate_file("quack")).resolve()
    if not source_root.is_dir():
        raise ValueError(f"QuACK source root does not exist: {source_root}")
    if not msa_source_root.is_dir():
        raise ValueError(f"MSA CuTe source root does not exist: {msa_source_root}")
    if str(msa_source_root) not in sys.path:
        sys.path.insert(0, str(msa_source_root))

    module_names = (
        "quack",
        *(f"quack.{name}" for name in SAFE_QUACK_MODULES),
        "quack.cute_dsl_utils",
        "src.common.cute_dsl_utils",
    )
    existing = {name: sys.modules.get(name) for name in module_names}
    package = ModuleType("quack")
    package.__file__ = str(source_root / "__init__.py")
    package.__package__ = "quack"
    package.__path__ = [str(source_root)]
    sys.modules["quack"] = package
    hashes: list[tuple[str, str]] = []
    loaded: list[str] = []
    try:
        for short_name in SAFE_QUACK_MODULES:
            source_path = source_root / f"{short_name}.py"
            source = source_path.read_text()
            module_name = f"quack.{short_name}"
            module = _load_source_module(module_name, source_path)
            setattr(package, short_name, module)
            hashes.append((module_name, _source_hash(source)))
            loaded.append(module_name)
        params, extracted_hash = _params_base_module(source_root / "cute_dsl_utils.py")
        package.cute_dsl_utils = params
        hashes.append(("quack.cute_dsl_utils:ParamsBase", extracted_hash))
        loaded.append("quack.cute_dsl_utils")
        common_package = importlib.import_module("src.common")
        alignment_source = msa_source_root / "src" / "common" / "cute_dsl_utils.py"
        alignment, alignment_hash = _msa_alignment_module(alignment_source)
        common_package.cute_dsl_utils = alignment
        hashes.append(("src.common.cute_dsl_utils:alignment", alignment_hash))
        loaded.append("src.common.cute_dsl_utils")
        if "torch" in sys.modules:
            raise RuntimeError("Torch was imported while loading physical support")
    except BaseException:
        for name, prior in existing.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
        common_package = sys.modules.get("src.common")
        prior_alignment = existing["src.common.cute_dsl_utils"]
        if common_package is not None and prior_alignment is None:
            common_package.__dict__.pop("cute_dsl_utils", None)
        elif common_package is not None:
            common_package.cute_dsl_utils = prior_alignment
        raise

    return TorchFreePhysicalSupport(
        distribution=QUACK_DISTRIBUTION,
        version=distribution.version,
        source_root=source_root,
        msa_source_root=msa_source_root,
        source_sha256=tuple(hashes),
        loaded_modules=tuple(loaded),
    )
