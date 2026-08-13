# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the rebuilt CUDA plugin's exported Shuttle typed-FFI bundle."""

import argparse
import ctypes
import importlib.util
from pathlib import Path
from types import ModuleType

TARGET = "shuttle.gpu.executable_bundle.v1"
STAGES = ("instantiate", "prepare", "initialize", "execute")
GPU_CUSTOM_CALL_EXTENSION = 0


class PjrtExtensionBase(ctypes.Structure):
    pass


PjrtExtensionBase._fields_ = [
    ("struct_size", ctypes.c_size_t),
    ("type", ctypes.c_int),
    ("next", ctypes.POINTER(PjrtExtensionBase)),
]


class PjrtGpuRegisterCustomCallArgs(ctypes.Structure):
    pass


PjrtGpuRegisterCustomCallArgs._fields_ = [
    ("struct_size", ctypes.c_size_t),
    ("function_name", ctypes.c_char_p),
    ("function_name_size", ctypes.c_size_t),
    ("api_version", ctypes.c_int),
    ("handler_instantiate", ctypes.c_void_p),
    ("handler_prepare", ctypes.c_void_p),
    ("handler_initialize", ctypes.c_void_p),
    ("handler_execute", ctypes.c_void_p),
]


RegisterCustomCall = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(PjrtGpuRegisterCustomCallArgs))


class PjrtGpuCustomCall(ctypes.Structure):
    pass


PjrtGpuCustomCall._fields_ = [
    ("base", PjrtExtensionBase),
    ("custom_call", RegisterCustomCall),
]


class PjrtApiPrefix(ctypes.Structure):
    pass


PjrtApiPrefix._fields_ = [
    ("struct_size", ctypes.c_size_t),
    ("extension_start", ctypes.POINTER(PjrtExtensionBase)),
]


def load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cuda_plugin_extension", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CUDA plugin extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capsule(pointer: int) -> object:
    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    new_capsule.restype = ctypes.py_object
    return new_capsule(pointer, None, None)


def capsule_pointer(value: object) -> int:
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(value, None)
    if pointer is None:
        raise RuntimeError("CUDA plugin exported a null handler capsule")
    return int(pointer)


def register_through_pjrt(module: ModuleType, bundle: dict[str, object]) -> None:
    observations: list[tuple[str, int, tuple[int, ...]]] = []

    @RegisterCustomCall
    def register(args_pointer: ctypes.POINTER(PjrtGpuRegisterCustomCallArgs)) -> None:
        args = args_pointer.contents
        target = ctypes.string_at(args.function_name, args.function_name_size).decode()
        observations.append(
            (
                target,
                args.api_version,
                tuple(
                    int(pointer)
                    for pointer in (
                        args.handler_instantiate,
                        args.handler_prepare,
                        args.handler_initialize,
                        args.handler_execute,
                    )
                ),
            )
        )
        return None

    extension = PjrtGpuCustomCall(
        base=PjrtExtensionBase(
            struct_size=ctypes.sizeof(PjrtGpuCustomCall),
            type=GPU_CUSTOM_CALL_EXTENSION,
            next=ctypes.POINTER(PjrtExtensionBase)(),
        ),
        custom_call=register,
    )
    api = PjrtApiPrefix(
        struct_size=ctypes.sizeof(PjrtApiPrefix),
        extension_start=ctypes.pointer(extension.base),
    )
    module.register_custom_call_target(
        capsule(ctypes.addressof(api)),
        TARGET,
        bundle,
        "CUDA",
        api_version=1,
        traits=0,
    )

    if len(observations) != 1:
        raise RuntimeError("CUDA plugin did not make one PJRT registration call")
    target, api_version, pointers = observations[0]
    exported_pointers = tuple(capsule_pointer(bundle[stage]) for stage in STAGES)
    if target != TARGET or api_version != 1 or pointers != exported_pointers:
        raise RuntimeError("CUDA plugin registered the wrong Shuttle typed-FFI bundle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-extension", type=Path, required=True)
    args = parser.parse_args()

    module = load_extension(args.plugin_extension.resolve())
    handlers = module.ffi_handlers()
    if TARGET not in handlers:
        raise RuntimeError(f"missing Shuttle CUDA typed-FFI target: {sorted(handlers)}")
    bundle = handlers[TARGET]
    if set(bundle) != {*STAGES, "api_version", "traits"}:
        raise RuntimeError("CUDA plugin exported an unknown Shuttle handler field")
    if bundle["api_version"] != 1 or bundle["traits"] != 0:
        raise RuntimeError("CUDA plugin exported the wrong Shuttle typed-FFI metadata")
    if any(bundle[stage] is None for stage in STAGES):
        raise RuntimeError("CUDA plugin did not export one complete Shuttle handler bundle")
    register_through_pjrt(module, bundle)


if __name__ == "__main__":
    main()
