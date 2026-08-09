# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tile_lifetime.ffi_command_buffer import (
    audit_ffi_command_buffer_eligibility,
    finalize_ffi_handler_source,
    require_custom_call_command_buffers_enabled,
)

_TEMPLATE = """
ffi::Error Handler(cudaStream_t stream) {
  Kernel<<<1, 128, 0, stream>>>();
  return ffi::Error::Success();
}
auto Binding() { return ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>(); }
XLA_FFI_DEFINE_HANDLER_SYMBOL(symbol, Handler, Binding()__SHUTTLE_FFI_HANDLER_TRAITS__);
"""


def test_capture_safe_handler_receives_command_buffer_trait() -> None:
    source = finalize_ffi_handler_source(_TEMPLATE, command_buffer_compatible=True)

    assert audit_ffi_command_buffer_eligibility(source).eligible
    assert "Binding(),\n    {ffi::Traits::kCmdBufferCompatible}" in source


def test_multiple_capture_safe_handlers_receive_command_buffer_traits() -> None:
    template = _TEMPLATE + _TEMPLATE.replace("symbol", "second_symbol")

    source = finalize_ffi_handler_source(
        template,
        command_buffer_compatible=True,
        expected_handler_count=2,
    )

    assert audit_ffi_command_buffer_eligibility(source).eligible
    assert source.count("{ffi::Traits::kCmdBufferCompatible}") == 2


@pytest.mark.parametrize(
    ("operation", "fragment"),
    (
        ("runtime scratch allocation", ".Ctx<ffi::ScratchAllocator>()"),
        ("runtime device allocation", "cudaMalloc(&pointer, bytes);"),
        ("lazy library handle creation", "std::call_once(once, [] { cublasCreate(&handle); });"),
        ("runtime autotuning", "select_autotuned_algorithm();"),
        ("runtime launch-status query", "cudaPeekAtLastError();"),
    ),
)
def test_ineligible_handler_cannot_receive_command_buffer_trait(operation: str, fragment: str) -> None:
    template = _TEMPLATE.replace("return ffi::Error::Success();", f"{fragment}\n  return ffi::Error::Success();")

    with pytest.raises(ValueError, match=operation):
        finalize_ffi_handler_source(template, command_buffer_compatible=True)


def test_custom_call_flag_accepts_default_and_explicit_selection() -> None:
    default = require_custom_call_command_buffers_enabled("")
    explicit = require_custom_call_command_buffers_enabled(
        "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=5"
    )
    incremental = require_custom_call_command_buffers_enabled("--xla_gpu_enable_command_buffer=+FUSION,-CUDNN")

    assert default.uses_xla_default
    assert default.selected_entries == ()
    assert explicit.selected_entries == ("FUSION", "CUBLAS", "CUSTOM_CALL")
    assert incremental.selected_entries == ("+FUSION", "-CUDNN")


def test_custom_call_flag_rejects_explicit_disable() -> None:
    with pytest.raises(ValueError, match="explicitly disables"):
        require_custom_call_command_buffers_enabled("--xla_gpu_enable_command_buffer=-CUSTOM_CALL")


def test_custom_call_flag_rejects_absolute_selection_without_custom_calls() -> None:
    with pytest.raises(ValueError, match="excludes CUSTOM_CALL"):
        require_custom_call_command_buffers_enabled("--xla_gpu_enable_command_buffer=FUSION,CUBLAS")
