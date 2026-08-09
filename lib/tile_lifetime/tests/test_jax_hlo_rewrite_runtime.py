# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest

from tile_lifetime.jax_hlo_rewrite_runtime import (
    HloRewriteRuntimeUnavailable,
    _public_compiler_ir_hlo_module_type,
    _resolve_hlo_text_parser,
    audit_hlo_rewrite_runtime,
    require_hlo_rewrite_runtime,
)


def test_public_compiler_ir_roundtrips_hlo_proto_and_private_parser_is_isolated() -> None:
    lowered = jax.jit(lambda value: value + 1).lower(jax.ShapeDtypeStruct((), jax.numpy.float32))
    compiler_ir = lowered.compiler_ir(dialect="hlo")
    serialized = compiler_ir.as_serialized_hlo_module_proto()

    module_type = _public_compiler_ir_hlo_module_type()
    restored = module_type.from_serialized_hlo_module_proto(serialized)
    parser, backend = _resolve_hlo_text_parser()
    parsed = parser(compiler_ir.as_hlo_text())

    assert restored.name == compiler_ir.name()
    assert parsed.name == compiler_ir.name()
    assert backend in ("jaxlib._jax.hlo_module_from_text", "jaxlib._hlo.hlo_module_from_text")


def test_repo_pinned_jax_reports_pre_scheduler_registration_boundary() -> None:
    audit = audit_hlo_rewrite_runtime()

    assert audit.compiler_ir_proto_roundtrip
    assert audit.compiler_ir_module_type is not None
    assert audit.text_parser_backend is not None
    if audit.transformation_api is None:
        with pytest.raises(HloRewriteRuntimeUnavailable, match=r"jax\.extend\.xla is unavailable"):
            require_hlo_rewrite_runtime()
    else:
        runtime = require_hlo_rewrite_runtime()
        assert runtime.transformation_api.__name__ == "jax.extend.xla"
