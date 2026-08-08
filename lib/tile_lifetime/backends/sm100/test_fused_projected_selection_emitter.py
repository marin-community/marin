# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

BACKEND = Path(__file__).parent
sys.path.insert(0, str(BACKEND))

from fused_projected_selection_emitter import (  # noqa: E402
    render_direct_projected_selection_template_text,
)


def test_direct_emitter_instantiates_contract_fold_without_public_kernel_call() -> None:
    sources = render_direct_projected_selection_template_text(
        "{{ func_name }} run_fmha_fwd< SparseAttnMode::{{ sparse_mode }} CausalMask",
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_{{ variant_name }}, binding);",
        source_sha256={"upstream-template": "pinned-digest"},
    )

    assert sources.clean
    assert sources.forbidden_tokens == ()
    assert set(sources.retained_physical_tokens) == {
        "run_fmha_fwd<",
        "SparseAttnMode::OnlyScore",
        "CausalMask",
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC",
    }
    assert "shuttle_projected_contract_maximum_fold_kernel" in sources.instantiation_source
    assert "run_shuttle_projected_contract_maximum_fold_bf16_m256n128k128" in sources.binding_source
    assert sources.source_sha256 == {"upstream-template": "pinned-digest"}
