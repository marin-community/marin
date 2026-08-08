# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from quack.activation import swiglu
from quack.epilogue import gemm_epilogue, unpack
from quack.operand_transform import a_transform


@a_transform(vec_size=8, args={"operand_0": "colvec_ktile_fp32"})
def generated_transform(activation, operand_0):

    return activation * operand_0


@gemm_epilogue(outputs=("output_0",), mode="acc_pair")
def generated_epilogue(acc):
    gate_0, up_0 = unpack(acc)
    value_1 = swiglu(gate_0, up_0)
    return {"output_0": value_1}
