# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from quack.activation import swiglu
from quack.epilogue import gemm_epilogue, unpack
from quack.epilogue.ops import ColVecLoad

generated_transform = None


@gemm_epilogue(outputs=("output_0",), ops={"operand_0": ColVecLoad("operand_0")}, mode="acc_pair")
def generated_epilogue(acc, operand_0):
    value_0 = acc * operand_0
    gate_1, up_1 = unpack(value_0)
    value_2 = swiglu(gate_1, up_1)
    return {"output_0": value_2}
