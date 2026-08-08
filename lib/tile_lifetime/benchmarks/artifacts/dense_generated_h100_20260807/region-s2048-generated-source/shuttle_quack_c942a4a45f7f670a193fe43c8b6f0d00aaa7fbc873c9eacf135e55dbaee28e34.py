# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from quack.epilogue import gemm_epilogue
from quack.epilogue.ops import ColVecReduce, RowVecLoad

generated_transform = None


@gemm_epilogue(
    outputs=("output_0",),
    reduces={"reduction_0": ColVecReduce("reduction_0", scaled=True)},
    ops={"operand_0": RowVecLoad("operand_0")},
)
def generated_epilogue(acc, c, operand_0):
    value_0 = acc + c
    return {"D": value_0 * operand_0, "output_0": value_0, "reduction_0": (value_0, value_0)}
