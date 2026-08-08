# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.rotary import rotary_cos_sin_load
from quack.operand_transform import a_transform


@a_transform(vec_size=8, args={"operand_0": "colvec_ktile_fp32"})
def generated_transform(activation, operand_0):

    return activation * operand_0


@gemm_epilogue(ops={"operand_1": rotary_cos_sin_load("operand_1")}, mode="acc_pair")
def generated_epilogue(acc, operand_1):
    rope_x, rope_y = unpack(acc)
    rope_cos, rope_sin = unpack(operand_1)
    rotated = pack(rope_x * rope_cos - rope_y * rope_sin, rope_x * rope_sin + rope_y * rope_cos)
    return {"D": rotated}
