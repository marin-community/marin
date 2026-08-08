# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.rotary import rotary_cos_sin_load

generated_transform = None


@gemm_epilogue(ops={"operand_0": rotary_cos_sin_load("operand_0")}, mode="acc_pair")
def generated_epilogue(acc, operand_0):
    rope_x, rope_y = unpack(acc)
    rope_cos, rope_sin = unpack(operand_0)
    rotated = pack(rope_x * rope_cos - rope_y * rope_sin, rope_x * rope_sin + rope_y * rope_cos)
    return {"D": rotated}
