import cutlass.cute as cute
from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, TileLoad
from quack.operand_transform import a_transform

generated_transform = None

@gemm_epilogue(outputs=('output_0',), ops={'operand_0': ColVecLoad('operand_0')}, mode='acc_pair')
def generated_epilogue(acc, operand_0):
    value_0 = acc * operand_0
    left_1, right_1 = unpack(value_0)
    value_2 = left_1 * (1.0 / (1.0 + cute.exp(-1.0 * left_1))) * right_1
    return {'output_0': value_2}
