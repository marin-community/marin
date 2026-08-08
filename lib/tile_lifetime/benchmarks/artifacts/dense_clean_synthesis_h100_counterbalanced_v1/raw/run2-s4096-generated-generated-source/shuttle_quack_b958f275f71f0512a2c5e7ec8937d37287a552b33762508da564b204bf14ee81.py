import cutlass.cute as cute
from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, TileLoad
from quack.operand_transform import a_transform

generated_transform = None

@gemm_epilogue(ops={'operand_0': ColVecLoad('operand_0'), 'operand_1': TileLoad('operand_1')}, mode='acc_pair')
def generated_epilogue(acc, operand_0, operand_1):
    value_0 = acc * operand_0
    pair_left, pair_right = unpack(value_0)
    coefficient_0, coefficient_1 = unpack(operand_1)
    mapped = pack(pair_left * coefficient_0 - pair_right * coefficient_1, pair_left * coefficient_1 + pair_right * coefficient_0)
    return {'D': mapped}
