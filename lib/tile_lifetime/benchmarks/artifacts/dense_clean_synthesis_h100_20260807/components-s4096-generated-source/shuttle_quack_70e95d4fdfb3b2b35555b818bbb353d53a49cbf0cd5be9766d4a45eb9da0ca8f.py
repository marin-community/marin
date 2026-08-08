import cutlass.cute as cute
from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, TileLoad
from quack.operand_transform import a_transform

generated_transform = None

@gemm_epilogue(ops={'operand_0': TileLoad('operand_0')}, mode='acc_pair')
def generated_epilogue(acc, operand_0):
    pair_left, pair_right = unpack(acc)
    coefficient_0, coefficient_1 = unpack(operand_0)
    mapped = pack(pair_left * coefficient_0 - pair_right * coefficient_1, pair_left * coefficient_1 + pair_right * coefficient_0)
    return {'D': mapped}
