import cutlass.cute as cute
from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, TileLoad
from quack.operand_transform import a_transform

@a_transform(vec_size=8, args={'operand_0': 'colvec_ktile_fp32'})
def generated_transform(activation, operand_0):

    return activation * operand_0

@gemm_epilogue(ops={'operand_1': TileLoad('operand_1')}, mode='acc_pair')
def generated_epilogue(acc, operand_1):
    pair_left, pair_right = unpack(acc)
    coefficient_0, coefficient_1 = unpack(operand_1)
    mapped = pack(pair_left * coefficient_0 - pair_right * coefficient_1, pair_left * coefficient_1 + pair_right * coefficient_0)
    return {'D': mapped}
