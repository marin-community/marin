import cutlass.cute as cute
from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, TileLoad
from quack.operand_transform import a_transform

@a_transform(vec_size=8, args={'operand_0': 'colvec_ktile_fp32'})
def generated_transform(activation, operand_0):

    return activation * operand_0

@gemm_epilogue(outputs=('output_0',), mode='acc_pair')
def generated_epilogue(acc):
    left_0, right_0 = unpack(acc)
    value_1 = left_0 * (1.0 / (1.0 + cute.exp(-1.0 * left_0))) * right_0
    return {'output_0': value_1}
