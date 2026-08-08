import cutlass.cute as cute
from quack.epilogue import gemm_epilogue, pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad, TileLoad
from quack.operand_transform import a_transform

generated_transform = None

@gemm_epilogue(outputs=('output_0',), reduces={'reduction_0': ColVecReduce('reduction_0', scaled=True)}, ops={'operand_0': TileLoad('operand_0'), 'operand_1': RowVecLoad('operand_1')})
def generated_epilogue(acc, operand_0, operand_1):
    value_0 = acc + operand_0
    return {'D': value_0 * operand_1, 'output_0': value_0, 'reduction_0': (value_0, value_0)}
