module @jit_transformed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x32xbf16>, %arg1: tensor<8x2xi32>, %arg2: tensor<8x2xf32>, %arg3: tensor<10x32xbf16>, %arg4: tensor<10xi32>, %arg5: tensor<10xi1>, %arg6: tensor<10x64xbf16>, %arg7: tensor<2x5x32xbf16>, %arg8: tensor<2x5x32xbf16>, %arg9: tensor<2x32x32xbf16>, %arg10: tensor<2x32x32xbf16>, %arg11: tensor<2x32x32xbf16>, %arg12: tensor<4x32xbf16>, %arg13: tensor<2x2xf32>, %arg14: tensor<2x32xbf16>, %arg15: tensor<32x8xbf16>, %arg16: tensor<2x2xi32>) -> (tensor<2x32xbf16> {jax.result_info = "result[0]"}, tensor<32x8xbf16> {jax.result_info = "result[1]"}, tensor<10x32xbf16> {jax.result_info = "result[2]"}, tensor<2x32x64xbf16> {jax.result_info = "result[3]"}, tensor<2x32x32xbf16> {jax.result_info = "result[4]"}) {
    %0:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.edge.rank0.edge_reverse(%arg0, %arg1, %arg2, %arg3) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<8x32xbf16>, tensor<8x2xi32>, tensor<8x2xf32>, tensor<10x32xbf16>) -> (tensor<10x32xbf16>, tensor<8x2xf32>)
    %1 = call @_one_hot(%arg4) : (tensor<10xi32>) -> tensor<10x2xbf16>
    %2 = stablehlo.broadcast_in_dim %arg5, dims = [0] : (tensor<10xi1>) -> tensor<10x1xi1>
    %3 = stablehlo.convert %2 : (tensor<10x1xi1>) -> tensor<10x1xbf16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<10x1xbf16>) -> tensor<10x2xbf16>
    %5 = stablehlo.multiply %1, %4 : tensor<10x2xbf16>
    %6 = stablehlo.broadcast_in_dim %0#0, dims = [0, 2] : (tensor<10x32xbf16>) -> tensor<10x1x32xbf16>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<10x2xbf16>) -> tensor<10x2x1xbf16>
    %8 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<10x1x32xbf16>) -> tensor<10x2x32xbf16>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<10x2x1xbf16>) -> tensor<10x2x32xbf16>
    %10 = stablehlo.multiply %8, %9 : tensor<10x2x32xbf16>
    %11 = stablehlo.reshape %10 : (tensor<10x2x32xbf16>) -> tensor<10x64xbf16>
    %12 = stablehlo.transpose %5, dims = [1, 0] : (tensor<10x2xbf16>) -> tensor<2x10xbf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x10xbf16>) -> tensor<2x10x1xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x10x1xbf16>
    %15 = stablehlo.compare NE, %13, %14, FLOAT : (tensor<2x10x1xbf16>, tensor<2x10x1xbf16>) -> tensor<2x10x1xi1>
    %16 = stablehlo.convert %15 : tensor<2x10x1xi1>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x10x1xi1>) -> tensor<2x10x64xi1>
    %18 = stablehlo.transpose %arg9, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
    %19 = stablehlo.reshape %18 : (tensor<2x32x32xbf16>) -> tensor<64x32xbf16>
    %20 = stablehlo.transpose %arg10, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
    %21 = stablehlo.transpose %arg11, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
    %22 = stablehlo.concatenate %20, %21, dim = 1 : (tensor<2x32x32xbf16>, tensor<2x32x32xbf16>) -> tensor<2x64x32xbf16>
    %23 = stablehlo.reshape %22 : (tensor<2x64x32xbf16>) -> tensor<128x32xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %24 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<10x32xbf16>
    %25 = stablehlo.iota dim = 0 : tensor<10xi32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<10xi32>) -> tensor<10x1xi32>
    %27:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.input_adjoint(%23, %24, %26, %17, %11, %19, %arg6) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<128x32xbf16>, tensor<10x32xbf16>, tensor<10x1xi32>, tensor<2x10x64xi1>, tensor<10x64xbf16>, tensor<64x32xbf16>, tensor<10x64xbf16>) -> (tensor<2x10x64xbf16>, tensor<10x32xbf16>)
    %28 = stablehlo.slice %27#0 [0:1, 0:5, 0:64] : (tensor<2x10x64xbf16>) -> tensor<1x5x64xbf16>
    %29 = stablehlo.reshape %28 : (tensor<1x5x64xbf16>) -> tensor<5x64xbf16>
    %30 = stablehlo.slice %27#0 [1:2, 5:10, 0:64] : (tensor<2x10x64xbf16>) -> tensor<1x5x64xbf16>
    %31 = stablehlo.reshape %30 : (tensor<1x5x64xbf16>) -> tensor<5x64xbf16>
    %32 = stablehlo.broadcast_in_dim %29, dims = [1, 2] : (tensor<5x64xbf16>) -> tensor<1x5x64xbf16>
    %33 = stablehlo.broadcast_in_dim %31, dims = [1, 2] : (tensor<5x64xbf16>) -> tensor<1x5x64xbf16>
    %34 = stablehlo.concatenate %32, %33, dim = 0 : (tensor<1x5x64xbf16>, tensor<1x5x64xbf16>) -> tensor<2x5x64xbf16>
    %35 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w13(%arg7, %34) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x64xbf16>) -> tensor<2x32x64xbf16>
    %36 = stablehlo.reshape %0#0 : (tensor<10x32xbf16>) -> tensor<2x5x32xbf16>
    %37 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w2(%arg8, %36) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x32xbf16>) -> tensor<2x32x32xbf16>
    %38 = stablehlo.iota dim = 0 : tensor<2xi32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0] : (tensor<2xi32>) -> tensor<2x2xi32>
    %40 = stablehlo.reshape %39 : (tensor<2x2xi32>) -> tensor<4xi32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %42 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<2x32xbf16>
    %43 = stablehlo.custom_call @shuttle.distributed_expert_cpu.source_fold(%42, %41, %arg12) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x32xbf16>, tensor<4x1xi32>, tensor<4x32xbf16>) -> tensor<2x32xbf16>
    %44 = stablehlo.convert %arg14 : (tensor<2x32xbf16>) -> tensor<2x32xf32>
    %45 = stablehlo.convert %arg15 : (tensor<32x8xbf16>) -> tensor<32x8xf32>
    %46 = stablehlo.dot_general %44, %45, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x32xf32>, tensor<32x8xf32>) -> tensor<2x8xf32>
    %47:2 = call @take_along_axis(%46, %arg16) : (tensor<2x8xf32>, tensor<2x2xi32>) -> (tensor<2x2xf32>, tensor<2x2x1xi32>)
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %48 = stablehlo.reduce(%47#0 init: %cst_2) applies stablehlo.maximum across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %50 = stablehlo.maximum %49, %48 : tensor<2xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
    %53 = stablehlo.subtract %47#0, %52 : tensor<2x2xf32>
    %54 = stablehlo.exponential %53 : tensor<2x2xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %55 = stablehlo.reduce(%54 init: %cst_4) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %57 = stablehlo.multiply %56, %56 : tensor<2x1xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %59 = stablehlo.divide %58, %57 : tensor<2x1xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
    %61 = stablehlo.multiply %arg13, %60 : tensor<2x2xf32>
    %62 = stablehlo.multiply %61, %54 : tensor<2x2xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %63 = stablehlo.reduce(%62 init: %cst_6) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %64 = stablehlo.reshape %63 : (tensor<2xf32>) -> tensor<2x1xf32>
    %65 = stablehlo.negate %64 : tensor<2x1xf32>
    %66 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
    %67 = stablehlo.divide %arg13, %66 : tensor<2x2xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %68 = stablehlo.reduce(%65 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<2xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
    %70 = stablehlo.add %67, %69 : tensor<2x2xf32>
    %71 = stablehlo.multiply %70, %54 : tensor<2x2xf32>
    %72 = call @take_along_axis_40(%47#1, %71) : (tensor<2x2x1xi32>, tensor<2x2xf32>) -> tensor<2x8xf32>
    %73 = stablehlo.dot_general %72, %44, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<2x32xf32>) -> tensor<8x32xf32>
    %74 = stablehlo.transpose %73, dims = [1, 0] : (tensor<8x32xf32>) -> tensor<32x8xf32>
    %75 = stablehlo.dot_general %72, %45, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<32x8xf32>) -> tensor<2x32xf32>
    %76 = stablehlo.convert %74 : (tensor<32x8xf32>) -> tensor<32x8xbf16>
    %77 = stablehlo.convert %75 : (tensor<2x32xf32>) -> tensor<2x32xbf16>
    %78 = stablehlo.add %43, %77 : tensor<2x32xbf16>
    return %78, %76, %27#1, %35, %37 : tensor<2x32xbf16>, tensor<32x8xbf16>, tensor<10x32xbf16>, tensor<2x32x64xbf16>, tensor<2x32x32xbf16>
  }
  func.func private @_one_hot(%arg0: tensor<10xi32>) -> tensor<10x2xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<10xi32>) -> tensor<10x1xi32>
    %1 = stablehlo.iota dim = 1 : tensor<1x2xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<10x1xi32>) -> tensor<10x2xi32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x2xi32>) -> tensor<10x2xi32>
    %4 = stablehlo.compare EQ, %2, %3, SIGNED : (tensor<10x2xi32>, tensor<10x2xi32>) -> tensor<10x2xi1>
    %5 = stablehlo.convert %4 : (tensor<10x2xi1>) -> tensor<10x2xbf16>
    return %5 : tensor<10x2xbf16>
  }
  func.func private @take_along_axis(%arg0: tensor<2x8xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xf32>, tensor<2x2x1xi32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x2xi32>
    %1 = stablehlo.compare LT, %arg1, %0, SIGNED : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %c_0 = stablehlo.constant dense<8> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x2xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<2x2xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<2x2xi1>, tensor<2x2xi32>
    %5 = stablehlo.reshape %4 : (tensor<2x2xi32>) -> tensor<2x2x1xi32>
    %c_1 = stablehlo.constant dense<7> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x2x1xi32>
    %7 = stablehlo.compare GE, %5, %6, SIGNED : (tensor<2x2x1xi32>, tensor<2x2x1xi32>) -> tensor<2x2x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<2x2x1xi32>
    %10 = stablehlo.compare LE, %5, %9, SIGNED : (tensor<2x2x1xi32>, tensor<2x2x1xi32>) -> tensor<2x2x1xi1>
    %11 = stablehlo.and %7, %10 : tensor<2x2x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.reduce(%11 init: %c_3) applies stablehlo.and across dimensions = [2] : (tensor<2x2x1xi1>, tensor<i1>) -> tensor<2x2xi1>
    %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x8xf32>, tensor<2x2x1xi32>) -> tensor<2x2xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %15 = stablehlo.select %12, %13, %14 : tensor<2x2xi1>, tensor<2x2xf32>
    return %15, %5 : tensor<2x2xf32>, tensor<2x2x1xi32>
  }
  func.func private @take_along_axis_40(%arg0: tensor<2x2x1xi32>, %arg1: tensor<2x2xf32>) -> tensor<2x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8xf32>
    %1 = "stablehlo.scatter"(%0, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<2x8xf32>, tensor<2x2x1xi32>, tensor<2x2xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}
