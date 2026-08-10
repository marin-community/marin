module @jit_transformed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x32xbf16>, %arg1: tensor<8x2xi32>, %arg2: tensor<8x2xf32>, %arg3: tensor<10x32xbf16>, %arg4: tensor<10xi1>, %arg5: tensor<10x64xbf16>, %arg6: tensor<2x5x32xbf16>, %arg7: tensor<2x5x32xbf16>, %arg8: tensor<2x32x32xbf16>, %arg9: tensor<2x32x32xbf16>, %arg10: tensor<2x32x32xbf16>, %arg11: tensor<4x32xbf16>, %arg12: tensor<2x2xf32>, %arg13: tensor<2x32xbf16>, %arg14: tensor<32x8xbf16>, %arg15: tensor<2x2xi32>) -> (tensor<2x32xbf16> {jax.result_info = "result[0]"}, tensor<32x8xbf16> {jax.result_info = "result[1]"}, tensor<10x32xbf16> {jax.result_info = "result[2]"}, tensor<2x32x32xbf16> {jax.result_info = "result[3]"}, tensor<2x32x32xbf16> {jax.result_info = "result[4]"}, tensor<2x32x32xbf16> {jax.result_info = "result[5]"}) {
    %0:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.edge.rank0.edge_reverse(%arg0, %arg1, %arg2, %arg3) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<8x32xbf16>, tensor<8x2xi32>, tensor<8x2xf32>, tensor<10x32xbf16>) -> (tensor<10x32xbf16>, tensor<8x2xf32>)
    %1 = stablehlo.transpose %arg8, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
    %2 = stablehlo.transpose %arg9, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
    %3 = stablehlo.transpose %arg10, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<2x32x32xbf16>, tensor<2x32x32xbf16>) -> tensor<2x64x32xbf16>
    %5 = stablehlo.reshape %0#0 : (tensor<10x32xbf16>) -> tensor<2x5x32xbf16>
    %6 = stablehlo.reshape %arg5 : (tensor<10x64xbf16>) -> tensor<2x5x64xbf16>
    %7 = stablehlo.reshape %arg4 : (tensor<10xi1>) -> tensor<2x5xi1>
    %8:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.input_adjoint(%5, %6, %7, %1, %4) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x64xbf16>, tensor<2x5xi1>, tensor<2x32x32xbf16>, tensor<2x64x32xbf16>) -> (tensor<2x5x64xbf16>, tensor<2x5x32xbf16>)
    %9 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w13(%arg6, %8#0) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x64xbf16>) -> tensor<2x32x64xbf16>
    %10 = stablehlo.reshape %0#0 : (tensor<10x32xbf16>) -> tensor<2x5x32xbf16>
    %11 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w2(%arg7, %10) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x32xbf16>) -> tensor<2x32x32xbf16>
    %12 = stablehlo.reshape %8#1 : (tensor<2x5x32xbf16>) -> tensor<10x32xbf16>
    %13 = stablehlo.iota dim = 0 : tensor<2xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<2xi32>) -> tensor<2x2xi32>
    %15 = stablehlo.reshape %14 : (tensor<2x2xi32>) -> tensor<4xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x32xbf16>
    %18 = stablehlo.custom_call @shuttle.distributed_expert_cpu.source_fold(%17, %16, %arg11) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x32xbf16>, tensor<4x1xi32>, tensor<4x32xbf16>) -> tensor<2x32xbf16>
    %19 = stablehlo.convert %arg13 : (tensor<2x32xbf16>) -> tensor<2x32xf32>
    %20 = stablehlo.convert %arg14 : (tensor<32x8xbf16>) -> tensor<32x8xf32>
    %21 = stablehlo.dot_general %19, %20, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x32xf32>, tensor<32x8xf32>) -> tensor<2x8xf32>
    %22:2 = call @take_along_axis(%21, %arg15) : (tensor<2x8xf32>, tensor<2x2xi32>) -> (tensor<2x2xf32>, tensor<2x2x1xi32>)
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %23 = stablehlo.reduce(%22#0 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<2xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
    %28 = stablehlo.subtract %22#0, %27 : tensor<2x2xf32>
    %29 = stablehlo.exponential %28 : tensor<2x2xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %30 = stablehlo.reduce(%29 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %32 = stablehlo.multiply %31, %31 : tensor<2x1xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %34 = stablehlo.divide %33, %32 : tensor<2x1xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
    %36 = stablehlo.multiply %arg12, %35 : tensor<2x2xf32>
    %37 = stablehlo.multiply %36, %29 : tensor<2x2xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = stablehlo.reduce(%37 init: %cst_4) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %39 = stablehlo.reshape %38 : (tensor<2xf32>) -> tensor<2x1xf32>
    %40 = stablehlo.negate %39 : tensor<2x1xf32>
    %41 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
    %42 = stablehlo.divide %arg12, %41 : tensor<2x2xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.reduce(%40 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<2xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
    %45 = stablehlo.add %42, %44 : tensor<2x2xf32>
    %46 = stablehlo.multiply %45, %29 : tensor<2x2xf32>
    %47 = call @take_along_axis_24(%22#1, %46) : (tensor<2x2x1xi32>, tensor<2x2xf32>) -> tensor<2x8xf32>
    %48 = stablehlo.dot_general %47, %19, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<2x32xf32>) -> tensor<8x32xf32>
    %49 = stablehlo.transpose %48, dims = [1, 0] : (tensor<8x32xf32>) -> tensor<32x8xf32>
    %50 = stablehlo.dot_general %47, %20, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<32x8xf32>) -> tensor<2x32xf32>
    %51 = stablehlo.convert %49 : (tensor<32x8xf32>) -> tensor<32x8xbf16>
    %52 = stablehlo.convert %50 : (tensor<2x32xf32>) -> tensor<2x32xbf16>
    %53 = stablehlo.add %18, %52 : tensor<2x32xbf16>
    %54 = stablehlo.slice %9 [0:2, 0:32, 0:32] : (tensor<2x32x64xbf16>) -> tensor<2x32x32xbf16>
    %55 = stablehlo.slice %9 [0:2, 0:32, 32:64] : (tensor<2x32x64xbf16>) -> tensor<2x32x32xbf16>
    return %53, %51, %12, %54, %55, %11 : tensor<2x32xbf16>, tensor<32x8xbf16>, tensor<10x32xbf16>, tensor<2x32x32xbf16>, tensor<2x32x32xbf16>, tensor<2x32x32xbf16>
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
  func.func private @take_along_axis_24(%arg0: tensor<2x2x1xi32>, %arg1: tensor<2x2xf32>) -> tensor<2x8xf32> {
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
