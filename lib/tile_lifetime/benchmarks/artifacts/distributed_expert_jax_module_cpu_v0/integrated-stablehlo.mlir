module @jit_local attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["expert"=4]> {stablehlo.mesh = {axes = [{name = "expert", size = 4 : i64}]}}
  func.func public @main(%arg0: tensor<4x4x2x32xbf16>, %arg1: tensor<4x8x2xi32>, %arg2: tensor<4x8x2xf32>, %arg3: tensor<4x10x32xbf16>, %arg4: tensor<4x10xi1>, %arg5: tensor<4x10x64xbf16>, %arg6: tensor<4x2x5x32xbf16>, %arg7: tensor<4x2x5x32xbf16>, %arg8: tensor<4x2x32x32xbf16>, %arg9: tensor<4x2x32x32xbf16>, %arg10: tensor<4x2x32x32xbf16>, %arg11: tensor<4x10xi32>, %arg12: tensor<4x10xi32>, %arg13: tensor<4x2x32xbf16>, %arg14: tensor<32x8xbf16>, %arg15: tensor<4x2x2xi32>) -> (tensor<4x2x32xbf16> {jax.result_info = "result[0]"}, tensor<32x8xbf16> {jax.result_info = "result[1]"}, tensor<4x2x32x32xbf16> {jax.result_info = "result[2]"}, tensor<4x2x32x32xbf16> {jax.result_info = "result[3]"}, tensor<4x2x32x32xbf16> {jax.result_info = "result[4]"}) {
    %0:5 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) in_shardings=[<@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}]>, <@mesh, [{"expert"}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"expert"}, {}, {}]>] out_shardings=[<@mesh, [{"expert"}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>] manual_axes={"expert"} (%arg16: tensor<1x4x2x32xbf16>, %arg17: tensor<1x8x2xi32>, %arg18: tensor<1x8x2xf32>, %arg19: tensor<1x10x32xbf16>, %arg20: tensor<1x10xi1>, %arg21: tensor<1x10x64xbf16>, %arg22: tensor<1x2x5x32xbf16>, %arg23: tensor<1x2x5x32xbf16>, %arg24: tensor<1x2x32x32xbf16>, %arg25: tensor<1x2x32x32xbf16>, %arg26: tensor<1x2x32x32xbf16>, %arg27: tensor<1x10xi32>, %arg28: tensor<1x10xi32>, %arg29: tensor<1x2x32xbf16>, %arg30: tensor<32x8xbf16>, %arg31: tensor<1x2x2xi32>) {
      %1 = stablehlo.reshape %arg16 : (tensor<1x4x2x32xbf16>) -> tensor<4x2x32xbf16>
      %2 = "stablehlo.all_to_all"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x2x32xbf16>) -> tensor<4x2x32xbf16>
      %3 = stablehlo.reshape %2 : (tensor<4x2x32xbf16>) -> tensor<8x32xbf16>
      %4 = stablehlo.reshape %arg17 : (tensor<1x8x2xi32>) -> tensor<8x2xi32>
      %5 = stablehlo.reshape %arg18 : (tensor<1x8x2xf32>) -> tensor<8x2xf32>
      %6 = stablehlo.reshape %arg19 : (tensor<1x10x32xbf16>) -> tensor<10x32xbf16>
      %7 = stablehlo.reshape %arg20 : (tensor<1x10xi1>) -> tensor<10xi1>
      %8 = stablehlo.reshape %arg21 : (tensor<1x10x64xbf16>) -> tensor<10x64xbf16>
      %9 = stablehlo.reshape %arg22 : (tensor<1x2x5x32xbf16>) -> tensor<2x5x32xbf16>
      %10 = stablehlo.reshape %arg23 : (tensor<1x2x5x32xbf16>) -> tensor<2x5x32xbf16>
      %11 = stablehlo.reshape %arg24 : (tensor<1x2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %12 = stablehlo.reshape %arg25 : (tensor<1x2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %13 = stablehlo.reshape %arg26 : (tensor<1x2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %14:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.edge.rank0.edge_reverse(%3, %4, %5, %6) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<8x32xbf16>, tensor<8x2xi32>, tensor<8x2xf32>, tensor<10x32xbf16>) -> (tensor<10x32xbf16>, tensor<8x2xf32>)
      %15 = stablehlo.transpose %11, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %16 = stablehlo.transpose %12, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %17 = stablehlo.transpose %13, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %18 = stablehlo.concatenate %16, %17, dim = 1 : (tensor<2x32x32xbf16>, tensor<2x32x32xbf16>) -> tensor<2x64x32xbf16>
      %19 = stablehlo.reshape %14#0 : (tensor<10x32xbf16>) -> tensor<2x5x32xbf16>
      %20 = stablehlo.reshape %8 : (tensor<10x64xbf16>) -> tensor<2x5x64xbf16>
      %21 = stablehlo.reshape %7 : (tensor<10xi1>) -> tensor<2x5xi1>
      %22:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.input_adjoint(%19, %20, %21, %15, %18) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x64xbf16>, tensor<2x5xi1>, tensor<2x32x32xbf16>, tensor<2x64x32xbf16>) -> (tensor<2x5x64xbf16>, tensor<2x5x32xbf16>)
      %23 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w13(%9, %22#0) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x64xbf16>) -> tensor<2x32x64xbf16>
      %24 = stablehlo.reshape %14#0 : (tensor<10x32xbf16>) -> tensor<2x5x32xbf16>
      %25 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w2(%10, %24) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x32xbf16>) -> tensor<2x32x32xbf16>
      %26 = stablehlo.reshape %22#1 : (tensor<2x5x32xbf16>) -> tensor<10x32xbf16>
      %27 = stablehlo.reshape %arg20 : (tensor<1x10xi1>) -> tensor<10xi1>
      %28 = stablehlo.reshape %arg27 : (tensor<1x10xi32>) -> tensor<10xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %29 = func.call @floor_divide(%28, %c) : (tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %c_0 = stablehlo.constant dense<4> : tensor<i32>
      %30 = func.call @_where_16(%27, %29, %c_0) : (tensor<10xi1>, tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %31 = stablehlo.reshape %arg27 : (tensor<1x10xi32>) -> tensor<10xi32>
      %c_1 = stablehlo.constant dense<2> : tensor<i32>
      %32 = func.call @remainder(%31, %c_1) : (tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %c_2 = stablehlo.constant dense<2> : tensor<i32>
      %33 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %34 = stablehlo.multiply %32, %33 : tensor<10xi32>
      %35 = stablehlo.reshape %arg28 : (tensor<1x10xi32>) -> tensor<10xi32>
      %36 = stablehlo.add %34, %35 : tensor<10xi32>
      %c_3 = stablehlo.constant dense<4> : tensor<i32>
      %37 = func.call @_where_16(%27, %36, %c_3) : (tensor<10xi1>, tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %38 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<4x4x32xbf16>
      %c_4 = stablehlo.constant dense<0> : tensor<i32>
      %39 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %40 = stablehlo.compare LT, %30, %39, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
      %c_5 = stablehlo.constant dense<4> : tensor<i32>
      %41 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %42 = stablehlo.add %30, %41 : tensor<10xi32>
      %43 = stablehlo.select %40, %42, %30 : tensor<10xi1>, tensor<10xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %44 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %45 = stablehlo.compare LT, %37, %44, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
      %c_7 = stablehlo.constant dense<4> : tensor<i32>
      %46 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %47 = stablehlo.add %37, %46 : tensor<10xi32>
      %48 = stablehlo.select %45, %47, %37 : tensor<10xi1>, tensor<10xi32>
      %49 = stablehlo.broadcast_in_dim %43, dims = [0] : (tensor<10xi32>) -> tensor<10x1xi32>
      %50 = stablehlo.broadcast_in_dim %48, dims = [0] : (tensor<10xi32>) -> tensor<10x1xi32>
      %51 = stablehlo.concatenate %49, %50, dim = 1 : (tensor<10x1xi32>, tensor<10x1xi32>) -> tensor<10x2xi32>
      %52 = "stablehlo.scatter"(%38, %51, %26) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg32: tensor<bf16>, %arg33: tensor<bf16>):
        stablehlo.return %arg33 : tensor<bf16>
      }) : (tensor<4x4x32xbf16>, tensor<10x2xi32>, tensor<10x32xbf16>) -> tensor<4x4x32xbf16>
      %53 = "stablehlo.all_to_all"(%52) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x4x32xbf16>) -> tensor<4x4x32xbf16>
      %54 = stablehlo.reshape %14#1 : (tensor<8x2xf32>) -> tensor<4x2x2xf32>
      %55 = "stablehlo.all_to_all"(%54) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
      %56 = stablehlo.reshape %arg31 : (tensor<1x2x2xi32>) -> tensor<2x2xi32>
      %c_8 = stablehlo.constant dense<2> : tensor<i32>
      %57 = func.call @floor_divide_30(%56, %c_8) : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x2xi32>
      %58 = stablehlo.reshape %57 : (tensor<2x2xi32>) -> tensor<4xi32>
      %59 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c_9 = stablehlo.constant dense<0> : tensor<i32>
      %60 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %61 = stablehlo.compare LT, %58, %60, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
      %c_10 = stablehlo.constant dense<4> : tensor<i32>
      %62 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %63 = stablehlo.add %58, %62 : tensor<4xi32>
      %64 = stablehlo.select %61, %63, %58 : tensor<4xi1>, tensor<4xi32>
      %c_11 = stablehlo.constant dense<0> : tensor<i32>
      %65 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %66 = stablehlo.compare LT, %59, %65, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
      %c_12 = stablehlo.constant dense<4> : tensor<i32>
      %67 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %68 = stablehlo.add %59, %67 : tensor<4xi32>
      %69 = stablehlo.select %66, %68, %59 : tensor<4xi1>, tensor<4xi32>
      %70 = stablehlo.broadcast_in_dim %64, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
      %71 = stablehlo.broadcast_in_dim %69, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
      %72 = stablehlo.concatenate %70, %71, dim = 1 : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
      %73 = "stablehlo.gather"(%53, %72) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 32>}> : (tensor<4x4x32xbf16>, tensor<4x2xi32>) -> tensor<4x32xbf16>
      %74 = stablehlo.transpose %55, dims = [1, 2, 0] : (tensor<4x2x2xf32>) -> tensor<2x2x4xf32>
      %75 = stablehlo.broadcast_in_dim %57, dims = [0, 1] : (tensor<2x2xi32>) -> tensor<2x2x1xi32>
      %76 = func.call @take_along_axis(%74, %75) : (tensor<2x2x4xf32>, tensor<2x2x1xi32>) -> tensor<2x2x1xf32>
      %77 = stablehlo.reshape %76 : (tensor<2x2x1xf32>) -> tensor<2x2xf32>
      %78 = stablehlo.reshape %arg29 : (tensor<1x2x32xbf16>) -> tensor<2x32xbf16>
      %79 = stablehlo.reshape %arg31 : (tensor<1x2x2xi32>) -> tensor<2x2xi32>
      %80 = stablehlo.iota dim = 0 : tensor<2xi32>
      %81 = stablehlo.broadcast_in_dim %80, dims = [0] : (tensor<2xi32>) -> tensor<2x2xi32>
      %82 = stablehlo.reshape %81 : (tensor<2x2xi32>) -> tensor<4xi32>
      %83 = stablehlo.broadcast_in_dim %82, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
      %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %84 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<bf16>) -> tensor<2x32xbf16>
      %85 = stablehlo.custom_call @shuttle.distributed_expert_cpu.source_fold(%84, %83, %73) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x32xbf16>, tensor<4x1xi32>, tensor<4x32xbf16>) -> tensor<2x32xbf16>
      %86 = stablehlo.convert %78 : (tensor<2x32xbf16>) -> tensor<2x32xf32>
      %87 = stablehlo.convert %arg30 : (tensor<32x8xbf16>) -> tensor<32x8xf32>
      %88 = stablehlo.dot_general %86, %87, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x32xf32>, tensor<32x8xf32>) -> tensor<2x8xf32>
      %89:2 = func.call @take_along_axis_65(%88, %79) : (tensor<2x8xf32>, tensor<2x2xi32>) -> (tensor<2x2xf32>, tensor<2x2x1xi32>)
      %cst_14 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %90 = stablehlo.reduce(%89#0 init: %cst_14) applies stablehlo.maximum across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
      %cst_15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %91 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<2xf32>
      %92 = stablehlo.maximum %91, %90 : tensor<2xf32>
      %93 = stablehlo.broadcast_in_dim %92, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
      %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
      %95 = stablehlo.subtract %89#0, %94 : tensor<2x2xf32>
      %96 = stablehlo.exponential %95 : tensor<2x2xf32>
      %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %97 = stablehlo.reduce(%96 init: %cst_16) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
      %98 = stablehlo.broadcast_in_dim %97, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
      %99 = stablehlo.multiply %98, %98 : tensor<2x1xf32>
      %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %100 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %101 = stablehlo.divide %100, %99 : tensor<2x1xf32>
      %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
      %103 = stablehlo.multiply %77, %102 : tensor<2x2xf32>
      %104 = stablehlo.multiply %103, %96 : tensor<2x2xf32>
      %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %105 = stablehlo.reduce(%104 init: %cst_18) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
      %106 = stablehlo.reshape %105 : (tensor<2xf32>) -> tensor<2x1xf32>
      %107 = stablehlo.negate %106 : tensor<2x1xf32>
      %108 = stablehlo.broadcast_in_dim %98, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
      %109 = stablehlo.divide %77, %108 : tensor<2x2xf32>
      %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %110 = stablehlo.reduce(%107 init: %cst_19) applies stablehlo.add across dimensions = [1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<2xf32>
      %111 = stablehlo.broadcast_in_dim %110, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
      %112 = stablehlo.add %109, %111 : tensor<2x2xf32>
      %113 = stablehlo.multiply %112, %96 : tensor<2x2xf32>
      %114 = func.call @take_along_axis_89(%89#1, %113) : (tensor<2x2x1xi32>, tensor<2x2xf32>) -> tensor<2x8xf32>
      %115 = stablehlo.dot_general %114, %86, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<2x32xf32>) -> tensor<8x32xf32>
      %116 = stablehlo.transpose %115, dims = [1, 0] : (tensor<8x32xf32>) -> tensor<32x8xf32>
      %117 = stablehlo.dot_general %114, %87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<32x8xf32>) -> tensor<2x32xf32>
      %118 = stablehlo.convert %116 : (tensor<32x8xf32>) -> tensor<32x8xbf16>
      %119 = stablehlo.convert %117 : (tensor<2x32xf32>) -> tensor<2x32xbf16>
      %120 = stablehlo.add %85, %119 : tensor<2x32xbf16>
      %121 = stablehlo.slice %23 [0:2, 0:32, 0:32] : (tensor<2x32x64xbf16>) -> tensor<2x32x32xbf16>
      %122 = stablehlo.slice %23 [0:2, 0:32, 32:64] : (tensor<2x32x64xbf16>) -> tensor<2x32x32xbf16>
      %123 = stablehlo.broadcast_in_dim %120, dims = [1, 2] : (tensor<2x32xbf16>) -> tensor<1x2x32xbf16>
      %124 = "stablehlo.all_reduce"(%118) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg32: tensor<bf16>, %arg33: tensor<bf16>):
        %128 = stablehlo.add %arg32, %arg33 : tensor<bf16>
        stablehlo.return %128 : tensor<bf16>
      }) : (tensor<32x8xbf16>) -> tensor<32x8xbf16>
      %125 = stablehlo.broadcast_in_dim %121, dims = [1, 2, 3] : (tensor<2x32x32xbf16>) -> tensor<1x2x32x32xbf16>
      %126 = stablehlo.broadcast_in_dim %122, dims = [1, 2, 3] : (tensor<2x32x32xbf16>) -> tensor<1x2x32x32xbf16>
      %127 = stablehlo.broadcast_in_dim %25, dims = [1, 2, 3] : (tensor<2x32x32xbf16>) -> tensor<1x2x32x32xbf16>
      sdy.return %123, %124, %125, %126, %127 : tensor<1x2x32xbf16>, tensor<32x8xbf16>, tensor<1x2x32x32xbf16>, tensor<1x2x32x32xbf16>, tensor<1x2x32x32xbf16>
    } : (tensor<4x4x2x32xbf16>, tensor<4x8x2xi32>, tensor<4x8x2xf32>, tensor<4x10x32xbf16>, tensor<4x10xi1>, tensor<4x10x64xbf16>, tensor<4x2x5x32xbf16>, tensor<4x2x5x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x10xi32>, tensor<4x10xi32>, tensor<4x2x32xbf16>, tensor<32x8xbf16>, tensor<4x2x2xi32>) -> (tensor<4x2x32xbf16>, tensor<32x8xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<4x2x32xbf16>, tensor<32x8xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>
  }
  func.func private @floor_divide(%arg0: tensor<10xi32>, %arg1: tensor<i32>) -> tensor<10xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %2 = stablehlo.divide %arg0, %1 : tensor<10xi32>
    %3 = stablehlo.sign %arg0 : tensor<10xi32>
    %4 = stablehlo.sign %0 : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %6 = stablehlo.compare NE, %3, %5, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %8 = stablehlo.remainder %arg0, %7 : tensor<10xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %10 = stablehlo.compare NE, %8, %9, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
    %11 = stablehlo.and %6, %10 : tensor<10xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %13 = stablehlo.subtract %2, %12 : tensor<10xi32>
    %14 = call @_where(%11, %13, %2) : (tensor<10xi1>, tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    return %14 : tensor<10xi32>
  }
  func.func private @_where(%arg0: tensor<10xi1>, %arg1: tensor<10xi32>, %arg2: tensor<10xi32>) -> tensor<10xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<10xi1>, tensor<10xi32>
    return %0 : tensor<10xi32>
  }
  func.func private @_where_16(%arg0: tensor<10xi1>, %arg1: tensor<10xi32>, %arg2: tensor<i32>) -> tensor<10xi32> {
    %0 = stablehlo.convert %arg2 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<10xi1>, tensor<10xi32>
    return %2 : tensor<10xi32>
  }
  func.func private @remainder(%arg0: tensor<10xi32>, %arg1: tensor<i32>) -> tensor<10xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare EQ, %0, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %2 = call @_where_17(%1, %c_0, %0) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %4 = stablehlo.remainder %arg0, %3 : tensor<10xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %6 = stablehlo.compare NE, %4, %5, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %8 = stablehlo.compare LT, %4, %7, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.compare LT, %2, %c_3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<10xi1>
    %11 = stablehlo.compare NE, %8, %10, UNSIGNED : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %12 = stablehlo.and %11, %6 : tensor<10xi1>
    %13 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<10xi32>
    %14 = stablehlo.add %4, %13 : tensor<10xi32>
    %15 = stablehlo.select %12, %14, %4 : tensor<10xi1>, tensor<10xi32>
    return %15 : tensor<10xi32>
  }
  func.func private @_where_17(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @floor_divide_30(%arg0: tensor<2x2xi32>, %arg1: tensor<i32>) -> tensor<2x2xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<2x2xi32>
    %2 = stablehlo.divide %arg0, %1 : tensor<2x2xi32>
    %3 = stablehlo.sign %arg0 : tensor<2x2xi32>
    %4 = stablehlo.sign %0 : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<2x2xi32>
    %6 = stablehlo.compare NE, %3, %5, SIGNED : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<2x2xi32>
    %8 = stablehlo.remainder %arg0, %7 : tensor<2x2xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x2xi32>
    %10 = stablehlo.compare NE, %8, %9, SIGNED : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi1>
    %11 = stablehlo.and %6, %10 : tensor<2x2xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x2xi32>
    %13 = stablehlo.subtract %2, %12 : tensor<2x2xi32>
    %14 = call @_where_37(%11, %13, %2) : (tensor<2x2xi1>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %14 : tensor<2x2xi32>
  }
  func.func private @_where_37(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<2x2xi1>, tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
  func.func private @take_along_axis(%arg0: tensor<2x2x4xf32>, %arg1: tensor<2x2x1xi32>) -> tensor<2x2x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x2x1xi32>
    %1 = stablehlo.compare LT, %arg1, %0, SIGNED : (tensor<2x2x1xi32>, tensor<2x2x1xi32>) -> tensor<2x2x1xi1>
    %c_0 = stablehlo.constant dense<4> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x2x1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<2x2x1xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<2x2x1xi1>, tensor<2x2x1xi32>
    %5 = stablehlo.reshape %4 : (tensor<2x2x1xi32>) -> tensor<2x2x1x1xi32>
    %c_1 = stablehlo.constant dense<3> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x2x1x1xi32>
    %7 = stablehlo.compare GE, %5, %6, SIGNED : (tensor<2x2x1x1xi32>, tensor<2x2x1x1xi32>) -> tensor<2x2x1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [3] : (tensor<1xi32>) -> tensor<1x1x1x1xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xi32>) -> tensor<2x2x1x1xi32>
    %10 = stablehlo.compare LE, %5, %9, SIGNED : (tensor<2x2x1x1xi32>, tensor<2x2x1x1xi32>) -> tensor<2x2x1x1xi1>
    %11 = stablehlo.and %7, %10 : tensor<2x2x1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.reduce(%11 init: %c_3) applies stablehlo.and across dimensions = [3] : (tensor<2x2x1x1xi1>, tensor<i1>) -> tensor<2x2x1xi1>
    %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [0, 1], start_index_map = [2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<2x2x4xf32>, tensor<2x2x1x1xi32>) -> tensor<2x2x1xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x2x1xf32>
    %15 = stablehlo.select %12, %13, %14 : tensor<2x2x1xi1>, tensor<2x2x1xf32>
    return %15 : tensor<2x2x1xf32>
  }
  func.func private @take_along_axis_65(%arg0: tensor<2x8xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xf32>, tensor<2x2x1xi32>) {
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
  func.func private @take_along_axis_89(%arg0: tensor<2x2x1xi32>, %arg1: tensor<2x2xf32>) -> tensor<2x8xf32> {
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
