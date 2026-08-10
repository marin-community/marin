module @jit_local attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["expert"=4]> {stablehlo.mesh = {axes = [{name = "expert", size = 4 : i64}]}}
  func.func public @main(%arg0: tensor<4x4x2x32xbf16>, %arg1: tensor<4x8x2xi32>, %arg2: tensor<4x8x2xf32>, %arg3: tensor<4x10x32xbf16>, %arg4: tensor<4x10xi32>, %arg5: tensor<4x10xi1>, %arg6: tensor<4x10x64xbf16>, %arg7: tensor<4x2x5x32xbf16>, %arg8: tensor<4x2x5x32xbf16>, %arg9: tensor<4x2x32x32xbf16>, %arg10: tensor<4x2x32x32xbf16>, %arg11: tensor<4x2x32x32xbf16>, %arg12: tensor<4x10xi32>, %arg13: tensor<4x10xi32>, %arg14: tensor<4x2x32xbf16>, %arg15: tensor<32x8xbf16>, %arg16: tensor<4x2x2xi32>) -> (tensor<4x2x32xbf16> {jax.result_info = "result[0]"}, tensor<32x8xbf16> {jax.result_info = "result[1]"}, tensor<4x2x32x64xbf16> {jax.result_info = "result[2]"}, tensor<4x2x32x32xbf16> {jax.result_info = "result[3]"}) {
    %0:4 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16) in_shardings=[<@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}]>, <@mesh, [{"expert"}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}]>, <@mesh, [{"expert"}, {}]>, <@mesh, [{"expert"}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"expert"}, {}, {}]>] out_shardings=[<@mesh, [{"expert"}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>, <@mesh, [{"expert"}, {}, {}, {}]>] manual_axes={"expert"} (%arg17: tensor<1x4x2x32xbf16>, %arg18: tensor<1x8x2xi32>, %arg19: tensor<1x8x2xf32>, %arg20: tensor<1x10x32xbf16>, %arg21: tensor<1x10xi32>, %arg22: tensor<1x10xi1>, %arg23: tensor<1x10x64xbf16>, %arg24: tensor<1x2x5x32xbf16>, %arg25: tensor<1x2x5x32xbf16>, %arg26: tensor<1x2x32x32xbf16>, %arg27: tensor<1x2x32x32xbf16>, %arg28: tensor<1x2x32x32xbf16>, %arg29: tensor<1x10xi32>, %arg30: tensor<1x10xi32>, %arg31: tensor<1x2x32xbf16>, %arg32: tensor<32x8xbf16>, %arg33: tensor<1x2x2xi32>) {
      %1 = stablehlo.reshape %arg17 : (tensor<1x4x2x32xbf16>) -> tensor<4x2x32xbf16>
      %2 = "stablehlo.all_to_all"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x2x32xbf16>) -> tensor<4x2x32xbf16>
      %3 = stablehlo.reshape %2 : (tensor<4x2x32xbf16>) -> tensor<8x32xbf16>
      %4 = stablehlo.reshape %arg18 : (tensor<1x8x2xi32>) -> tensor<8x2xi32>
      %5 = stablehlo.reshape %arg19 : (tensor<1x8x2xf32>) -> tensor<8x2xf32>
      %6 = stablehlo.reshape %arg20 : (tensor<1x10x32xbf16>) -> tensor<10x32xbf16>
      %7 = stablehlo.reshape %arg21 : (tensor<1x10xi32>) -> tensor<10xi32>
      %8 = stablehlo.reshape %arg22 : (tensor<1x10xi1>) -> tensor<10xi1>
      %9 = stablehlo.reshape %arg23 : (tensor<1x10x64xbf16>) -> tensor<10x64xbf16>
      %10 = stablehlo.reshape %arg24 : (tensor<1x2x5x32xbf16>) -> tensor<2x5x32xbf16>
      %11 = stablehlo.reshape %arg25 : (tensor<1x2x5x32xbf16>) -> tensor<2x5x32xbf16>
      %12 = stablehlo.reshape %arg26 : (tensor<1x2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %13 = stablehlo.reshape %arg27 : (tensor<1x2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %14 = stablehlo.reshape %arg28 : (tensor<1x2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %15:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.edge.rank0.edge_reverse(%3, %4, %5, %6) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<8x32xbf16>, tensor<8x2xi32>, tensor<8x2xf32>, tensor<10x32xbf16>) -> (tensor<10x32xbf16>, tensor<8x2xf32>)
      %16 = func.call @_one_hot(%7) : (tensor<10xi32>) -> tensor<10x2xbf16>
      %17 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<10xi1>) -> tensor<10x1xi1>
      %18 = stablehlo.convert %17 : (tensor<10x1xi1>) -> tensor<10x1xbf16>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<10x1xbf16>) -> tensor<10x2xbf16>
      %20 = stablehlo.multiply %16, %19 : tensor<10x2xbf16>
      %21 = stablehlo.broadcast_in_dim %15#0, dims = [0, 2] : (tensor<10x32xbf16>) -> tensor<10x1x32xbf16>
      %22 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<10x2xbf16>) -> tensor<10x2x1xbf16>
      %23 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<10x1x32xbf16>) -> tensor<10x2x32xbf16>
      %24 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<10x2x1xbf16>) -> tensor<10x2x32xbf16>
      %25 = stablehlo.multiply %23, %24 : tensor<10x2x32xbf16>
      %26 = stablehlo.reshape %25 : (tensor<10x2x32xbf16>) -> tensor<10x64xbf16>
      %27 = stablehlo.transpose %20, dims = [1, 0] : (tensor<10x2xbf16>) -> tensor<2x10xbf16>
      %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<2x10xbf16>) -> tensor<2x10x1xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %29 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x10x1xbf16>
      %30 = stablehlo.compare NE, %28, %29, FLOAT : (tensor<2x10x1xbf16>, tensor<2x10x1xbf16>) -> tensor<2x10x1xi1>
      %31 = stablehlo.convert %30 : tensor<2x10x1xi1>
      %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2] : (tensor<2x10x1xi1>) -> tensor<2x10x64xi1>
      %33 = stablehlo.transpose %12, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %34 = stablehlo.reshape %33 : (tensor<2x32x32xbf16>) -> tensor<64x32xbf16>
      %35 = stablehlo.transpose %13, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %36 = stablehlo.transpose %14, dims = [0, 2, 1] : (tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
      %37 = stablehlo.concatenate %35, %36, dim = 1 : (tensor<2x32x32xbf16>, tensor<2x32x32xbf16>) -> tensor<2x64x32xbf16>
      %38 = stablehlo.reshape %37 : (tensor<2x64x32xbf16>) -> tensor<128x32xbf16>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %39 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<10x32xbf16>
      %40 = stablehlo.iota dim = 0 : tensor<10xi32>
      %41 = stablehlo.broadcast_in_dim %40, dims = [0] : (tensor<10xi32>) -> tensor<10x1xi32>
      %42:2 = stablehlo.custom_call @shuttle.distributed_expert_cpu.input_adjoint(%38, %39, %41, %32, %26, %34, %9) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<128x32xbf16>, tensor<10x32xbf16>, tensor<10x1xi32>, tensor<2x10x64xi1>, tensor<10x64xbf16>, tensor<64x32xbf16>, tensor<10x64xbf16>) -> (tensor<2x10x64xbf16>, tensor<10x32xbf16>)
      %43 = stablehlo.slice %42#0 [0:1, 0:5, 0:64] : (tensor<2x10x64xbf16>) -> tensor<1x5x64xbf16>
      %44 = stablehlo.reshape %43 : (tensor<1x5x64xbf16>) -> tensor<5x64xbf16>
      %45 = stablehlo.slice %42#0 [1:2, 5:10, 0:64] : (tensor<2x10x64xbf16>) -> tensor<1x5x64xbf16>
      %46 = stablehlo.reshape %45 : (tensor<1x5x64xbf16>) -> tensor<5x64xbf16>
      %47 = stablehlo.broadcast_in_dim %44, dims = [1, 2] : (tensor<5x64xbf16>) -> tensor<1x5x64xbf16>
      %48 = stablehlo.broadcast_in_dim %46, dims = [1, 2] : (tensor<5x64xbf16>) -> tensor<1x5x64xbf16>
      %49 = stablehlo.concatenate %47, %48, dim = 0 : (tensor<1x5x64xbf16>, tensor<1x5x64xbf16>) -> tensor<2x5x64xbf16>
      %50 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w13(%10, %49) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x64xbf16>) -> tensor<2x32x64xbf16>
      %51 = stablehlo.reshape %15#0 : (tensor<10x32xbf16>) -> tensor<2x5x32xbf16>
      %52 = stablehlo.custom_call @shuttle.distributed_expert_cpu.weight_gradient.w2(%11, %51) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x5x32xbf16>, tensor<2x5x32xbf16>) -> tensor<2x32x32xbf16>
      %53 = stablehlo.reshape %arg22 : (tensor<1x10xi1>) -> tensor<10xi1>
      %54 = stablehlo.reshape %arg29 : (tensor<1x10xi32>) -> tensor<10xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %55 = func.call @floor_divide(%54, %c) : (tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %c_1 = stablehlo.constant dense<4> : tensor<i32>
      %56 = func.call @_where_30(%53, %55, %c_1) : (tensor<10xi1>, tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %57 = stablehlo.reshape %arg29 : (tensor<1x10xi32>) -> tensor<10xi32>
      %c_2 = stablehlo.constant dense<2> : tensor<i32>
      %58 = func.call @remainder(%57, %c_2) : (tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %c_3 = stablehlo.constant dense<2> : tensor<i32>
      %59 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %60 = stablehlo.multiply %58, %59 : tensor<10xi32>
      %61 = stablehlo.reshape %arg30 : (tensor<1x10xi32>) -> tensor<10xi32>
      %62 = stablehlo.add %60, %61 : tensor<10xi32>
      %c_4 = stablehlo.constant dense<4> : tensor<i32>
      %63 = func.call @_where_30(%53, %62, %c_4) : (tensor<10xi1>, tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
      %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %64 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<bf16>) -> tensor<4x4x32xbf16>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %65 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %66 = stablehlo.compare LT, %56, %65, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
      %c_7 = stablehlo.constant dense<4> : tensor<i32>
      %67 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %68 = stablehlo.add %56, %67 : tensor<10xi32>
      %69 = stablehlo.select %66, %68, %56 : tensor<10xi1>, tensor<10xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %70 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %71 = stablehlo.compare LT, %63, %70, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
      %c_9 = stablehlo.constant dense<4> : tensor<i32>
      %72 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<10xi32>
      %73 = stablehlo.add %63, %72 : tensor<10xi32>
      %74 = stablehlo.select %71, %73, %63 : tensor<10xi1>, tensor<10xi32>
      %75 = stablehlo.broadcast_in_dim %69, dims = [0] : (tensor<10xi32>) -> tensor<10x1xi32>
      %76 = stablehlo.broadcast_in_dim %74, dims = [0] : (tensor<10xi32>) -> tensor<10x1xi32>
      %77 = stablehlo.concatenate %75, %76, dim = 1 : (tensor<10x1xi32>, tensor<10x1xi32>) -> tensor<10x2xi32>
      %78 = "stablehlo.scatter"(%64, %77, %42#1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg34: tensor<bf16>, %arg35: tensor<bf16>):
        stablehlo.return %arg35 : tensor<bf16>
      }) : (tensor<4x4x32xbf16>, tensor<10x2xi32>, tensor<10x32xbf16>) -> tensor<4x4x32xbf16>
      %79 = "stablehlo.all_to_all"(%78) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x4x32xbf16>) -> tensor<4x4x32xbf16>
      %80 = stablehlo.reshape %15#1 : (tensor<8x2xf32>) -> tensor<4x2x2xf32>
      %81 = "stablehlo.all_to_all"(%80) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
      %82 = stablehlo.reshape %arg33 : (tensor<1x2x2xi32>) -> tensor<2x2xi32>
      %c_10 = stablehlo.constant dense<2> : tensor<i32>
      %83 = func.call @floor_divide_46(%82, %c_10) : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x2xi32>
      %84 = stablehlo.reshape %83 : (tensor<2x2xi32>) -> tensor<4xi32>
      %85 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c_11 = stablehlo.constant dense<0> : tensor<i32>
      %86 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %87 = stablehlo.compare LT, %84, %86, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
      %c_12 = stablehlo.constant dense<4> : tensor<i32>
      %88 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %89 = stablehlo.add %84, %88 : tensor<4xi32>
      %90 = stablehlo.select %87, %89, %84 : tensor<4xi1>, tensor<4xi32>
      %c_13 = stablehlo.constant dense<0> : tensor<i32>
      %91 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %92 = stablehlo.compare LT, %85, %91, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
      %c_14 = stablehlo.constant dense<4> : tensor<i32>
      %93 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %94 = stablehlo.add %85, %93 : tensor<4xi32>
      %95 = stablehlo.select %92, %94, %85 : tensor<4xi1>, tensor<4xi32>
      %96 = stablehlo.broadcast_in_dim %90, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
      %97 = stablehlo.broadcast_in_dim %95, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
      %98 = stablehlo.concatenate %96, %97, dim = 1 : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
      %99 = "stablehlo.gather"(%79, %98) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 32>}> : (tensor<4x4x32xbf16>, tensor<4x2xi32>) -> tensor<4x32xbf16>
      %100 = stablehlo.transpose %81, dims = [1, 2, 0] : (tensor<4x2x2xf32>) -> tensor<2x2x4xf32>
      %101 = stablehlo.broadcast_in_dim %83, dims = [0, 1] : (tensor<2x2xi32>) -> tensor<2x2x1xi32>
      %102 = func.call @take_along_axis(%100, %101) : (tensor<2x2x4xf32>, tensor<2x2x1xi32>) -> tensor<2x2x1xf32>
      %103 = stablehlo.reshape %102 : (tensor<2x2x1xf32>) -> tensor<2x2xf32>
      %104 = stablehlo.reshape %arg31 : (tensor<1x2x32xbf16>) -> tensor<2x32xbf16>
      %105 = stablehlo.reshape %arg33 : (tensor<1x2x2xi32>) -> tensor<2x2xi32>
      %106 = stablehlo.iota dim = 0 : tensor<2xi32>
      %107 = stablehlo.broadcast_in_dim %106, dims = [0] : (tensor<2xi32>) -> tensor<2x2xi32>
      %108 = stablehlo.reshape %107 : (tensor<2x2xi32>) -> tensor<4xi32>
      %109 = stablehlo.broadcast_in_dim %108, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
      %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %110 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<bf16>) -> tensor<2x32xbf16>
      %111 = stablehlo.custom_call @shuttle.distributed_expert_cpu.source_fold(%110, %109, %99) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x32xbf16>, tensor<4x1xi32>, tensor<4x32xbf16>) -> tensor<2x32xbf16>
      %112 = stablehlo.convert %104 : (tensor<2x32xbf16>) -> tensor<2x32xf32>
      %113 = stablehlo.convert %arg32 : (tensor<32x8xbf16>) -> tensor<32x8xf32>
      %114 = stablehlo.dot_general %112, %113, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x32xf32>, tensor<32x8xf32>) -> tensor<2x8xf32>
      %115:2 = func.call @take_along_axis_82(%114, %105) : (tensor<2x8xf32>, tensor<2x2xi32>) -> (tensor<2x2xf32>, tensor<2x2x1xi32>)
      %cst_16 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %116 = stablehlo.reduce(%115#0 init: %cst_16) applies stablehlo.maximum across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
      %cst_17 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %117 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<2xf32>
      %118 = stablehlo.maximum %117, %116 : tensor<2xf32>
      %119 = stablehlo.broadcast_in_dim %118, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
      %120 = stablehlo.broadcast_in_dim %119, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
      %121 = stablehlo.subtract %115#0, %120 : tensor<2x2xf32>
      %122 = stablehlo.exponential %121 : tensor<2x2xf32>
      %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %123 = stablehlo.reduce(%122 init: %cst_18) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
      %124 = stablehlo.broadcast_in_dim %123, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
      %125 = stablehlo.multiply %124, %124 : tensor<2x1xf32>
      %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %127 = stablehlo.divide %126, %125 : tensor<2x1xf32>
      %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
      %129 = stablehlo.multiply %103, %128 : tensor<2x2xf32>
      %130 = stablehlo.multiply %129, %122 : tensor<2x2xf32>
      %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %131 = stablehlo.reduce(%130 init: %cst_20) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
      %132 = stablehlo.reshape %131 : (tensor<2xf32>) -> tensor<2x1xf32>
      %133 = stablehlo.negate %132 : tensor<2x1xf32>
      %134 = stablehlo.broadcast_in_dim %124, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
      %135 = stablehlo.divide %103, %134 : tensor<2x2xf32>
      %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %136 = stablehlo.reduce(%133 init: %cst_21) applies stablehlo.add across dimensions = [1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<2xf32>
      %137 = stablehlo.broadcast_in_dim %136, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
      %138 = stablehlo.add %135, %137 : tensor<2x2xf32>
      %139 = stablehlo.multiply %138, %122 : tensor<2x2xf32>
      %140 = func.call @take_along_axis_106(%115#1, %139) : (tensor<2x2x1xi32>, tensor<2x2xf32>) -> tensor<2x8xf32>
      %141 = stablehlo.dot_general %140, %112, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<2x32xf32>) -> tensor<8x32xf32>
      %142 = stablehlo.transpose %141, dims = [1, 0] : (tensor<8x32xf32>) -> tensor<32x8xf32>
      %143 = stablehlo.dot_general %140, %113, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<32x8xf32>) -> tensor<2x32xf32>
      %144 = stablehlo.convert %142 : (tensor<32x8xf32>) -> tensor<32x8xbf16>
      %145 = stablehlo.convert %143 : (tensor<2x32xf32>) -> tensor<2x32xbf16>
      %146 = stablehlo.add %111, %145 : tensor<2x32xbf16>
      %147 = stablehlo.broadcast_in_dim %146, dims = [1, 2] : (tensor<2x32xbf16>) -> tensor<1x2x32xbf16>
      %148 = "stablehlo.all_reduce"(%144) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg34: tensor<bf16>, %arg35: tensor<bf16>):
        %151 = stablehlo.add %arg34, %arg35 : tensor<bf16>
        stablehlo.return %151 : tensor<bf16>
      }) : (tensor<32x8xbf16>) -> tensor<32x8xbf16>
      %149 = stablehlo.broadcast_in_dim %50, dims = [1, 2, 3] : (tensor<2x32x64xbf16>) -> tensor<1x2x32x64xbf16>
      %150 = stablehlo.broadcast_in_dim %52, dims = [1, 2, 3] : (tensor<2x32x32xbf16>) -> tensor<1x2x32x32xbf16>
      sdy.return %147, %148, %149, %150 : tensor<1x2x32xbf16>, tensor<32x8xbf16>, tensor<1x2x32x64xbf16>, tensor<1x2x32x32xbf16>
    } : (tensor<4x4x2x32xbf16>, tensor<4x8x2xi32>, tensor<4x8x2xf32>, tensor<4x10x32xbf16>, tensor<4x10xi32>, tensor<4x10xi1>, tensor<4x10x64xbf16>, tensor<4x2x5x32xbf16>, tensor<4x2x5x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x2x32x32xbf16>, tensor<4x10xi32>, tensor<4x10xi32>, tensor<4x2x32xbf16>, tensor<32x8xbf16>, tensor<4x2x2xi32>) -> (tensor<4x2x32xbf16>, tensor<32x8xbf16>, tensor<4x2x32x64xbf16>, tensor<4x2x32x32xbf16>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x2x32xbf16>, tensor<32x8xbf16>, tensor<4x2x32x64xbf16>, tensor<4x2x32x32xbf16>
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
  func.func private @_where_30(%arg0: tensor<10xi1>, %arg1: tensor<10xi32>, %arg2: tensor<i32>) -> tensor<10xi32> {
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
    %2 = call @_where_33(%1, %c_0, %0) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_33(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @floor_divide_46(%arg0: tensor<2x2xi32>, %arg1: tensor<i32>) -> tensor<2x2xi32> {
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
    %14 = call @_where_53(%11, %13, %2) : (tensor<2x2xi1>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    return %14 : tensor<2x2xi32>
  }
  func.func private @_where_53(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2x2xi32> {
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
  func.func private @take_along_axis_82(%arg0: tensor<2x8xf32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xf32>, tensor<2x2x1xi32>) {
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
  func.func private @take_along_axis_106(%arg0: tensor<2x2x1xi32>, %arg1: tensor<2x2xf32>) -> tensor<2x8xf32> {
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
