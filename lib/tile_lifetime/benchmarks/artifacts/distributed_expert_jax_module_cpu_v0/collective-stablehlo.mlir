module @jit_local attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["expert"=4]> {stablehlo.mesh = {axes = [{name = "expert", size = 4 : i64}]}}
  func.func public @main(%arg0: tensor<4x4x2x32xi32>) -> (tensor<4x4x2x32xi32> {jax.result_info = "result"}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"expert"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"expert"}, {}, {}, {}]>] manual_axes={"expert"} (%arg1: tensor<1x4x2x32xi32>) {
      %1 = stablehlo.reshape %arg1 : (tensor<1x4x2x32xi32>) -> tensor<4x2x32xi32>
      %2 = "stablehlo.all_to_all"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x2x32xi32>) -> tensor<4x2x32xi32>
      %3 = "stablehlo.all_to_all"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x2x32xi32>) -> tensor<4x2x32xi32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2, 3] : (tensor<4x2x32xi32>) -> tensor<1x4x2x32xi32>
      sdy.return %4 : tensor<1x4x2x32xi32>
    } : (tensor<4x4x2x32xi32>) -> tensor<4x4x2x32xi32>
    return %0 : tensor<4x4x2x32xi32>
  }
}
