module @jit_train_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["replica_dcn"=1, "data"=1, "expert"=1, "model"=1]> {stablehlo.mesh = {axes = [{name = "replica_dcn", size = 1 : i64}, {name = "data", size = 1 : i64}, {name = "expert", size = 1 : i64}, {name = "model", size = 1 : i64}]}}
  func.func public @main(%arg0: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>, tf.aliasing_output = 0 : i32}, %arg1: tensor<64x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>, tf.aliasing_output = 1 : i32}, %arg2: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 2 : i32}, %arg3: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 3 : i32}, %arg4: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 4 : i32}, %arg5: tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>, tf.aliasing_output = 5 : i32}, %arg6: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 6 : i32}, %arg7: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 7 : i32}, %arg8: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 8 : i32}, %arg9: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 9 : i32}, %arg10: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 10 : i32}, %arg11: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 11 : i32}, %arg12: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 12 : i32}, %arg13: tensor<32x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 13 : i32}, %arg14: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 14 : i32}, %arg15: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 15 : i32}, %arg16: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 16 : i32}, %arg17: tensor<32x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 17 : i32}, %arg18: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 19 : i32}, %arg19: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 20 : i32}, %arg20: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 21 : i32}, %arg21: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 22 : i32}, %arg22: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 23 : i32}, %arg23: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>, tf.aliasing_output = 24 : i32}, %arg24: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 25 : i32}, %arg25: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 26 : i32}, %arg26: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 27 : i32}, %arg27: tensor<1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 28 : i32}, %arg28: tensor<2x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg29: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<i32> {jax.result_info = "result[0].step", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<64x32xf32> {jax.result_info = "result[0].params.token_embed", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>}, tensor<32xf32> {jax.result_info = "result[0].params.embed_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.embed_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.embed_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x64xf32> {jax.result_info = "result[0].params.output_proj", sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>}, tensor<32xf32> {jax.result_info = "result[0].params.blocks[0].rms_attn.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.blocks[0].attn_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.blocks[0].attn_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_q", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_k", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_v", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_o", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<32x2xf32> {jax.result_info = "result[0].params.blocks[0].attn.attn_gate", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32xf32> {jax.result_info = "result[0].params.blocks[0].rms_mlp.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.blocks[0].mlp_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.blocks[0].mlp_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x4xf32> {jax.result_info = "result[0].params.blocks[0].mlp.router", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<4xf32> {jax.result_info = "result[0].params.blocks[0].mlp.router_bias", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].shared.w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].shared.w_up", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].shared.w_down", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].params.expert_banks[0].w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].params.expert_banks[0].w_up", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].params.expert_banks[0].w_down", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>}, tensor<32xf32> {jax.result_info = "result[0].params.final_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.final_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.final_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1x4xf32> {jax.result_info = "result[0].pending_qb_betas", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1x4xf32> {jax.result_info = "result[1]['qb_beta_per_layer']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<f32> {jax.result_info = "result[1]['train/cross_entropy_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/aux_loss_weighted']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/capacity_overflow_rate_mean']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/capacity_overflow_rate']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/load_balancing_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/router_z_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_entropy']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].min", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].max", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].num", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<i32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].nonzero_count", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].sum", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].sum_squares", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].mean", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].variance", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].rms", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<5xf32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].histogram.bucket_limits", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<4xf32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].histogram.bucket_counts", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<f32> {jax.result_info = "result[1]['train/router/load_balancing_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/router_z_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<1x4xf32> {jax.result_info = "result[1]['train/router/routing_counts_per_layer']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<f32> {jax.result_info = "result[1]['train/router/routing_entropy_mean']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['tying/activation_norm_by_layer/layer_0']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['tying/expert_gradient_norm_by_bank/bank_0']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['tying/top1_cross_loop_agreement']"}, tensor<f32> {jax.result_info = "result[1]['tying/topk_set_overlap']"}, tensor<f32> {jax.result_info = "result[1]['tying/update_norm_by_bank/bank_0']", sdy.sharding = #sdy.sharding<@mesh, []>}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.reshape %arg27 : (tensor<1x4xf32>) -> tensor<4xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<4xf32>
    %2 = stablehlo.negate %1 : tensor<4xf32>
    %3 = sdy.sharding_constraint %2 <@mesh, [{}]> : tensor<4xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %5 = sdy.sharding_constraint %4 <@mesh, []> : tensor<f32>
    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %6 = sdy.sharding_constraint %cst_0 <@mesh, []> : tensor<f32>
    %7 = stablehlo.divide %5, %6 : tensor<f32>
    %8 = sdy.sharding_constraint %7 <@mesh, []> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %10 = sdy.sharding_constraint %9 <@mesh, [{}]> : tensor<4xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<4xf32>
    %12 = sdy.sharding_constraint %11 <@mesh, [{}]> : tensor<4xf32>
    %13 = stablehlo.convert %arg1 : (tensor<64x32xf32>) -> tensor<64x32xbf16>
    %14 = sdy.sharding_constraint %13 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xbf16>
    %15 = stablehlo.convert %arg2 : (tensor<32xf32>) -> tensor<32xbf16>
    %16 = sdy.sharding_constraint %15 <@mesh, [{}]> : tensor<32xbf16>
    %17 = stablehlo.convert %arg3 : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %18 = sdy.sharding_constraint %17 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %19 = stablehlo.convert %arg4 : (tensor<128x32xf32>) -> tensor<128x32xbf16>
    %20 = sdy.sharding_constraint %19 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %21 = stablehlo.convert %arg5 : (tensor<32x64xf32>) -> tensor<32x64xbf16>
    %22 = sdy.sharding_constraint %21 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xbf16>
    %23 = stablehlo.convert %arg6 : (tensor<32xf32>) -> tensor<32xbf16>
    %24 = sdy.sharding_constraint %23 <@mesh, [{}]> : tensor<32xbf16>
    %25 = stablehlo.convert %arg7 : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %26 = sdy.sharding_constraint %25 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %27 = stablehlo.convert %arg8 : (tensor<128x32xf32>) -> tensor<128x32xbf16>
    %28 = sdy.sharding_constraint %27 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %29 = stablehlo.convert %arg9 : (tensor<32x32xf32>) -> tensor<32x32xbf16>
    %30 = sdy.sharding_constraint %29 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %31 = stablehlo.convert %arg10 : (tensor<32x16xf32>) -> tensor<32x16xbf16>
    %32 = sdy.sharding_constraint %31 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %33 = stablehlo.convert %arg11 : (tensor<32x16xf32>) -> tensor<32x16xbf16>
    %34 = sdy.sharding_constraint %33 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %35 = stablehlo.convert %arg12 : (tensor<32x32xf32>) -> tensor<32x32xbf16>
    %36 = sdy.sharding_constraint %35 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %37 = stablehlo.convert %arg13 : (tensor<32x2xf32>) -> tensor<32x2xbf16>
    %38 = sdy.sharding_constraint %37 <@mesh, [{}, {}]> : tensor<32x2xbf16>
    %39 = stablehlo.convert %arg14 : (tensor<32xf32>) -> tensor<32xbf16>
    %40 = sdy.sharding_constraint %39 <@mesh, [{}]> : tensor<32xbf16>
    %41 = stablehlo.convert %arg15 : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %42 = sdy.sharding_constraint %41 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %43 = stablehlo.convert %arg16 : (tensor<128x32xf32>) -> tensor<128x32xbf16>
    %44 = sdy.sharding_constraint %43 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %45 = stablehlo.convert %arg17 : (tensor<32x4xf32>) -> tensor<32x4xbf16>
    %46 = sdy.sharding_constraint %45 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %47 = stablehlo.convert %12 : (tensor<4xf32>) -> tensor<4xbf16>
    %48 = sdy.sharding_constraint %47 <@mesh, [{}]> : tensor<4xbf16>
    %49 = stablehlo.convert %arg18 : (tensor<32x32xf32>) -> tensor<32x32xbf16>
    %50 = sdy.sharding_constraint %49 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %51 = stablehlo.convert %arg19 : (tensor<32x32xf32>) -> tensor<32x32xbf16>
    %52 = sdy.sharding_constraint %51 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %53 = stablehlo.convert %arg20 : (tensor<32x32xf32>) -> tensor<32x32xbf16>
    %54 = sdy.sharding_constraint %53 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %55 = stablehlo.convert %arg21 : (tensor<4x32x32xf32>) -> tensor<4x32x32xbf16>
    %56 = sdy.sharding_constraint %55 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %57 = stablehlo.convert %arg22 : (tensor<4x32x32xf32>) -> tensor<4x32x32xbf16>
    %58 = sdy.sharding_constraint %57 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %59 = stablehlo.convert %arg23 : (tensor<4x32x32xf32>) -> tensor<4x32x32xbf16>
    %60 = sdy.sharding_constraint %59 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xbf16>
    %61 = stablehlo.convert %arg24 : (tensor<32xf32>) -> tensor<32xbf16>
    %62 = sdy.sharding_constraint %61 <@mesh, [{}]> : tensor<32xbf16>
    %63 = stablehlo.convert %arg25 : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %64 = sdy.sharding_constraint %63 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %65 = stablehlo.convert %arg26 : (tensor<128x32xf32>) -> tensor<128x32xbf16>
    %66 = sdy.sharding_constraint %65 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %67 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %68 = stablehlo.compare LT, %arg28, %67, SIGNED : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %c_2 = stablehlo.constant dense<64> : tensor<i32>
    %69 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %70 = stablehlo.add %arg28, %69 : tensor<2x4xi32>
    %71 = stablehlo.select %68, %70, %arg28 : tensor<2x4xi1>, tensor<2x4xi32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x1xi32>
    %73 = "stablehlo.gather"(%14, %72) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xbf16>, tensor<2x4x1xi32>) -> tensor<2x4x32xbf16>
    %74 = sdy.sharding_constraint %73 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %75 = sdy.sharding_constraint %16 <@mesh, [{}]> : tensor<32xbf16>
    %76 = stablehlo.convert %74 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %77 = sdy.sharding_constraint %76 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %78 = chlo.square %77 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %79 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2x4x32xf32>
    %80 = sdy.sharding_constraint %79 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %81 = stablehlo.multiply %80, %77 : tensor<2x4x32xf32>
    %82 = sdy.sharding_constraint %81 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %83 = stablehlo.reduce(%78 init: %cst_4) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %84 = sdy.sharding_constraint %83 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %86 = sdy.sharding_constraint %85 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_5 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %88 = sdy.sharding_constraint %87 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %89 = stablehlo.divide %86, %88 : tensor<2x4x1xf32>
    %90 = sdy.sharding_constraint %89 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %91 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %92 = sdy.sharding_constraint %91 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %93 = stablehlo.add %90, %92 : tensor<2x4x1xf32>
    %94 = sdy.sharding_constraint %93 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %95 = stablehlo.rsqrt %94 : tensor<2x4x1xf32>
    %96 = sdy.sharding_constraint %95 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %97 = stablehlo.divide %96, %94 : tensor<2x4x1xf32>
    %98 = sdy.sharding_constraint %97 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_7 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %99 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %100 = sdy.sharding_constraint %99 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %101 = stablehlo.multiply %100, %98 : tensor<2x4x1xf32>
    %102 = sdy.sharding_constraint %101 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %103 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %104 = sdy.sharding_constraint %103 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %105 = stablehlo.multiply %77, %104 : tensor<2x4x32xf32>
    %106 = sdy.sharding_constraint %105 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %107 = stablehlo.convert %75 : (tensor<32xbf16>) -> tensor<32xf32>
    %108 = sdy.sharding_constraint %107 <@mesh, [{}]> : tensor<32xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %110 = sdy.sharding_constraint %109 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %112 = sdy.sharding_constraint %111 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %113 = stablehlo.multiply %106, %112 : tensor<2x4x32xf32>
    %114 = sdy.sharding_constraint %113 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %115 = stablehlo.convert %114 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %116 = sdy.sharding_constraint %115 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %117 = stablehlo.dot_general %116, %18, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %118 = sdy.sharding_constraint %117 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %119:3 = call @silu(%118) : (tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>)
    %120 = sdy.sharding_constraint %119#0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %121 = sdy.sharding_constraint %119#1 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %122 = sdy.sharding_constraint %119#2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %123 = stablehlo.dot_general %120, %20, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %124 = sdy.sharding_constraint %123 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %125 = stablehlo.negate %124 : tensor<2x4x32xbf16>
    %126 = sdy.sharding_constraint %125 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %127 = stablehlo.exponential %126 : tensor<2x4x32xbf16>
    %128 = sdy.sharding_constraint %127 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %129 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %130 = sdy.sharding_constraint %129 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %131 = stablehlo.add %130, %128 : tensor<2x4x32xbf16>
    %132 = sdy.sharding_constraint %131 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %133 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %134 = sdy.sharding_constraint %133 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %135 = stablehlo.divide %134, %132 : tensor<2x4x32xbf16>
    %136 = sdy.sharding_constraint %135 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %137 = sdy.sharding_constraint %cst_10 <@mesh, []> : tensor<bf16>
    %138 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %139 = sdy.sharding_constraint %138 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %140 = stablehlo.subtract %139, %136 : tensor<2x4x32xbf16>
    %141 = sdy.sharding_constraint %140 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %142 = stablehlo.multiply %136, %141 : tensor<2x4x32xbf16>
    %143 = sdy.sharding_constraint %142 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %144 = stablehlo.multiply %116, %136 : tensor<2x4x32xbf16>
    %145 = sdy.sharding_constraint %144 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %146 = sdy.sharding_constraint %24 <@mesh, [{}]> : tensor<32xbf16>
    %147 = stablehlo.convert %145 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %148 = sdy.sharding_constraint %147 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %149 = chlo.square %148 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %150 = stablehlo.reduce(%149 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %151 = sdy.sharding_constraint %150 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %153 = sdy.sharding_constraint %152 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_12 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %154 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %155 = sdy.sharding_constraint %154 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %156 = stablehlo.divide %153, %155 : tensor<2x4x1xf32>
    %157 = sdy.sharding_constraint %156 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %158 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %159 = sdy.sharding_constraint %158 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %160 = stablehlo.add %157, %159 : tensor<2x4x1xf32>
    %161 = sdy.sharding_constraint %160 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %162 = stablehlo.rsqrt %161 : tensor<2x4x1xf32>
    %163 = sdy.sharding_constraint %162 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %165 = sdy.sharding_constraint %164 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %166 = stablehlo.multiply %148, %165 : tensor<2x4x32xf32>
    %167 = sdy.sharding_constraint %166 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %168 = stablehlo.convert %146 : (tensor<32xbf16>) -> tensor<32xf32>
    %169 = sdy.sharding_constraint %168 <@mesh, [{}]> : tensor<32xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %171 = sdy.sharding_constraint %170 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %173 = sdy.sharding_constraint %172 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %174 = stablehlo.multiply %167, %173 : tensor<2x4x32xf32>
    %175 = sdy.sharding_constraint %174 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %176 = stablehlo.convert %175 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %177 = sdy.sharding_constraint %176 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %178 = stablehlo.dot_general %177, %26, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %179 = sdy.sharding_constraint %178 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %180 = call @silu_0(%179) : (tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %181 = sdy.sharding_constraint %180 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %182 = stablehlo.dot_general %181, %28, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %183 = sdy.sharding_constraint %182 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %184 = stablehlo.negate %183 : tensor<2x4x32xbf16>
    %185 = sdy.sharding_constraint %184 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %186 = stablehlo.exponential %185 : tensor<2x4x32xbf16>
    %187 = sdy.sharding_constraint %186 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %188 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %189 = sdy.sharding_constraint %188 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %190 = stablehlo.add %189, %187 : tensor<2x4x32xbf16>
    %191 = sdy.sharding_constraint %190 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %192 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %193 = sdy.sharding_constraint %192 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %194 = stablehlo.divide %193, %191 : tensor<2x4x32xbf16>
    %195 = sdy.sharding_constraint %194 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %196 = stablehlo.multiply %177, %195 : tensor<2x4x32xbf16>
    %197 = sdy.sharding_constraint %196 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %198 = stablehlo.dot_general %197, %30, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %199 = sdy.sharding_constraint %198 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %200 = stablehlo.reshape %199 : (tensor<2x4x32xbf16>) -> tensor<2x4x2x16xbf16>
    %201 = sdy.sharding_constraint %200 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %202 = stablehlo.dot_general %197, %32, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %203 = sdy.sharding_constraint %202 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %204 = stablehlo.reshape %203 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %205 = sdy.sharding_constraint %204 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %206 = stablehlo.dot_general %197, %34, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %207 = sdy.sharding_constraint %206 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %208 = stablehlo.reshape %207 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %209 = sdy.sharding_constraint %208 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %210 = stablehlo.convert %201 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %211 = sdy.sharding_constraint %210 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %212 = chlo.square %211 : tensor<2x4x2x16xf32> -> tensor<2x4x2x16xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %213 = stablehlo.reduce(%212 init: %cst_16) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %214 = sdy.sharding_constraint %213 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %216 = sdy.sharding_constraint %215 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_17 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %217 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %218 = sdy.sharding_constraint %217 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %219 = stablehlo.divide %216, %218 : tensor<2x4x2x1xf32>
    %220 = sdy.sharding_constraint %219 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_18 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %221 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %222 = sdy.sharding_constraint %221 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %223 = stablehlo.add %220, %222 : tensor<2x4x2x1xf32>
    %224 = sdy.sharding_constraint %223 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %225 = stablehlo.rsqrt %224 : tensor<2x4x2x1xf32>
    %226 = sdy.sharding_constraint %225 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %227 = stablehlo.convert %201 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %228 = sdy.sharding_constraint %227 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %229 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x16xf32>
    %230 = sdy.sharding_constraint %229 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %231 = stablehlo.multiply %228, %230 : tensor<2x4x2x16xf32>
    %232 = sdy.sharding_constraint %231 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %233 = stablehlo.convert %232 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %234 = sdy.sharding_constraint %233 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %235 = stablehlo.convert %205 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %236 = sdy.sharding_constraint %235 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %237 = chlo.square %236 : tensor<2x4x1x16xf32> -> tensor<2x4x1x16xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %238 = stablehlo.reduce(%237 init: %cst_19) applies stablehlo.add across dimensions = [3] : (tensor<2x4x1x16xf32>, tensor<f32>) -> tensor<2x4x1xf32>
    %239 = sdy.sharding_constraint %238 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x1x1xf32>
    %241 = sdy.sharding_constraint %240 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_20 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %242 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %243 = sdy.sharding_constraint %242 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %244 = stablehlo.divide %241, %243 : tensor<2x4x1x1xf32>
    %245 = sdy.sharding_constraint %244 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_21 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %246 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %247 = sdy.sharding_constraint %246 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %248 = stablehlo.add %245, %247 : tensor<2x4x1x1xf32>
    %249 = sdy.sharding_constraint %248 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %250 = stablehlo.rsqrt %249 : tensor<2x4x1x1xf32>
    %251 = sdy.sharding_constraint %250 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %252 = stablehlo.convert %205 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %253 = sdy.sharding_constraint %252 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %254 = stablehlo.broadcast_in_dim %251, dims = [0, 1, 2, 3] : (tensor<2x4x1x1xf32>) -> tensor<2x4x1x16xf32>
    %255 = sdy.sharding_constraint %254 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %256 = stablehlo.multiply %253, %255 : tensor<2x4x1x16xf32>
    %257 = sdy.sharding_constraint %256 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %258 = stablehlo.convert %257 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %259 = sdy.sharding_constraint %258 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %cst_22 = stablehlo.constant dense<1.296880e+00> : tensor<bf16>
    %260 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %261 = sdy.sharding_constraint %260 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %262 = stablehlo.multiply %234, %261 : tensor<2x4x2x16xbf16>
    %263 = sdy.sharding_constraint %262 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %264 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %265 = sdy.sharding_constraint %264 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %267 = sdy.sharding_constraint %266 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %268 = stablehlo.reshape %267 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %269 = sdy.sharding_constraint %268 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %270 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %271 = sdy.sharding_constraint %270 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %273 = sdy.sharding_constraint %272 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %274 = stablehlo.reshape %273 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %275 = sdy.sharding_constraint %274 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_23 = stablehlo.constant dense<2.500000e-01> : tensor<bf16>
    %276 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %277 = sdy.sharding_constraint %276 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %278 = stablehlo.multiply %263, %277 : tensor<2x4x2x16xbf16>
    %279 = sdy.sharding_constraint %278 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %280 = stablehlo.dot_general %279, %269, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x4xbf16>
    %281 = sdy.sharding_constraint %280 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %282 = stablehlo.iota dim = 0 : tensor<4xi32>
    %283 = sdy.sharding_constraint %282 <@mesh, [{}]> : tensor<4xi32>
    %284 = stablehlo.broadcast_in_dim %283, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %285 = sdy.sharding_constraint %284 <@mesh, [{}, {}]> : tensor<4x1xi32>
    %286 = stablehlo.iota dim = 0 : tensor<4xi32>
    %287 = sdy.sharding_constraint %286 <@mesh, [{}]> : tensor<4xi32>
    %288 = stablehlo.broadcast_in_dim %287, dims = [1] : (tensor<4xi32>) -> tensor<1x4xi32>
    %289 = sdy.sharding_constraint %288 <@mesh, [{}, {}]> : tensor<1x4xi32>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %291 = sdy.sharding_constraint %290 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %292 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<4x1xi32>) -> tensor<4x4xi32>
    %293 = sdy.sharding_constraint %292 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %294 = stablehlo.compare LE, %291, %293, SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    %295 = stablehlo.broadcast_in_dim %294, dims = [1, 2] : (tensor<4x4xi1>) -> tensor<1x4x4xi1>
    %296 = sdy.sharding_constraint %295 <@mesh, [{}, {}, {}]> : tensor<1x4x4xi1>
    %297 = stablehlo.broadcast_in_dim %296, dims = [0, 2, 3] : (tensor<1x4x4xi1>) -> tensor<1x1x4x4xi1>
    %298 = sdy.sharding_constraint %297 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x4x4xi1>
    %cst_24 = stablehlo.constant dense<-9.982440e+08> : tensor<bf16>
    %299 = call @_where(%298, %281, %cst_24) : (tensor<1x1x4x4xi1>, tensor<2x2x4x4xbf16>, tensor<bf16>) -> tensor<2x2x4x4xbf16>
    %300 = sdy.sharding_constraint %299 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %301 = stablehlo.convert %300 : (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %302 = sdy.sharding_constraint %301 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_25 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %303 = stablehlo.reduce(%302 init: %cst_25) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %304 = sdy.sharding_constraint %303 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %cst_26 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %305 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<2x2x4xf32>
    %306 = sdy.sharding_constraint %305 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %307 = stablehlo.maximum %306, %304 : tensor<2x2x4xf32>
    %308 = sdy.sharding_constraint %307 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %310 = sdy.sharding_constraint %309 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %312 = sdy.sharding_constraint %311 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %313 = stablehlo.subtract %302, %312 : tensor<2x2x4x4xf32>
    %314 = sdy.sharding_constraint %313 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %315 = stablehlo.exponential %314 : tensor<2x2x4x4xf32>
    %316 = sdy.sharding_constraint %315 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %317 = stablehlo.reduce(%316 init: %cst_27) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %318 = sdy.sharding_constraint %317 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %320 = sdy.sharding_constraint %319 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %322 = sdy.sharding_constraint %321 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %323 = stablehlo.divide %316, %322 : tensor<2x2x4x4xf32>
    %324 = sdy.sharding_constraint %323 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %325 = stablehlo.convert %324 : (tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xbf16>
    %326 = sdy.sharding_constraint %325 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %327 = stablehlo.dot_general %275, %326, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x2x4x4xbf16>) -> tensor<2x2x16x4xbf16>
    %328 = sdy.sharding_constraint %327 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %329 = stablehlo.transpose %328, dims = [0, 3, 1, 2] : (tensor<2x2x16x4xbf16>) -> tensor<2x4x2x16xbf16>
    %330 = sdy.sharding_constraint %329 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %331 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %332 = sdy.sharding_constraint %331 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %334 = sdy.sharding_constraint %333 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %335 = stablehlo.reshape %334 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %336 = sdy.sharding_constraint %335 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %337 = sdy.sharding_constraint %336 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %338 = stablehlo.multiply %330, %337 : tensor<2x4x2x16xbf16>
    %339 = sdy.sharding_constraint %338 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %340 = stablehlo.convert %339 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %341 = sdy.sharding_constraint %340 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %342 = stablehlo.reduce(%341 init: %cst_28) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %343 = sdy.sharding_constraint %342 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %345 = sdy.sharding_constraint %344 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %346 = stablehlo.convert %345 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %347 = sdy.sharding_constraint %346 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %348 = stablehlo.multiply %337, %337 : tensor<2x4x2x16xbf16>
    %349 = sdy.sharding_constraint %348 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %350 = stablehlo.convert %349 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %351 = sdy.sharding_constraint %350 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %352 = stablehlo.reduce(%351 init: %cst_29) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %353 = sdy.sharding_constraint %352 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %355 = sdy.sharding_constraint %354 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %356 = stablehlo.convert %355 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %357 = sdy.sharding_constraint %356 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_30 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %358 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %359 = sdy.sharding_constraint %358 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %360 = stablehlo.add %357, %359 : tensor<2x4x2x1xbf16>
    %361 = sdy.sharding_constraint %360 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %362 = stablehlo.divide %347, %361 : tensor<2x4x2x1xbf16>
    %363 = sdy.sharding_constraint %362 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %365 = sdy.sharding_constraint %364 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %366 = stablehlo.multiply %365, %337 : tensor<2x4x2x16xbf16>
    %367 = sdy.sharding_constraint %366 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %368 = stablehlo.subtract %330, %367 : tensor<2x4x2x16xbf16>
    %369 = sdy.sharding_constraint %368 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %370 = stablehlo.dot_general %197, %38, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x2xbf16>) -> tensor<2x4x2xbf16>
    %371 = sdy.sharding_constraint %370 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %372 = stablehlo.negate %371 : tensor<2x4x2xbf16>
    %373 = sdy.sharding_constraint %372 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %374 = stablehlo.exponential %373 : tensor<2x4x2xbf16>
    %375 = sdy.sharding_constraint %374 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_31 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %376 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %377 = sdy.sharding_constraint %376 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %378 = stablehlo.add %377, %375 : tensor<2x4x2xbf16>
    %379 = sdy.sharding_constraint %378 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_32 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %380 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %381 = sdy.sharding_constraint %380 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %382 = stablehlo.divide %381, %379 : tensor<2x4x2xbf16>
    %383 = sdy.sharding_constraint %382 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2] : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %385 = sdy.sharding_constraint %384 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_33 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %386 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %387 = sdy.sharding_constraint %386 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %388 = stablehlo.multiply %387, %385 : tensor<2x4x2x1xbf16>
    %389 = sdy.sharding_constraint %388 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %391 = sdy.sharding_constraint %390 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %392 = stablehlo.multiply %391, %369 : tensor<2x4x2x16xbf16>
    %393 = sdy.sharding_constraint %392 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %394 = stablehlo.reshape %393 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x32xbf16>
    %395 = sdy.sharding_constraint %394 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %396 = stablehlo.dot_general %395, %36, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %397 = sdy.sharding_constraint %396 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %398 = stablehlo.add %145, %397 : tensor<2x4x32xbf16>
    %399 = sdy.sharding_constraint %398 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %400 = sdy.sharding_constraint %40 <@mesh, [{}]> : tensor<32xbf16>
    %401 = stablehlo.convert %399 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %402 = sdy.sharding_constraint %401 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %403 = chlo.square %402 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %404 = stablehlo.reduce(%403 init: %cst_34) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %405 = sdy.sharding_constraint %404 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %406 = stablehlo.broadcast_in_dim %405, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %407 = sdy.sharding_constraint %406 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_35 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %408 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %409 = sdy.sharding_constraint %408 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %410 = stablehlo.divide %407, %409 : tensor<2x4x1xf32>
    %411 = sdy.sharding_constraint %410 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_36 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %412 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %413 = sdy.sharding_constraint %412 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %414 = stablehlo.add %411, %413 : tensor<2x4x1xf32>
    %415 = sdy.sharding_constraint %414 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %416 = stablehlo.rsqrt %415 : tensor<2x4x1xf32>
    %417 = sdy.sharding_constraint %416 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %419 = sdy.sharding_constraint %418 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %420 = stablehlo.multiply %402, %419 : tensor<2x4x32xf32>
    %421 = sdy.sharding_constraint %420 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %422 = stablehlo.convert %400 : (tensor<32xbf16>) -> tensor<32xf32>
    %423 = sdy.sharding_constraint %422 <@mesh, [{}]> : tensor<32xf32>
    %424 = stablehlo.broadcast_in_dim %423, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %425 = sdy.sharding_constraint %424 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %427 = sdy.sharding_constraint %426 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %428 = stablehlo.multiply %421, %427 : tensor<2x4x32xf32>
    %429 = sdy.sharding_constraint %428 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %430 = stablehlo.convert %429 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %431 = sdy.sharding_constraint %430 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %432 = stablehlo.dot_general %431, %42, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %433 = sdy.sharding_constraint %432 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %434 = call @silu_0(%433) : (tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %435 = sdy.sharding_constraint %434 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %436 = stablehlo.dot_general %435, %44, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %437 = sdy.sharding_constraint %436 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %438 = stablehlo.negate %437 : tensor<2x4x32xbf16>
    %439 = sdy.sharding_constraint %438 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %440 = stablehlo.exponential %439 : tensor<2x4x32xbf16>
    %441 = sdy.sharding_constraint %440 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %442 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %443 = sdy.sharding_constraint %442 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %444 = stablehlo.add %443, %441 : tensor<2x4x32xbf16>
    %445 = sdy.sharding_constraint %444 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %446 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %447 = sdy.sharding_constraint %446 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %448 = stablehlo.divide %447, %445 : tensor<2x4x32xbf16>
    %449 = sdy.sharding_constraint %448 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %450 = stablehlo.multiply %431, %449 : tensor<2x4x32xbf16>
    %451 = sdy.sharding_constraint %450 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %452 = stablehlo.reshape %451 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %453 = sdy.sharding_constraint %452 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %454 = sdy.sharding_constraint %46 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %455 = stablehlo.dot_general %453, %454, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x4xbf16>) -> tensor<8x4xbf16>
    %456 = sdy.sharding_constraint %455 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xbf16>
    %457 = stablehlo.convert %456 : (tensor<8x4xbf16>) -> tensor<8x4xf32>
    %458 = sdy.sharding_constraint %457 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %459 = stablehlo.convert %48 : (tensor<4xbf16>) -> tensor<4xf32>
    %460 = sdy.sharding_constraint %459 <@mesh, [{}]> : tensor<4xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %462 = sdy.sharding_constraint %461 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<8x4xf32>
    %464 = sdy.sharding_constraint %463 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %465 = stablehlo.add %458, %464 : tensor<8x4xf32>
    %466 = sdy.sharding_constraint %465 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_39 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %467 = stablehlo.reduce(%458 init: %cst_39) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %468 = sdy.sharding_constraint %467 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_40 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %469 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %470 = sdy.sharding_constraint %469 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %471 = stablehlo.maximum %470, %468 : tensor<8xf32>
    %472 = sdy.sharding_constraint %471 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %473 = stablehlo.broadcast_in_dim %472, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %474 = sdy.sharding_constraint %473 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %475 = stablehlo.broadcast_in_dim %474, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %476 = sdy.sharding_constraint %475 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %477 = stablehlo.subtract %458, %476 : tensor<8x4xf32>
    %478 = sdy.sharding_constraint %477 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %479 = stablehlo.exponential %478 : tensor<8x4xf32>
    %480 = sdy.sharding_constraint %479 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %481 = stablehlo.reduce(%480 init: %cst_41) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %482 = sdy.sharding_constraint %481 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %484 = sdy.sharding_constraint %483 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %486 = sdy.sharding_constraint %485 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %487 = stablehlo.divide %480, %486 : tensor<8x4xf32>
    %488 = sdy.sharding_constraint %487 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %values, %indices = chlo.top_k(%466, k = 3) : tensor<8x4xf32> -> (tensor<8x3xf32>, tensor<8x3xi32>)
    %489 = sdy.sharding_constraint %values <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xf32>
    %490 = sdy.sharding_constraint %indices <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xi32>
    %491 = stablehlo.slice %489 [0:8, 2:3] : (tensor<8x3xf32>) -> tensor<8x1xf32>
    %492 = sdy.sharding_constraint %491 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %493 = stablehlo.slice %490 [0:8, 0:2] : (tensor<8x3xi32>) -> tensor<8x2xi32>
    %494 = sdy.sharding_constraint %493 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %495 = call @take_along_axis(%458, %494) : (tensor<8x4xf32>, tensor<8x2xi32>) -> tensor<8x2xf32>
    %496 = sdy.sharding_constraint %495 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %497 = stablehlo.negate %496 : tensor<8x2xf32>
    %498 = sdy.sharding_constraint %497 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %499 = stablehlo.exponential %498 : tensor<8x2xf32>
    %500 = sdy.sharding_constraint %499 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_42 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %501 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %502 = sdy.sharding_constraint %501 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %503 = stablehlo.add %502, %500 : tensor<8x2xf32>
    %504 = sdy.sharding_constraint %503 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_43 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %505 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %506 = sdy.sharding_constraint %505 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %507 = stablehlo.divide %506, %504 : tensor<8x2xf32>
    %508 = sdy.sharding_constraint %507 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %509 = stablehlo.reduce(%508 init: %cst_44) applies stablehlo.add across dimensions = [1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8xf32>
    %510 = sdy.sharding_constraint %509 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %511 = stablehlo.broadcast_in_dim %510, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %512 = sdy.sharding_constraint %511 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_45 = stablehlo.constant dense<9.99999971E-10> : tensor<f32>
    %513 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %514 = sdy.sharding_constraint %513 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %515 = stablehlo.add %512, %514 : tensor<8x1xf32>
    %516 = sdy.sharding_constraint %515 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_46 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %517 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %518 = sdy.sharding_constraint %517 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %519 = stablehlo.divide %518, %516 : tensor<8x1xf32>
    %520 = sdy.sharding_constraint %519 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %521 = stablehlo.broadcast_in_dim %520, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x2xf32>
    %522 = sdy.sharding_constraint %521 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %523 = stablehlo.multiply %508, %522 : tensor<8x2xf32>
    %524 = sdy.sharding_constraint %523 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %525 = stablehlo.convert %524 : (tensor<8x2xf32>) -> tensor<8x2xbf16>
    %526 = sdy.sharding_constraint %525 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %527 = call @_one_hot(%494) : (tensor<8x2xi32>) -> tensor<8x2x4xf32>
    %528 = sdy.sharding_constraint %527 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x4xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %529 = stablehlo.reduce(%528 init: %cst_47) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x2x4xf32>, tensor<f32>) -> tensor<4xf32>
    %530 = sdy.sharding_constraint %529 <@mesh, [{}]> : tensor<4xf32>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %531 = stablehlo.reduce(%530 init: %cst_48) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %532 = sdy.sharding_constraint %531 <@mesh, []> : tensor<f32>
    %cst_49 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %533 = sdy.sharding_constraint %cst_49 <@mesh, []> : tensor<f32>
    %534 = stablehlo.maximum %532, %533 : tensor<f32>
    %535 = sdy.sharding_constraint %534 <@mesh, []> : tensor<f32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %537 = sdy.sharding_constraint %536 <@mesh, [{}]> : tensor<4xf32>
    %538 = stablehlo.divide %530, %537 : tensor<4xf32>
    %539 = sdy.sharding_constraint %538 <@mesh, [{}]> : tensor<4xf32>
    %cst_50 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %540 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %541 = sdy.sharding_constraint %540 <@mesh, [{}]> : tensor<4xf32>
    %542 = stablehlo.add %539, %541 : tensor<4xf32>
    %543 = sdy.sharding_constraint %542 <@mesh, [{}]> : tensor<4xf32>
    %544 = stablehlo.log %543 : tensor<4xf32>
    %545 = sdy.sharding_constraint %544 <@mesh, [{}]> : tensor<4xf32>
    %546 = stablehlo.multiply %539, %545 : tensor<4xf32>
    %547 = sdy.sharding_constraint %546 <@mesh, [{}]> : tensor<4xf32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %548 = stablehlo.reduce(%547 init: %cst_51) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %549 = sdy.sharding_constraint %548 <@mesh, []> : tensor<f32>
    %550 = stablehlo.negate %549 : tensor<f32>
    %551 = sdy.sharding_constraint %550 <@mesh, []> : tensor<f32>
    %cst_52 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %552 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %553 = sdy.sharding_constraint %552 <@mesh, [{}]> : tensor<4xf32>
    %554 = stablehlo.multiply %539, %553 : tensor<4xf32>
    %555 = sdy.sharding_constraint %554 <@mesh, [{}]> : tensor<4xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %556 = stablehlo.reduce(%488 init: %cst_53) applies stablehlo.add across dimensions = [0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<4xf32>
    %557 = sdy.sharding_constraint %556 <@mesh, [{}]> : tensor<4xf32>
    %cst_54 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %558 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %559 = sdy.sharding_constraint %558 <@mesh, [{}]> : tensor<4xf32>
    %560 = stablehlo.divide %557, %559 : tensor<4xf32>
    %561 = sdy.sharding_constraint %560 <@mesh, [{}]> : tensor<4xf32>
    %562 = stablehlo.multiply %555, %561 : tensor<4xf32>
    %563 = sdy.sharding_constraint %562 <@mesh, [{}]> : tensor<4xf32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %564 = stablehlo.reduce(%563 init: %cst_55) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %565 = sdy.sharding_constraint %564 <@mesh, []> : tensor<f32>
    %cst_56 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %566 = sdy.sharding_constraint %cst_56 <@mesh, []> : tensor<f32>
    %567 = stablehlo.multiply %566, %565 : tensor<f32>
    %568 = sdy.sharding_constraint %567 <@mesh, []> : tensor<f32>
    %cst_57 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %569 = stablehlo.reduce(%458 init: %cst_57) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %570 = sdy.sharding_constraint %569 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_58 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %571 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %572 = sdy.sharding_constraint %571 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %573 = stablehlo.maximum %572, %570 : tensor<8xf32>
    %574 = sdy.sharding_constraint %573 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %575 = stablehlo.is_finite %574 : (tensor<8xf32>) -> tensor<8xi1>
    %576 = sdy.sharding_constraint %575 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xi1>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %577 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %578 = sdy.sharding_constraint %577 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %579 = stablehlo.select %576, %574, %578 : tensor<8xi1>, tensor<8xf32>
    %580 = sdy.sharding_constraint %579 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %582 = sdy.sharding_constraint %581 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %583 = stablehlo.broadcast_in_dim %582, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %584 = sdy.sharding_constraint %583 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %585 = stablehlo.subtract %458, %584 : tensor<8x4xf32>
    %586 = sdy.sharding_constraint %585 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %587 = stablehlo.exponential %586 : tensor<8x4xf32>
    %588 = sdy.sharding_constraint %587 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %589 = stablehlo.reduce(%588 init: %cst_60) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %590 = sdy.sharding_constraint %589 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %591 = stablehlo.abs %590 : tensor<8xf32>
    %592 = sdy.sharding_constraint %591 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %593 = stablehlo.log %592 : tensor<8xf32>
    %594 = sdy.sharding_constraint %593 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %595 = stablehlo.add %594, %580 : tensor<8xf32>
    %596 = sdy.sharding_constraint %595 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %597 = stablehlo.multiply %596, %596 : tensor<8xf32>
    %598 = sdy.sharding_constraint %597 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %599 = stablehlo.reduce(%598 init: %cst_61) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %600 = sdy.sharding_constraint %599 <@mesh, []> : tensor<f32>
    %cst_62 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %601 = sdy.sharding_constraint %cst_62 <@mesh, []> : tensor<f32>
    %602 = stablehlo.divide %600, %601 : tensor<f32>
    %603 = sdy.sharding_constraint %602 <@mesh, []> : tensor<f32>
    %604 = stablehlo.broadcast_in_dim %492, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %605 = sdy.sharding_constraint %604 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %606 = stablehlo.subtract %458, %605 : tensor<8x4xf32>
    %607 = sdy.sharding_constraint %606 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %608 = sdy.sharding_constraint %607 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %609 = stablehlo.transpose %608, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
    %values_63, %indices_64 = chlo.top_k(%609, k = 4) : tensor<4x8xf32> -> (tensor<4x4xf32>, tensor<4x4xi32>)
    %610 = stablehlo.slice %values_63 [0:4, 3:4] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %611 = stablehlo.reshape %610 : (tensor<4x1xf32>) -> tensor<4xf32>
    %612 = "stablehlo.all_reduce"(%611) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<f32>, %arg31: tensor<f32>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<f32>
      stablehlo.return %2665 : tensor<f32>
    }) : (tensor<4xf32>) -> tensor<4xf32>
    %cst_65 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %613 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %614 = stablehlo.divide %612, %613 : tensor<4xf32>
    %615 = stablehlo.concatenate %56, %58, dim = 2 : (tensor<4x32x32xbf16>, tensor<4x32x32xbf16>) -> tensor<4x32x64xbf16>
    %616 = sdy.sharding_constraint %615 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x64xbf16>
    %617 = sdy.sharding_constraint %453 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %618 = sdy.sharding_constraint %494 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %619 = sdy.sharding_constraint %526 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %620 = sdy.sharding_constraint %616 <@mesh, [{}, {}, {}]> : tensor<4x32x64xbf16>
    %621 = sdy.sharding_constraint %60 <@mesh, [{}, {}, {}]> : tensor<4x32x32xbf16>
    %622 = stablehlo.reshape %618 : (tensor<8x2xi32>) -> tensor<16xi32>
    %623 = stablehlo.reshape %619 : (tensor<8x2xbf16>) -> tensor<16xbf16>
    %624 = call @argsort(%622) : (tensor<16xi32>) -> tensor<16xi32>
    %625 = stablehlo.iota dim = 0 : tensor<16xi32>
    %c_66 = stablehlo.constant dense<2> : tensor<i32>
    %626 = call @floor_divide(%625, %c_66) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_67 = stablehlo.constant dense<0> : tensor<i32>
    %627 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %628 = stablehlo.compare LT, %624, %627, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_68 = stablehlo.constant dense<16> : tensor<i32>
    %629 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %630 = stablehlo.add %624, %629 : tensor<16xi32>
    %631 = stablehlo.select %628, %630, %624 : tensor<16xi1>, tensor<16xi32>
    %632 = stablehlo.broadcast_in_dim %631, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %633 = "stablehlo.gather"(%626, %632) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xi32>, tensor<16x1xi32>) -> tensor<16xi32>
    %c_69 = stablehlo.constant dense<0> : tensor<i32>
    %634 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %635 = stablehlo.compare LT, %633, %634, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_70 = stablehlo.constant dense<8> : tensor<i32>
    %636 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %637 = stablehlo.add %633, %636 : tensor<16xi32>
    %638 = stablehlo.select %635, %637, %633 : tensor<16xi1>, tensor<16xi32>
    %639 = stablehlo.broadcast_in_dim %638, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %640 = "stablehlo.gather"(%617, %639) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<8x32xbf16>, tensor<16x1xi32>) -> tensor<16x32xbf16>
    %c_71 = stablehlo.constant dense<0> : tensor<i32>
    %641 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %642 = stablehlo.compare LT, %624, %641, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_72 = stablehlo.constant dense<16> : tensor<i32>
    %643 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %644 = stablehlo.add %624, %643 : tensor<16xi32>
    %645 = stablehlo.select %642, %644, %624 : tensor<16xi1>, tensor<16xi32>
    %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %647 = "stablehlo.gather"(%623, %646) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xbf16>, tensor<16x1xi32>) -> tensor<16xbf16>
    %c_73 = stablehlo.constant dense<0> : tensor<i32>
    %648 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %c_74 = stablehlo.constant dense<0> : tensor<i32>
    %649 = call @clip(%622, %c_74) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_75 = stablehlo.constant dense<0> : tensor<i32>
    %650 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %651 = stablehlo.compare LT, %649, %650, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_76 = stablehlo.constant dense<4> : tensor<i32>
    %652 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %653 = stablehlo.add %649, %652 : tensor<16xi32>
    %654 = stablehlo.select %651, %653, %649 : tensor<16xi1>, tensor<16xi32>
    %655 = stablehlo.broadcast_in_dim %654, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %c_77 = stablehlo.constant dense<1> : tensor<i32>
    %656 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %657 = "stablehlo.scatter"(%648, %655, %656) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg30: tensor<i32>, %arg31: tensor<i32>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<i32>
      stablehlo.return %2665 : tensor<i32>
    }) : (tensor<4xi32>, tensor<16x1xi32>, tensor<16xi32>) -> tensor<4xi32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %658 = stablehlo.pad %640, %cst_78, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %659 = stablehlo.broadcast_in_dim %658, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %660 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %661 = call @cumsum(%657) : (tensor<4xi32>) -> tensor<4xi32>
    %c_79 = stablehlo.constant dense<0> : tensor<i32>
    %662 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %663 = stablehlo.slice %662 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %664 = stablehlo.slice %661 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %665 = stablehlo.concatenate %663, %664, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %666 = stablehlo.broadcast_in_dim %661, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %667 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %668 = stablehlo.compare LE, %667, %660, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %669 = stablehlo.compare LT, %660, %666, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %670 = stablehlo.and %668, %669 : tensor<4x512x32xi1>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %671 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %672 = stablehlo.select %670, %659, %671 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %673 = stablehlo.dot_general %672, %620, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x64xbf16>) -> tensor<512x64xbf16>
    %674 = stablehlo.slice %673 [0:16, 0:64] : (tensor<512x64xbf16>) -> tensor<16x64xbf16>
    %675 = stablehlo.slice %674 [0:16, 0:32] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %676 = stablehlo.slice %674 [0:16, 32:64] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %677 = call @silu_2(%675) : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %678 = stablehlo.multiply %677, %676 : tensor<16x32xbf16>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %679 = stablehlo.pad %678, %cst_81, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %680 = stablehlo.broadcast_in_dim %679, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %681 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %682 = call @cumsum(%657) : (tensor<4xi32>) -> tensor<4xi32>
    %c_82 = stablehlo.constant dense<0> : tensor<i32>
    %683 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %684 = stablehlo.slice %683 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %685 = stablehlo.slice %682 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %686 = stablehlo.concatenate %684, %685, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %687 = stablehlo.broadcast_in_dim %682, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %688 = stablehlo.broadcast_in_dim %686, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %689 = stablehlo.compare LE, %688, %681, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %690 = stablehlo.compare LT, %681, %687, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %691 = stablehlo.and %689, %690 : tensor<4x512x32xi1>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %692 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %693 = stablehlo.select %691, %680, %692 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %694 = stablehlo.dot_general %693, %621, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x32xbf16>) -> tensor<512x32xbf16>
    %695 = stablehlo.slice %694 [0:16, 0:32] : (tensor<512x32xbf16>) -> tensor<16x32xbf16>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %696 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %697 = stablehlo.broadcast_in_dim %647, dims = [0] : (tensor<16xbf16>) -> tensor<16x1xbf16>
    %698 = stablehlo.broadcast_in_dim %697, dims = [0, 1] : (tensor<16x1xbf16>) -> tensor<16x32xbf16>
    %699 = stablehlo.multiply %695, %698 : tensor<16x32xbf16>
    %c_85 = stablehlo.constant dense<0> : tensor<i32>
    %700 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %701 = stablehlo.compare LT, %633, %700, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_86 = stablehlo.constant dense<8> : tensor<i32>
    %702 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %703 = stablehlo.add %633, %702 : tensor<16xi32>
    %704 = stablehlo.select %701, %703, %633 : tensor<16xi1>, tensor<16xi32>
    %705 = stablehlo.broadcast_in_dim %704, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %706 = "stablehlo.scatter"(%696, %705, %699) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<8x32xbf16>, tensor<16x1xi32>, tensor<16x32xbf16>) -> tensor<8x32xbf16>
    %c_87 = stablehlo.constant dense<0> : tensor<i32>
    %707 = stablehlo.convert %c_87 : (tensor<i32>) -> tensor<f32>
    %708 = sdy.sharding_constraint %707 <@mesh, []> : tensor<f32>
    %709 = stablehlo.reshape %706 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %710 = sdy.sharding_constraint %709 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %711 = sdy.sharding_constraint %710 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %712 = stablehlo.reshape %451 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %713 = sdy.sharding_constraint %712 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %714 = stablehlo.dot_general %713, %50, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %715 = sdy.sharding_constraint %714 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %716 = stablehlo.dot_general %713, %52, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %717 = sdy.sharding_constraint %716 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %718 = call @silu_3(%715) : (tensor<8x32xbf16>) -> tensor<8x32xbf16>
    %719 = sdy.sharding_constraint %718 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %720 = stablehlo.multiply %719, %717 : tensor<8x32xbf16>
    %721 = sdy.sharding_constraint %720 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %722 = stablehlo.dot_general %721, %54, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %723 = sdy.sharding_constraint %722 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %724 = stablehlo.reshape %723 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %725 = sdy.sharding_constraint %724 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %726 = sdy.sharding_constraint %725 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %727 = stablehlo.add %711, %726 : tensor<2x4x32xbf16>
    %728 = sdy.sharding_constraint %727 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %729 = stablehlo.add %399, %728 : tensor<2x4x32xbf16>
    %730 = sdy.sharding_constraint %729 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %731 = stablehlo.convert %730 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %732 = sdy.sharding_constraint %731 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %733 = chlo.square %732 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %734 = stablehlo.reduce(%733 init: %cst_88) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<f32>
    %735 = sdy.sharding_constraint %734 <@mesh, []> : tensor<f32>
    %cst_89 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %736 = sdy.sharding_constraint %cst_89 <@mesh, []> : tensor<f32>
    %737 = stablehlo.divide %735, %736 : tensor<f32>
    %738 = sdy.sharding_constraint %737 <@mesh, []> : tensor<f32>
    %739 = stablehlo.sqrt %738 : tensor<f32>
    %740 = sdy.sharding_constraint %739 <@mesh, []> : tensor<f32>
    %741 = stablehlo.broadcast_in_dim %551, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %742 = sdy.sharding_constraint %741 <@mesh, [{}]> : tensor<1xf32>
    %743 = stablehlo.broadcast_in_dim %530, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %744 = sdy.sharding_constraint %743 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %745 = stablehlo.broadcast_in_dim %568, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %746 = sdy.sharding_constraint %745 <@mesh, [{}]> : tensor<1xf32>
    %747 = stablehlo.broadcast_in_dim %603, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %748 = sdy.sharding_constraint %747 <@mesh, [{}]> : tensor<1xf32>
    %749 = stablehlo.broadcast_in_dim %614, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %750 = sdy.sharding_constraint %749 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %751 = stablehlo.broadcast_in_dim %708, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %752 = sdy.sharding_constraint %751 <@mesh, [{}]> : tensor<1xf32>
    %753 = stablehlo.broadcast_in_dim %740, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %754 = sdy.sharding_constraint %753 <@mesh, [{}]> : tensor<1xf32>
    %755 = sdy.sharding_constraint %62 <@mesh, [{}]> : tensor<32xbf16>
    %756 = stablehlo.convert %730 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %757 = sdy.sharding_constraint %756 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %758 = chlo.square %757 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_90 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %759 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<2x4x32xf32>
    %760 = sdy.sharding_constraint %759 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %761 = stablehlo.multiply %760, %757 : tensor<2x4x32xf32>
    %762 = sdy.sharding_constraint %761 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %763 = stablehlo.reduce(%758 init: %cst_91) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %764 = sdy.sharding_constraint %763 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %766 = sdy.sharding_constraint %765 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_92 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %767 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %768 = sdy.sharding_constraint %767 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %769 = stablehlo.divide %766, %768 : tensor<2x4x1xf32>
    %770 = sdy.sharding_constraint %769 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_93 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %771 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %772 = sdy.sharding_constraint %771 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %773 = stablehlo.add %770, %772 : tensor<2x4x1xf32>
    %774 = sdy.sharding_constraint %773 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %775 = stablehlo.rsqrt %774 : tensor<2x4x1xf32>
    %776 = sdy.sharding_constraint %775 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %777 = stablehlo.divide %776, %774 : tensor<2x4x1xf32>
    %778 = sdy.sharding_constraint %777 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_94 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %779 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %780 = sdy.sharding_constraint %779 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %781 = stablehlo.multiply %780, %778 : tensor<2x4x1xf32>
    %782 = sdy.sharding_constraint %781 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %783 = stablehlo.broadcast_in_dim %776, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %784 = sdy.sharding_constraint %783 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %785 = stablehlo.multiply %757, %784 : tensor<2x4x32xf32>
    %786 = sdy.sharding_constraint %785 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %787 = stablehlo.convert %755 : (tensor<32xbf16>) -> tensor<32xf32>
    %788 = sdy.sharding_constraint %787 <@mesh, [{}]> : tensor<32xf32>
    %789 = stablehlo.broadcast_in_dim %788, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %790 = sdy.sharding_constraint %789 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %792 = sdy.sharding_constraint %791 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %793 = stablehlo.multiply %786, %792 : tensor<2x4x32xf32>
    %794 = sdy.sharding_constraint %793 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %795 = stablehlo.convert %794 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %796 = sdy.sharding_constraint %795 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %797 = stablehlo.dot_general %796, %64, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %798 = sdy.sharding_constraint %797 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %799:3 = call @silu(%798) : (tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>)
    %800 = sdy.sharding_constraint %799#0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %801 = sdy.sharding_constraint %799#1 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %802 = sdy.sharding_constraint %799#2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %803 = stablehlo.dot_general %800, %66, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %804 = sdy.sharding_constraint %803 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %805 = stablehlo.negate %804 : tensor<2x4x32xbf16>
    %806 = sdy.sharding_constraint %805 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %807 = stablehlo.exponential %806 : tensor<2x4x32xbf16>
    %808 = sdy.sharding_constraint %807 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_95 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %809 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %810 = sdy.sharding_constraint %809 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %811 = stablehlo.add %810, %808 : tensor<2x4x32xbf16>
    %812 = sdy.sharding_constraint %811 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_96 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %813 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %814 = sdy.sharding_constraint %813 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %815 = stablehlo.divide %814, %812 : tensor<2x4x32xbf16>
    %816 = sdy.sharding_constraint %815 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_97 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %817 = sdy.sharding_constraint %cst_97 <@mesh, []> : tensor<bf16>
    %818 = stablehlo.broadcast_in_dim %817, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %819 = sdy.sharding_constraint %818 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %820 = stablehlo.subtract %819, %816 : tensor<2x4x32xbf16>
    %821 = sdy.sharding_constraint %820 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %822 = stablehlo.multiply %816, %821 : tensor<2x4x32xbf16>
    %823 = sdy.sharding_constraint %822 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %824 = stablehlo.multiply %796, %816 : tensor<2x4x32xbf16>
    %825 = sdy.sharding_constraint %824 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %826 = stablehlo.slice %arg28 [0:2, 1:4] : (tensor<2x4xi32>) -> tensor<2x3xi32>
    %827 = sdy.sharding_constraint %826 <@mesh, [{}, {}]> : tensor<2x3xi32>
    %828 = stablehlo.slice %arg28 [0:2, 0:1] : (tensor<2x4xi32>) -> tensor<2x1xi32>
    %829 = sdy.sharding_constraint %828 <@mesh, [{}, {}]> : tensor<2x1xi32>
    %c_98 = stablehlo.constant dense<0> : tensor<i32>
    %830 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %831 = sdy.sharding_constraint %830 <@mesh, [{}, {}]> : tensor<2x1xi32>
    %832 = stablehlo.multiply %829, %831 : tensor<2x1xi32>
    %833 = sdy.sharding_constraint %832 <@mesh, [{}, {}]> : tensor<2x1xi32>
    %834 = stablehlo.concatenate %827, %833, dim = 1 : (tensor<2x3xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
    %835 = sdy.sharding_constraint %834 <@mesh, [{}, {}]> : tensor<2x4xi32>
    %836 = sdy.sharding_constraint %825 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %837 = sdy.sharding_constraint %22 <@mesh, [{}, {}]> : tensor<32x64xbf16>
    %838 = sdy.sharding_constraint %835 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xi32>
    %839 = sdy.sharding_constraint %arg29 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %840 = stablehlo.reshape %836 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %841 = stablehlo.reshape %838 : (tensor<2x4xi32>) -> tensor<8xi32>
    %842 = stablehlo.reshape %839 : (tensor<2x4xf32>) -> tensor<8xf32>
    %843 = stablehlo.dot_general %840, %837, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x64xbf16>) -> tensor<8x64xbf16>
    %844 = stablehlo.convert %843 : (tensor<8x64xbf16>) -> tensor<8x64xf32>
    %cst_99 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %845 = stablehlo.reduce(%844 init: %cst_99) applies stablehlo.maximum across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %cst_100 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %846 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %847 = stablehlo.maximum %846, %845 : tensor<8xf32>
    %848 = stablehlo.is_finite %847 : (tensor<8xf32>) -> tensor<8xi1>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %849 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %850 = stablehlo.select %848, %847, %849 : tensor<8xi1>, tensor<8xf32>
    %851 = stablehlo.broadcast_in_dim %850, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %852 = stablehlo.broadcast_in_dim %851, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %853 = stablehlo.subtract %844, %852 : tensor<8x64xf32>
    %854 = stablehlo.exponential %853 : tensor<8x64xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %855 = stablehlo.reduce(%854 init: %cst_102) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %856 = stablehlo.abs %855 : tensor<8xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %857 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %858 = stablehlo.compare GE, %855, %857, FLOAT : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xi1>
    %859 = stablehlo.log %856 : tensor<8xf32>
    %860 = stablehlo.add %859, %850 : tensor<8xf32>
    %861 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c_104 = stablehlo.constant dense<0> : tensor<i32>
    %862 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %863 = stablehlo.compare LT, %861, %862, SIGNED : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
    %c_105 = stablehlo.constant dense<8> : tensor<i32>
    %864 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %865 = stablehlo.add %861, %864 : tensor<8xi32>
    %866 = stablehlo.select %863, %865, %861 : tensor<8xi1>, tensor<8xi32>
    %c_106 = stablehlo.constant dense<0> : tensor<i32>
    %867 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %868 = stablehlo.compare LT, %841, %867, SIGNED : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
    %c_107 = stablehlo.constant dense<64> : tensor<i32>
    %869 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %870 = stablehlo.add %841, %869 : tensor<8xi32>
    %871 = stablehlo.select %868, %870, %841 : tensor<8xi1>, tensor<8xi32>
    %872 = stablehlo.broadcast_in_dim %866, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %873 = stablehlo.broadcast_in_dim %871, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %874 = stablehlo.concatenate %872, %873, dim = 1 : (tensor<8x1xi32>, tensor<8x1xi32>) -> tensor<8x2xi32>
    %875 = "stablehlo.gather"(%844, %874) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x64xf32>, tensor<8x2xi32>) -> tensor<8xf32>
    %876 = stablehlo.subtract %860, %875 : tensor<8xf32>
    %877 = stablehlo.multiply %876, %842 : tensor<8xf32>
    %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %878 = stablehlo.reduce(%877 init: %cst_108) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %879 = stablehlo.reduce(%842 init: %cst_109) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %880 = "stablehlo.all_reduce"(%878) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<f32>, %arg31: tensor<f32>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<f32>
      stablehlo.return %2665 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %881 = "stablehlo.all_reduce"(%879) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<f32>, %arg31: tensor<f32>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<f32>
      stablehlo.return %2665 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %882 = stablehlo.compare NE, %881, %cst_110, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %883 = stablehlo.divide %880, %881 : tensor<f32>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %884 = call @_where_4(%882, %883, %cst_111) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %885 = stablehlo.broadcast_in_dim %881, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %886 = stablehlo.broadcast_in_dim %882, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %cst_112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %887 = stablehlo.reduce(%748 init: %cst_112) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %888 = sdy.sharding_constraint %887 <@mesh, []> : tensor<f32>
    %cst_113 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %889 = sdy.sharding_constraint %cst_113 <@mesh, []> : tensor<f32>
    %890 = stablehlo.divide %888, %889 : tensor<f32>
    %891 = sdy.sharding_constraint %890 <@mesh, []> : tensor<f32>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %892 = sdy.sharding_constraint %cst_114 <@mesh, []> : tensor<f32>
    %893 = stablehlo.multiply %892, %891 : tensor<f32>
    %894 = sdy.sharding_constraint %893 <@mesh, []> : tensor<f32>
    %895 = stablehlo.add %884, %894 : tensor<f32>
    %896 = sdy.sharding_constraint %895 <@mesh, []> : tensor<f32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %897 = stablehlo.reduce(%744 init: %cst_115) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
    %898 = sdy.sharding_constraint %897 <@mesh, [{}]> : tensor<1xf32>
    %cst_116 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %899 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %900 = sdy.sharding_constraint %899 <@mesh, [{}]> : tensor<1xf32>
    %901 = stablehlo.maximum %898, %900 : tensor<1xf32>
    %902 = sdy.sharding_constraint %901 <@mesh, [{}]> : tensor<1xf32>
    %903 = stablehlo.divide %752, %902 : tensor<1xf32>
    %904 = sdy.sharding_constraint %903 <@mesh, [{}]> : tensor<1xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %905 = stablehlo.reduce(%742 init: %cst_117) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %906 = sdy.sharding_constraint %905 <@mesh, []> : tensor<f32>
    %cst_118 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %907 = sdy.sharding_constraint %cst_118 <@mesh, []> : tensor<f32>
    %908 = stablehlo.divide %906, %907 : tensor<f32>
    %909 = sdy.sharding_constraint %908 <@mesh, []> : tensor<f32>
    %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %910 = stablehlo.reduce(%746 init: %cst_119) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %911 = sdy.sharding_constraint %910 <@mesh, []> : tensor<f32>
    %cst_120 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %912 = sdy.sharding_constraint %cst_120 <@mesh, []> : tensor<f32>
    %913 = stablehlo.divide %911, %912 : tensor<f32>
    %914 = sdy.sharding_constraint %913 <@mesh, []> : tensor<f32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %915 = stablehlo.reduce(%748 init: %cst_121) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %916 = sdy.sharding_constraint %915 <@mesh, []> : tensor<f32>
    %cst_122 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %917 = sdy.sharding_constraint %cst_122 <@mesh, []> : tensor<f32>
    %918 = stablehlo.divide %916, %917 : tensor<f32>
    %919 = sdy.sharding_constraint %918 <@mesh, []> : tensor<f32>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %920 = stablehlo.reduce(%904 init: %cst_123) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %921 = sdy.sharding_constraint %920 <@mesh, []> : tensor<f32>
    %cst_124 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %922 = sdy.sharding_constraint %cst_124 <@mesh, []> : tensor<f32>
    %923 = stablehlo.divide %921, %922 : tensor<f32>
    %924 = sdy.sharding_constraint %923 <@mesh, []> : tensor<f32>
    %925 = stablehlo.reshape %742 : (tensor<1xf32>) -> tensor<f32>
    %926 = sdy.sharding_constraint %925 <@mesh, []> : tensor<f32>
    %927 = stablehlo.reshape %746 : (tensor<1xf32>) -> tensor<f32>
    %928 = sdy.sharding_constraint %927 <@mesh, []> : tensor<f32>
    %929 = stablehlo.reshape %748 : (tensor<1xf32>) -> tensor<f32>
    %930 = sdy.sharding_constraint %929 <@mesh, []> : tensor<f32>
    %931 = stablehlo.reshape %744 : (tensor<1x4xf32>) -> tensor<4xf32>
    %932 = sdy.sharding_constraint %931 <@mesh, [{}]> : tensor<4xf32>
    %933 = stablehlo.iota dim = 0 : tensor<4xf32>
    %934 = sdy.sharding_constraint %933 <@mesh, [{}]> : tensor<4xf32>
    %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %935 = stablehlo.reduce(%932 init: %cst_125) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %936 = sdy.sharding_constraint %935 <@mesh, []> : tensor<f32>
    %937 = stablehlo.multiply %932, %934 : tensor<4xf32>
    %938 = sdy.sharding_constraint %937 <@mesh, [{}]> : tensor<4xf32>
    %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %939 = stablehlo.reduce(%938 init: %cst_126) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %940 = sdy.sharding_constraint %939 <@mesh, []> : tensor<f32>
    %941 = stablehlo.multiply %932, %934 : tensor<4xf32>
    %942 = sdy.sharding_constraint %941 <@mesh, [{}]> : tensor<4xf32>
    %943 = stablehlo.multiply %942, %934 : tensor<4xf32>
    %944 = sdy.sharding_constraint %943 <@mesh, [{}]> : tensor<4xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %945 = stablehlo.reduce(%944 init: %cst_127) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %946 = sdy.sharding_constraint %945 <@mesh, []> : tensor<f32>
    %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %947 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %948 = sdy.sharding_constraint %947 <@mesh, [{}]> : tensor<4xf32>
    %949 = stablehlo.compare GT, %932, %948, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %cst_129 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %950 = call @_where_5(%949, %934, %cst_129) : (tensor<4xi1>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    %951 = sdy.sharding_constraint %950 <@mesh, [{}]> : tensor<4xf32>
    %cst_130 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %952 = stablehlo.reduce(%951 init: %cst_130) applies stablehlo.minimum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %953 = sdy.sharding_constraint %952 <@mesh, []> : tensor<f32>
    %cst_131 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %954 = call @_where_5(%949, %934, %cst_131) : (tensor<4xi1>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    %955 = sdy.sharding_constraint %954 <@mesh, [{}]> : tensor<4xf32>
    %cst_132 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %956 = stablehlo.reduce(%955 init: %cst_132) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %957 = sdy.sharding_constraint %956 <@mesh, []> : tensor<f32>
    %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %958 = sdy.sharding_constraint %cst_133 <@mesh, []> : tensor<f32>
    %959 = stablehlo.compare GT, %936, %958, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %960 = call @_where_6(%959, %953, %cst_134) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %961 = sdy.sharding_constraint %960 <@mesh, []> : tensor<f32>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %962 = sdy.sharding_constraint %cst_135 <@mesh, []> : tensor<f32>
    %963 = stablehlo.compare GT, %936, %962, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %964 = call @_where_6(%963, %957, %cst_136) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %965 = sdy.sharding_constraint %964 <@mesh, []> : tensor<f32>
    %966 = stablehlo.iota dim = 0 : tensor<5xf32>
    %967 = sdy.sharding_constraint %966 <@mesh, [{}]> : tensor<5xf32>
    %968 = stablehlo.convert %949 : (tensor<4xi1>) -> tensor<4xi32>
    %969 = sdy.sharding_constraint %968 <@mesh, [{}]> : tensor<4xi32>
    %c_137 = stablehlo.constant dense<0> : tensor<i32>
    %970 = stablehlo.reduce(%969 init: %c_137) applies stablehlo.add across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
    %971 = sdy.sharding_constraint %970 <@mesh, []> : tensor<i32>
    %972 = stablehlo.divide %940, %936 : tensor<f32>
    %973 = sdy.sharding_constraint %972 <@mesh, []> : tensor<f32>
    %974 = stablehlo.divide %946, %936 : tensor<f32>
    %975 = sdy.sharding_constraint %974 <@mesh, []> : tensor<f32>
    %976 = stablehlo.multiply %973, %973 : tensor<f32>
    %977 = sdy.sharding_constraint %976 <@mesh, []> : tensor<f32>
    %978 = stablehlo.subtract %975, %977 : tensor<f32>
    %979 = sdy.sharding_constraint %978 <@mesh, []> : tensor<f32>
    %980 = stablehlo.divide %946, %936 : tensor<f32>
    %981 = sdy.sharding_constraint %980 <@mesh, []> : tensor<f32>
    %982 = stablehlo.sqrt %981 : tensor<f32>
    %983 = sdy.sharding_constraint %982 <@mesh, []> : tensor<f32>
    %984 = stablehlo.reshape %904 : (tensor<1xf32>) -> tensor<f32>
    %985 = sdy.sharding_constraint %984 <@mesh, []> : tensor<f32>
    %986 = stablehlo.reshape %754 : (tensor<1xf32>) -> tensor<f32>
    %987 = sdy.sharding_constraint %986 <@mesh, []> : tensor<f32>
    %cst_138 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %988 = sdy.sharding_constraint %cst_138 <@mesh, []> : tensor<f32>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %989 = sdy.sharding_constraint %cst_139 <@mesh, []> : tensor<f32>
    %990 = stablehlo.multiply %989, %988 : tensor<f32>
    %991 = sdy.sharding_constraint %990 <@mesh, []> : tensor<f32>
    %cst_140 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %992 = sdy.sharding_constraint %cst_140 <@mesh, []> : tensor<f32>
    %993 = stablehlo.divide %991, %992 : tensor<f32>
    %994 = sdy.sharding_constraint %993 <@mesh, []> : tensor<f32>
    %995 = stablehlo.broadcast_in_dim %994, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %996 = sdy.sharding_constraint %995 <@mesh, [{}]> : tensor<1xf32>
    %997 = stablehlo.reshape %885 : (tensor<1xf32>) -> tensor<f32>
    %998 = stablehlo.reshape %886 : (tensor<1xi1>) -> tensor<i1>
    %999 = call @_where_7(%998, %988) : (tensor<i1>, tensor<f32>) -> tensor<f32>
    %1000 = stablehlo.divide %999, %997 : tensor<f32>
    %1001 = "stablehlo.all_reduce"(%1000) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<f32>, %arg31: tensor<f32>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<f32>
      stablehlo.return %2665 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %1002 = stablehlo.broadcast_in_dim %1001, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1003 = stablehlo.multiply %1002, %842 : tensor<8xf32>
    %1004 = stablehlo.negate %1003 : tensor<8xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1005 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<8x64xf32>
    %1006 = "stablehlo.scatter"(%1005, %874, %1004) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg30: tensor<f32>, %arg31: tensor<f32>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<f32>
      stablehlo.return %2665 : tensor<f32>
    }) : (tensor<8x64xf32>, tensor<8x2xi32>, tensor<8xf32>) -> tensor<8x64xf32>
    %1007 = stablehlo.divide %1003, %856 : tensor<8xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1008 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1009 = stablehlo.select %858, %1008, %1007 : tensor<8xi1>, tensor<8xf32>
    %1010 = stablehlo.select %858, %1007, %1008 : tensor<8xi1>, tensor<8xf32>
    %1011 = stablehlo.negate %1009 : tensor<8xf32>
    %1012 = stablehlo.add %1010, %1011 : tensor<8xf32>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [0] : (tensor<8xf32>) -> tensor<8x64xf32>
    %1014 = stablehlo.multiply %1013, %854 : tensor<8x64xf32>
    %1015 = stablehlo.add %1006, %1014 : tensor<8x64xf32>
    %1016 = stablehlo.convert %1015 : (tensor<8x64xf32>) -> tensor<8x64xbf16>
    %1017 = stablehlo.dot_general %1016, %840, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xbf16>, tensor<8x32xbf16>) -> tensor<64x32xbf16>
    %1018 = stablehlo.transpose %1017, dims = [1, 0] : (tensor<64x32xbf16>) -> tensor<32x64xbf16>
    %1019 = stablehlo.dot_general %1016, %837, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x64xbf16>, tensor<32x64xbf16>) -> tensor<8x32xbf16>
    %1020 = stablehlo.reshape %1019 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %1021 = "stablehlo.all_reduce"(%1020) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<2x4x32xbf16>) -> tensor<2x4x32xbf16>
    %1022 = "stablehlo.all_reduce"(%1018) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %1023 = sdy.sharding_constraint %1022 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xbf16>
    %1024 = sdy.sharding_constraint %1021 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1025 = stablehlo.multiply %796, %1024 : tensor<2x4x32xbf16>
    %1026 = sdy.sharding_constraint %1025 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1027 = stablehlo.multiply %1024, %816 : tensor<2x4x32xbf16>
    %1028 = sdy.sharding_constraint %1027 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1029 = stablehlo.multiply %1026, %823 : tensor<2x4x32xbf16>
    %1030 = sdy.sharding_constraint %1029 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1031 = stablehlo.dot_general %1030, %800, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %1032 = sdy.sharding_constraint %1031 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1033 = stablehlo.transpose %1032, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %1034 = sdy.sharding_constraint %1033 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1035 = stablehlo.dot_general %1030, %66, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %1036 = sdy.sharding_constraint %1035 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1037 = call @silu_8(%801, %802, %798, %1036) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %1038 = sdy.sharding_constraint %1037 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1039 = stablehlo.dot_general %1038, %796, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %1040 = sdy.sharding_constraint %1039 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1041 = stablehlo.transpose %1040, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %1042 = sdy.sharding_constraint %1041 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1043 = stablehlo.dot_general %1038, %64, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %1044 = sdy.sharding_constraint %1043 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1045 = stablehlo.add %1028, %1044 : tensor<2x4x32xbf16>
    %1046 = sdy.sharding_constraint %1045 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1047 = stablehlo.convert %1046 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1048 = sdy.sharding_constraint %1047 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1049 = stablehlo.multiply %786, %1048 : tensor<2x4x32xf32>
    %1050 = sdy.sharding_constraint %1049 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1051 = stablehlo.reduce(%1050 init: %cst_143) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1052 = sdy.sharding_constraint %1051 <@mesh, [{}]> : tensor<32xf32>
    %1053 = stablehlo.reshape %1052 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1054 = sdy.sharding_constraint %1053 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1055 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1056 = sdy.sharding_constraint %1055 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1057 = stablehlo.multiply %1048, %1056 : tensor<2x4x32xf32>
    %1058 = sdy.sharding_constraint %1057 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1059 = stablehlo.reduce(%1054 init: %cst_144) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1060 = sdy.sharding_constraint %1059 <@mesh, [{}]> : tensor<32xf32>
    %1061 = stablehlo.convert %1060 : (tensor<32xf32>) -> tensor<32xbf16>
    %1062 = sdy.sharding_constraint %1061 <@mesh, [{}]> : tensor<32xbf16>
    %1063 = stablehlo.multiply %757, %1058 : tensor<2x4x32xf32>
    %1064 = sdy.sharding_constraint %1063 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1065 = stablehlo.reduce(%1064 init: %cst_145) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1066 = sdy.sharding_constraint %1065 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1067 = stablehlo.reshape %1066 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1068 = sdy.sharding_constraint %1067 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1069 = stablehlo.broadcast_in_dim %776, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1070 = sdy.sharding_constraint %1069 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1071 = stablehlo.multiply %1058, %1070 : tensor<2x4x32xf32>
    %1072 = sdy.sharding_constraint %1071 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1073 = stablehlo.multiply %1068, %782 : tensor<2x4x1xf32>
    %1074 = sdy.sharding_constraint %1073 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_146 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1075 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1076 = sdy.sharding_constraint %1075 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1077 = stablehlo.divide %1074, %1076 : tensor<2x4x1xf32>
    %1078 = sdy.sharding_constraint %1077 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1079 = stablehlo.reduce(%1078 init: %cst_147) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1080 = sdy.sharding_constraint %1079 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1081 = stablehlo.broadcast_in_dim %1080, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %1082 = sdy.sharding_constraint %1081 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1083 = stablehlo.multiply %1082, %762 : tensor<2x4x32xf32>
    %1084 = sdy.sharding_constraint %1083 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1085 = stablehlo.add %1072, %1084 : tensor<2x4x32xf32>
    %1086 = sdy.sharding_constraint %1085 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1087 = stablehlo.convert %1086 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %1088 = sdy.sharding_constraint %1087 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1089 = sdy.sharding_constraint %1062 <@mesh, [{}]> : tensor<32xbf16>
    %1090 = stablehlo.slice %996 [0:1] : (tensor<1xf32>) -> tensor<1xf32>
    %1091 = stablehlo.reshape %1090 : (tensor<1xf32>) -> tensor<f32>
    %1092 = sdy.sharding_constraint %1091 <@mesh, []> : tensor<f32>
    %1093:22 = stablehlo.optimization_barrier %24, %26, %28, %30, %32, %34, %38, %36, %40, %42, %44, %46, %48, %50, %52, %54, %145, %56, %58, %60, %1088, %1092 : tensor<32xbf16>, tensor<32x128xbf16>, tensor<128x32xbf16>, tensor<32x32xbf16>, tensor<32x16xbf16>, tensor<32x16xbf16>, tensor<32x2xbf16>, tensor<32x32xbf16>, tensor<32xbf16>, tensor<32x128xbf16>, tensor<128x32xbf16>, tensor<32x4xbf16>, tensor<4xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<2x4x32xbf16>, tensor<4x32x32xbf16>, tensor<4x32x32xbf16>, tensor<4x32x32xbf16>, tensor<2x4x32xbf16>, tensor<f32>
    %1094 = sdy.sharding_constraint %1093#0 <@mesh, [{}]> : tensor<32xbf16>
    %1095 = sdy.sharding_constraint %1093#1 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1096 = sdy.sharding_constraint %1093#2 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1097 = sdy.sharding_constraint %1093#3 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1098 = sdy.sharding_constraint %1093#4 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %1099 = sdy.sharding_constraint %1093#5 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %1100 = sdy.sharding_constraint %1093#6 <@mesh, [{}, {}]> : tensor<32x2xbf16>
    %1101 = sdy.sharding_constraint %1093#7 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1102 = sdy.sharding_constraint %1093#8 <@mesh, [{}]> : tensor<32xbf16>
    %1103 = sdy.sharding_constraint %1093#9 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1104 = sdy.sharding_constraint %1093#10 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1105 = sdy.sharding_constraint %1093#11 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %1106 = sdy.sharding_constraint %1093#12 <@mesh, [{}]> : tensor<4xbf16>
    %1107 = sdy.sharding_constraint %1093#13 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1108 = sdy.sharding_constraint %1093#14 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1109 = sdy.sharding_constraint %1093#15 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1110 = sdy.sharding_constraint %1093#16 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1111 = sdy.sharding_constraint %1093#17 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %1112 = sdy.sharding_constraint %1093#18 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %1113 = sdy.sharding_constraint %1093#19 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xbf16>
    %1114 = sdy.sharding_constraint %1093#20 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1115 = sdy.sharding_constraint %1093#21 <@mesh, []> : tensor<f32>
    %1116 = sdy.sharding_constraint %1094 <@mesh, [{}]> : tensor<32xbf16>
    %1117 = stablehlo.convert %1110 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1118 = sdy.sharding_constraint %1117 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1119 = chlo.square %1118 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_148 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1120 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<2x4x32xf32>
    %1121 = sdy.sharding_constraint %1120 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1122 = stablehlo.multiply %1121, %1118 : tensor<2x4x32xf32>
    %1123 = sdy.sharding_constraint %1122 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1124 = stablehlo.reduce(%1119 init: %cst_149) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1125 = sdy.sharding_constraint %1124 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1126 = stablehlo.broadcast_in_dim %1125, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1127 = sdy.sharding_constraint %1126 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_150 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1128 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1129 = sdy.sharding_constraint %1128 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1130 = stablehlo.divide %1127, %1129 : tensor<2x4x1xf32>
    %1131 = sdy.sharding_constraint %1130 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_151 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1132 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1133 = sdy.sharding_constraint %1132 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1134 = stablehlo.add %1131, %1133 : tensor<2x4x1xf32>
    %1135 = sdy.sharding_constraint %1134 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1136 = stablehlo.rsqrt %1135 : tensor<2x4x1xf32>
    %1137 = sdy.sharding_constraint %1136 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1138 = stablehlo.divide %1137, %1135 : tensor<2x4x1xf32>
    %1139 = sdy.sharding_constraint %1138 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_152 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1140 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1141 = sdy.sharding_constraint %1140 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1142 = stablehlo.multiply %1141, %1139 : tensor<2x4x1xf32>
    %1143 = sdy.sharding_constraint %1142 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1144 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1145 = sdy.sharding_constraint %1144 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1146 = stablehlo.multiply %1118, %1145 : tensor<2x4x32xf32>
    %1147 = sdy.sharding_constraint %1146 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1148 = stablehlo.convert %1116 : (tensor<32xbf16>) -> tensor<32xf32>
    %1149 = sdy.sharding_constraint %1148 <@mesh, [{}]> : tensor<32xf32>
    %1150 = stablehlo.broadcast_in_dim %1149, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1151 = sdy.sharding_constraint %1150 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1152 = stablehlo.broadcast_in_dim %1151, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1153 = sdy.sharding_constraint %1152 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1154 = stablehlo.multiply %1147, %1153 : tensor<2x4x32xf32>
    %1155 = sdy.sharding_constraint %1154 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1156 = stablehlo.convert %1155 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %1157 = sdy.sharding_constraint %1156 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1158 = stablehlo.dot_general %1157, %1095, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %1159 = sdy.sharding_constraint %1158 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1160:3 = call @silu_9(%1159) : (tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>)
    %1161 = sdy.sharding_constraint %1160#0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1162 = sdy.sharding_constraint %1160#1 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1163 = sdy.sharding_constraint %1160#2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1164 = stablehlo.dot_general %1161, %1096, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %1165 = sdy.sharding_constraint %1164 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1166 = stablehlo.negate %1165 : tensor<2x4x32xbf16>
    %1167 = sdy.sharding_constraint %1166 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1168 = stablehlo.exponential %1167 : tensor<2x4x32xbf16>
    %1169 = sdy.sharding_constraint %1168 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_153 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1170 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1171 = sdy.sharding_constraint %1170 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1172 = stablehlo.add %1171, %1169 : tensor<2x4x32xbf16>
    %1173 = sdy.sharding_constraint %1172 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_154 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1174 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1175 = sdy.sharding_constraint %1174 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1176 = stablehlo.divide %1175, %1173 : tensor<2x4x32xbf16>
    %1177 = sdy.sharding_constraint %1176 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_155 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1178 = sdy.sharding_constraint %cst_155 <@mesh, []> : tensor<bf16>
    %1179 = stablehlo.broadcast_in_dim %1178, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1180 = sdy.sharding_constraint %1179 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1181 = stablehlo.subtract %1180, %1177 : tensor<2x4x32xbf16>
    %1182 = sdy.sharding_constraint %1181 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1183 = stablehlo.multiply %1177, %1182 : tensor<2x4x32xbf16>
    %1184 = sdy.sharding_constraint %1183 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1185 = stablehlo.multiply %1157, %1177 : tensor<2x4x32xbf16>
    %1186 = sdy.sharding_constraint %1185 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1187 = stablehlo.dot_general %1186, %1097, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %1188 = sdy.sharding_constraint %1187 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %1189 = stablehlo.reshape %1188 : (tensor<2x4x32xbf16>) -> tensor<2x4x2x16xbf16>
    %1190 = sdy.sharding_constraint %1189 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1191 = stablehlo.dot_general %1186, %1098, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %1192 = sdy.sharding_constraint %1191 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %1193 = stablehlo.reshape %1192 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %1194 = sdy.sharding_constraint %1193 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %1195 = stablehlo.dot_general %1186, %1099, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %1196 = sdy.sharding_constraint %1195 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %1197 = stablehlo.reshape %1196 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %1198 = sdy.sharding_constraint %1197 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %1199 = stablehlo.convert %1190 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1200 = sdy.sharding_constraint %1199 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1201 = chlo.square %1200 : tensor<2x4x2x16xf32> -> tensor<2x4x2x16xf32>
    %cst_156 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1202 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f32>) -> tensor<2x4x2x16xf32>
    %1203 = sdy.sharding_constraint %1202 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1204 = stablehlo.multiply %1203, %1200 : tensor<2x4x2x16xf32>
    %1205 = sdy.sharding_constraint %1204 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1206 = stablehlo.reduce(%1201 init: %cst_157) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %1207 = sdy.sharding_constraint %1206 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %1208 = stablehlo.broadcast_in_dim %1207, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %1209 = sdy.sharding_constraint %1208 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_158 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %1210 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %1211 = sdy.sharding_constraint %1210 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1212 = stablehlo.divide %1209, %1211 : tensor<2x4x2x1xf32>
    %1213 = sdy.sharding_constraint %1212 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_159 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1214 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %1215 = sdy.sharding_constraint %1214 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1216 = stablehlo.add %1213, %1215 : tensor<2x4x2x1xf32>
    %1217 = sdy.sharding_constraint %1216 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1218 = stablehlo.rsqrt %1217 : tensor<2x4x2x1xf32>
    %1219 = sdy.sharding_constraint %1218 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1220 = stablehlo.divide %1219, %1217 : tensor<2x4x2x1xf32>
    %1221 = sdy.sharding_constraint %1220 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_160 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1222 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %1223 = sdy.sharding_constraint %1222 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1224 = stablehlo.multiply %1223, %1221 : tensor<2x4x2x1xf32>
    %1225 = sdy.sharding_constraint %1224 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1226 = stablehlo.convert %1190 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1227 = sdy.sharding_constraint %1226 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1228 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x16xf32>
    %1229 = sdy.sharding_constraint %1228 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1230 = stablehlo.multiply %1227, %1229 : tensor<2x4x2x16xf32>
    %1231 = sdy.sharding_constraint %1230 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1232 = stablehlo.convert %1231 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %1233 = sdy.sharding_constraint %1232 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1234 = stablehlo.convert %1194 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %1235 = sdy.sharding_constraint %1234 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1236 = chlo.square %1235 : tensor<2x4x1x16xf32> -> tensor<2x4x1x16xf32>
    %cst_161 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1237 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<2x4x1x16xf32>
    %1238 = sdy.sharding_constraint %1237 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1239 = stablehlo.multiply %1238, %1235 : tensor<2x4x1x16xf32>
    %1240 = sdy.sharding_constraint %1239 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1241 = stablehlo.reduce(%1236 init: %cst_162) applies stablehlo.add across dimensions = [3] : (tensor<2x4x1x16xf32>, tensor<f32>) -> tensor<2x4x1xf32>
    %1242 = sdy.sharding_constraint %1241 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1243 = stablehlo.broadcast_in_dim %1242, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x1x1xf32>
    %1244 = sdy.sharding_constraint %1243 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_163 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %1245 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %1246 = sdy.sharding_constraint %1245 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1247 = stablehlo.divide %1244, %1246 : tensor<2x4x1x1xf32>
    %1248 = sdy.sharding_constraint %1247 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_164 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1249 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %1250 = sdy.sharding_constraint %1249 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1251 = stablehlo.add %1248, %1250 : tensor<2x4x1x1xf32>
    %1252 = sdy.sharding_constraint %1251 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1253 = stablehlo.rsqrt %1252 : tensor<2x4x1x1xf32>
    %1254 = sdy.sharding_constraint %1253 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1255 = stablehlo.divide %1254, %1252 : tensor<2x4x1x1xf32>
    %1256 = sdy.sharding_constraint %1255 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_165 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1257 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %1258 = sdy.sharding_constraint %1257 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1259 = stablehlo.multiply %1258, %1256 : tensor<2x4x1x1xf32>
    %1260 = sdy.sharding_constraint %1259 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1261 = stablehlo.convert %1194 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %1262 = sdy.sharding_constraint %1261 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1263 = stablehlo.broadcast_in_dim %1254, dims = [0, 1, 2, 3] : (tensor<2x4x1x1xf32>) -> tensor<2x4x1x16xf32>
    %1264 = sdy.sharding_constraint %1263 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1265 = stablehlo.multiply %1262, %1264 : tensor<2x4x1x16xf32>
    %1266 = sdy.sharding_constraint %1265 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1267 = stablehlo.convert %1266 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %1268 = sdy.sharding_constraint %1267 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %cst_166 = stablehlo.constant dense<1.296880e+00> : tensor<bf16>
    %1269 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %1270 = sdy.sharding_constraint %1269 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1271 = stablehlo.multiply %1233, %1270 : tensor<2x4x2x16xbf16>
    %1272 = sdy.sharding_constraint %1271 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1273 = stablehlo.broadcast_in_dim %1268, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %1274 = sdy.sharding_constraint %1273 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %1275 = stablehlo.broadcast_in_dim %1274, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %1276 = sdy.sharding_constraint %1275 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %1277 = stablehlo.reshape %1276 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %1278 = sdy.sharding_constraint %1277 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1279 = stablehlo.broadcast_in_dim %1198, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %1280 = sdy.sharding_constraint %1279 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %1281 = stablehlo.broadcast_in_dim %1280, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %1282 = sdy.sharding_constraint %1281 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %1283 = stablehlo.reshape %1282 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %1284 = sdy.sharding_constraint %1283 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_167 = stablehlo.constant dense<2.500000e-01> : tensor<bf16>
    %1285 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %1286 = sdy.sharding_constraint %1285 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1287 = stablehlo.multiply %1272, %1286 : tensor<2x4x2x16xbf16>
    %1288 = sdy.sharding_constraint %1287 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1289 = stablehlo.dot_general %1288, %1278, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x4xbf16>
    %1290 = sdy.sharding_constraint %1289 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %1291 = stablehlo.iota dim = 0 : tensor<4xi32>
    %1292 = sdy.sharding_constraint %1291 <@mesh, [{}]> : tensor<4xi32>
    %1293 = stablehlo.broadcast_in_dim %1292, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %1294 = sdy.sharding_constraint %1293 <@mesh, [{}, {}]> : tensor<4x1xi32>
    %1295 = stablehlo.iota dim = 0 : tensor<4xi32>
    %1296 = sdy.sharding_constraint %1295 <@mesh, [{}]> : tensor<4xi32>
    %1297 = stablehlo.broadcast_in_dim %1296, dims = [1] : (tensor<4xi32>) -> tensor<1x4xi32>
    %1298 = sdy.sharding_constraint %1297 <@mesh, [{}, {}]> : tensor<1x4xi32>
    %1299 = stablehlo.broadcast_in_dim %1298, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %1300 = sdy.sharding_constraint %1299 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %1301 = stablehlo.broadcast_in_dim %1294, dims = [0, 1] : (tensor<4x1xi32>) -> tensor<4x4xi32>
    %1302 = sdy.sharding_constraint %1301 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %1303 = stablehlo.compare LE, %1300, %1302, SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    %1304 = stablehlo.broadcast_in_dim %1303, dims = [1, 2] : (tensor<4x4xi1>) -> tensor<1x4x4xi1>
    %1305 = sdy.sharding_constraint %1304 <@mesh, [{}, {}, {}]> : tensor<1x4x4xi1>
    %1306 = stablehlo.broadcast_in_dim %1305, dims = [0, 2, 3] : (tensor<1x4x4xi1>) -> tensor<1x1x4x4xi1>
    %1307 = sdy.sharding_constraint %1306 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x4x4xi1>
    %cst_168 = stablehlo.constant dense<-9.982440e+08> : tensor<bf16>
    %1308:2 = call @_where_10(%1307, %1290, %cst_168) : (tensor<1x1x4x4xi1>, tensor<2x2x4x4xbf16>, tensor<bf16>) -> (tensor<2x2x4x4xbf16>, tensor<2x2x4x4xi1>)
    %1309 = sdy.sharding_constraint %1308#0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %1310 = sdy.sharding_constraint %1308#1 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xi1>
    %1311 = stablehlo.convert %1309 : (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %1312 = sdy.sharding_constraint %1311 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_169 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1313 = stablehlo.reduce(%1312 init: %cst_169) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %1314 = sdy.sharding_constraint %1313 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %cst_170 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1315 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<2x2x4xf32>
    %1316 = sdy.sharding_constraint %1315 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %1317 = stablehlo.maximum %1316, %1314 : tensor<2x2x4xf32>
    %1318 = sdy.sharding_constraint %1317 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %1319 = stablehlo.broadcast_in_dim %1318, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %1320 = sdy.sharding_constraint %1319 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1321 = stablehlo.broadcast_in_dim %1320, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %1322 = sdy.sharding_constraint %1321 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1323 = stablehlo.subtract %1312, %1322 : tensor<2x2x4x4xf32>
    %1324 = sdy.sharding_constraint %1323 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1325 = stablehlo.exponential %1324 : tensor<2x2x4x4xf32>
    %1326 = sdy.sharding_constraint %1325 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1327 = stablehlo.reduce(%1326 init: %cst_171) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %1328 = sdy.sharding_constraint %1327 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %1329 = stablehlo.broadcast_in_dim %1328, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %1330 = sdy.sharding_constraint %1329 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1331 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %1332 = sdy.sharding_constraint %1331 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1333 = stablehlo.divide %1326, %1332 : tensor<2x2x4x4xf32>
    %1334 = sdy.sharding_constraint %1333 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1335 = stablehlo.multiply %1330, %1330 : tensor<2x2x4x1xf32>
    %1336 = sdy.sharding_constraint %1335 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %cst_172 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1337 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<2x2x4x1xf32>
    %1338 = sdy.sharding_constraint %1337 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1339 = stablehlo.divide %1338, %1336 : tensor<2x2x4x1xf32>
    %1340 = sdy.sharding_constraint %1339 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1341 = sdy.sharding_constraint %1340 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1342 = stablehlo.convert %1334 : (tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xbf16>
    %1343 = sdy.sharding_constraint %1342 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %1344 = stablehlo.dot_general %1284, %1343, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x2x4x4xbf16>) -> tensor<2x2x16x4xbf16>
    %1345 = sdy.sharding_constraint %1344 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %1346 = stablehlo.transpose %1345, dims = [0, 3, 1, 2] : (tensor<2x2x16x4xbf16>) -> tensor<2x4x2x16xbf16>
    %1347 = sdy.sharding_constraint %1346 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1348 = stablehlo.broadcast_in_dim %1198, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %1349 = sdy.sharding_constraint %1348 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %1350 = stablehlo.broadcast_in_dim %1349, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %1351 = sdy.sharding_constraint %1350 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %1352 = stablehlo.reshape %1351 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %1353 = sdy.sharding_constraint %1352 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1354 = sdy.sharding_constraint %1353 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1355 = stablehlo.multiply %1347, %1354 : tensor<2x4x2x16xbf16>
    %1356 = sdy.sharding_constraint %1355 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1357 = stablehlo.convert %1356 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1358 = sdy.sharding_constraint %1357 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1359 = stablehlo.reduce(%1358 init: %cst_173) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %1360 = sdy.sharding_constraint %1359 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %1362 = sdy.sharding_constraint %1361 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %1363 = stablehlo.convert %1362 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %1364 = sdy.sharding_constraint %1363 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1365 = stablehlo.multiply %1354, %1354 : tensor<2x4x2x16xbf16>
    %1366 = sdy.sharding_constraint %1365 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1367 = stablehlo.convert %1366 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1368 = sdy.sharding_constraint %1367 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1369 = stablehlo.reduce(%1368 init: %cst_174) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %1370 = sdy.sharding_constraint %1369 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %1371 = stablehlo.broadcast_in_dim %1370, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %1372 = sdy.sharding_constraint %1371 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %1373 = stablehlo.convert %1372 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %1374 = sdy.sharding_constraint %1373 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_175 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1375 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %1376 = sdy.sharding_constraint %1375 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1377 = stablehlo.add %1374, %1376 : tensor<2x4x2x1xbf16>
    %1378 = sdy.sharding_constraint %1377 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1379 = stablehlo.divide %1364, %1378 : tensor<2x4x2x1xbf16>
    %1380 = sdy.sharding_constraint %1379 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1381 = stablehlo.multiply %1378, %1378 : tensor<2x4x2x1xbf16>
    %1382 = sdy.sharding_constraint %1381 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_176 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1383 = stablehlo.broadcast_in_dim %cst_176, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %1384 = sdy.sharding_constraint %1383 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1385 = stablehlo.divide %1384, %1382 : tensor<2x4x2x1xbf16>
    %1386 = sdy.sharding_constraint %1385 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1387 = sdy.sharding_constraint %1386 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1388 = stablehlo.broadcast_in_dim %1380, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %1389 = sdy.sharding_constraint %1388 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1390 = stablehlo.multiply %1389, %1354 : tensor<2x4x2x16xbf16>
    %1391 = sdy.sharding_constraint %1390 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1392 = stablehlo.subtract %1347, %1391 : tensor<2x4x2x16xbf16>
    %1393 = sdy.sharding_constraint %1392 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1394 = stablehlo.dot_general %1186, %1100, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x2xbf16>) -> tensor<2x4x2xbf16>
    %1395 = sdy.sharding_constraint %1394 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1396 = stablehlo.negate %1395 : tensor<2x4x2xbf16>
    %1397 = sdy.sharding_constraint %1396 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1398 = stablehlo.exponential %1397 : tensor<2x4x2xbf16>
    %1399 = sdy.sharding_constraint %1398 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_177 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1400 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %1401 = sdy.sharding_constraint %1400 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1402 = stablehlo.add %1401, %1399 : tensor<2x4x2xbf16>
    %1403 = sdy.sharding_constraint %1402 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_178 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1404 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %1405 = sdy.sharding_constraint %1404 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1406 = stablehlo.divide %1405, %1403 : tensor<2x4x2xbf16>
    %1407 = sdy.sharding_constraint %1406 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_179 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1408 = sdy.sharding_constraint %cst_179 <@mesh, []> : tensor<bf16>
    %1409 = stablehlo.broadcast_in_dim %1408, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %1410 = sdy.sharding_constraint %1409 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1411 = stablehlo.subtract %1410, %1407 : tensor<2x4x2xbf16>
    %1412 = sdy.sharding_constraint %1411 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1413 = stablehlo.multiply %1407, %1412 : tensor<2x4x2xbf16>
    %1414 = sdy.sharding_constraint %1413 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1415 = stablehlo.broadcast_in_dim %1407, dims = [0, 1, 2] : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %1416 = sdy.sharding_constraint %1415 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_180 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %1417 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %1418 = sdy.sharding_constraint %1417 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1419 = stablehlo.multiply %1418, %1416 : tensor<2x4x2x1xbf16>
    %1420 = sdy.sharding_constraint %1419 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1421 = stablehlo.broadcast_in_dim %1420, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %1422 = sdy.sharding_constraint %1421 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1423 = stablehlo.multiply %1422, %1393 : tensor<2x4x2x16xbf16>
    %1424 = sdy.sharding_constraint %1423 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1425 = stablehlo.reshape %1424 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x32xbf16>
    %1426 = sdy.sharding_constraint %1425 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %1427 = stablehlo.dot_general %1426, %1101, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %1428 = sdy.sharding_constraint %1427 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1429 = stablehlo.add %1110, %1428 : tensor<2x4x32xbf16>
    %1430 = sdy.sharding_constraint %1429 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1431 = sdy.sharding_constraint %1102 <@mesh, [{}]> : tensor<32xbf16>
    %1432 = stablehlo.convert %1430 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1433 = sdy.sharding_constraint %1432 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1434 = chlo.square %1433 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_181 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1435 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<2x4x32xf32>
    %1436 = sdy.sharding_constraint %1435 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1437 = stablehlo.multiply %1436, %1433 : tensor<2x4x32xf32>
    %1438 = sdy.sharding_constraint %1437 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1439 = stablehlo.reduce(%1434 init: %cst_182) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1440 = sdy.sharding_constraint %1439 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1441 = stablehlo.broadcast_in_dim %1440, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1442 = sdy.sharding_constraint %1441 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_183 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1443 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1444 = sdy.sharding_constraint %1443 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1445 = stablehlo.divide %1442, %1444 : tensor<2x4x1xf32>
    %1446 = sdy.sharding_constraint %1445 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_184 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1447 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1448 = sdy.sharding_constraint %1447 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1449 = stablehlo.add %1446, %1448 : tensor<2x4x1xf32>
    %1450 = sdy.sharding_constraint %1449 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1451 = stablehlo.rsqrt %1450 : tensor<2x4x1xf32>
    %1452 = sdy.sharding_constraint %1451 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1453 = stablehlo.divide %1452, %1450 : tensor<2x4x1xf32>
    %1454 = sdy.sharding_constraint %1453 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_185 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1455 = stablehlo.broadcast_in_dim %cst_185, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1456 = sdy.sharding_constraint %1455 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1457 = stablehlo.multiply %1456, %1454 : tensor<2x4x1xf32>
    %1458 = sdy.sharding_constraint %1457 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1459 = stablehlo.broadcast_in_dim %1452, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1460 = sdy.sharding_constraint %1459 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1461 = stablehlo.multiply %1433, %1460 : tensor<2x4x32xf32>
    %1462 = sdy.sharding_constraint %1461 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1463 = stablehlo.convert %1431 : (tensor<32xbf16>) -> tensor<32xf32>
    %1464 = sdy.sharding_constraint %1463 <@mesh, [{}]> : tensor<32xf32>
    %1465 = stablehlo.broadcast_in_dim %1464, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1466 = sdy.sharding_constraint %1465 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1467 = stablehlo.broadcast_in_dim %1466, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1468 = sdy.sharding_constraint %1467 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1469 = stablehlo.multiply %1462, %1468 : tensor<2x4x32xf32>
    %1470 = sdy.sharding_constraint %1469 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1471 = stablehlo.convert %1470 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %1472 = sdy.sharding_constraint %1471 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1473 = stablehlo.dot_general %1472, %1103, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %1474 = sdy.sharding_constraint %1473 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1475:3 = call @silu_9(%1474) : (tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>)
    %1476 = sdy.sharding_constraint %1475#0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1477 = sdy.sharding_constraint %1475#1 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1478 = sdy.sharding_constraint %1475#2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1479 = stablehlo.dot_general %1476, %1104, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %1480 = sdy.sharding_constraint %1479 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1481 = stablehlo.negate %1480 : tensor<2x4x32xbf16>
    %1482 = sdy.sharding_constraint %1481 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1483 = stablehlo.exponential %1482 : tensor<2x4x32xbf16>
    %1484 = sdy.sharding_constraint %1483 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_186 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1485 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1486 = sdy.sharding_constraint %1485 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1487 = stablehlo.add %1486, %1484 : tensor<2x4x32xbf16>
    %1488 = sdy.sharding_constraint %1487 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_187 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1489 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1490 = sdy.sharding_constraint %1489 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1491 = stablehlo.divide %1490, %1488 : tensor<2x4x32xbf16>
    %1492 = sdy.sharding_constraint %1491 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_188 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1493 = sdy.sharding_constraint %cst_188 <@mesh, []> : tensor<bf16>
    %1494 = stablehlo.broadcast_in_dim %1493, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1495 = sdy.sharding_constraint %1494 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1496 = stablehlo.subtract %1495, %1492 : tensor<2x4x32xbf16>
    %1497 = sdy.sharding_constraint %1496 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1498 = stablehlo.multiply %1492, %1497 : tensor<2x4x32xbf16>
    %1499 = sdy.sharding_constraint %1498 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1500 = stablehlo.multiply %1472, %1492 : tensor<2x4x32xbf16>
    %1501 = sdy.sharding_constraint %1500 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1502 = stablehlo.reshape %1501 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1503 = sdy.sharding_constraint %1502 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1504 = sdy.sharding_constraint %1105 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %1505 = stablehlo.dot_general %1503, %1504, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x4xbf16>) -> tensor<8x4xbf16>
    %1506 = sdy.sharding_constraint %1505 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xbf16>
    %1507 = stablehlo.convert %1506 : (tensor<8x4xbf16>) -> tensor<8x4xf32>
    %1508 = sdy.sharding_constraint %1507 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1509 = stablehlo.convert %1106 : (tensor<4xbf16>) -> tensor<4xf32>
    %1510 = sdy.sharding_constraint %1509 <@mesh, [{}]> : tensor<4xf32>
    %1511 = stablehlo.broadcast_in_dim %1510, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %1512 = sdy.sharding_constraint %1511 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %1513 = stablehlo.broadcast_in_dim %1512, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<8x4xf32>
    %1514 = sdy.sharding_constraint %1513 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1515 = stablehlo.add %1508, %1514 : tensor<8x4xf32>
    %1516 = sdy.sharding_constraint %1515 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %values_189, %indices_190 = chlo.top_k(%1516, k = 3) : tensor<8x4xf32> -> (tensor<8x3xf32>, tensor<8x3xi32>)
    %1517 = sdy.sharding_constraint %values_189 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xf32>
    %1518 = sdy.sharding_constraint %indices_190 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xi32>
    %1519 = stablehlo.slice %1518 [0:8, 0:2] : (tensor<8x3xi32>) -> tensor<8x2xi32>
    %1520 = sdy.sharding_constraint %1519 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %1521:2 = call @take_along_axis_11(%1508, %1520) : (tensor<8x4xf32>, tensor<8x2xi32>) -> (tensor<8x2xf32>, tensor<8x2x1xi32>)
    %1522 = sdy.sharding_constraint %1521#0 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1523 = sdy.sharding_constraint %1521#1 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %1524 = stablehlo.negate %1522 : tensor<8x2xf32>
    %1525 = sdy.sharding_constraint %1524 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1526 = stablehlo.exponential %1525 : tensor<8x2xf32>
    %1527 = sdy.sharding_constraint %1526 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_191 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1528 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %1529 = sdy.sharding_constraint %1528 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1530 = stablehlo.add %1529, %1527 : tensor<8x2xf32>
    %1531 = sdy.sharding_constraint %1530 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_192 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1532 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %1533 = sdy.sharding_constraint %1532 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1534 = stablehlo.divide %1533, %1531 : tensor<8x2xf32>
    %1535 = sdy.sharding_constraint %1534 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_193 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1536 = sdy.sharding_constraint %cst_193 <@mesh, []> : tensor<f32>
    %1537 = stablehlo.broadcast_in_dim %1536, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %1538 = sdy.sharding_constraint %1537 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1539 = stablehlo.subtract %1538, %1535 : tensor<8x2xf32>
    %1540 = sdy.sharding_constraint %1539 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1541 = stablehlo.multiply %1535, %1540 : tensor<8x2xf32>
    %1542 = sdy.sharding_constraint %1541 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1543 = stablehlo.reduce(%1535 init: %cst_194) applies stablehlo.add across dimensions = [1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8xf32>
    %1544 = sdy.sharding_constraint %1543 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1545 = stablehlo.broadcast_in_dim %1544, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %1546 = sdy.sharding_constraint %1545 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_195 = stablehlo.constant dense<9.99999971E-10> : tensor<f32>
    %1547 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1548 = sdy.sharding_constraint %1547 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1549 = stablehlo.add %1546, %1548 : tensor<8x1xf32>
    %1550 = sdy.sharding_constraint %1549 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_196 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %1551 = stablehlo.broadcast_in_dim %cst_196, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1552 = sdy.sharding_constraint %1551 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1553 = stablehlo.divide %1552, %1550 : tensor<8x1xf32>
    %1554 = sdy.sharding_constraint %1553 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1555 = stablehlo.multiply %1550, %1550 : tensor<8x1xf32>
    %1556 = sdy.sharding_constraint %1555 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_197 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1557 = stablehlo.broadcast_in_dim %cst_197, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1558 = sdy.sharding_constraint %1557 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1559 = stablehlo.divide %1558, %1556 : tensor<8x1xf32>
    %1560 = sdy.sharding_constraint %1559 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1561 = sdy.sharding_constraint %1560 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1562 = stablehlo.broadcast_in_dim %1554, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x2xf32>
    %1563 = sdy.sharding_constraint %1562 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1564 = stablehlo.multiply %1535, %1563 : tensor<8x2xf32>
    %1565 = sdy.sharding_constraint %1564 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1566 = stablehlo.convert %1565 : (tensor<8x2xf32>) -> tensor<8x2xbf16>
    %1567 = sdy.sharding_constraint %1566 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %cst_198 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1568 = stablehlo.reduce(%1508 init: %cst_198) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %1569 = sdy.sharding_constraint %1568 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_199 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1570 = stablehlo.broadcast_in_dim %cst_199, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1571 = sdy.sharding_constraint %1570 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1572 = stablehlo.maximum %1571, %1569 : tensor<8xf32>
    %1573 = sdy.sharding_constraint %1572 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1574 = stablehlo.is_finite %1573 : (tensor<8xf32>) -> tensor<8xi1>
    %1575 = sdy.sharding_constraint %1574 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xi1>
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1576 = stablehlo.broadcast_in_dim %cst_200, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1577 = sdy.sharding_constraint %1576 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1578 = stablehlo.select %1575, %1573, %1577 : tensor<8xi1>, tensor<8xf32>
    %1579 = sdy.sharding_constraint %1578 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1580 = stablehlo.broadcast_in_dim %1579, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %1581 = sdy.sharding_constraint %1580 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1582 = stablehlo.broadcast_in_dim %1581, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %1583 = sdy.sharding_constraint %1582 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1584 = stablehlo.subtract %1508, %1583 : tensor<8x4xf32>
    %1585 = sdy.sharding_constraint %1584 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1586 = stablehlo.exponential %1585 : tensor<8x4xf32>
    %1587 = sdy.sharding_constraint %1586 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1588 = stablehlo.reduce(%1587 init: %cst_201) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %1589 = sdy.sharding_constraint %1588 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1590 = stablehlo.abs %1589 : tensor<8xf32>
    %1591 = sdy.sharding_constraint %1590 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1592 = sdy.sharding_constraint %cst_202 <@mesh, []> : tensor<f32>
    %1593 = stablehlo.broadcast_in_dim %1592, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1594 = sdy.sharding_constraint %1593 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1595 = stablehlo.compare GE, %1589, %1594, FLOAT : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xi1>
    %1596 = stablehlo.log %1591 : tensor<8xf32>
    %1597 = sdy.sharding_constraint %1596 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1598 = stablehlo.add %1597, %1579 : tensor<8xf32>
    %1599 = sdy.sharding_constraint %1598 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_203 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1600 = stablehlo.broadcast_in_dim %cst_203, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1601 = sdy.sharding_constraint %1600 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1602 = stablehlo.multiply %1601, %1599 : tensor<8xf32>
    %1603 = sdy.sharding_constraint %1602 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1604 = stablehlo.concatenate %1111, %1112, dim = 2 : (tensor<4x32x32xbf16>, tensor<4x32x32xbf16>) -> tensor<4x32x64xbf16>
    %1605 = sdy.sharding_constraint %1604 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x64xbf16>
    %1606 = sdy.sharding_constraint %1503 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1607 = sdy.sharding_constraint %1520 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %1608 = sdy.sharding_constraint %1567 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %1609 = sdy.sharding_constraint %1605 <@mesh, [{}, {}, {}]> : tensor<4x32x64xbf16>
    %1610 = sdy.sharding_constraint %1113 <@mesh, [{}, {}, {}]> : tensor<4x32x32xbf16>
    %1611 = stablehlo.reshape %1607 : (tensor<8x2xi32>) -> tensor<16xi32>
    %1612 = stablehlo.reshape %1608 : (tensor<8x2xbf16>) -> tensor<16xbf16>
    %1613 = call @argsort_12(%1611) : (tensor<16xi32>) -> tensor<16xi32>
    %1614 = stablehlo.iota dim = 0 : tensor<16xi32>
    %c_204 = stablehlo.constant dense<2> : tensor<i32>
    %1615 = call @floor_divide_13(%1614, %c_204) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_205 = stablehlo.constant dense<0> : tensor<i32>
    %1616 = stablehlo.broadcast_in_dim %c_205, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1617 = stablehlo.compare LT, %1613, %1616, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_206 = stablehlo.constant dense<16> : tensor<i32>
    %1618 = stablehlo.broadcast_in_dim %c_206, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1619 = stablehlo.add %1613, %1618 : tensor<16xi32>
    %1620 = stablehlo.select %1617, %1619, %1613 : tensor<16xi1>, tensor<16xi32>
    %1621 = stablehlo.broadcast_in_dim %1620, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1622 = "stablehlo.gather"(%1615, %1621) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xi32>, tensor<16x1xi32>) -> tensor<16xi32>
    %c_207 = stablehlo.constant dense<0> : tensor<i32>
    %1623 = stablehlo.broadcast_in_dim %c_207, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1624 = stablehlo.compare LT, %1622, %1623, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_208 = stablehlo.constant dense<8> : tensor<i32>
    %1625 = stablehlo.broadcast_in_dim %c_208, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1626 = stablehlo.add %1622, %1625 : tensor<16xi32>
    %1627 = stablehlo.select %1624, %1626, %1622 : tensor<16xi1>, tensor<16xi32>
    %1628 = stablehlo.broadcast_in_dim %1627, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1629 = "stablehlo.gather"(%1606, %1628) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<8x32xbf16>, tensor<16x1xi32>) -> tensor<16x32xbf16>
    %c_209 = stablehlo.constant dense<0> : tensor<i32>
    %1630 = stablehlo.broadcast_in_dim %c_209, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1631 = stablehlo.compare LT, %1613, %1630, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_210 = stablehlo.constant dense<16> : tensor<i32>
    %1632 = stablehlo.broadcast_in_dim %c_210, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1633 = stablehlo.add %1613, %1632 : tensor<16xi32>
    %1634 = stablehlo.select %1631, %1633, %1613 : tensor<16xi1>, tensor<16xi32>
    %1635 = stablehlo.broadcast_in_dim %1634, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1636 = "stablehlo.gather"(%1612, %1635) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xbf16>, tensor<16x1xi32>) -> tensor<16xbf16>
    %c_211 = stablehlo.constant dense<0> : tensor<i32>
    %1637 = stablehlo.broadcast_in_dim %c_211, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %c_212 = stablehlo.constant dense<0> : tensor<i32>
    %1638 = call @clip_15(%1611, %c_212) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_213 = stablehlo.constant dense<0> : tensor<i32>
    %1639 = stablehlo.broadcast_in_dim %c_213, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1640 = stablehlo.compare LT, %1638, %1639, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_214 = stablehlo.constant dense<4> : tensor<i32>
    %1641 = stablehlo.broadcast_in_dim %c_214, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1642 = stablehlo.add %1638, %1641 : tensor<16xi32>
    %1643 = stablehlo.select %1640, %1642, %1638 : tensor<16xi1>, tensor<16xi32>
    %1644 = stablehlo.broadcast_in_dim %1643, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %c_215 = stablehlo.constant dense<1> : tensor<i32>
    %1645 = stablehlo.broadcast_in_dim %c_215, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1646 = "stablehlo.scatter"(%1637, %1644, %1645) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg30: tensor<i32>, %arg31: tensor<i32>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<i32>
      stablehlo.return %2665 : tensor<i32>
    }) : (tensor<4xi32>, tensor<16x1xi32>, tensor<16xi32>) -> tensor<4xi32>
    %cst_216 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1647 = stablehlo.pad %1629, %cst_216, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %1648 = stablehlo.broadcast_in_dim %1647, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1649 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1650 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_217 = stablehlo.constant dense<0> : tensor<i32>
    %1651 = stablehlo.broadcast_in_dim %c_217, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1652 = stablehlo.slice %1651 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1653 = stablehlo.slice %1650 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1654 = stablehlo.concatenate %1652, %1653, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1655 = stablehlo.broadcast_in_dim %1650, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1656 = stablehlo.broadcast_in_dim %1654, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1657 = stablehlo.compare LE, %1656, %1649, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1658 = stablehlo.compare LT, %1649, %1655, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1659 = stablehlo.and %1657, %1658 : tensor<4x512x32xi1>
    %cst_218 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1660 = stablehlo.broadcast_in_dim %cst_218, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1661 = stablehlo.select %1659, %1648, %1660 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1662 = stablehlo.dot_general %1661, %1609, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x64xbf16>) -> tensor<512x64xbf16>
    %1663 = stablehlo.slice %1662 [0:16, 0:64] : (tensor<512x64xbf16>) -> tensor<16x64xbf16>
    %1664 = stablehlo.slice %1663 [0:16, 0:32] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %1665 = stablehlo.slice %1663 [0:16, 32:64] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %1666:3 = call @silu_16(%1664) : (tensor<16x32xbf16>) -> (tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>)
    %1667 = stablehlo.multiply %1666#0, %1665 : tensor<16x32xbf16>
    %cst_219 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1668 = stablehlo.pad %1667, %cst_219, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %1669 = stablehlo.broadcast_in_dim %1668, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1670 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1671 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_220 = stablehlo.constant dense<0> : tensor<i32>
    %1672 = stablehlo.broadcast_in_dim %c_220, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1673 = stablehlo.slice %1672 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1674 = stablehlo.slice %1671 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1675 = stablehlo.concatenate %1673, %1674, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1676 = stablehlo.broadcast_in_dim %1671, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1677 = stablehlo.broadcast_in_dim %1675, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1678 = stablehlo.compare LE, %1677, %1670, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1679 = stablehlo.compare LT, %1670, %1676, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1680 = stablehlo.and %1678, %1679 : tensor<4x512x32xi1>
    %cst_221 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1681 = stablehlo.broadcast_in_dim %cst_221, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1682 = stablehlo.select %1680, %1669, %1681 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1683 = stablehlo.dot_general %1682, %1610, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x32xbf16>) -> tensor<512x32xbf16>
    %1684 = stablehlo.slice %1683 [0:16, 0:32] : (tensor<512x32xbf16>) -> tensor<16x32xbf16>
    %1685 = stablehlo.broadcast_in_dim %1636, dims = [0] : (tensor<16xbf16>) -> tensor<16x1xbf16>
    %c_222 = stablehlo.constant dense<0> : tensor<i32>
    %1686 = stablehlo.broadcast_in_dim %c_222, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1687 = stablehlo.compare LT, %1622, %1686, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_223 = stablehlo.constant dense<8> : tensor<i32>
    %1688 = stablehlo.broadcast_in_dim %c_223, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1689 = stablehlo.add %1622, %1688 : tensor<16xi32>
    %1690 = stablehlo.select %1687, %1689, %1622 : tensor<16xi1>, tensor<16xi32>
    %1691 = stablehlo.broadcast_in_dim %1690, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1692 = stablehlo.reshape %1501 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1693 = sdy.sharding_constraint %1692 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1694 = stablehlo.dot_general %1693, %1107, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1695 = sdy.sharding_constraint %1694 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1696 = stablehlo.dot_general %1693, %1108, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1697 = sdy.sharding_constraint %1696 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1698:3 = call @silu_17(%1695) : (tensor<8x32xbf16>) -> (tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>)
    %1699 = sdy.sharding_constraint %1698#0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1700 = sdy.sharding_constraint %1698#1 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1701 = sdy.sharding_constraint %1698#2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1702 = stablehlo.multiply %1699, %1697 : tensor<8x32xbf16>
    %1703 = sdy.sharding_constraint %1702 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1704 = sdy.sharding_constraint %1114 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1705 = stablehlo.reshape %1704 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1706 = sdy.sharding_constraint %1705 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1707 = stablehlo.dot_general %1706, %1703, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<32x32xbf16>
    %1708 = sdy.sharding_constraint %1707 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1709 = stablehlo.transpose %1708, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1710 = sdy.sharding_constraint %1709 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1711 = stablehlo.dot_general %1706, %1109, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1712 = sdy.sharding_constraint %1711 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1713 = stablehlo.multiply %1699, %1712 : tensor<8x32xbf16>
    %1714 = sdy.sharding_constraint %1713 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1715 = stablehlo.multiply %1712, %1697 : tensor<8x32xbf16>
    %1716 = sdy.sharding_constraint %1715 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1717 = call @silu_18(%1700, %1695, %1701, %1716) : (tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<8x32xbf16>
    %1718 = sdy.sharding_constraint %1717 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1719 = stablehlo.dot_general %1714, %1693, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<32x32xbf16>
    %1720 = sdy.sharding_constraint %1719 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1721 = stablehlo.transpose %1720, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1722 = sdy.sharding_constraint %1721 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1723 = stablehlo.dot_general %1714, %1108, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1724 = sdy.sharding_constraint %1723 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1725 = stablehlo.dot_general %1718, %1693, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<32x32xbf16>
    %1726 = sdy.sharding_constraint %1725 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1727 = stablehlo.transpose %1726, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1728 = sdy.sharding_constraint %1727 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1729 = stablehlo.dot_general %1718, %1107, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1730 = sdy.sharding_constraint %1729 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1731 = stablehlo.add %1724, %1730 : tensor<8x32xbf16>
    %1732 = sdy.sharding_constraint %1731 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1733 = stablehlo.reshape %1732 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %1734 = sdy.sharding_constraint %1733 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1735 = sdy.sharding_constraint %1114 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1736 = stablehlo.reshape %1735 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1737 = sdy.sharding_constraint %1736 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %c_224 = stablehlo.constant dense<7> : tensor<1xi32>
    %c_225 = stablehlo.constant dense<0> : tensor<i32>
    %1738 = stablehlo.broadcast_in_dim %c_225, dims = [] : (tensor<i32>) -> tensor<16x1xi32>
    %1739 = stablehlo.compare GE, %1691, %1738, SIGNED : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x1xi1>
    %1740 = stablehlo.broadcast_in_dim %c_224, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1741 = stablehlo.broadcast_in_dim %1740, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<16x1xi32>
    %1742 = stablehlo.compare LE, %1691, %1741, SIGNED : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x1xi1>
    %1743 = stablehlo.and %1739, %1742 : tensor<16x1xi1>
    %c_226 = stablehlo.constant dense<true> : tensor<i1>
    %1744 = stablehlo.reduce(%1743 init: %c_226) applies stablehlo.and across dimensions = [1] : (tensor<16x1xi1>, tensor<i1>) -> tensor<16xi1>
    %1745 = "stablehlo.gather"(%1737, %1691) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<8x32xbf16>, tensor<16x1xi32>) -> tensor<16x32xbf16>
    %1746 = stablehlo.broadcast_in_dim %1744, dims = [0] : (tensor<16xi1>) -> tensor<16x32xi1>
    %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1747 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<bf16>) -> tensor<16x32xbf16>
    %1748 = stablehlo.select %1746, %1745, %1747 : tensor<16x32xi1>, tensor<16x32xbf16>
    %1749 = stablehlo.multiply %1684, %1748 : tensor<16x32xbf16>
    %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1750 = stablehlo.reduce(%1749 init: %cst_228) applies stablehlo.add across dimensions = [1] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<16xbf16>
    %1751 = stablehlo.reshape %1750 : (tensor<16xbf16>) -> tensor<16x1xbf16>
    %1752 = stablehlo.broadcast_in_dim %1685, dims = [0, 1] : (tensor<16x1xbf16>) -> tensor<16x32xbf16>
    %1753 = stablehlo.multiply %1748, %1752 : tensor<16x32xbf16>
    %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1754 = stablehlo.reduce(%1751 init: %cst_229) applies stablehlo.add across dimensions = [1] : (tensor<16x1xbf16>, tensor<bf16>) -> tensor<16xbf16>
    %cst_230 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1755 = stablehlo.pad %1753, %cst_230, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %1756 = stablehlo.broadcast_in_dim %1668, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1757 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1758 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_231 = stablehlo.constant dense<0> : tensor<i32>
    %1759 = stablehlo.broadcast_in_dim %c_231, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1760 = stablehlo.slice %1759 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1761 = stablehlo.slice %1758 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1762 = stablehlo.concatenate %1760, %1761, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1763 = stablehlo.broadcast_in_dim %1758, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1764 = stablehlo.broadcast_in_dim %1762, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1765 = stablehlo.compare LE, %1764, %1757, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1766 = stablehlo.compare LT, %1757, %1763, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1767 = stablehlo.and %1765, %1766 : tensor<4x512x32xi1>
    %cst_232 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1768 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1769 = stablehlo.select %1767, %1756, %1768 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1770 = stablehlo.broadcast_in_dim %1755, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1771 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1772 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_233 = stablehlo.constant dense<0> : tensor<i32>
    %1773 = stablehlo.broadcast_in_dim %c_233, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1774 = stablehlo.slice %1773 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1775 = stablehlo.slice %1772 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1776 = stablehlo.concatenate %1774, %1775, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1777 = stablehlo.broadcast_in_dim %1772, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1778 = stablehlo.broadcast_in_dim %1776, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1779 = stablehlo.compare LE, %1778, %1771, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1780 = stablehlo.compare LT, %1771, %1777, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1781 = stablehlo.and %1779, %1780 : tensor<4x512x32xi1>
    %cst_234 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1782 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1783 = stablehlo.select %1781, %1770, %1782 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1784 = stablehlo.dot_general %1769, %1783, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x512x32xbf16>) -> tensor<4x32x32xbf16>
    %1785 = stablehlo.broadcast_in_dim %1755, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1786 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1787 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_235 = stablehlo.constant dense<0> : tensor<i32>
    %1788 = stablehlo.broadcast_in_dim %c_235, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1789 = stablehlo.slice %1788 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1790 = stablehlo.slice %1787 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1791 = stablehlo.concatenate %1789, %1790, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1792 = stablehlo.broadcast_in_dim %1787, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1793 = stablehlo.broadcast_in_dim %1791, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1794 = stablehlo.compare LE, %1793, %1786, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1795 = stablehlo.compare LT, %1786, %1792, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1796 = stablehlo.and %1794, %1795 : tensor<4x512x32xi1>
    %cst_236 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1797 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1798 = stablehlo.select %1796, %1785, %1797 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1799 = stablehlo.dot_general %1798, %1610, contracting_dims = [2, 0] x [2, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x32xbf16>) -> tensor<512x32xbf16>
    %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1800 = stablehlo.pad %1799, %cst_237, low = [0, 0], high = [-496, 0], interior = [0, 0] : (tensor<512x32xbf16>, tensor<bf16>) -> tensor<16x32xbf16>
    %1801 = stablehlo.slice %1800 [0:16, 0:32] : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %1802 = stablehlo.multiply %1666#0, %1801 : tensor<16x32xbf16>
    %1803 = stablehlo.multiply %1801, %1665 : tensor<16x32xbf16>
    %1804 = call @silu_19(%1666#1, %1664, %1666#2, %1803) : (tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %1805 = stablehlo.concatenate %1804, %1802, dim = 1 : (tensor<16x32xbf16>, tensor<16x32xbf16>) -> tensor<16x64xbf16>
    %cst_238 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1806 = stablehlo.pad %1805, %cst_238, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x64xbf16>, tensor<bf16>) -> tensor<512x64xbf16>
    %1807 = stablehlo.broadcast_in_dim %1647, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1808 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1809 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_239 = stablehlo.constant dense<0> : tensor<i32>
    %1810 = stablehlo.broadcast_in_dim %c_239, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1811 = stablehlo.slice %1810 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1812 = stablehlo.slice %1809 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1813 = stablehlo.concatenate %1811, %1812, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1814 = stablehlo.broadcast_in_dim %1809, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1815 = stablehlo.broadcast_in_dim %1813, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1816 = stablehlo.compare LE, %1815, %1808, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1817 = stablehlo.compare LT, %1808, %1814, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1818 = stablehlo.and %1816, %1817 : tensor<4x512x32xi1>
    %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1819 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1820 = stablehlo.select %1818, %1807, %1819 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1821 = stablehlo.broadcast_in_dim %1806, dims = [1, 2] : (tensor<512x64xbf16>) -> tensor<4x512x64xbf16>
    %1822 = stablehlo.iota dim = 1 : tensor<4x512x64xi32>
    %1823 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_241 = stablehlo.constant dense<0> : tensor<i32>
    %1824 = stablehlo.broadcast_in_dim %c_241, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1825 = stablehlo.slice %1824 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1826 = stablehlo.slice %1823 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1827 = stablehlo.concatenate %1825, %1826, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1828 = stablehlo.broadcast_in_dim %1823, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1829 = stablehlo.broadcast_in_dim %1827, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1830 = stablehlo.compare LE, %1829, %1822, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1831 = stablehlo.compare LT, %1822, %1828, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1832 = stablehlo.and %1830, %1831 : tensor<4x512x64xi1>
    %cst_242 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1833 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<bf16>) -> tensor<4x512x64xbf16>
    %1834 = stablehlo.select %1832, %1821, %1833 : tensor<4x512x64xi1>, tensor<4x512x64xbf16>
    %1835 = stablehlo.dot_general %1820, %1834, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x512x64xbf16>) -> tensor<4x32x64xbf16>
    %1836 = stablehlo.broadcast_in_dim %1806, dims = [1, 2] : (tensor<512x64xbf16>) -> tensor<4x512x64xbf16>
    %1837 = stablehlo.iota dim = 1 : tensor<4x512x64xi32>
    %1838 = call @cumsum(%1646) : (tensor<4xi32>) -> tensor<4xi32>
    %c_243 = stablehlo.constant dense<0> : tensor<i32>
    %1839 = stablehlo.broadcast_in_dim %c_243, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1840 = stablehlo.slice %1839 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1841 = stablehlo.slice %1838 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1842 = stablehlo.concatenate %1840, %1841, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1843 = stablehlo.broadcast_in_dim %1838, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1844 = stablehlo.broadcast_in_dim %1842, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1845 = stablehlo.compare LE, %1844, %1837, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1846 = stablehlo.compare LT, %1837, %1843, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1847 = stablehlo.and %1845, %1846 : tensor<4x512x64xi1>
    %cst_244 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1848 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<bf16>) -> tensor<4x512x64xbf16>
    %1849 = stablehlo.select %1847, %1836, %1848 : tensor<4x512x64xi1>, tensor<4x512x64xbf16>
    %1850 = stablehlo.dot_general %1849, %1609, contracting_dims = [2, 0] x [2, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x64xbf16>, tensor<4x32x64xbf16>) -> tensor<512x32xbf16>
    %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1851 = stablehlo.pad %1850, %cst_245, low = [0, 0], high = [-496, 0], interior = [0, 0] : (tensor<512x32xbf16>, tensor<bf16>) -> tensor<16x32xbf16>
    %1852 = stablehlo.slice %1851 [0:16, 0:32] : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1853 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<bf16>) -> tensor<16xbf16>
    %1854 = "stablehlo.scatter"(%1853, %1635, %1754) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<16xbf16>, tensor<16x1xi32>, tensor<16xbf16>) -> tensor<16xbf16>
    %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1855 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %1856 = "stablehlo.scatter"(%1855, %1628, %1852) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<8x32xbf16>, tensor<16x1xi32>, tensor<16x32xbf16>) -> tensor<8x32xbf16>
    %1857 = stablehlo.reshape %1854 : (tensor<16xbf16>) -> tensor<8x2xbf16>
    %1858 = "stablehlo.all_reduce"(%1856) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<8x32xbf16>) -> tensor<8x32xbf16>
    %1859 = "stablehlo.all_reduce"(%1857) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<8x2xbf16>) -> tensor<8x2xbf16>
    %1860 = "stablehlo.all_reduce"(%1835) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<4x32x64xbf16>) -> tensor<4x32x64xbf16>
    %1861 = "stablehlo.all_reduce"(%1784) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<4x32x32xbf16>) -> tensor<4x32x32xbf16>
    %1862 = sdy.sharding_constraint %1861 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xbf16>
    %1863 = sdy.sharding_constraint %1860 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x64xbf16>
    %1864 = sdy.sharding_constraint %1859 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %1865 = sdy.sharding_constraint %1858 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1866 = stablehlo.slice %1863 [0:4, 0:32, 0:32] : (tensor<4x32x64xbf16>) -> tensor<4x32x32xbf16>
    %1867 = sdy.sharding_constraint %1866 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %1868 = stablehlo.slice %1863 [0:4, 0:32, 32:64] : (tensor<4x32x64xbf16>) -> tensor<4x32x32xbf16>
    %1869 = sdy.sharding_constraint %1868 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %cst_248 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %1870 = sdy.sharding_constraint %cst_248 <@mesh, []> : tensor<f32>
    %1871 = stablehlo.divide %1115, %1870 : tensor<f32>
    %1872 = sdy.sharding_constraint %1871 <@mesh, []> : tensor<f32>
    %1873 = stablehlo.broadcast_in_dim %1872, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1874 = sdy.sharding_constraint %1873 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1875 = stablehlo.multiply %1874, %1603 : tensor<8xf32>
    %1876 = sdy.sharding_constraint %1875 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1877 = stablehlo.divide %1876, %1591 : tensor<8xf32>
    %1878 = sdy.sharding_constraint %1877 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_249 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1879 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1880 = sdy.sharding_constraint %1879 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1881 = stablehlo.select %1595, %1880, %1878 : tensor<8xi1>, tensor<8xf32>
    %1882 = sdy.sharding_constraint %1881 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1883 = stablehlo.select %1595, %1878, %1880 : tensor<8xi1>, tensor<8xf32>
    %1884 = sdy.sharding_constraint %1883 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1885 = stablehlo.negate %1882 : tensor<8xf32>
    %1886 = sdy.sharding_constraint %1885 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1887 = stablehlo.add %1884, %1886 : tensor<8xf32>
    %1888 = sdy.sharding_constraint %1887 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1889 = stablehlo.broadcast_in_dim %1888, dims = [0] : (tensor<8xf32>) -> tensor<8x4xf32>
    %1890 = sdy.sharding_constraint %1889 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1891 = stablehlo.multiply %1890, %1587 : tensor<8x4xf32>
    %1892 = sdy.sharding_constraint %1891 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1893 = stablehlo.convert %1864 : (tensor<8x2xbf16>) -> tensor<8x2xf32>
    %1894 = sdy.sharding_constraint %1893 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1895 = stablehlo.multiply %1535, %1894 : tensor<8x2xf32>
    %1896 = sdy.sharding_constraint %1895 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1897 = stablehlo.reduce(%1896 init: %cst_250) applies stablehlo.add across dimensions = [1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8xf32>
    %1898 = sdy.sharding_constraint %1897 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1899 = stablehlo.reshape %1898 : (tensor<8xf32>) -> tensor<8x1xf32>
    %1900 = sdy.sharding_constraint %1899 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1901 = stablehlo.broadcast_in_dim %1554, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x2xf32>
    %1902 = sdy.sharding_constraint %1901 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1903 = stablehlo.multiply %1894, %1902 : tensor<8x2xf32>
    %1904 = sdy.sharding_constraint %1903 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1905 = stablehlo.multiply %1900, %1561 : tensor<8x1xf32>
    %1906 = sdy.sharding_constraint %1905 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_251 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %1907 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1908 = sdy.sharding_constraint %1907 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1909 = stablehlo.multiply %1906, %1908 : tensor<8x1xf32>
    %1910 = sdy.sharding_constraint %1909 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1911 = stablehlo.negate %1910 : tensor<8x1xf32>
    %1912 = sdy.sharding_constraint %1911 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1913 = stablehlo.reduce(%1912 init: %cst_252) applies stablehlo.add across dimensions = [1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8xf32>
    %1914 = sdy.sharding_constraint %1913 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1915 = stablehlo.broadcast_in_dim %1914, dims = [0] : (tensor<8xf32>) -> tensor<8x2xf32>
    %1916 = sdy.sharding_constraint %1915 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1917 = stablehlo.add %1904, %1916 : tensor<8x2xf32>
    %1918 = sdy.sharding_constraint %1917 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1919 = stablehlo.multiply %1918, %1542 : tensor<8x2xf32>
    %1920 = sdy.sharding_constraint %1919 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1921 = call @take_along_axis_20(%1523, %1920) : (tensor<8x2x1xi32>, tensor<8x2xf32>) -> tensor<8x4xf32>
    %1922 = sdy.sharding_constraint %1921 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1923 = stablehlo.add %1892, %1922 : tensor<8x4xf32>
    %1924 = sdy.sharding_constraint %1923 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1925 = stablehlo.convert %1924 : (tensor<8x4xf32>) -> tensor<8x4xbf16>
    %1926 = sdy.sharding_constraint %1925 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xbf16>
    %1927 = stablehlo.dot_general %1926, %1503, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4xbf16>, tensor<8x32xbf16>) -> tensor<4x32xbf16>
    %1928 = sdy.sharding_constraint %1927 <@mesh, [{}, {}]> : tensor<4x32xbf16>
    %1929 = stablehlo.transpose %1928, dims = [1, 0] : (tensor<4x32xbf16>) -> tensor<32x4xbf16>
    %1930 = sdy.sharding_constraint %1929 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %1931 = stablehlo.dot_general %1926, %1504, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4xbf16>, tensor<32x4xbf16>) -> tensor<8x32xbf16>
    %1932 = sdy.sharding_constraint %1931 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1933 = stablehlo.add %1865, %1932 : tensor<8x32xbf16>
    %1934 = sdy.sharding_constraint %1933 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1935 = sdy.sharding_constraint %1930 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %1936 = stablehlo.reshape %1934 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %1937 = sdy.sharding_constraint %1936 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1938 = stablehlo.add %1734, %1937 : tensor<2x4x32xbf16>
    %1939 = sdy.sharding_constraint %1938 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1940 = stablehlo.multiply %1472, %1939 : tensor<2x4x32xbf16>
    %1941 = sdy.sharding_constraint %1940 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1942 = stablehlo.multiply %1939, %1492 : tensor<2x4x32xbf16>
    %1943 = sdy.sharding_constraint %1942 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1944 = stablehlo.multiply %1941, %1499 : tensor<2x4x32xbf16>
    %1945 = sdy.sharding_constraint %1944 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1946 = stablehlo.dot_general %1945, %1476, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %1947 = sdy.sharding_constraint %1946 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1948 = stablehlo.transpose %1947, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %1949 = sdy.sharding_constraint %1948 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1950 = stablehlo.dot_general %1945, %1104, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %1951 = sdy.sharding_constraint %1950 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1952 = call @silu_21(%1477, %1474, %1478, %1951) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %1953 = sdy.sharding_constraint %1952 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1954 = stablehlo.dot_general %1953, %1472, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %1955 = sdy.sharding_constraint %1954 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1956 = stablehlo.transpose %1955, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %1957 = sdy.sharding_constraint %1956 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1958 = stablehlo.dot_general %1953, %1103, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %1959 = sdy.sharding_constraint %1958 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1960 = stablehlo.add %1943, %1959 : tensor<2x4x32xbf16>
    %1961 = sdy.sharding_constraint %1960 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1962 = stablehlo.convert %1961 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1963 = sdy.sharding_constraint %1962 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1964 = stablehlo.multiply %1462, %1963 : tensor<2x4x32xf32>
    %1965 = sdy.sharding_constraint %1964 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_253 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1966 = stablehlo.reduce(%1965 init: %cst_253) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1967 = sdy.sharding_constraint %1966 <@mesh, [{}]> : tensor<32xf32>
    %1968 = stablehlo.reshape %1967 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1969 = sdy.sharding_constraint %1968 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1970 = stablehlo.broadcast_in_dim %1466, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1971 = sdy.sharding_constraint %1970 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1972 = stablehlo.multiply %1963, %1971 : tensor<2x4x32xf32>
    %1973 = sdy.sharding_constraint %1972 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1974 = stablehlo.reduce(%1969 init: %cst_254) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1975 = sdy.sharding_constraint %1974 <@mesh, [{}]> : tensor<32xf32>
    %1976 = stablehlo.convert %1975 : (tensor<32xf32>) -> tensor<32xbf16>
    %1977 = sdy.sharding_constraint %1976 <@mesh, [{}]> : tensor<32xbf16>
    %1978 = stablehlo.multiply %1433, %1973 : tensor<2x4x32xf32>
    %1979 = sdy.sharding_constraint %1978 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_255 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1980 = stablehlo.reduce(%1979 init: %cst_255) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1981 = sdy.sharding_constraint %1980 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1982 = stablehlo.reshape %1981 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1983 = sdy.sharding_constraint %1982 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1984 = stablehlo.broadcast_in_dim %1452, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1985 = sdy.sharding_constraint %1984 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1986 = stablehlo.multiply %1973, %1985 : tensor<2x4x32xf32>
    %1987 = sdy.sharding_constraint %1986 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1988 = stablehlo.multiply %1983, %1458 : tensor<2x4x1xf32>
    %1989 = sdy.sharding_constraint %1988 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_256 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1990 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1991 = sdy.sharding_constraint %1990 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1992 = stablehlo.divide %1989, %1991 : tensor<2x4x1xf32>
    %1993 = sdy.sharding_constraint %1992 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_257 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1994 = stablehlo.reduce(%1993 init: %cst_257) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1995 = sdy.sharding_constraint %1994 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1996 = stablehlo.broadcast_in_dim %1995, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %1997 = sdy.sharding_constraint %1996 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1998 = stablehlo.multiply %1997, %1438 : tensor<2x4x32xf32>
    %1999 = sdy.sharding_constraint %1998 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2000 = stablehlo.add %1987, %1999 : tensor<2x4x32xf32>
    %2001 = sdy.sharding_constraint %2000 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2002 = stablehlo.convert %2001 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %2003 = sdy.sharding_constraint %2002 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2004 = stablehlo.add %1114, %2003 : tensor<2x4x32xbf16>
    %2005 = sdy.sharding_constraint %2004 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2006 = sdy.sharding_constraint %1977 <@mesh, [{}]> : tensor<32xbf16>
    %2007 = stablehlo.dot_general %2005, %1426, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x32xbf16>) -> tensor<32x32xbf16>
    %2008 = sdy.sharding_constraint %2007 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %2009 = stablehlo.transpose %2008, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2010 = sdy.sharding_constraint %2009 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %2011 = stablehlo.dot_general %2005, %1101, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %2012 = sdy.sharding_constraint %2011 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %2013 = stablehlo.reshape %2012 : (tensor<2x4x32xbf16>) -> tensor<2x4x2x16xbf16>
    %2014 = sdy.sharding_constraint %2013 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2015 = stablehlo.broadcast_in_dim %1420, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %2016 = sdy.sharding_constraint %2015 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2017 = stablehlo.multiply %2016, %2014 : tensor<2x4x2x16xbf16>
    %2018 = sdy.sharding_constraint %2017 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2019 = stablehlo.multiply %2014, %1393 : tensor<2x4x2x16xbf16>
    %2020 = sdy.sharding_constraint %2019 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_258 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2021 = stablehlo.reduce(%2020 init: %cst_258) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xbf16>, tensor<bf16>) -> tensor<2x4x2xbf16>
    %2022 = sdy.sharding_constraint %2021 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %2023 = stablehlo.reshape %2022 : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %2024 = sdy.sharding_constraint %2023 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_259 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %2025 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %2026 = sdy.sharding_constraint %2025 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2027 = stablehlo.multiply %2026, %2024 : tensor<2x4x2x1xbf16>
    %2028 = sdy.sharding_constraint %2027 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_260 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2029 = stablehlo.reduce(%2028 init: %cst_260) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xbf16>, tensor<bf16>) -> tensor<2x4x2xbf16>
    %2030 = sdy.sharding_constraint %2029 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %2031 = stablehlo.multiply %2030, %1414 : tensor<2x4x2xbf16>
    %2032 = sdy.sharding_constraint %2031 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %2033 = stablehlo.dot_general %2032, %1186, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2xbf16>, tensor<2x4x32xbf16>) -> tensor<2x32xbf16>
    %2034 = sdy.sharding_constraint %2033 <@mesh, [{}, {}]> : tensor<2x32xbf16>
    %2035 = stablehlo.transpose %2034, dims = [1, 0] : (tensor<2x32xbf16>) -> tensor<32x2xbf16>
    %2036 = sdy.sharding_constraint %2035 <@mesh, [{}, {}]> : tensor<32x2xbf16>
    %2037 = stablehlo.dot_general %2032, %1100, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2xbf16>, tensor<32x2xbf16>) -> tensor<2x4x32xbf16>
    %2038 = sdy.sharding_constraint %2037 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2039 = stablehlo.negate %2018 : tensor<2x4x2x16xbf16>
    %2040 = sdy.sharding_constraint %2039 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2041 = stablehlo.broadcast_in_dim %1380, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %2042 = sdy.sharding_constraint %2041 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2043 = stablehlo.multiply %2042, %2040 : tensor<2x4x2x16xbf16>
    %2044 = sdy.sharding_constraint %2043 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2045 = stablehlo.multiply %2040, %1354 : tensor<2x4x2x16xbf16>
    %2046 = sdy.sharding_constraint %2045 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_261 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2047 = stablehlo.reduce(%2046 init: %cst_261) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xbf16>, tensor<bf16>) -> tensor<2x4x2xbf16>
    %2048 = sdy.sharding_constraint %2047 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %2049 = stablehlo.reshape %2048 : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %2050 = sdy.sharding_constraint %2049 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2051 = stablehlo.multiply %2050, %1387 : tensor<2x4x2x1xbf16>
    %2052 = sdy.sharding_constraint %2051 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2053 = stablehlo.multiply %2052, %1364 : tensor<2x4x2x1xbf16>
    %2054 = sdy.sharding_constraint %2053 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2055 = stablehlo.negate %2054 : tensor<2x4x2x1xbf16>
    %2056 = sdy.sharding_constraint %2055 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2057 = stablehlo.divide %2050, %1378 : tensor<2x4x2x1xbf16>
    %2058 = sdy.sharding_constraint %2057 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2059 = stablehlo.convert %2056 : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x1xf32>
    %2060 = sdy.sharding_constraint %2059 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %cst_262 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2061 = stablehlo.reduce(%2060 init: %cst_262) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2062 = sdy.sharding_constraint %2061 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %2063 = stablehlo.broadcast_in_dim %2062, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x16xf32>
    %2064 = sdy.sharding_constraint %2063 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %2065 = stablehlo.convert %2064 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2066 = sdy.sharding_constraint %2065 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2067 = stablehlo.multiply %1354, %2066 : tensor<2x4x2x16xbf16>
    %2068 = sdy.sharding_constraint %2067 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2069 = stablehlo.add %2044, %2068 : tensor<2x4x2x16xbf16>
    %2070 = sdy.sharding_constraint %2069 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2071 = stablehlo.multiply %2066, %1354 : tensor<2x4x2x16xbf16>
    %2072 = sdy.sharding_constraint %2071 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2073 = stablehlo.add %2070, %2072 : tensor<2x4x2x16xbf16>
    %2074 = sdy.sharding_constraint %2073 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2075 = stablehlo.convert %2058 : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x1xf32>
    %2076 = sdy.sharding_constraint %2075 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2077 = stablehlo.reduce(%2076 init: %cst_263) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2078 = sdy.sharding_constraint %2077 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %2079 = stablehlo.broadcast_in_dim %2078, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x16xf32>
    %2080 = sdy.sharding_constraint %2079 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %2081 = stablehlo.convert %2080 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2082 = sdy.sharding_constraint %2081 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2083 = stablehlo.multiply %1347, %2082 : tensor<2x4x2x16xbf16>
    %2084 = sdy.sharding_constraint %2083 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2085 = stablehlo.add %2074, %2084 : tensor<2x4x2x16xbf16>
    %2086 = sdy.sharding_constraint %2085 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2087 = stablehlo.multiply %2082, %1354 : tensor<2x4x2x16xbf16>
    %2088 = sdy.sharding_constraint %2087 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2089 = stablehlo.add %2018, %2088 : tensor<2x4x2x16xbf16>
    %2090 = sdy.sharding_constraint %2089 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2091 = sdy.sharding_constraint %2086 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2092 = stablehlo.reshape %2091 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %2093 = sdy.sharding_constraint %2092 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %cst_264 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2094 = stablehlo.reduce(%2093 init: %cst_264) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x2x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2095 = sdy.sharding_constraint %2094 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2096 = stablehlo.broadcast_in_dim %2095, dims = [0, 1, 4] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %2097 = sdy.sharding_constraint %2096 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %cst_265 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2098 = stablehlo.reduce(%2097 init: %cst_265) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2099 = sdy.sharding_constraint %2098 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2100 = stablehlo.broadcast_in_dim %2099, dims = [0, 1, 3] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %2101 = sdy.sharding_constraint %2100 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2102 = stablehlo.transpose %2090, dims = [0, 2, 3, 1] : (tensor<2x4x2x16xbf16>) -> tensor<2x2x16x4xbf16>
    %2103 = sdy.sharding_constraint %2102 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %2104 = stablehlo.dot_general %2103, %1284, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x2x16x4xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x4xbf16>
    %2105 = sdy.sharding_constraint %2104 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %2106 = stablehlo.dot_general %2103, %1343, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x16x4xbf16>, tensor<2x2x4x4xbf16>) -> tensor<2x2x16x4xbf16>
    %2107 = sdy.sharding_constraint %2106 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %2108 = stablehlo.transpose %2107, dims = [0, 3, 1, 2] : (tensor<2x2x16x4xbf16>) -> tensor<2x4x2x16xbf16>
    %2109 = sdy.sharding_constraint %2108 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2110 = stablehlo.convert %2105 : (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %2111 = sdy.sharding_constraint %2110 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2112 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %2113 = sdy.sharding_constraint %2112 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2114 = stablehlo.multiply %2111, %2113 : tensor<2x2x4x4xf32>
    %2115 = sdy.sharding_constraint %2114 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2116 = stablehlo.multiply %2115, %1326 : tensor<2x2x4x4xf32>
    %2117 = sdy.sharding_constraint %2116 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_266 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2118 = stablehlo.reduce(%2117 init: %cst_266) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %2119 = sdy.sharding_constraint %2118 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %2120 = stablehlo.reshape %2119 : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %2121 = sdy.sharding_constraint %2120 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %2122 = stablehlo.negate %2121 : tensor<2x2x4x1xf32>
    %2123 = sdy.sharding_constraint %2122 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %2124 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %2125 = sdy.sharding_constraint %2124 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2126 = stablehlo.divide %2111, %2125 : tensor<2x2x4x4xf32>
    %2127 = sdy.sharding_constraint %2126 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_267 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2128 = stablehlo.reduce(%2123 init: %cst_267) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x1xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %2129 = sdy.sharding_constraint %2128 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %2130 = stablehlo.broadcast_in_dim %2129, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x4xf32>
    %2131 = sdy.sharding_constraint %2130 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2132 = stablehlo.add %2127, %2131 : tensor<2x2x4x4xf32>
    %2133 = sdy.sharding_constraint %2132 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2134 = stablehlo.multiply %2133, %1326 : tensor<2x2x4x4xf32>
    %2135 = sdy.sharding_constraint %2134 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2136 = stablehlo.convert %2135 : (tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xbf16>
    %2137 = sdy.sharding_constraint %2136 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %2138 = call @_where_22(%1310, %2137) : (tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xbf16>
    %2139 = sdy.sharding_constraint %2138 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %2140 = stablehlo.dot_general %2139, %1288, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2x4x4xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x16xbf16>
    %2141 = sdy.sharding_constraint %2140 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x2x4x16xbf16>
    %2142 = stablehlo.transpose %2141, dims = [0, 2, 1, 3] : (tensor<2x2x4x16xbf16>) -> tensor<2x4x2x16xbf16>
    %2143 = sdy.sharding_constraint %2142 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2144 = stablehlo.dot_general %2139, %1278, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2x4x4xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x16xbf16>
    %2145 = sdy.sharding_constraint %2144 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x16xbf16>
    %2146 = stablehlo.transpose %2145, dims = [0, 2, 1, 3] : (tensor<2x2x4x16xbf16>) -> tensor<2x4x2x16xbf16>
    %2147 = sdy.sharding_constraint %2146 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %cst_268 = stablehlo.constant dense<2.500000e-01> : tensor<bf16>
    %2148 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %2149 = sdy.sharding_constraint %2148 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2150 = stablehlo.multiply %2147, %2149 : tensor<2x4x2x16xbf16>
    %2151 = sdy.sharding_constraint %2150 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2152 = stablehlo.reshape %2109 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %2153 = sdy.sharding_constraint %2152 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2154 = stablehlo.reduce(%2153 init: %cst_269) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x2x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2155 = sdy.sharding_constraint %2154 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2156 = stablehlo.broadcast_in_dim %2155, dims = [0, 1, 4] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %2157 = sdy.sharding_constraint %2156 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %cst_270 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2158 = stablehlo.reduce(%2157 init: %cst_270) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2159 = sdy.sharding_constraint %2158 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2160 = stablehlo.broadcast_in_dim %2159, dims = [0, 1, 3] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %2161 = sdy.sharding_constraint %2160 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2162 = stablehlo.add %2101, %2161 : tensor<2x4x1x16xbf16>
    %2163 = sdy.sharding_constraint %2162 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2164 = stablehlo.reshape %2143 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %2165 = sdy.sharding_constraint %2164 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %cst_271 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2166 = stablehlo.reduce(%2165 init: %cst_271) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x2x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2167 = sdy.sharding_constraint %2166 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2168 = stablehlo.broadcast_in_dim %2167, dims = [0, 1, 4] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %2169 = sdy.sharding_constraint %2168 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %cst_272 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2170 = stablehlo.reduce(%2169 init: %cst_272) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2171 = sdy.sharding_constraint %2170 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2172 = stablehlo.broadcast_in_dim %2171, dims = [0, 1, 3] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %2173 = sdy.sharding_constraint %2172 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %cst_273 = stablehlo.constant dense<1.296880e+00> : tensor<bf16>
    %2174 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %2175 = sdy.sharding_constraint %2174 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2176 = stablehlo.multiply %2151, %2175 : tensor<2x4x2x16xbf16>
    %2177 = sdy.sharding_constraint %2176 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2178 = stablehlo.convert %2173 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %2179 = sdy.sharding_constraint %2178 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2180 = stablehlo.multiply %1262, %2179 : tensor<2x4x1x16xf32>
    %2181 = sdy.sharding_constraint %2180 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %cst_274 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2182 = stablehlo.reduce(%2181 init: %cst_274) applies stablehlo.add across dimensions = [3] : (tensor<2x4x1x16xf32>, tensor<f32>) -> tensor<2x4x1xf32>
    %2183 = sdy.sharding_constraint %2182 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2184 = stablehlo.reshape %2183 : (tensor<2x4x1xf32>) -> tensor<2x4x1x1xf32>
    %2185 = sdy.sharding_constraint %2184 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %2186 = stablehlo.broadcast_in_dim %1254, dims = [0, 1, 2, 3] : (tensor<2x4x1x1xf32>) -> tensor<2x4x1x16xf32>
    %2187 = sdy.sharding_constraint %2186 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2188 = stablehlo.multiply %2179, %2187 : tensor<2x4x1x16xf32>
    %2189 = sdy.sharding_constraint %2188 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2190 = stablehlo.convert %2189 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %2191 = sdy.sharding_constraint %2190 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2192 = stablehlo.multiply %2185, %1260 : tensor<2x4x1x1xf32>
    %2193 = sdy.sharding_constraint %2192 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_275 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %2194 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %2195 = sdy.sharding_constraint %2194 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %2196 = stablehlo.divide %2193, %2195 : tensor<2x4x1x1xf32>
    %2197 = sdy.sharding_constraint %2196 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2198 = stablehlo.reduce(%2197 init: %cst_276) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2199 = sdy.sharding_constraint %2198 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2200 = stablehlo.broadcast_in_dim %2199, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %2201 = sdy.sharding_constraint %2200 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2202 = stablehlo.broadcast_in_dim %2201, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x1x16xf32>
    %2203 = sdy.sharding_constraint %2202 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2204 = stablehlo.multiply %2203, %1240 : tensor<2x4x1x16xf32>
    %2205 = sdy.sharding_constraint %2204 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2206 = stablehlo.convert %2205 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %2207 = sdy.sharding_constraint %2206 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2208 = stablehlo.add %2191, %2207 : tensor<2x4x1x16xbf16>
    %2209 = sdy.sharding_constraint %2208 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2210 = stablehlo.convert %2177 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %2211 = sdy.sharding_constraint %2210 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2212 = stablehlo.multiply %1227, %2211 : tensor<2x4x2x16xf32>
    %2213 = sdy.sharding_constraint %2212 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %cst_277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2214 = stablehlo.reduce(%2213 init: %cst_277) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2215 = sdy.sharding_constraint %2214 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %2216 = stablehlo.reshape %2215 : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %2217 = sdy.sharding_constraint %2216 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %2218 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x16xf32>
    %2219 = sdy.sharding_constraint %2218 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2220 = stablehlo.multiply %2211, %2219 : tensor<2x4x2x16xf32>
    %2221 = sdy.sharding_constraint %2220 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2222 = stablehlo.convert %2221 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2223 = sdy.sharding_constraint %2222 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2224 = stablehlo.multiply %2217, %1225 : tensor<2x4x2x1xf32>
    %2225 = sdy.sharding_constraint %2224 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_278 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %2226 = stablehlo.broadcast_in_dim %cst_278, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %2227 = sdy.sharding_constraint %2226 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %2228 = stablehlo.divide %2225, %2227 : tensor<2x4x2x1xf32>
    %2229 = sdy.sharding_constraint %2228 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_279 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2230 = stablehlo.reduce(%2229 init: %cst_279) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2231 = sdy.sharding_constraint %2230 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %2232 = stablehlo.broadcast_in_dim %2231, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x16xf32>
    %2233 = sdy.sharding_constraint %2232 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2234 = stablehlo.multiply %2233, %1205 : tensor<2x4x2x16xf32>
    %2235 = sdy.sharding_constraint %2234 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2236 = stablehlo.convert %2235 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2237 = sdy.sharding_constraint %2236 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2238 = stablehlo.add %2223, %2237 : tensor<2x4x2x16xbf16>
    %2239 = sdy.sharding_constraint %2238 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2240 = stablehlo.reshape %2163 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x16xbf16>
    %2241 = sdy.sharding_constraint %2240 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2242 = stablehlo.dot_general %2241, %1186, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<2x4x32xbf16>) -> tensor<16x32xbf16>
    %2243 = sdy.sharding_constraint %2242 <@mesh, [{"model"}, {"data"}]> : tensor<16x32xbf16>
    %2244 = stablehlo.transpose %2243, dims = [1, 0] : (tensor<16x32xbf16>) -> tensor<32x16xbf16>
    %2245 = sdy.sharding_constraint %2244 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %2246 = stablehlo.dot_general %2241, %1099, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<32x16xbf16>) -> tensor<2x4x32xbf16>
    %2247 = sdy.sharding_constraint %2246 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2248 = stablehlo.add %2038, %2247 : tensor<2x4x32xbf16>
    %2249 = sdy.sharding_constraint %2248 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2250 = stablehlo.reshape %2209 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x16xbf16>
    %2251 = sdy.sharding_constraint %2250 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2252 = stablehlo.dot_general %2251, %1186, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<2x4x32xbf16>) -> tensor<16x32xbf16>
    %2253 = sdy.sharding_constraint %2252 <@mesh, [{"model"}, {"data"}]> : tensor<16x32xbf16>
    %2254 = stablehlo.transpose %2253, dims = [1, 0] : (tensor<16x32xbf16>) -> tensor<32x16xbf16>
    %2255 = sdy.sharding_constraint %2254 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %2256 = stablehlo.dot_general %2251, %1098, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<32x16xbf16>) -> tensor<2x4x32xbf16>
    %2257 = sdy.sharding_constraint %2256 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2258 = stablehlo.add %2249, %2257 : tensor<2x4x32xbf16>
    %2259 = sdy.sharding_constraint %2258 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2260 = stablehlo.reshape %2239 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x32xbf16>
    %2261 = sdy.sharding_constraint %2260 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %2262 = stablehlo.dot_general %2261, %1186, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x32xbf16>) -> tensor<32x32xbf16>
    %2263 = sdy.sharding_constraint %2262 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %2264 = stablehlo.transpose %2263, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2265 = sdy.sharding_constraint %2264 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %2266 = stablehlo.dot_general %2261, %1097, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %2267 = sdy.sharding_constraint %2266 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2268 = stablehlo.add %2259, %2267 : tensor<2x4x32xbf16>
    %2269 = sdy.sharding_constraint %2268 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2270 = stablehlo.multiply %1157, %2269 : tensor<2x4x32xbf16>
    %2271 = sdy.sharding_constraint %2270 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2272 = stablehlo.multiply %2269, %1177 : tensor<2x4x32xbf16>
    %2273 = sdy.sharding_constraint %2272 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2274 = stablehlo.multiply %2271, %1184 : tensor<2x4x32xbf16>
    %2275 = sdy.sharding_constraint %2274 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2276 = stablehlo.dot_general %2275, %1161, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %2277 = sdy.sharding_constraint %2276 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2278 = stablehlo.transpose %2277, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %2279 = sdy.sharding_constraint %2278 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2280 = stablehlo.dot_general %2275, %1096, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %2281 = sdy.sharding_constraint %2280 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2282 = call @silu_21(%1162, %1159, %1163, %2281) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %2283 = sdy.sharding_constraint %2282 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2284 = stablehlo.dot_general %2283, %1157, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %2285 = sdy.sharding_constraint %2284 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2286 = stablehlo.transpose %2285, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %2287 = sdy.sharding_constraint %2286 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2288 = stablehlo.dot_general %2283, %1095, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %2289 = sdy.sharding_constraint %2288 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2290 = stablehlo.add %2273, %2289 : tensor<2x4x32xbf16>
    %2291 = sdy.sharding_constraint %2290 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2292 = stablehlo.convert %2291 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %2293 = sdy.sharding_constraint %2292 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2294 = stablehlo.multiply %1147, %2293 : tensor<2x4x32xf32>
    %2295 = sdy.sharding_constraint %2294 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2296 = stablehlo.reduce(%2295 init: %cst_280) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2297 = sdy.sharding_constraint %2296 <@mesh, [{}]> : tensor<32xf32>
    %2298 = stablehlo.reshape %2297 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %2299 = sdy.sharding_constraint %2298 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %2300 = stablehlo.broadcast_in_dim %1151, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %2301 = sdy.sharding_constraint %2300 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2302 = stablehlo.multiply %2293, %2301 : tensor<2x4x32xf32>
    %2303 = sdy.sharding_constraint %2302 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_281 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2304 = stablehlo.reduce(%2299 init: %cst_281) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2305 = sdy.sharding_constraint %2304 <@mesh, [{}]> : tensor<32xf32>
    %2306 = stablehlo.convert %2305 : (tensor<32xf32>) -> tensor<32xbf16>
    %2307 = sdy.sharding_constraint %2306 <@mesh, [{}]> : tensor<32xbf16>
    %2308 = stablehlo.multiply %1118, %2303 : tensor<2x4x32xf32>
    %2309 = sdy.sharding_constraint %2308 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_282 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2310 = stablehlo.reduce(%2309 init: %cst_282) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2311 = sdy.sharding_constraint %2310 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2312 = stablehlo.reshape %2311 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %2313 = sdy.sharding_constraint %2312 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2314 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %2315 = sdy.sharding_constraint %2314 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2316 = stablehlo.multiply %2303, %2315 : tensor<2x4x32xf32>
    %2317 = sdy.sharding_constraint %2316 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2318 = stablehlo.multiply %2313, %1143 : tensor<2x4x1xf32>
    %2319 = sdy.sharding_constraint %2318 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_283 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %2320 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %2321 = sdy.sharding_constraint %2320 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2322 = stablehlo.divide %2319, %2321 : tensor<2x4x1xf32>
    %2323 = sdy.sharding_constraint %2322 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2324 = stablehlo.reduce(%2323 init: %cst_284) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2325 = sdy.sharding_constraint %2324 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2326 = stablehlo.broadcast_in_dim %2325, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %2327 = sdy.sharding_constraint %2326 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2328 = stablehlo.multiply %2327, %1123 : tensor<2x4x32xf32>
    %2329 = sdy.sharding_constraint %2328 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2330 = stablehlo.add %2317, %2329 : tensor<2x4x32xf32>
    %2331 = sdy.sharding_constraint %2330 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2332 = stablehlo.convert %2331 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %2333 = sdy.sharding_constraint %2332 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2334 = stablehlo.add %2005, %2333 : tensor<2x4x32xbf16>
    %2335 = sdy.sharding_constraint %2334 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2336 = sdy.sharding_constraint %2307 <@mesh, [{}]> : tensor<32xbf16>
    %2337 = stablehlo.multiply %116, %2335 : tensor<2x4x32xbf16>
    %2338 = sdy.sharding_constraint %2337 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2339 = stablehlo.multiply %2335, %136 : tensor<2x4x32xbf16>
    %2340 = sdy.sharding_constraint %2339 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2341 = stablehlo.multiply %2338, %143 : tensor<2x4x32xbf16>
    %2342 = sdy.sharding_constraint %2341 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2343 = stablehlo.dot_general %2342, %120, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %2344 = sdy.sharding_constraint %2343 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2345 = stablehlo.transpose %2344, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %2346 = sdy.sharding_constraint %2345 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2347 = stablehlo.dot_general %2342, %20, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %2348 = sdy.sharding_constraint %2347 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2349 = call @silu_8(%121, %122, %118, %2348) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %2350 = sdy.sharding_constraint %2349 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2351 = stablehlo.dot_general %2350, %116, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %2352 = sdy.sharding_constraint %2351 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2353 = stablehlo.transpose %2352, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %2354 = sdy.sharding_constraint %2353 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2355 = stablehlo.dot_general %2350, %18, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %2356 = sdy.sharding_constraint %2355 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2357 = stablehlo.add %2340, %2356 : tensor<2x4x32xbf16>
    %2358 = sdy.sharding_constraint %2357 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2359 = stablehlo.convert %2358 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %2360 = sdy.sharding_constraint %2359 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2361 = stablehlo.multiply %106, %2360 : tensor<2x4x32xf32>
    %2362 = sdy.sharding_constraint %2361 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2363 = stablehlo.reduce(%2362 init: %cst_285) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2364 = sdy.sharding_constraint %2363 <@mesh, [{}]> : tensor<32xf32>
    %2365 = stablehlo.reshape %2364 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %2366 = sdy.sharding_constraint %2365 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %2367 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %2368 = sdy.sharding_constraint %2367 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2369 = stablehlo.multiply %2360, %2368 : tensor<2x4x32xf32>
    %2370 = sdy.sharding_constraint %2369 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2371 = stablehlo.reduce(%2366 init: %cst_286) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2372 = sdy.sharding_constraint %2371 <@mesh, [{}]> : tensor<32xf32>
    %2373 = stablehlo.convert %2372 : (tensor<32xf32>) -> tensor<32xbf16>
    %2374 = sdy.sharding_constraint %2373 <@mesh, [{}]> : tensor<32xbf16>
    %2375 = stablehlo.multiply %77, %2370 : tensor<2x4x32xf32>
    %2376 = sdy.sharding_constraint %2375 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_287 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2377 = stablehlo.reduce(%2376 init: %cst_287) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2378 = sdy.sharding_constraint %2377 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2379 = stablehlo.reshape %2378 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %2380 = sdy.sharding_constraint %2379 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2381 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %2382 = sdy.sharding_constraint %2381 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2383 = stablehlo.multiply %2370, %2382 : tensor<2x4x32xf32>
    %2384 = sdy.sharding_constraint %2383 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2385 = stablehlo.multiply %2380, %102 : tensor<2x4x1xf32>
    %2386 = sdy.sharding_constraint %2385 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_288 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %2387 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %2388 = sdy.sharding_constraint %2387 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2389 = stablehlo.divide %2386, %2388 : tensor<2x4x1xf32>
    %2390 = sdy.sharding_constraint %2389 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2391 = stablehlo.reduce(%2390 init: %cst_289) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2392 = sdy.sharding_constraint %2391 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2393 = stablehlo.broadcast_in_dim %2392, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %2394 = sdy.sharding_constraint %2393 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2395 = stablehlo.multiply %2394, %82 : tensor<2x4x32xf32>
    %2396 = sdy.sharding_constraint %2395 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2397 = stablehlo.add %2384, %2396 : tensor<2x4x32xf32>
    %2398 = sdy.sharding_constraint %2397 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2399 = stablehlo.convert %2398 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %2400 = sdy.sharding_constraint %2399 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2401 = sdy.sharding_constraint %2374 <@mesh, [{}]> : tensor<32xbf16>
    %cst_290 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2402 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<bf16>) -> tensor<64x32xbf16>
    %2403 = "stablehlo.scatter"(%2402, %72, %2400) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg30: tensor<bf16>, %arg31: tensor<bf16>):
      %2665 = stablehlo.add %arg30, %arg31 : tensor<bf16>
      stablehlo.return %2665 : tensor<bf16>
    }) : (tensor<64x32xbf16>, tensor<2x4x1xi32>, tensor<2x4x32xbf16>) -> tensor<64x32xbf16>
    %2404 = sdy.sharding_constraint %2403 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xbf16>
    %2405 = stablehlo.convert %1034 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2406 = sdy.sharding_constraint %2405 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2407 = stablehlo.convert %1042 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2408 = sdy.sharding_constraint %2407 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2409 = stablehlo.convert %1089 : (tensor<32xbf16>) -> tensor<32xf32>
    %2410 = sdy.sharding_constraint %2409 <@mesh, [{}]> : tensor<32xf32>
    %2411 = stablehlo.convert %1862 : (tensor<4x32x32xbf16>) -> tensor<4x32x32xf32>
    %2412 = sdy.sharding_constraint %2411 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2413 = stablehlo.convert %1869 : (tensor<4x32x32xbf16>) -> tensor<4x32x32xf32>
    %2414 = sdy.sharding_constraint %2413 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2415 = stablehlo.convert %1867 : (tensor<4x32x32xbf16>) -> tensor<4x32x32xf32>
    %2416 = sdy.sharding_constraint %2415 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2417 = stablehlo.convert %1710 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2418 = sdy.sharding_constraint %2417 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2419 = stablehlo.convert %1722 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2420 = sdy.sharding_constraint %2419 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2421 = stablehlo.convert %1728 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2422 = sdy.sharding_constraint %2421 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2423 = stablehlo.convert %1935 : (tensor<32x4xbf16>) -> tensor<32x4xf32>
    %2424 = sdy.sharding_constraint %2423 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2425 = stablehlo.convert %1949 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2426 = sdy.sharding_constraint %2425 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2427 = stablehlo.convert %1957 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2428 = sdy.sharding_constraint %2427 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2429 = stablehlo.convert %2006 : (tensor<32xbf16>) -> tensor<32xf32>
    %2430 = sdy.sharding_constraint %2429 <@mesh, [{}]> : tensor<32xf32>
    %2431 = stablehlo.convert %2036 : (tensor<32x2xbf16>) -> tensor<32x2xf32>
    %2432 = sdy.sharding_constraint %2431 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2433 = stablehlo.convert %2010 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2434 = sdy.sharding_constraint %2433 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2435 = stablehlo.convert %2245 : (tensor<32x16xbf16>) -> tensor<32x16xf32>
    %2436 = sdy.sharding_constraint %2435 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2437 = stablehlo.convert %2255 : (tensor<32x16xbf16>) -> tensor<32x16xf32>
    %2438 = sdy.sharding_constraint %2437 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2439 = stablehlo.convert %2265 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2440 = sdy.sharding_constraint %2439 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2441 = stablehlo.convert %2279 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2442 = sdy.sharding_constraint %2441 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2443 = stablehlo.convert %2287 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2444 = sdy.sharding_constraint %2443 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2445 = stablehlo.convert %2336 : (tensor<32xbf16>) -> tensor<32xf32>
    %2446 = sdy.sharding_constraint %2445 <@mesh, [{}]> : tensor<32xf32>
    %2447 = stablehlo.convert %1023 : (tensor<32x64xbf16>) -> tensor<32x64xf32>
    %2448 = sdy.sharding_constraint %2447 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2449 = stablehlo.convert %2346 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2450 = sdy.sharding_constraint %2449 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2451 = stablehlo.convert %2354 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2452 = sdy.sharding_constraint %2451 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2453 = stablehlo.convert %2401 : (tensor<32xbf16>) -> tensor<32xf32>
    %2454 = sdy.sharding_constraint %2453 <@mesh, [{}]> : tensor<32xf32>
    %2455 = stablehlo.convert %2404 : (tensor<64x32xbf16>) -> tensor<64x32xf32>
    %2456 = sdy.sharding_constraint %2455 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_291 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2457 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2458 = sdy.sharding_constraint %2457 <@mesh, [{}]> : tensor<4xf32>
    %cst_292 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2459 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2460 = sdy.sharding_constraint %2459 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2461 = stablehlo.multiply %2460, %2456 : tensor<64x32xf32>
    %2462 = sdy.sharding_constraint %2461 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_293 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2463 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2464 = sdy.sharding_constraint %2463 <@mesh, [{}]> : tensor<32xf32>
    %2465 = stablehlo.multiply %2464, %2454 : tensor<32xf32>
    %2466 = sdy.sharding_constraint %2465 <@mesh, [{}]> : tensor<32xf32>
    %cst_294 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2467 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2468 = sdy.sharding_constraint %2467 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2469 = stablehlo.multiply %2468, %2452 : tensor<32x128xf32>
    %2470 = sdy.sharding_constraint %2469 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_295 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2471 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2472 = sdy.sharding_constraint %2471 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2473 = stablehlo.multiply %2472, %2450 : tensor<128x32xf32>
    %2474 = sdy.sharding_constraint %2473 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_296 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2475 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %2476 = sdy.sharding_constraint %2475 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2477 = stablehlo.multiply %2476, %2448 : tensor<32x64xf32>
    %2478 = sdy.sharding_constraint %2477 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_297 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2479 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2480 = sdy.sharding_constraint %2479 <@mesh, [{}]> : tensor<32xf32>
    %2481 = stablehlo.multiply %2480, %2446 : tensor<32xf32>
    %2482 = sdy.sharding_constraint %2481 <@mesh, [{}]> : tensor<32xf32>
    %cst_298 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2483 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2484 = sdy.sharding_constraint %2483 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2485 = stablehlo.multiply %2484, %2444 : tensor<32x128xf32>
    %2486 = sdy.sharding_constraint %2485 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_299 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2487 = stablehlo.broadcast_in_dim %cst_299, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2488 = sdy.sharding_constraint %2487 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2489 = stablehlo.multiply %2488, %2442 : tensor<128x32xf32>
    %2490 = sdy.sharding_constraint %2489 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_300 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2491 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2492 = sdy.sharding_constraint %2491 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2493 = stablehlo.multiply %2492, %2440 : tensor<32x32xf32>
    %2494 = sdy.sharding_constraint %2493 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_301 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2495 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2496 = sdy.sharding_constraint %2495 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2497 = stablehlo.multiply %2496, %2438 : tensor<32x16xf32>
    %2498 = sdy.sharding_constraint %2497 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_302 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2499 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2500 = sdy.sharding_constraint %2499 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2501 = stablehlo.multiply %2500, %2436 : tensor<32x16xf32>
    %2502 = sdy.sharding_constraint %2501 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_303 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2503 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2504 = sdy.sharding_constraint %2503 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2505 = stablehlo.multiply %2504, %2434 : tensor<32x32xf32>
    %2506 = sdy.sharding_constraint %2505 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_304 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2507 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %2508 = sdy.sharding_constraint %2507 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2509 = stablehlo.multiply %2508, %2432 : tensor<32x2xf32>
    %2510 = sdy.sharding_constraint %2509 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_305 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2511 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2512 = sdy.sharding_constraint %2511 <@mesh, [{}]> : tensor<32xf32>
    %2513 = stablehlo.multiply %2512, %2430 : tensor<32xf32>
    %2514 = sdy.sharding_constraint %2513 <@mesh, [{}]> : tensor<32xf32>
    %cst_306 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2515 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2516 = sdy.sharding_constraint %2515 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2517 = stablehlo.multiply %2516, %2428 : tensor<32x128xf32>
    %2518 = sdy.sharding_constraint %2517 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_307 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2519 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2520 = sdy.sharding_constraint %2519 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2521 = stablehlo.multiply %2520, %2426 : tensor<128x32xf32>
    %2522 = sdy.sharding_constraint %2521 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_308 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2523 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %2524 = sdy.sharding_constraint %2523 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2525 = stablehlo.multiply %2524, %2424 : tensor<32x4xf32>
    %2526 = sdy.sharding_constraint %2525 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_309 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2527 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2528 = sdy.sharding_constraint %2527 <@mesh, [{}]> : tensor<4xf32>
    %2529 = stablehlo.multiply %2528, %2458 : tensor<4xf32>
    %2530 = sdy.sharding_constraint %2529 <@mesh, [{}]> : tensor<4xf32>
    %cst_310 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2531 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2532 = sdy.sharding_constraint %2531 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2533 = stablehlo.multiply %2532, %2422 : tensor<32x32xf32>
    %2534 = sdy.sharding_constraint %2533 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_311 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2535 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2536 = sdy.sharding_constraint %2535 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2537 = stablehlo.multiply %2536, %2420 : tensor<32x32xf32>
    %2538 = sdy.sharding_constraint %2537 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_312 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2539 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2540 = sdy.sharding_constraint %2539 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2541 = stablehlo.multiply %2540, %2418 : tensor<32x32xf32>
    %2542 = sdy.sharding_constraint %2541 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_313 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2543 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2544 = sdy.sharding_constraint %2543 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2545 = stablehlo.multiply %2544, %2416 : tensor<4x32x32xf32>
    %2546 = sdy.sharding_constraint %2545 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_314 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2547 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2548 = sdy.sharding_constraint %2547 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2549 = stablehlo.multiply %2548, %2414 : tensor<4x32x32xf32>
    %2550 = sdy.sharding_constraint %2549 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_315 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2551 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2552 = sdy.sharding_constraint %2551 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2553 = stablehlo.multiply %2552, %2412 : tensor<4x32x32xf32>
    %2554 = sdy.sharding_constraint %2553 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_316 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2555 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2556 = sdy.sharding_constraint %2555 <@mesh, [{}]> : tensor<32xf32>
    %2557 = stablehlo.multiply %2556, %2410 : tensor<32xf32>
    %2558 = sdy.sharding_constraint %2557 <@mesh, [{}]> : tensor<32xf32>
    %cst_317 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2559 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2560 = sdy.sharding_constraint %2559 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2561 = stablehlo.multiply %2560, %2408 : tensor<32x128xf32>
    %2562 = sdy.sharding_constraint %2561 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_318 = stablehlo.constant dense<-1.000000e-03> : tensor<f32>
    %2563 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2564 = sdy.sharding_constraint %2563 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2565 = stablehlo.multiply %2564, %2406 : tensor<128x32xf32>
    %2566 = sdy.sharding_constraint %2565 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2567 = stablehlo.multiply %2416, %2416 : tensor<4x32x32xf32>
    %2568 = sdy.sharding_constraint %2567 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_319 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2569 = stablehlo.reduce(%2568 init: %cst_319) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %2570 = sdy.sharding_constraint %2569 <@mesh, []> : tensor<f32>
    %cst_320 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2571 = sdy.sharding_constraint %cst_320 <@mesh, []> : tensor<f32>
    %2572 = stablehlo.add %2571, %2570 : tensor<f32>
    %2573 = sdy.sharding_constraint %2572 <@mesh, []> : tensor<f32>
    %2574 = stablehlo.multiply %2414, %2414 : tensor<4x32x32xf32>
    %2575 = sdy.sharding_constraint %2574 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_321 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2576 = stablehlo.reduce(%2575 init: %cst_321) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %2577 = sdy.sharding_constraint %2576 <@mesh, []> : tensor<f32>
    %2578 = stablehlo.add %2573, %2577 : tensor<f32>
    %2579 = sdy.sharding_constraint %2578 <@mesh, []> : tensor<f32>
    %2580 = stablehlo.multiply %2412, %2412 : tensor<4x32x32xf32>
    %2581 = sdy.sharding_constraint %2580 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_322 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2582 = stablehlo.reduce(%2581 init: %cst_322) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %2583 = sdy.sharding_constraint %2582 <@mesh, []> : tensor<f32>
    %2584 = stablehlo.add %2579, %2583 : tensor<f32>
    %2585 = sdy.sharding_constraint %2584 <@mesh, []> : tensor<f32>
    %2586 = stablehlo.sqrt %2585 : tensor<f32>
    %2587 = sdy.sharding_constraint %2586 <@mesh, []> : tensor<f32>
    %2588 = stablehlo.multiply %2546, %2546 : tensor<4x32x32xf32>
    %2589 = sdy.sharding_constraint %2588 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_323 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2590 = stablehlo.reduce(%2589 init: %cst_323) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %2591 = sdy.sharding_constraint %2590 <@mesh, []> : tensor<f32>
    %cst_324 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2592 = sdy.sharding_constraint %cst_324 <@mesh, []> : tensor<f32>
    %2593 = stablehlo.add %2592, %2591 : tensor<f32>
    %2594 = sdy.sharding_constraint %2593 <@mesh, []> : tensor<f32>
    %2595 = stablehlo.multiply %2550, %2550 : tensor<4x32x32xf32>
    %2596 = sdy.sharding_constraint %2595 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2597 = stablehlo.reduce(%2596 init: %cst_325) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %2598 = sdy.sharding_constraint %2597 <@mesh, []> : tensor<f32>
    %2599 = stablehlo.add %2594, %2598 : tensor<f32>
    %2600 = sdy.sharding_constraint %2599 <@mesh, []> : tensor<f32>
    %2601 = stablehlo.multiply %2554, %2554 : tensor<4x32x32xf32>
    %2602 = sdy.sharding_constraint %2601 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_326 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2603 = stablehlo.reduce(%2602 init: %cst_326) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %2604 = sdy.sharding_constraint %2603 <@mesh, []> : tensor<f32>
    %2605 = stablehlo.add %2600, %2604 : tensor<f32>
    %2606 = sdy.sharding_constraint %2605 <@mesh, []> : tensor<f32>
    %2607 = stablehlo.sqrt %2606 : tensor<f32>
    %2608 = sdy.sharding_constraint %2607 <@mesh, []> : tensor<f32>
    %2609 = stablehlo.add %arg1, %2462 : tensor<64x32xf32>
    %2610 = sdy.sharding_constraint %2609 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2611 = stablehlo.add %arg2, %2466 : tensor<32xf32>
    %2612 = sdy.sharding_constraint %2611 <@mesh, [{}]> : tensor<32xf32>
    %2613 = stablehlo.add %arg3, %2470 : tensor<32x128xf32>
    %2614 = sdy.sharding_constraint %2613 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2615 = stablehlo.add %arg4, %2474 : tensor<128x32xf32>
    %2616 = sdy.sharding_constraint %2615 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2617 = stablehlo.add %arg5, %2478 : tensor<32x64xf32>
    %2618 = sdy.sharding_constraint %2617 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2619 = stablehlo.add %arg6, %2482 : tensor<32xf32>
    %2620 = sdy.sharding_constraint %2619 <@mesh, [{}]> : tensor<32xf32>
    %2621 = stablehlo.add %arg7, %2486 : tensor<32x128xf32>
    %2622 = sdy.sharding_constraint %2621 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2623 = stablehlo.add %arg8, %2490 : tensor<128x32xf32>
    %2624 = sdy.sharding_constraint %2623 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2625 = stablehlo.add %arg9, %2494 : tensor<32x32xf32>
    %2626 = sdy.sharding_constraint %2625 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2627 = stablehlo.add %arg10, %2498 : tensor<32x16xf32>
    %2628 = sdy.sharding_constraint %2627 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2629 = stablehlo.add %arg11, %2502 : tensor<32x16xf32>
    %2630 = sdy.sharding_constraint %2629 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2631 = stablehlo.add %arg12, %2506 : tensor<32x32xf32>
    %2632 = sdy.sharding_constraint %2631 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2633 = stablehlo.add %arg13, %2510 : tensor<32x2xf32>
    %2634 = sdy.sharding_constraint %2633 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2635 = stablehlo.add %arg14, %2514 : tensor<32xf32>
    %2636 = sdy.sharding_constraint %2635 <@mesh, [{}]> : tensor<32xf32>
    %2637 = stablehlo.add %arg15, %2518 : tensor<32x128xf32>
    %2638 = sdy.sharding_constraint %2637 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2639 = stablehlo.add %arg16, %2522 : tensor<128x32xf32>
    %2640 = sdy.sharding_constraint %2639 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2641 = stablehlo.add %arg17, %2526 : tensor<32x4xf32>
    %2642 = sdy.sharding_constraint %2641 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2643 = stablehlo.add %12, %2530 : tensor<4xf32>
    %2644 = sdy.sharding_constraint %2643 <@mesh, [{}]> : tensor<4xf32>
    %2645 = stablehlo.add %arg18, %2534 : tensor<32x32xf32>
    %2646 = sdy.sharding_constraint %2645 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2647 = stablehlo.add %arg19, %2538 : tensor<32x32xf32>
    %2648 = sdy.sharding_constraint %2647 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2649 = stablehlo.add %arg20, %2542 : tensor<32x32xf32>
    %2650 = sdy.sharding_constraint %2649 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2651 = stablehlo.add %arg21, %2546 : tensor<4x32x32xf32>
    %2652 = sdy.sharding_constraint %2651 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2653 = stablehlo.add %arg22, %2550 : tensor<4x32x32xf32>
    %2654 = sdy.sharding_constraint %2653 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2655 = stablehlo.add %arg23, %2554 : tensor<4x32x32xf32>
    %2656 = sdy.sharding_constraint %2655 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2657 = stablehlo.add %arg24, %2558 : tensor<32xf32>
    %2658 = sdy.sharding_constraint %2657 <@mesh, [{}]> : tensor<32xf32>
    %2659 = stablehlo.add %arg25, %2562 : tensor<32x128xf32>
    %2660 = sdy.sharding_constraint %2659 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2661 = stablehlo.add %arg26, %2566 : tensor<128x32xf32>
    %2662 = sdy.sharding_constraint %2661 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2663 = stablehlo.add %arg0, %c : tensor<i32>
    %2664 = sdy.sharding_constraint %2663 <@mesh, []> : tensor<i32>
    %cst_327 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %cst_328 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    return %2664, %2610, %2612, %2614, %2616, %2618, %2620, %2622, %2624, %2626, %2628, %2630, %2632, %2634, %2636, %2638, %2640, %2642, %2644, %2646, %2648, %2650, %2652, %2654, %2656, %2658, %2660, %2662, %750, %750, %884, %896, %894, %924, %985, %928, %930, %926, %961, %965, %936, %971, %940, %946, %973, %979, %983, %967, %932, %914, %919, %744, %909, %987, %2587, %cst_327, %cst_328, %2608 : tensor<i32>, tensor<64x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x64xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x32xf32>, tensor<32x16xf32>, tensor<32x16xf32>, tensor<32x32xf32>, tensor<32x2xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x4xf32>, tensor<4xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<5xf32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @silu(%arg0: tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) {
    %0 = stablehlo.negate %arg0 : tensor<2x4x128xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2 = stablehlo.exponential %1 : tensor<2x4x128xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %6 = stablehlo.add %5, %3 : tensor<2x4x128xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %9 = sdy.sharding_constraint %8 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %10 = stablehlo.divide %9, %7 : tensor<2x4x128xbf16>
    %11 = sdy.sharding_constraint %10 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %12 = sdy.sharding_constraint %cst_1 <@mesh, []> : tensor<bf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %14 = sdy.sharding_constraint %13 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %15 = stablehlo.subtract %14, %11 : tensor<2x4x128xbf16>
    %16 = sdy.sharding_constraint %15 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %17 = stablehlo.multiply %11, %16 : tensor<2x4x128xbf16>
    %18 = sdy.sharding_constraint %17 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %19 = stablehlo.multiply %arg0, %11 : tensor<2x4x128xbf16>
    %20 = sdy.sharding_constraint %19 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    return %20, %18, %11 : tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>
  }
  func.func private @silu_0(%arg0: tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16> {
    %0 = stablehlo.negate %arg0 : tensor<2x4x128xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2 = stablehlo.exponential %1 : tensor<2x4x128xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %6 = stablehlo.add %5, %3 : tensor<2x4x128xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %9 = sdy.sharding_constraint %8 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %10 = stablehlo.divide %9, %7 : tensor<2x4x128xbf16>
    %11 = sdy.sharding_constraint %10 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %12 = stablehlo.multiply %arg0, %11 : tensor<2x4x128xbf16>
    %13 = sdy.sharding_constraint %12 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    return %13 : tensor<2x4x128xbf16>
  }
  func.func private @_where(%arg0: tensor<1x1x4x4xi1>, %arg1: tensor<2x2x4x4xbf16>, %arg2: tensor<bf16>) -> tensor<2x2x4x4xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x4x4xi1>) -> tensor<2x2x4x4xi1>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xi1>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<bf16>) -> tensor<2x2x4x4xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %4 = stablehlo.select %1, %arg1, %3 : tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    return %5 : tensor<2x2x4x4xbf16>
  }
  func.func private @take_along_axis(%arg0: tensor<8x4xf32>, %arg1: tensor<8x2xi32>) -> tensor<8x2xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x2xi32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %2 = stablehlo.compare LT, %arg1, %1, SIGNED : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi1>
    %c_0 = stablehlo.constant dense<4> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<8x2xi32>
    %4 = sdy.sharding_constraint %3 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<8x2xi32>
    %6 = sdy.sharding_constraint %5 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %7 = stablehlo.select %2, %6, %arg1 : tensor<8x2xi1>, tensor<8x2xi32>
    %8 = sdy.sharding_constraint %7 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %9 = stablehlo.reshape %8 : (tensor<8x2xi32>) -> tensor<8x2x1xi32>
    %10 = sdy.sharding_constraint %9 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %c_1 = stablehlo.constant dense<3> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<8x2x1xi32>
    %12 = sdy.sharding_constraint %11 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %13 = stablehlo.compare GE, %10, %12, SIGNED : (tensor<8x2x1xi32>, tensor<8x2x1xi32>) -> tensor<8x2x1xi1>
    %14 = stablehlo.broadcast_in_dim %c_1, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %15 = sdy.sharding_constraint %14 <@mesh, [{}, {}, {}]> : tensor<1x1x1xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<8x2x1xi32>
    %17 = sdy.sharding_constraint %16 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %18 = stablehlo.compare LE, %10, %17, SIGNED : (tensor<8x2x1xi32>, tensor<8x2x1xi32>) -> tensor<8x2x1xi1>
    %19 = stablehlo.and %13, %18 : tensor<8x2x1xi1>
    %20 = sdy.sharding_constraint %19 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %21 = stablehlo.reduce(%20 init: %c_3) applies stablehlo.and across dimensions = [2] : (tensor<8x2x1xi1>, tensor<i1>) -> tensor<8x2xi1>
    %22 = sdy.sharding_constraint %21 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi1>
    %23 = "stablehlo.gather"(%arg0, %10) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x4xf32>, tensor<8x2x1xi32>) -> tensor<8x2xf32>
    %24 = sdy.sharding_constraint %23 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %26 = sdy.sharding_constraint %25 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %27 = stablehlo.select %22, %24, %26 : tensor<8x2xi1>, tensor<8x2xf32>
    %28 = sdy.sharding_constraint %27 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    return %28 : tensor<8x2xf32>
  }
  func.func private @_one_hot(%arg0: tensor<8x2xi32>) -> tensor<8x2x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<8x2xi32>) -> tensor<8x2x1xi32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %2 = stablehlo.iota dim = 2 : tensor<1x1x4xi32>
    %3 = sdy.sharding_constraint %2 <@mesh, [{}, {}, {}]> : tensor<1x1x4xi32>
    %4 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<8x2x1xi32>) -> tensor<8x2x4xi32>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x4xi32>
    %6 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x1x4xi32>) -> tensor<8x2x4xi32>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x4xi32>
    %8 = stablehlo.compare EQ, %5, %7, SIGNED : (tensor<8x2x4xi32>, tensor<8x2x4xi32>) -> tensor<8x2x4xi1>
    %9 = stablehlo.convert %8 : (tensor<8x2x4xi1>) -> tensor<8x2x4xf32>
    %10 = sdy.sharding_constraint %9 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x4xf32>
    return %10 : tensor<8x2x4xf32>
  }
  func.func private @argsort(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<16xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare LT, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) : (tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>)
    return %1#1 : tensor<16xi32>
  }
  func.func private @floor_divide(%arg0: tensor<16xi32>, %arg1: tensor<i32>) -> tensor<16xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16xi32>
    %3 = stablehlo.sign %arg0 : tensor<16xi32>
    %4 = stablehlo.sign %0 : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %6 = stablehlo.compare NE, %3, %5, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %8 = stablehlo.remainder %arg0, %7 : tensor<16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %10 = stablehlo.compare NE, %8, %9, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %11 = stablehlo.and %6, %10 : tensor<16xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %13 = stablehlo.subtract %2, %12 : tensor<16xi32>
    %14 = call @_where_1(%11, %13, %2) : (tensor<16xi1>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %14 : tensor<16xi32>
  }
  func.func private @_where_1(%arg0: tensor<16xi1>, %arg1: tensor<16xi32>, %arg2: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<16xi1>, tensor<16xi32>
    return %0 : tensor<16xi32>
  }
  func.func private @clip(%arg0: tensor<16xi32>, %arg1: tensor<i32>) -> tensor<16xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16xi32>
    return %2 : tensor<16xi32>
  }
  func.func private @cumsum(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{base_dilations = array<i64: 1>, padding = dense<[[3, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 4>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }
  func.func private @silu_2(%arg0: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
    %0 = stablehlo.negate %arg0 : tensor<16x32xbf16>
    %1 = stablehlo.exponential %0 : tensor<16x32xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<16x32xbf16>
    %3 = stablehlo.add %2, %1 : tensor<16x32xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<16x32xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<16x32xbf16>
    %6 = stablehlo.multiply %arg0, %5 : tensor<16x32xbf16>
    return %6 : tensor<16x32xbf16>
  }
  func.func private @silu_3(%arg0: tensor<8x32xbf16>) -> tensor<8x32xbf16> {
    %0 = stablehlo.negate %arg0 : tensor<8x32xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %2 = stablehlo.exponential %1 : tensor<8x32xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %6 = stablehlo.add %5, %3 : tensor<8x32xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %9 = sdy.sharding_constraint %8 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %10 = stablehlo.divide %9, %7 : tensor<8x32xbf16>
    %11 = sdy.sharding_constraint %10 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %12 = stablehlo.multiply %arg0, %11 : tensor<8x32xbf16>
    %13 = sdy.sharding_constraint %12 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    return %13 : tensor<8x32xbf16>
  }
  func.func private @_where_4(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_5(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<f32>) -> tensor<4xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = sdy.sharding_constraint %1 <@mesh, [{}]> : tensor<4xf32>
    %3 = stablehlo.select %arg0, %arg1, %2 : tensor<4xi1>, tensor<4xf32>
    %4 = sdy.sharding_constraint %3 <@mesh, [{}]> : tensor<4xf32>
    return %4 : tensor<4xf32>
  }
  func.func private @_where_6(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    %2 = sdy.sharding_constraint %1 <@mesh, []> : tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @_where_7(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.select %arg0, %arg1, %cst : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @silu_8(%arg0: tensor<2x4x128xbf16>, %arg1: tensor<2x4x128xbf16>, %arg2: tensor<2x4x128xbf16>, %arg3: tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16> {
    %0 = stablehlo.multiply %arg2, %arg3 : tensor<2x4x128xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2 = stablehlo.multiply %arg3, %arg1 : tensor<2x4x128xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %4 = stablehlo.multiply %1, %arg0 : tensor<2x4x128xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %6 = stablehlo.add %3, %5 : tensor<2x4x128xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    return %7 : tensor<2x4x128xbf16>
  }
  func.func private @silu_9(%arg0: tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) {
    %0 = stablehlo.negate %arg0 : tensor<2x4x128xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2 = stablehlo.exponential %1 : tensor<2x4x128xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %6 = stablehlo.add %5, %3 : tensor<2x4x128xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %9 = sdy.sharding_constraint %8 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %10 = stablehlo.divide %9, %7 : tensor<2x4x128xbf16>
    %11 = sdy.sharding_constraint %10 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %12 = sdy.sharding_constraint %cst_1 <@mesh, []> : tensor<bf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<bf16>) -> tensor<2x4x128xbf16>
    %14 = sdy.sharding_constraint %13 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %15 = stablehlo.subtract %14, %11 : tensor<2x4x128xbf16>
    %16 = sdy.sharding_constraint %15 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %17 = stablehlo.multiply %11, %16 : tensor<2x4x128xbf16>
    %18 = sdy.sharding_constraint %17 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %19 = stablehlo.multiply %arg0, %11 : tensor<2x4x128xbf16>
    %20 = sdy.sharding_constraint %19 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    return %20, %11, %18 : tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>
  }
  func.func private @_where_10(%arg0: tensor<1x1x4x4xi1>, %arg1: tensor<2x2x4x4xbf16>, %arg2: tensor<bf16>) -> (tensor<2x2x4x4xbf16>, tensor<2x2x4x4xi1>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x4x4xi1>) -> tensor<2x2x4x4xi1>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xi1>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<bf16>) -> tensor<2x2x4x4xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %4 = stablehlo.select %1, %arg1, %3 : tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    return %5, %1 : tensor<2x2x4x4xbf16>, tensor<2x2x4x4xi1>
  }
  func.func private @take_along_axis_11(%arg0: tensor<8x4xf32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xf32>, tensor<8x2x1xi32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x2xi32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %2 = stablehlo.compare LT, %arg1, %1, SIGNED : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi1>
    %c_0 = stablehlo.constant dense<4> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<8x2xi32>
    %4 = sdy.sharding_constraint %3 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<8x2xi32>
    %6 = sdy.sharding_constraint %5 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %7 = stablehlo.select %2, %6, %arg1 : tensor<8x2xi1>, tensor<8x2xi32>
    %8 = sdy.sharding_constraint %7 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %9 = stablehlo.reshape %8 : (tensor<8x2xi32>) -> tensor<8x2x1xi32>
    %10 = sdy.sharding_constraint %9 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %c_1 = stablehlo.constant dense<3> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<8x2x1xi32>
    %12 = sdy.sharding_constraint %11 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %13 = stablehlo.compare GE, %10, %12, SIGNED : (tensor<8x2x1xi32>, tensor<8x2x1xi32>) -> tensor<8x2x1xi1>
    %14 = stablehlo.broadcast_in_dim %c_1, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %15 = sdy.sharding_constraint %14 <@mesh, [{}, {}, {}]> : tensor<1x1x1xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<8x2x1xi32>
    %17 = sdy.sharding_constraint %16 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi32>
    %18 = stablehlo.compare LE, %10, %17, SIGNED : (tensor<8x2x1xi32>, tensor<8x2x1xi32>) -> tensor<8x2x1xi1>
    %19 = stablehlo.and %13, %18 : tensor<8x2x1xi1>
    %20 = sdy.sharding_constraint %19 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<8x2x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %21 = stablehlo.reduce(%20 init: %c_3) applies stablehlo.and across dimensions = [2] : (tensor<8x2x1xi1>, tensor<i1>) -> tensor<8x2xi1>
    %22 = sdy.sharding_constraint %21 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi1>
    %23 = "stablehlo.gather"(%arg0, %10) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x4xf32>, tensor<8x2x1xi32>) -> tensor<8x2xf32>
    %24 = sdy.sharding_constraint %23 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %26 = sdy.sharding_constraint %25 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %27 = stablehlo.select %22, %24, %26 : tensor<8x2xi1>, tensor<8x2xf32>
    %28 = sdy.sharding_constraint %27 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    return %28, %10 : tensor<8x2xf32>, tensor<8x2x1xi32>
  }
  func.func private @argsort_12(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<16xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare LT, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) : (tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>)
    return %1#1 : tensor<16xi32>
  }
  func.func private @floor_divide_13(%arg0: tensor<16xi32>, %arg1: tensor<i32>) -> tensor<16xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16xi32>
    %3 = stablehlo.sign %arg0 : tensor<16xi32>
    %4 = stablehlo.sign %0 : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %6 = stablehlo.compare NE, %3, %5, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %8 = stablehlo.remainder %arg0, %7 : tensor<16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %10 = stablehlo.compare NE, %8, %9, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %11 = stablehlo.and %6, %10 : tensor<16xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %13 = stablehlo.subtract %2, %12 : tensor<16xi32>
    %14 = call @_where_14(%11, %13, %2) : (tensor<16xi1>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %14 : tensor<16xi32>
  }
  func.func private @_where_14(%arg0: tensor<16xi1>, %arg1: tensor<16xi32>, %arg2: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<16xi1>, tensor<16xi32>
    return %0 : tensor<16xi32>
  }
  func.func private @clip_15(%arg0: tensor<16xi32>, %arg1: tensor<i32>) -> tensor<16xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16xi32>
    return %2 : tensor<16xi32>
  }
  func.func private @silu_16(%arg0: tensor<16x32xbf16>) -> (tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>) {
    %0 = stablehlo.negate %arg0 : tensor<16x32xbf16>
    %1 = stablehlo.exponential %0 : tensor<16x32xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<16x32xbf16>
    %3 = stablehlo.add %2, %1 : tensor<16x32xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<16x32xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<16x32xbf16>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<16x32xbf16>
    %7 = stablehlo.subtract %6, %5 : tensor<16x32xbf16>
    %8 = stablehlo.multiply %5, %7 : tensor<16x32xbf16>
    %9 = stablehlo.multiply %arg0, %5 : tensor<16x32xbf16>
    return %9, %5, %8 : tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>
  }
  func.func private @silu_17(%arg0: tensor<8x32xbf16>) -> (tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>) {
    %0 = stablehlo.negate %arg0 : tensor<8x32xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %2 = stablehlo.exponential %1 : tensor<8x32xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %6 = stablehlo.add %5, %3 : tensor<8x32xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %9 = sdy.sharding_constraint %8 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %10 = stablehlo.divide %9, %7 : tensor<8x32xbf16>
    %11 = sdy.sharding_constraint %10 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %12 = sdy.sharding_constraint %cst_1 <@mesh, []> : tensor<bf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %14 = sdy.sharding_constraint %13 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %15 = stablehlo.subtract %14, %11 : tensor<8x32xbf16>
    %16 = sdy.sharding_constraint %15 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %17 = stablehlo.multiply %11, %16 : tensor<8x32xbf16>
    %18 = sdy.sharding_constraint %17 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %19 = stablehlo.multiply %arg0, %11 : tensor<8x32xbf16>
    %20 = sdy.sharding_constraint %19 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    return %20, %11, %18 : tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>
  }
  func.func private @silu_18(%arg0: tensor<8x32xbf16>, %arg1: tensor<8x32xbf16>, %arg2: tensor<8x32xbf16>, %arg3: tensor<8x32xbf16>) -> tensor<8x32xbf16> {
    %0 = stablehlo.multiply %arg1, %arg3 : tensor<8x32xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %2 = stablehlo.multiply %arg3, %arg0 : tensor<8x32xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %4 = stablehlo.multiply %1, %arg2 : tensor<8x32xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %6 = stablehlo.add %3, %5 : tensor<8x32xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    return %7 : tensor<8x32xbf16>
  }
  func.func private @silu_19(%arg0: tensor<16x32xbf16>, %arg1: tensor<16x32xbf16>, %arg2: tensor<16x32xbf16>, %arg3: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
    %0 = stablehlo.multiply %arg1, %arg3 : tensor<16x32xbf16>
    %1 = stablehlo.multiply %arg3, %arg0 : tensor<16x32xbf16>
    %2 = stablehlo.multiply %0, %arg2 : tensor<16x32xbf16>
    %3 = stablehlo.add %1, %2 : tensor<16x32xbf16>
    return %3 : tensor<16x32xbf16>
  }
  func.func private @take_along_axis_20(%arg0: tensor<8x2x1xi32>, %arg1: tensor<8x2xf32>) -> tensor<8x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x4xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %2 = "stablehlo.scatter"(%1, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = sdy.sharding_constraint %arg2 <@mesh, []> : tensor<f32>
      %5 = sdy.sharding_constraint %arg3 <@mesh, []> : tensor<f32>
      %6 = stablehlo.add %4, %5 : tensor<f32>
      %7 = sdy.sharding_constraint %6 <@mesh, []> : tensor<f32>
      stablehlo.return %7 : tensor<f32>
    }) : (tensor<8x4xf32>, tensor<8x2x1xi32>, tensor<8x2xf32>) -> tensor<8x4xf32>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    return %3 : tensor<8x4xf32>
  }
  func.func private @silu_21(%arg0: tensor<2x4x128xbf16>, %arg1: tensor<2x4x128xbf16>, %arg2: tensor<2x4x128xbf16>, %arg3: tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16> {
    %0 = stablehlo.multiply %arg1, %arg3 : tensor<2x4x128xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2 = stablehlo.multiply %arg3, %arg0 : tensor<2x4x128xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %4 = stablehlo.multiply %1, %arg2 : tensor<2x4x128xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %6 = stablehlo.add %3, %5 : tensor<2x4x128xbf16>
    %7 = sdy.sharding_constraint %6 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    return %7 : tensor<2x4x128xbf16>
  }
  func.func private @_where_22(%arg0: tensor<2x2x4x4xi1>, %arg1: tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x2x4x4xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    return %3 : tensor<2x2x4x4xbf16>
  }
}
