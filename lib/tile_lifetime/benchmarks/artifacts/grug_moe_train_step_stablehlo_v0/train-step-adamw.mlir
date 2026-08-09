module @jit_train_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["replica_dcn"=1, "data"=1, "expert"=1, "model"=1]> {stablehlo.mesh = {axes = [{name = "replica_dcn", size = 1 : i64}, {name = "data", size = 1 : i64}, {name = "expert", size = 1 : i64}, {name = "model", size = 1 : i64}]}}
  func.func public @main(%arg0: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>, tf.aliasing_output = 0 : i32}, %arg1: tensor<64x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>, tf.aliasing_output = 1 : i32}, %arg2: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 2 : i32}, %arg3: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 3 : i32}, %arg4: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 4 : i32}, %arg5: tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>, tf.aliasing_output = 5 : i32}, %arg6: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 6 : i32}, %arg7: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 7 : i32}, %arg8: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 8 : i32}, %arg9: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 9 : i32}, %arg10: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 10 : i32}, %arg11: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 11 : i32}, %arg12: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 12 : i32}, %arg13: tensor<32x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 13 : i32}, %arg14: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 14 : i32}, %arg15: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 15 : i32}, %arg16: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 16 : i32}, %arg17: tensor<32x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 17 : i32}, %arg18: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 19 : i32}, %arg19: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 20 : i32}, %arg20: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 21 : i32}, %arg21: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 22 : i32}, %arg22: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 23 : i32}, %arg23: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>, tf.aliasing_output = 24 : i32}, %arg24: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 25 : i32}, %arg25: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 26 : i32}, %arg26: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 27 : i32}, %arg27: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>, tf.aliasing_output = 28 : i32}, %arg28: tensor<64x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>, tf.aliasing_output = 29 : i32}, %arg29: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 30 : i32}, %arg30: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 31 : i32}, %arg31: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 32 : i32}, %arg32: tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>, tf.aliasing_output = 33 : i32}, %arg33: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 34 : i32}, %arg34: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 35 : i32}, %arg35: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 36 : i32}, %arg36: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 37 : i32}, %arg37: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 38 : i32}, %arg38: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 39 : i32}, %arg39: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 40 : i32}, %arg40: tensor<32x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 41 : i32}, %arg41: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 42 : i32}, %arg42: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 43 : i32}, %arg43: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 44 : i32}, %arg44: tensor<32x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 45 : i32}, %arg45: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 18 : i32}, %arg46: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 47 : i32}, %arg47: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 48 : i32}, %arg48: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 49 : i32}, %arg49: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 50 : i32}, %arg50: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 51 : i32}, %arg51: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>, tf.aliasing_output = 52 : i32}, %arg52: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 53 : i32}, %arg53: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 54 : i32}, %arg54: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 55 : i32}, %arg55: tensor<64x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>, tf.aliasing_output = 56 : i32}, %arg56: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 57 : i32}, %arg57: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 58 : i32}, %arg58: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 59 : i32}, %arg59: tensor<32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>, tf.aliasing_output = 60 : i32}, %arg60: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 61 : i32}, %arg61: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 62 : i32}, %arg62: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 63 : i32}, %arg63: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 64 : i32}, %arg64: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 65 : i32}, %arg65: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 66 : i32}, %arg66: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 67 : i32}, %arg67: tensor<32x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 68 : i32}, %arg68: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 69 : i32}, %arg69: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 70 : i32}, %arg70: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 71 : i32}, %arg71: tensor<32x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 72 : i32}, %arg72: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 46 : i32}, %arg73: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 74 : i32}, %arg74: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>, tf.aliasing_output = 75 : i32}, %arg75: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>, tf.aliasing_output = 76 : i32}, %arg76: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 77 : i32}, %arg77: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>, tf.aliasing_output = 78 : i32}, %arg78: tensor<4x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>, tf.aliasing_output = 79 : i32}, %arg79: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 80 : i32}, %arg80: tensor<32x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 81 : i32}, %arg81: tensor<128x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 82 : i32}, %arg82: tensor<1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, tf.aliasing_output = 83 : i32}, %arg83: tensor<2x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg84: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<i32> {jax.result_info = "result[0].step", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<64x32xf32> {jax.result_info = "result[0].params.token_embed", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>}, tensor<32xf32> {jax.result_info = "result[0].params.embed_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.embed_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.embed_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x64xf32> {jax.result_info = "result[0].params.output_proj", sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>}, tensor<32xf32> {jax.result_info = "result[0].params.blocks[0].rms_attn.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.blocks[0].attn_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.blocks[0].attn_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_q", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_k", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_v", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].attn.w_o", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<32x2xf32> {jax.result_info = "result[0].params.blocks[0].attn.attn_gate", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32xf32> {jax.result_info = "result[0].params.blocks[0].rms_mlp.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.blocks[0].mlp_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.blocks[0].mlp_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x4xf32> {jax.result_info = "result[0].params.blocks[0].mlp.router", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<4xf32> {jax.result_info = "result[0].params.blocks[0].mlp.router_bias", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].shared.w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].shared.w_up", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].params.blocks[0].shared.w_down", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].params.expert_banks[0].w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].params.expert_banks[0].w_up", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].params.expert_banks[0].w_down", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>}, tensor<32xf32> {jax.result_info = "result[0].params.final_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].params.final_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].params.final_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<i32> {jax.result_info = "result[0].opt_state[0].count", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<64x32xf32> {jax.result_info = "result[0].opt_state[0].mu.token_embed", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].mu.embed_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].mu.embed_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].mu.embed_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x64xf32> {jax.result_info = "result[0].opt_state[0].mu.output_proj", sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].rms_attn.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].attn_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].attn_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].attn.w_q", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].attn.w_k", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].attn.w_v", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].attn.w_o", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<32x2xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].attn.attn_gate", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].rms_mlp.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].mlp_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].mlp_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x4xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].mlp.router", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<4xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].mlp.router_bias", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].shared.w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].shared.w_up", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.blocks[0].shared.w_down", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.expert_banks[0].w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.expert_banks[0].w_up", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].opt_state[0].mu.expert_banks[0].w_down", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].mu.final_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].mu.final_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].mu.final_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<64x32xf32> {jax.result_info = "result[0].opt_state[0].nu.token_embed", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"replica_dcn", "data"}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].nu.embed_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].nu.embed_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].nu.embed_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x64xf32> {jax.result_info = "result[0].opt_state[0].nu.output_proj", sdy.sharding = #sdy.sharding<@mesh, [{"replica_dcn", "data"}, {"model"}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].rms_attn.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].attn_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].attn_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].attn.w_q", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].attn.w_k", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x16xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].attn.w_v", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].attn.w_o", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<32x2xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].attn.attn_gate", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].rms_mlp.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].mlp_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].mlp_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<32x4xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].mlp.router", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<4xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].mlp.router_bias", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].shared.w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].shared.w_up", sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, tensor<32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.blocks[0].shared.w_down", sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"data"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.expert_banks[0].w_gate", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.expert_banks[0].w_up", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"data"}, {"model"}]>}, tensor<4x32x32xf32> {jax.result_info = "result[0].opt_state[0].nu.expert_banks[0].w_down", sdy.sharding = #sdy.sharding<@mesh, [{"expert"}, {"model"}, {"data"}]>}, tensor<32xf32> {jax.result_info = "result[0].opt_state[0].nu.final_norm.weight", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<32x128xf32> {jax.result_info = "result[0].opt_state[0].nu.final_gated_norm.w_down", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<128x32xf32> {jax.result_info = "result[0].opt_state[0].nu.final_gated_norm.w_up", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1x4xf32> {jax.result_info = "result[0].pending_qb_betas", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1x4xf32> {jax.result_info = "result[1]['qb_beta_per_layer']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<f32> {jax.result_info = "result[1]['train/cross_entropy_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/aux_loss_weighted']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/capacity_overflow_rate_mean']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/capacity_overflow_rate']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/load_balancing_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/router_z_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_entropy']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].min", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].max", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].num", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<i32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].nonzero_count", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].sum", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].sum_squares", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].mean", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].variance", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].rms", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<5xf32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].histogram.bucket_limits", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<4xf32> {jax.result_info = "result[1]['train/router/layer_0/routing_hist'].histogram.bucket_counts", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<f32> {jax.result_info = "result[1]['train/router/load_balancing_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['train/router/router_z_loss']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<1x4xf32> {jax.result_info = "result[1]['train/router/routing_counts_per_layer']", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<f32> {jax.result_info = "result[1]['train/router/routing_entropy_mean']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['tying/activation_norm_by_layer/layer_0']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['tying/expert_gradient_norm_by_bank/bank_0']", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<f32> {jax.result_info = "result[1]['tying/top1_cross_loop_agreement']"}, tensor<f32> {jax.result_info = "result[1]['tying/topk_set_overlap']"}, tensor<f32> {jax.result_info = "result[1]['tying/update_norm_by_bank/bank_0']", sdy.sharding = #sdy.sharding<@mesh, []>}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.reshape %arg82 : (tensor<1x4xf32>) -> tensor<4xf32>
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
    %68 = stablehlo.compare LT, %arg83, %67, SIGNED : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %c_2 = stablehlo.constant dense<64> : tensor<i32>
    %69 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %70 = stablehlo.add %arg83, %69 : tensor<2x4xi32>
    %71 = stablehlo.select %68, %70, %arg83 : tensor<2x4xi1>, tensor<2x4xi32>
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
    %120 = stablehlo.dot_general %119#0, %20, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %121 = sdy.sharding_constraint %120 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %122 = stablehlo.negate %121 : tensor<2x4x32xbf16>
    %123 = sdy.sharding_constraint %122 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %124 = stablehlo.exponential %123 : tensor<2x4x32xbf16>
    %125 = sdy.sharding_constraint %124 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %126 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %127 = sdy.sharding_constraint %126 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %128 = stablehlo.add %127, %125 : tensor<2x4x32xbf16>
    %129 = sdy.sharding_constraint %128 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %130 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %131 = sdy.sharding_constraint %130 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %132 = stablehlo.divide %131, %129 : tensor<2x4x32xbf16>
    %133 = sdy.sharding_constraint %132 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %134 = sdy.sharding_constraint %cst_10 <@mesh, []> : tensor<bf16>
    %135 = stablehlo.broadcast_in_dim %134, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %136 = sdy.sharding_constraint %135 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %137 = stablehlo.subtract %136, %133 : tensor<2x4x32xbf16>
    %138 = sdy.sharding_constraint %137 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %139 = stablehlo.multiply %133, %138 : tensor<2x4x32xbf16>
    %140 = sdy.sharding_constraint %139 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %141 = stablehlo.multiply %116, %133 : tensor<2x4x32xbf16>
    %142 = sdy.sharding_constraint %141 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %143 = sdy.sharding_constraint %24 <@mesh, [{}]> : tensor<32xbf16>
    %144 = stablehlo.convert %142 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %145 = sdy.sharding_constraint %144 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %146 = chlo.square %145 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %147 = stablehlo.reduce(%146 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %148 = sdy.sharding_constraint %147 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %150 = sdy.sharding_constraint %149 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_12 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %151 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %152 = sdy.sharding_constraint %151 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %153 = stablehlo.divide %150, %152 : tensor<2x4x1xf32>
    %154 = sdy.sharding_constraint %153 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %155 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %156 = sdy.sharding_constraint %155 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %157 = stablehlo.add %154, %156 : tensor<2x4x1xf32>
    %158 = sdy.sharding_constraint %157 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %159 = stablehlo.rsqrt %158 : tensor<2x4x1xf32>
    %160 = sdy.sharding_constraint %159 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %162 = sdy.sharding_constraint %161 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %163 = stablehlo.multiply %145, %162 : tensor<2x4x32xf32>
    %164 = sdy.sharding_constraint %163 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %165 = stablehlo.convert %143 : (tensor<32xbf16>) -> tensor<32xf32>
    %166 = sdy.sharding_constraint %165 <@mesh, [{}]> : tensor<32xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %168 = sdy.sharding_constraint %167 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %170 = sdy.sharding_constraint %169 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %171 = stablehlo.multiply %164, %170 : tensor<2x4x32xf32>
    %172 = sdy.sharding_constraint %171 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %173 = stablehlo.convert %172 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %174 = sdy.sharding_constraint %173 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %175 = stablehlo.dot_general %174, %26, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %176 = sdy.sharding_constraint %175 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %177 = call @silu_41(%176) : (tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %178 = stablehlo.dot_general %177, %28, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %179 = sdy.sharding_constraint %178 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %180 = stablehlo.negate %179 : tensor<2x4x32xbf16>
    %181 = sdy.sharding_constraint %180 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %182 = stablehlo.exponential %181 : tensor<2x4x32xbf16>
    %183 = sdy.sharding_constraint %182 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %184 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %185 = sdy.sharding_constraint %184 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %186 = stablehlo.add %185, %183 : tensor<2x4x32xbf16>
    %187 = sdy.sharding_constraint %186 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %188 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %189 = sdy.sharding_constraint %188 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %190 = stablehlo.divide %189, %187 : tensor<2x4x32xbf16>
    %191 = sdy.sharding_constraint %190 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %192 = stablehlo.multiply %174, %191 : tensor<2x4x32xbf16>
    %193 = sdy.sharding_constraint %192 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %194 = stablehlo.dot_general %193, %30, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %195 = sdy.sharding_constraint %194 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %196 = stablehlo.reshape %195 : (tensor<2x4x32xbf16>) -> tensor<2x4x2x16xbf16>
    %197 = sdy.sharding_constraint %196 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %198 = stablehlo.dot_general %193, %32, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %199 = sdy.sharding_constraint %198 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %200 = stablehlo.reshape %199 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %201 = sdy.sharding_constraint %200 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %202 = stablehlo.dot_general %193, %34, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %203 = sdy.sharding_constraint %202 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %204 = stablehlo.reshape %203 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %205 = sdy.sharding_constraint %204 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %206 = stablehlo.convert %197 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %207 = sdy.sharding_constraint %206 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %208 = chlo.square %207 : tensor<2x4x2x16xf32> -> tensor<2x4x2x16xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %209 = stablehlo.reduce(%208 init: %cst_16) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %210 = sdy.sharding_constraint %209 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %212 = sdy.sharding_constraint %211 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_17 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %213 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %214 = sdy.sharding_constraint %213 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %215 = stablehlo.divide %212, %214 : tensor<2x4x2x1xf32>
    %216 = sdy.sharding_constraint %215 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_18 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %217 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %218 = sdy.sharding_constraint %217 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %219 = stablehlo.add %216, %218 : tensor<2x4x2x1xf32>
    %220 = sdy.sharding_constraint %219 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %221 = stablehlo.rsqrt %220 : tensor<2x4x2x1xf32>
    %222 = sdy.sharding_constraint %221 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %223 = stablehlo.convert %197 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %224 = sdy.sharding_constraint %223 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %225 = stablehlo.broadcast_in_dim %222, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x16xf32>
    %226 = sdy.sharding_constraint %225 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %227 = stablehlo.multiply %224, %226 : tensor<2x4x2x16xf32>
    %228 = sdy.sharding_constraint %227 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %229 = stablehlo.convert %228 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %230 = sdy.sharding_constraint %229 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %231 = stablehlo.convert %201 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %232 = sdy.sharding_constraint %231 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %233 = chlo.square %232 : tensor<2x4x1x16xf32> -> tensor<2x4x1x16xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %234 = stablehlo.reduce(%233 init: %cst_19) applies stablehlo.add across dimensions = [3] : (tensor<2x4x1x16xf32>, tensor<f32>) -> tensor<2x4x1xf32>
    %235 = sdy.sharding_constraint %234 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %236 = stablehlo.broadcast_in_dim %235, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x1x1xf32>
    %237 = sdy.sharding_constraint %236 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_20 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %238 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %239 = sdy.sharding_constraint %238 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %240 = stablehlo.divide %237, %239 : tensor<2x4x1x1xf32>
    %241 = sdy.sharding_constraint %240 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_21 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %242 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %243 = sdy.sharding_constraint %242 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %244 = stablehlo.add %241, %243 : tensor<2x4x1x1xf32>
    %245 = sdy.sharding_constraint %244 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %246 = stablehlo.rsqrt %245 : tensor<2x4x1x1xf32>
    %247 = sdy.sharding_constraint %246 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %248 = stablehlo.convert %201 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %249 = sdy.sharding_constraint %248 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %250 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2, 3] : (tensor<2x4x1x1xf32>) -> tensor<2x4x1x16xf32>
    %251 = sdy.sharding_constraint %250 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %252 = stablehlo.multiply %249, %251 : tensor<2x4x1x16xf32>
    %253 = sdy.sharding_constraint %252 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %254 = stablehlo.convert %253 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %255 = sdy.sharding_constraint %254 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %cst_22 = stablehlo.constant dense<1.296880e+00> : tensor<bf16>
    %256 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %257 = sdy.sharding_constraint %256 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %258 = stablehlo.multiply %230, %257 : tensor<2x4x2x16xbf16>
    %259 = sdy.sharding_constraint %258 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %260 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %261 = sdy.sharding_constraint %260 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %262 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %263 = sdy.sharding_constraint %262 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %264 = stablehlo.reshape %263 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %265 = sdy.sharding_constraint %264 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %266 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %267 = sdy.sharding_constraint %266 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %269 = sdy.sharding_constraint %268 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %270 = stablehlo.reshape %269 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %271 = sdy.sharding_constraint %270 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_23 = stablehlo.constant dense<2.500000e-01> : tensor<bf16>
    %272 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %273 = sdy.sharding_constraint %272 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %274 = stablehlo.multiply %259, %273 : tensor<2x4x2x16xbf16>
    %275 = sdy.sharding_constraint %274 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %276 = stablehlo.dot_general %275, %265, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x4xbf16>
    %277 = sdy.sharding_constraint %276 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %278 = stablehlo.iota dim = 0 : tensor<4xi32>
    %279 = sdy.sharding_constraint %278 <@mesh, [{}]> : tensor<4xi32>
    %280 = stablehlo.broadcast_in_dim %279, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %281 = sdy.sharding_constraint %280 <@mesh, [{}, {}]> : tensor<4x1xi32>
    %282 = stablehlo.iota dim = 0 : tensor<4xi32>
    %283 = sdy.sharding_constraint %282 <@mesh, [{}]> : tensor<4xi32>
    %284 = stablehlo.broadcast_in_dim %283, dims = [1] : (tensor<4xi32>) -> tensor<1x4xi32>
    %285 = sdy.sharding_constraint %284 <@mesh, [{}, {}]> : tensor<1x4xi32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %287 = sdy.sharding_constraint %286 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %288 = stablehlo.broadcast_in_dim %281, dims = [0, 1] : (tensor<4x1xi32>) -> tensor<4x4xi32>
    %289 = sdy.sharding_constraint %288 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %290 = stablehlo.compare LE, %287, %289, SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    %291 = stablehlo.broadcast_in_dim %290, dims = [1, 2] : (tensor<4x4xi1>) -> tensor<1x4x4xi1>
    %292 = sdy.sharding_constraint %291 <@mesh, [{}, {}, {}]> : tensor<1x4x4xi1>
    %293 = stablehlo.broadcast_in_dim %292, dims = [0, 2, 3] : (tensor<1x4x4xi1>) -> tensor<1x1x4x4xi1>
    %294 = sdy.sharding_constraint %293 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x4x4xi1>
    %cst_24 = stablehlo.constant dense<-9.982440e+08> : tensor<bf16>
    %295 = call @_where(%294, %277, %cst_24) : (tensor<1x1x4x4xi1>, tensor<2x2x4x4xbf16>, tensor<bf16>) -> tensor<2x2x4x4xbf16>
    %296 = stablehlo.convert %295 : (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %297 = sdy.sharding_constraint %296 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_25 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %298 = stablehlo.reduce(%297 init: %cst_25) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %299 = sdy.sharding_constraint %298 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %cst_26 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %300 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<2x2x4xf32>
    %301 = sdy.sharding_constraint %300 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %302 = stablehlo.maximum %301, %299 : tensor<2x2x4xf32>
    %303 = sdy.sharding_constraint %302 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %305 = sdy.sharding_constraint %304 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %307 = sdy.sharding_constraint %306 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %308 = stablehlo.subtract %297, %307 : tensor<2x2x4x4xf32>
    %309 = sdy.sharding_constraint %308 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %310 = stablehlo.exponential %309 : tensor<2x2x4x4xf32>
    %311 = sdy.sharding_constraint %310 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %312 = stablehlo.reduce(%311 init: %cst_27) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %313 = sdy.sharding_constraint %312 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %315 = sdy.sharding_constraint %314 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %317 = sdy.sharding_constraint %316 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %318 = stablehlo.divide %311, %317 : tensor<2x2x4x4xf32>
    %319 = sdy.sharding_constraint %318 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %320 = stablehlo.convert %319 : (tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xbf16>
    %321 = sdy.sharding_constraint %320 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %322 = stablehlo.dot_general %271, %321, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x2x4x4xbf16>) -> tensor<2x2x16x4xbf16>
    %323 = sdy.sharding_constraint %322 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %324 = stablehlo.transpose %323, dims = [0, 3, 1, 2] : (tensor<2x2x16x4xbf16>) -> tensor<2x4x2x16xbf16>
    %325 = sdy.sharding_constraint %324 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %326 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %327 = sdy.sharding_constraint %326 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %329 = sdy.sharding_constraint %328 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %330 = stablehlo.reshape %329 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %331 = sdy.sharding_constraint %330 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %332 = sdy.sharding_constraint %331 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %333 = stablehlo.multiply %325, %332 : tensor<2x4x2x16xbf16>
    %334 = sdy.sharding_constraint %333 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %335 = stablehlo.convert %334 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %336 = sdy.sharding_constraint %335 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %337 = stablehlo.reduce(%336 init: %cst_28) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %338 = sdy.sharding_constraint %337 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %339 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %340 = sdy.sharding_constraint %339 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %341 = stablehlo.convert %340 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %342 = sdy.sharding_constraint %341 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %343 = stablehlo.multiply %332, %332 : tensor<2x4x2x16xbf16>
    %344 = sdy.sharding_constraint %343 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %345 = stablehlo.convert %344 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %346 = sdy.sharding_constraint %345 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %347 = stablehlo.reduce(%346 init: %cst_29) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %348 = sdy.sharding_constraint %347 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %350 = sdy.sharding_constraint %349 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %351 = stablehlo.convert %350 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %352 = sdy.sharding_constraint %351 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_30 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %353 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %354 = sdy.sharding_constraint %353 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %355 = stablehlo.add %352, %354 : tensor<2x4x2x1xbf16>
    %356 = sdy.sharding_constraint %355 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %357 = stablehlo.divide %342, %356 : tensor<2x4x2x1xbf16>
    %358 = sdy.sharding_constraint %357 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %359 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %360 = sdy.sharding_constraint %359 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %361 = stablehlo.multiply %360, %332 : tensor<2x4x2x16xbf16>
    %362 = sdy.sharding_constraint %361 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %363 = stablehlo.subtract %325, %362 : tensor<2x4x2x16xbf16>
    %364 = sdy.sharding_constraint %363 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %365 = stablehlo.dot_general %193, %38, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x2xbf16>) -> tensor<2x4x2xbf16>
    %366 = sdy.sharding_constraint %365 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %367 = stablehlo.negate %366 : tensor<2x4x2xbf16>
    %368 = sdy.sharding_constraint %367 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %369 = stablehlo.exponential %368 : tensor<2x4x2xbf16>
    %370 = sdy.sharding_constraint %369 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_31 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %371 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %372 = sdy.sharding_constraint %371 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %373 = stablehlo.add %372, %370 : tensor<2x4x2xbf16>
    %374 = sdy.sharding_constraint %373 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_32 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %375 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %376 = sdy.sharding_constraint %375 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %377 = stablehlo.divide %376, %374 : tensor<2x4x2xbf16>
    %378 = sdy.sharding_constraint %377 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %379 = stablehlo.broadcast_in_dim %378, dims = [0, 1, 2] : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %380 = sdy.sharding_constraint %379 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_33 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %381 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %382 = sdy.sharding_constraint %381 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %383 = stablehlo.multiply %382, %380 : tensor<2x4x2x1xbf16>
    %384 = sdy.sharding_constraint %383 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %386 = sdy.sharding_constraint %385 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %387 = stablehlo.multiply %386, %364 : tensor<2x4x2x16xbf16>
    %388 = sdy.sharding_constraint %387 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %389 = stablehlo.reshape %388 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x32xbf16>
    %390 = sdy.sharding_constraint %389 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %391 = stablehlo.dot_general %390, %36, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %392 = sdy.sharding_constraint %391 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %393 = stablehlo.add %142, %392 : tensor<2x4x32xbf16>
    %394 = sdy.sharding_constraint %393 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %395 = sdy.sharding_constraint %40 <@mesh, [{}]> : tensor<32xbf16>
    %396 = stablehlo.convert %394 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %397 = sdy.sharding_constraint %396 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %398 = chlo.square %397 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %399 = stablehlo.reduce(%398 init: %cst_34) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %400 = sdy.sharding_constraint %399 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %401 = stablehlo.broadcast_in_dim %400, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %402 = sdy.sharding_constraint %401 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_35 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %403 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %404 = sdy.sharding_constraint %403 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %405 = stablehlo.divide %402, %404 : tensor<2x4x1xf32>
    %406 = sdy.sharding_constraint %405 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_36 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %407 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %408 = sdy.sharding_constraint %407 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %409 = stablehlo.add %406, %408 : tensor<2x4x1xf32>
    %410 = sdy.sharding_constraint %409 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %411 = stablehlo.rsqrt %410 : tensor<2x4x1xf32>
    %412 = sdy.sharding_constraint %411 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %413 = stablehlo.broadcast_in_dim %412, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %414 = sdy.sharding_constraint %413 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %415 = stablehlo.multiply %397, %414 : tensor<2x4x32xf32>
    %416 = sdy.sharding_constraint %415 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %417 = stablehlo.convert %395 : (tensor<32xbf16>) -> tensor<32xf32>
    %418 = sdy.sharding_constraint %417 <@mesh, [{}]> : tensor<32xf32>
    %419 = stablehlo.broadcast_in_dim %418, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %420 = sdy.sharding_constraint %419 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %421 = stablehlo.broadcast_in_dim %420, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %422 = sdy.sharding_constraint %421 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %423 = stablehlo.multiply %416, %422 : tensor<2x4x32xf32>
    %424 = sdy.sharding_constraint %423 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %425 = stablehlo.convert %424 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %426 = sdy.sharding_constraint %425 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %427 = stablehlo.dot_general %426, %42, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %428 = sdy.sharding_constraint %427 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %429 = call @silu_41(%428) : (tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %430 = stablehlo.dot_general %429, %44, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %431 = sdy.sharding_constraint %430 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %432 = stablehlo.negate %431 : tensor<2x4x32xbf16>
    %433 = sdy.sharding_constraint %432 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %434 = stablehlo.exponential %433 : tensor<2x4x32xbf16>
    %435 = sdy.sharding_constraint %434 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %436 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %437 = sdy.sharding_constraint %436 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %438 = stablehlo.add %437, %435 : tensor<2x4x32xbf16>
    %439 = sdy.sharding_constraint %438 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %440 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %441 = sdy.sharding_constraint %440 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %442 = stablehlo.divide %441, %439 : tensor<2x4x32xbf16>
    %443 = sdy.sharding_constraint %442 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %444 = stablehlo.multiply %426, %443 : tensor<2x4x32xbf16>
    %445 = sdy.sharding_constraint %444 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %446 = stablehlo.reshape %445 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %447 = sdy.sharding_constraint %446 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %448 = sdy.sharding_constraint %46 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %449 = stablehlo.dot_general %447, %448, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x4xbf16>) -> tensor<8x4xbf16>
    %450 = sdy.sharding_constraint %449 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xbf16>
    %451 = stablehlo.convert %450 : (tensor<8x4xbf16>) -> tensor<8x4xf32>
    %452 = sdy.sharding_constraint %451 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %453 = stablehlo.convert %48 : (tensor<4xbf16>) -> tensor<4xf32>
    %454 = sdy.sharding_constraint %453 <@mesh, [{}]> : tensor<4xf32>
    %455 = stablehlo.broadcast_in_dim %454, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %456 = sdy.sharding_constraint %455 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<8x4xf32>
    %458 = sdy.sharding_constraint %457 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %459 = stablehlo.add %452, %458 : tensor<8x4xf32>
    %460 = sdy.sharding_constraint %459 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_39 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %461 = stablehlo.reduce(%452 init: %cst_39) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %462 = sdy.sharding_constraint %461 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_40 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %463 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %464 = sdy.sharding_constraint %463 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %465 = stablehlo.maximum %464, %462 : tensor<8xf32>
    %466 = sdy.sharding_constraint %465 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %467 = stablehlo.broadcast_in_dim %466, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %468 = sdy.sharding_constraint %467 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %469 = stablehlo.broadcast_in_dim %468, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %470 = sdy.sharding_constraint %469 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %471 = stablehlo.subtract %452, %470 : tensor<8x4xf32>
    %472 = sdy.sharding_constraint %471 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %473 = stablehlo.exponential %472 : tensor<8x4xf32>
    %474 = sdy.sharding_constraint %473 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %475 = stablehlo.reduce(%474 init: %cst_41) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %476 = sdy.sharding_constraint %475 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %478 = sdy.sharding_constraint %477 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %479 = stablehlo.broadcast_in_dim %478, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %480 = sdy.sharding_constraint %479 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %481 = stablehlo.divide %474, %480 : tensor<8x4xf32>
    %482 = sdy.sharding_constraint %481 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %values, %indices = chlo.top_k(%460, k = 3) : tensor<8x4xf32> -> (tensor<8x3xf32>, tensor<8x3xi32>)
    %483 = sdy.sharding_constraint %values <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xf32>
    %484 = sdy.sharding_constraint %indices <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xi32>
    %485 = stablehlo.slice %483 [0:8, 2:3] : (tensor<8x3xf32>) -> tensor<8x1xf32>
    %486 = sdy.sharding_constraint %485 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %487 = stablehlo.slice %484 [0:8, 0:2] : (tensor<8x3xi32>) -> tensor<8x2xi32>
    %488 = sdy.sharding_constraint %487 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %489 = call @take_along_axis(%452, %488) : (tensor<8x4xf32>, tensor<8x2xi32>) -> tensor<8x2xf32>
    %490 = stablehlo.negate %489 : tensor<8x2xf32>
    %491 = sdy.sharding_constraint %490 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %492 = stablehlo.exponential %491 : tensor<8x2xf32>
    %493 = sdy.sharding_constraint %492 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_42 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %494 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %495 = sdy.sharding_constraint %494 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %496 = stablehlo.add %495, %493 : tensor<8x2xf32>
    %497 = sdy.sharding_constraint %496 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_43 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %498 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %499 = sdy.sharding_constraint %498 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %500 = stablehlo.divide %499, %497 : tensor<8x2xf32>
    %501 = sdy.sharding_constraint %500 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %502 = stablehlo.reduce(%501 init: %cst_44) applies stablehlo.add across dimensions = [1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8xf32>
    %503 = sdy.sharding_constraint %502 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %504 = stablehlo.broadcast_in_dim %503, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %505 = sdy.sharding_constraint %504 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_45 = stablehlo.constant dense<9.99999971E-10> : tensor<f32>
    %506 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %507 = sdy.sharding_constraint %506 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %508 = stablehlo.add %505, %507 : tensor<8x1xf32>
    %509 = sdy.sharding_constraint %508 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_46 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %510 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %511 = sdy.sharding_constraint %510 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %512 = stablehlo.divide %511, %509 : tensor<8x1xf32>
    %513 = sdy.sharding_constraint %512 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %514 = stablehlo.broadcast_in_dim %513, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x2xf32>
    %515 = sdy.sharding_constraint %514 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %516 = stablehlo.multiply %501, %515 : tensor<8x2xf32>
    %517 = sdy.sharding_constraint %516 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %518 = stablehlo.convert %517 : (tensor<8x2xf32>) -> tensor<8x2xbf16>
    %519 = sdy.sharding_constraint %518 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %520 = call @_one_hot(%488) : (tensor<8x2xi32>) -> tensor<8x2x4xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %521 = stablehlo.reduce(%520 init: %cst_47) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x2x4xf32>, tensor<f32>) -> tensor<4xf32>
    %522 = sdy.sharding_constraint %521 <@mesh, [{}]> : tensor<4xf32>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %523 = stablehlo.reduce(%522 init: %cst_48) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %524 = sdy.sharding_constraint %523 <@mesh, []> : tensor<f32>
    %cst_49 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %525 = sdy.sharding_constraint %cst_49 <@mesh, []> : tensor<f32>
    %526 = stablehlo.maximum %524, %525 : tensor<f32>
    %527 = sdy.sharding_constraint %526 <@mesh, []> : tensor<f32>
    %528 = stablehlo.broadcast_in_dim %527, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %529 = sdy.sharding_constraint %528 <@mesh, [{}]> : tensor<4xf32>
    %530 = stablehlo.divide %522, %529 : tensor<4xf32>
    %531 = sdy.sharding_constraint %530 <@mesh, [{}]> : tensor<4xf32>
    %cst_50 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %532 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %533 = sdy.sharding_constraint %532 <@mesh, [{}]> : tensor<4xf32>
    %534 = stablehlo.add %531, %533 : tensor<4xf32>
    %535 = sdy.sharding_constraint %534 <@mesh, [{}]> : tensor<4xf32>
    %536 = stablehlo.log %535 : tensor<4xf32>
    %537 = sdy.sharding_constraint %536 <@mesh, [{}]> : tensor<4xf32>
    %538 = stablehlo.multiply %531, %537 : tensor<4xf32>
    %539 = sdy.sharding_constraint %538 <@mesh, [{}]> : tensor<4xf32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %540 = stablehlo.reduce(%539 init: %cst_51) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %541 = sdy.sharding_constraint %540 <@mesh, []> : tensor<f32>
    %542 = stablehlo.negate %541 : tensor<f32>
    %543 = sdy.sharding_constraint %542 <@mesh, []> : tensor<f32>
    %cst_52 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %544 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %545 = sdy.sharding_constraint %544 <@mesh, [{}]> : tensor<4xf32>
    %546 = stablehlo.multiply %531, %545 : tensor<4xf32>
    %547 = sdy.sharding_constraint %546 <@mesh, [{}]> : tensor<4xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %548 = stablehlo.reduce(%482 init: %cst_53) applies stablehlo.add across dimensions = [0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<4xf32>
    %549 = sdy.sharding_constraint %548 <@mesh, [{}]> : tensor<4xf32>
    %cst_54 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %550 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %551 = sdy.sharding_constraint %550 <@mesh, [{}]> : tensor<4xf32>
    %552 = stablehlo.divide %549, %551 : tensor<4xf32>
    %553 = sdy.sharding_constraint %552 <@mesh, [{}]> : tensor<4xf32>
    %554 = stablehlo.multiply %547, %553 : tensor<4xf32>
    %555 = sdy.sharding_constraint %554 <@mesh, [{}]> : tensor<4xf32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %556 = stablehlo.reduce(%555 init: %cst_55) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %557 = sdy.sharding_constraint %556 <@mesh, []> : tensor<f32>
    %cst_56 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %558 = sdy.sharding_constraint %cst_56 <@mesh, []> : tensor<f32>
    %559 = stablehlo.multiply %558, %557 : tensor<f32>
    %560 = sdy.sharding_constraint %559 <@mesh, []> : tensor<f32>
    %cst_57 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %561 = stablehlo.reduce(%452 init: %cst_57) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %562 = sdy.sharding_constraint %561 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_58 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %563 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %564 = sdy.sharding_constraint %563 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %565 = stablehlo.maximum %564, %562 : tensor<8xf32>
    %566 = sdy.sharding_constraint %565 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %567 = stablehlo.is_finite %566 : (tensor<8xf32>) -> tensor<8xi1>
    %568 = sdy.sharding_constraint %567 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xi1>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %569 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %570 = sdy.sharding_constraint %569 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %571 = stablehlo.select %568, %566, %570 : tensor<8xi1>, tensor<8xf32>
    %572 = sdy.sharding_constraint %571 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %573 = stablehlo.broadcast_in_dim %572, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %574 = sdy.sharding_constraint %573 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %576 = sdy.sharding_constraint %575 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %577 = stablehlo.subtract %452, %576 : tensor<8x4xf32>
    %578 = sdy.sharding_constraint %577 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %579 = stablehlo.exponential %578 : tensor<8x4xf32>
    %580 = sdy.sharding_constraint %579 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %581 = stablehlo.reduce(%580 init: %cst_60) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %582 = sdy.sharding_constraint %581 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %583 = stablehlo.abs %582 : tensor<8xf32>
    %584 = sdy.sharding_constraint %583 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %585 = stablehlo.log %584 : tensor<8xf32>
    %586 = sdy.sharding_constraint %585 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %587 = stablehlo.add %586, %572 : tensor<8xf32>
    %588 = sdy.sharding_constraint %587 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %589 = stablehlo.multiply %588, %588 : tensor<8xf32>
    %590 = sdy.sharding_constraint %589 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %591 = stablehlo.reduce(%590 init: %cst_61) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %592 = sdy.sharding_constraint %591 <@mesh, []> : tensor<f32>
    %cst_62 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %593 = sdy.sharding_constraint %cst_62 <@mesh, []> : tensor<f32>
    %594 = stablehlo.divide %592, %593 : tensor<f32>
    %595 = sdy.sharding_constraint %594 <@mesh, []> : tensor<f32>
    %596 = stablehlo.broadcast_in_dim %486, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %597 = sdy.sharding_constraint %596 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %598 = stablehlo.subtract %452, %597 : tensor<8x4xf32>
    %599 = sdy.sharding_constraint %598 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %600 = sdy.sharding_constraint %599 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %601 = stablehlo.transpose %600, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
    %values_63, %indices_64 = chlo.top_k(%601, k = 4) : tensor<4x8xf32> -> (tensor<4x4xf32>, tensor<4x4xi32>)
    %602 = stablehlo.slice %values_63 [0:4, 3:4] : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %603 = stablehlo.reshape %602 : (tensor<4x1xf32>) -> tensor<4xf32>
    %604 = "stablehlo.all_reduce"(%603) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<f32>, %arg86: tensor<f32>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<f32>
      stablehlo.return %3950 : tensor<f32>
    }) : (tensor<4xf32>) -> tensor<4xf32>
    %cst_65 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %605 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %606 = stablehlo.divide %604, %605 : tensor<4xf32>
    %607 = stablehlo.concatenate %56, %58, dim = 2 : (tensor<4x32x32xbf16>, tensor<4x32x32xbf16>) -> tensor<4x32x64xbf16>
    %608 = sdy.sharding_constraint %607 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x64xbf16>
    %609 = sdy.sharding_constraint %447 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %610 = sdy.sharding_constraint %488 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %611 = sdy.sharding_constraint %519 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %612 = sdy.sharding_constraint %608 <@mesh, [{}, {}, {}]> : tensor<4x32x64xbf16>
    %613 = sdy.sharding_constraint %60 <@mesh, [{}, {}, {}]> : tensor<4x32x32xbf16>
    %614 = stablehlo.reshape %610 : (tensor<8x2xi32>) -> tensor<16xi32>
    %615 = stablehlo.reshape %611 : (tensor<8x2xbf16>) -> tensor<16xbf16>
    %616 = call @argsort(%614) : (tensor<16xi32>) -> tensor<16xi32>
    %617 = stablehlo.iota dim = 0 : tensor<16xi32>
    %c_66 = stablehlo.constant dense<2> : tensor<i32>
    %618 = call @floor_divide(%617, %c_66) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_67 = stablehlo.constant dense<0> : tensor<i32>
    %619 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %620 = stablehlo.compare LT, %616, %619, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_68 = stablehlo.constant dense<16> : tensor<i32>
    %621 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %622 = stablehlo.add %616, %621 : tensor<16xi32>
    %623 = stablehlo.select %620, %622, %616 : tensor<16xi1>, tensor<16xi32>
    %624 = stablehlo.broadcast_in_dim %623, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %625 = "stablehlo.gather"(%618, %624) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xi32>, tensor<16x1xi32>) -> tensor<16xi32>
    %c_69 = stablehlo.constant dense<0> : tensor<i32>
    %626 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %627 = stablehlo.compare LT, %625, %626, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_70 = stablehlo.constant dense<8> : tensor<i32>
    %628 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %629 = stablehlo.add %625, %628 : tensor<16xi32>
    %630 = stablehlo.select %627, %629, %625 : tensor<16xi1>, tensor<16xi32>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %632 = "stablehlo.gather"(%609, %631) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<8x32xbf16>, tensor<16x1xi32>) -> tensor<16x32xbf16>
    %c_71 = stablehlo.constant dense<0> : tensor<i32>
    %633 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %634 = stablehlo.compare LT, %616, %633, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_72 = stablehlo.constant dense<16> : tensor<i32>
    %635 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %636 = stablehlo.add %616, %635 : tensor<16xi32>
    %637 = stablehlo.select %634, %636, %616 : tensor<16xi1>, tensor<16xi32>
    %638 = stablehlo.broadcast_in_dim %637, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %639 = "stablehlo.gather"(%615, %638) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xbf16>, tensor<16x1xi32>) -> tensor<16xbf16>
    %c_73 = stablehlo.constant dense<0> : tensor<i32>
    %640 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %c_74 = stablehlo.constant dense<0> : tensor<i32>
    %641 = call @clip(%614, %c_74) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_75 = stablehlo.constant dense<0> : tensor<i32>
    %642 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %643 = stablehlo.compare LT, %641, %642, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_76 = stablehlo.constant dense<4> : tensor<i32>
    %644 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %645 = stablehlo.add %641, %644 : tensor<16xi32>
    %646 = stablehlo.select %643, %645, %641 : tensor<16xi1>, tensor<16xi32>
    %647 = stablehlo.broadcast_in_dim %646, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %c_77 = stablehlo.constant dense<1> : tensor<i32>
    %648 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %649 = "stablehlo.scatter"(%640, %647, %648) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg85: tensor<i32>, %arg86: tensor<i32>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<i32>
      stablehlo.return %3950 : tensor<i32>
    }) : (tensor<4xi32>, tensor<16x1xi32>, tensor<16xi32>) -> tensor<4xi32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %650 = stablehlo.pad %632, %cst_78, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %651 = stablehlo.broadcast_in_dim %650, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %652 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %653 = call @cumsum(%649) : (tensor<4xi32>) -> tensor<4xi32>
    %c_79 = stablehlo.constant dense<0> : tensor<i32>
    %654 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %655 = stablehlo.slice %654 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %656 = stablehlo.slice %653 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %657 = stablehlo.concatenate %655, %656, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %658 = stablehlo.broadcast_in_dim %653, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %659 = stablehlo.broadcast_in_dim %657, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %660 = stablehlo.compare LE, %659, %652, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %661 = stablehlo.compare LT, %652, %658, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %662 = stablehlo.and %660, %661 : tensor<4x512x32xi1>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %663 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %664 = stablehlo.select %662, %651, %663 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %665 = stablehlo.dot_general %664, %612, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x64xbf16>) -> tensor<512x64xbf16>
    %666 = stablehlo.slice %665 [0:16, 0:64] : (tensor<512x64xbf16>) -> tensor<16x64xbf16>
    %667 = stablehlo.slice %666 [0:16, 0:32] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %668 = stablehlo.slice %666 [0:16, 32:64] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %669 = call @silu_209(%667) : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %670 = stablehlo.multiply %669, %668 : tensor<16x32xbf16>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %671 = stablehlo.pad %670, %cst_81, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %672 = stablehlo.broadcast_in_dim %671, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %673 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %674 = call @cumsum(%649) : (tensor<4xi32>) -> tensor<4xi32>
    %c_82 = stablehlo.constant dense<0> : tensor<i32>
    %675 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %676 = stablehlo.slice %675 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %677 = stablehlo.slice %674 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %678 = stablehlo.concatenate %676, %677, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %679 = stablehlo.broadcast_in_dim %674, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %680 = stablehlo.broadcast_in_dim %678, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %681 = stablehlo.compare LE, %680, %673, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %682 = stablehlo.compare LT, %673, %679, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %683 = stablehlo.and %681, %682 : tensor<4x512x32xi1>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %684 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %685 = stablehlo.select %683, %672, %684 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %686 = stablehlo.dot_general %685, %613, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x32xbf16>) -> tensor<512x32xbf16>
    %687 = stablehlo.slice %686 [0:16, 0:32] : (tensor<512x32xbf16>) -> tensor<16x32xbf16>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %688 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %689 = stablehlo.broadcast_in_dim %639, dims = [0] : (tensor<16xbf16>) -> tensor<16x1xbf16>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1] : (tensor<16x1xbf16>) -> tensor<16x32xbf16>
    %691 = stablehlo.multiply %687, %690 : tensor<16x32xbf16>
    %c_85 = stablehlo.constant dense<0> : tensor<i32>
    %692 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %693 = stablehlo.compare LT, %625, %692, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_86 = stablehlo.constant dense<8> : tensor<i32>
    %694 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %695 = stablehlo.add %625, %694 : tensor<16xi32>
    %696 = stablehlo.select %693, %695, %625 : tensor<16xi1>, tensor<16xi32>
    %697 = stablehlo.broadcast_in_dim %696, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %698 = "stablehlo.scatter"(%688, %697, %691) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<8x32xbf16>, tensor<16x1xi32>, tensor<16x32xbf16>) -> tensor<8x32xbf16>
    %c_87 = stablehlo.constant dense<0> : tensor<i32>
    %699 = stablehlo.convert %c_87 : (tensor<i32>) -> tensor<f32>
    %700 = sdy.sharding_constraint %699 <@mesh, []> : tensor<f32>
    %701 = stablehlo.reshape %698 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %702 = sdy.sharding_constraint %701 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %703 = sdy.sharding_constraint %702 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %704 = stablehlo.reshape %445 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %705 = sdy.sharding_constraint %704 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %706 = stablehlo.dot_general %705, %50, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %707 = sdy.sharding_constraint %706 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %708 = stablehlo.dot_general %705, %52, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %709 = sdy.sharding_constraint %708 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %710 = call @silu_229(%707) : (tensor<8x32xbf16>) -> tensor<8x32xbf16>
    %711 = stablehlo.multiply %710, %709 : tensor<8x32xbf16>
    %712 = sdy.sharding_constraint %711 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %713 = stablehlo.dot_general %712, %54, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %714 = sdy.sharding_constraint %713 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %715 = stablehlo.reshape %714 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %716 = sdy.sharding_constraint %715 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %717 = sdy.sharding_constraint %716 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %718 = stablehlo.add %703, %717 : tensor<2x4x32xbf16>
    %719 = sdy.sharding_constraint %718 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %720 = stablehlo.add %394, %719 : tensor<2x4x32xbf16>
    %721 = sdy.sharding_constraint %720 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %722 = stablehlo.convert %721 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %723 = sdy.sharding_constraint %722 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %724 = chlo.square %723 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %725 = stablehlo.reduce(%724 init: %cst_88) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<f32>
    %726 = sdy.sharding_constraint %725 <@mesh, []> : tensor<f32>
    %cst_89 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %727 = sdy.sharding_constraint %cst_89 <@mesh, []> : tensor<f32>
    %728 = stablehlo.divide %726, %727 : tensor<f32>
    %729 = sdy.sharding_constraint %728 <@mesh, []> : tensor<f32>
    %730 = stablehlo.sqrt %729 : tensor<f32>
    %731 = sdy.sharding_constraint %730 <@mesh, []> : tensor<f32>
    %732 = stablehlo.broadcast_in_dim %543, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %733 = sdy.sharding_constraint %732 <@mesh, [{}]> : tensor<1xf32>
    %734 = stablehlo.broadcast_in_dim %522, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %735 = sdy.sharding_constraint %734 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %736 = stablehlo.broadcast_in_dim %560, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %737 = sdy.sharding_constraint %736 <@mesh, [{}]> : tensor<1xf32>
    %738 = stablehlo.broadcast_in_dim %595, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %739 = sdy.sharding_constraint %738 <@mesh, [{}]> : tensor<1xf32>
    %740 = stablehlo.broadcast_in_dim %606, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %741 = sdy.sharding_constraint %740 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %742 = stablehlo.broadcast_in_dim %700, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %743 = sdy.sharding_constraint %742 <@mesh, [{}]> : tensor<1xf32>
    %744 = stablehlo.broadcast_in_dim %731, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %745 = sdy.sharding_constraint %744 <@mesh, [{}]> : tensor<1xf32>
    %746 = sdy.sharding_constraint %62 <@mesh, [{}]> : tensor<32xbf16>
    %747 = stablehlo.convert %721 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %748 = sdy.sharding_constraint %747 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %749 = chlo.square %748 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_90 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %750 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<2x4x32xf32>
    %751 = sdy.sharding_constraint %750 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %752 = stablehlo.multiply %751, %748 : tensor<2x4x32xf32>
    %753 = sdy.sharding_constraint %752 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %754 = stablehlo.reduce(%749 init: %cst_91) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %755 = sdy.sharding_constraint %754 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %756 = stablehlo.broadcast_in_dim %755, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %757 = sdy.sharding_constraint %756 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_92 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %758 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %759 = sdy.sharding_constraint %758 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %760 = stablehlo.divide %757, %759 : tensor<2x4x1xf32>
    %761 = sdy.sharding_constraint %760 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_93 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %762 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %763 = sdy.sharding_constraint %762 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %764 = stablehlo.add %761, %763 : tensor<2x4x1xf32>
    %765 = sdy.sharding_constraint %764 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %766 = stablehlo.rsqrt %765 : tensor<2x4x1xf32>
    %767 = sdy.sharding_constraint %766 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %768 = stablehlo.divide %767, %765 : tensor<2x4x1xf32>
    %769 = sdy.sharding_constraint %768 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_94 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %770 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %771 = sdy.sharding_constraint %770 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %772 = stablehlo.multiply %771, %769 : tensor<2x4x1xf32>
    %773 = sdy.sharding_constraint %772 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %774 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %775 = sdy.sharding_constraint %774 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %776 = stablehlo.multiply %748, %775 : tensor<2x4x32xf32>
    %777 = sdy.sharding_constraint %776 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %778 = stablehlo.convert %746 : (tensor<32xbf16>) -> tensor<32xf32>
    %779 = sdy.sharding_constraint %778 <@mesh, [{}]> : tensor<32xf32>
    %780 = stablehlo.broadcast_in_dim %779, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %781 = sdy.sharding_constraint %780 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %782 = stablehlo.broadcast_in_dim %781, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %783 = sdy.sharding_constraint %782 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %784 = stablehlo.multiply %777, %783 : tensor<2x4x32xf32>
    %785 = sdy.sharding_constraint %784 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %786 = stablehlo.convert %785 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %787 = sdy.sharding_constraint %786 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %788 = stablehlo.dot_general %787, %64, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %789 = sdy.sharding_constraint %788 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %790:3 = call @silu(%789) : (tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>)
    %791 = stablehlo.dot_general %790#0, %66, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %792 = sdy.sharding_constraint %791 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %793 = stablehlo.negate %792 : tensor<2x4x32xbf16>
    %794 = sdy.sharding_constraint %793 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %795 = stablehlo.exponential %794 : tensor<2x4x32xbf16>
    %796 = sdy.sharding_constraint %795 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_95 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %797 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %798 = sdy.sharding_constraint %797 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %799 = stablehlo.add %798, %796 : tensor<2x4x32xbf16>
    %800 = sdy.sharding_constraint %799 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_96 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %801 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %802 = sdy.sharding_constraint %801 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %803 = stablehlo.divide %802, %800 : tensor<2x4x32xbf16>
    %804 = sdy.sharding_constraint %803 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_97 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %805 = sdy.sharding_constraint %cst_97 <@mesh, []> : tensor<bf16>
    %806 = stablehlo.broadcast_in_dim %805, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %807 = sdy.sharding_constraint %806 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %808 = stablehlo.subtract %807, %804 : tensor<2x4x32xbf16>
    %809 = sdy.sharding_constraint %808 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %810 = stablehlo.multiply %804, %809 : tensor<2x4x32xbf16>
    %811 = sdy.sharding_constraint %810 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %812 = stablehlo.multiply %787, %804 : tensor<2x4x32xbf16>
    %813 = sdy.sharding_constraint %812 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %814 = stablehlo.slice %arg83 [0:2, 1:4] : (tensor<2x4xi32>) -> tensor<2x3xi32>
    %815 = sdy.sharding_constraint %814 <@mesh, [{}, {}]> : tensor<2x3xi32>
    %816 = stablehlo.slice %arg83 [0:2, 0:1] : (tensor<2x4xi32>) -> tensor<2x1xi32>
    %817 = sdy.sharding_constraint %816 <@mesh, [{}, {}]> : tensor<2x1xi32>
    %c_98 = stablehlo.constant dense<0> : tensor<i32>
    %818 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %819 = sdy.sharding_constraint %818 <@mesh, [{}, {}]> : tensor<2x1xi32>
    %820 = stablehlo.multiply %817, %819 : tensor<2x1xi32>
    %821 = sdy.sharding_constraint %820 <@mesh, [{}, {}]> : tensor<2x1xi32>
    %822 = stablehlo.concatenate %815, %821, dim = 1 : (tensor<2x3xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
    %823 = sdy.sharding_constraint %822 <@mesh, [{}, {}]> : tensor<2x4xi32>
    %824 = sdy.sharding_constraint %813 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %825 = sdy.sharding_constraint %22 <@mesh, [{}, {}]> : tensor<32x64xbf16>
    %826 = sdy.sharding_constraint %823 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xi32>
    %827 = sdy.sharding_constraint %arg84 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %828 = stablehlo.reshape %824 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %829 = stablehlo.reshape %826 : (tensor<2x4xi32>) -> tensor<8xi32>
    %830 = stablehlo.reshape %827 : (tensor<2x4xf32>) -> tensor<8xf32>
    %831 = stablehlo.dot_general %828, %825, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x64xbf16>) -> tensor<8x64xbf16>
    %832 = stablehlo.convert %831 : (tensor<8x64xbf16>) -> tensor<8x64xf32>
    %cst_99 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %833 = stablehlo.reduce(%832 init: %cst_99) applies stablehlo.maximum across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %cst_100 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %834 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %835 = stablehlo.maximum %834, %833 : tensor<8xf32>
    %836 = stablehlo.is_finite %835 : (tensor<8xf32>) -> tensor<8xi1>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %837 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %838 = stablehlo.select %836, %835, %837 : tensor<8xi1>, tensor<8xf32>
    %839 = stablehlo.broadcast_in_dim %838, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %840 = stablehlo.broadcast_in_dim %839, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %841 = stablehlo.subtract %832, %840 : tensor<8x64xf32>
    %842 = stablehlo.exponential %841 : tensor<8x64xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %843 = stablehlo.reduce(%842 init: %cst_102) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %844 = stablehlo.abs %843 : tensor<8xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %845 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %846 = stablehlo.compare GE, %843, %845, FLOAT : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xi1>
    %847 = stablehlo.log %844 : tensor<8xf32>
    %848 = stablehlo.add %847, %838 : tensor<8xf32>
    %849 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c_104 = stablehlo.constant dense<0> : tensor<i32>
    %850 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %851 = stablehlo.compare LT, %849, %850, SIGNED : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
    %c_105 = stablehlo.constant dense<8> : tensor<i32>
    %852 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %853 = stablehlo.add %849, %852 : tensor<8xi32>
    %854 = stablehlo.select %851, %853, %849 : tensor<8xi1>, tensor<8xi32>
    %c_106 = stablehlo.constant dense<0> : tensor<i32>
    %855 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %856 = stablehlo.compare LT, %829, %855, SIGNED : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
    %c_107 = stablehlo.constant dense<64> : tensor<i32>
    %857 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %858 = stablehlo.add %829, %857 : tensor<8xi32>
    %859 = stablehlo.select %856, %858, %829 : tensor<8xi1>, tensor<8xi32>
    %860 = stablehlo.broadcast_in_dim %854, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %861 = stablehlo.broadcast_in_dim %859, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %862 = stablehlo.concatenate %860, %861, dim = 1 : (tensor<8x1xi32>, tensor<8x1xi32>) -> tensor<8x2xi32>
    %863 = "stablehlo.gather"(%832, %862) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<8x64xf32>, tensor<8x2xi32>) -> tensor<8xf32>
    %864 = stablehlo.subtract %848, %863 : tensor<8xf32>
    %865 = stablehlo.multiply %864, %830 : tensor<8xf32>
    %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %866 = stablehlo.reduce(%865 init: %cst_108) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %867 = stablehlo.reduce(%830 init: %cst_109) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %868 = "stablehlo.all_reduce"(%866) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<f32>, %arg86: tensor<f32>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<f32>
      stablehlo.return %3950 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %869 = "stablehlo.all_reduce"(%867) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<f32>, %arg86: tensor<f32>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<f32>
      stablehlo.return %3950 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %870 = stablehlo.compare NE, %869, %cst_110, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %871 = stablehlo.divide %868, %869 : tensor<f32>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %872 = call @_where_278(%870, %871, %cst_111) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %873 = stablehlo.broadcast_in_dim %869, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %874 = stablehlo.broadcast_in_dim %870, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %cst_112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %875 = stablehlo.reduce(%739 init: %cst_112) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %876 = sdy.sharding_constraint %875 <@mesh, []> : tensor<f32>
    %cst_113 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %877 = sdy.sharding_constraint %cst_113 <@mesh, []> : tensor<f32>
    %878 = stablehlo.divide %876, %877 : tensor<f32>
    %879 = sdy.sharding_constraint %878 <@mesh, []> : tensor<f32>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %880 = sdy.sharding_constraint %cst_114 <@mesh, []> : tensor<f32>
    %881 = stablehlo.multiply %880, %879 : tensor<f32>
    %882 = sdy.sharding_constraint %881 <@mesh, []> : tensor<f32>
    %883 = stablehlo.add %872, %882 : tensor<f32>
    %884 = sdy.sharding_constraint %883 <@mesh, []> : tensor<f32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %885 = stablehlo.reduce(%735 init: %cst_115) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
    %886 = sdy.sharding_constraint %885 <@mesh, [{}]> : tensor<1xf32>
    %cst_116 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %887 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %888 = sdy.sharding_constraint %887 <@mesh, [{}]> : tensor<1xf32>
    %889 = stablehlo.maximum %886, %888 : tensor<1xf32>
    %890 = sdy.sharding_constraint %889 <@mesh, [{}]> : tensor<1xf32>
    %891 = stablehlo.divide %743, %890 : tensor<1xf32>
    %892 = sdy.sharding_constraint %891 <@mesh, [{}]> : tensor<1xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %893 = stablehlo.reduce(%733 init: %cst_117) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %894 = sdy.sharding_constraint %893 <@mesh, []> : tensor<f32>
    %cst_118 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %895 = sdy.sharding_constraint %cst_118 <@mesh, []> : tensor<f32>
    %896 = stablehlo.divide %894, %895 : tensor<f32>
    %897 = sdy.sharding_constraint %896 <@mesh, []> : tensor<f32>
    %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %898 = stablehlo.reduce(%737 init: %cst_119) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %899 = sdy.sharding_constraint %898 <@mesh, []> : tensor<f32>
    %cst_120 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %900 = sdy.sharding_constraint %cst_120 <@mesh, []> : tensor<f32>
    %901 = stablehlo.divide %899, %900 : tensor<f32>
    %902 = sdy.sharding_constraint %901 <@mesh, []> : tensor<f32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %903 = stablehlo.reduce(%739 init: %cst_121) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %904 = sdy.sharding_constraint %903 <@mesh, []> : tensor<f32>
    %cst_122 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %905 = sdy.sharding_constraint %cst_122 <@mesh, []> : tensor<f32>
    %906 = stablehlo.divide %904, %905 : tensor<f32>
    %907 = sdy.sharding_constraint %906 <@mesh, []> : tensor<f32>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %908 = stablehlo.reduce(%892 init: %cst_123) applies stablehlo.add across dimensions = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<f32>
    %909 = sdy.sharding_constraint %908 <@mesh, []> : tensor<f32>
    %cst_124 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %910 = sdy.sharding_constraint %cst_124 <@mesh, []> : tensor<f32>
    %911 = stablehlo.divide %909, %910 : tensor<f32>
    %912 = sdy.sharding_constraint %911 <@mesh, []> : tensor<f32>
    %913 = stablehlo.reshape %733 : (tensor<1xf32>) -> tensor<f32>
    %914 = sdy.sharding_constraint %913 <@mesh, []> : tensor<f32>
    %915 = stablehlo.reshape %737 : (tensor<1xf32>) -> tensor<f32>
    %916 = sdy.sharding_constraint %915 <@mesh, []> : tensor<f32>
    %917 = stablehlo.reshape %739 : (tensor<1xf32>) -> tensor<f32>
    %918 = sdy.sharding_constraint %917 <@mesh, []> : tensor<f32>
    %919 = stablehlo.reshape %735 : (tensor<1x4xf32>) -> tensor<4xf32>
    %920 = sdy.sharding_constraint %919 <@mesh, [{}]> : tensor<4xf32>
    %921 = stablehlo.iota dim = 0 : tensor<4xf32>
    %922 = sdy.sharding_constraint %921 <@mesh, [{}]> : tensor<4xf32>
    %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %923 = stablehlo.reduce(%920 init: %cst_125) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %924 = sdy.sharding_constraint %923 <@mesh, []> : tensor<f32>
    %925 = stablehlo.multiply %920, %922 : tensor<4xf32>
    %926 = sdy.sharding_constraint %925 <@mesh, [{}]> : tensor<4xf32>
    %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %927 = stablehlo.reduce(%926 init: %cst_126) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %928 = sdy.sharding_constraint %927 <@mesh, []> : tensor<f32>
    %929 = stablehlo.multiply %920, %922 : tensor<4xf32>
    %930 = sdy.sharding_constraint %929 <@mesh, [{}]> : tensor<4xf32>
    %931 = stablehlo.multiply %930, %922 : tensor<4xf32>
    %932 = sdy.sharding_constraint %931 <@mesh, [{}]> : tensor<4xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %933 = stablehlo.reduce(%932 init: %cst_127) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %934 = sdy.sharding_constraint %933 <@mesh, []> : tensor<f32>
    %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %935 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %936 = sdy.sharding_constraint %935 <@mesh, [{}]> : tensor<4xf32>
    %937 = stablehlo.compare GT, %920, %936, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %cst_129 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %938 = call @_where_289(%937, %922, %cst_129) : (tensor<4xi1>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    %cst_130 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %939 = stablehlo.reduce(%938 init: %cst_130) applies stablehlo.minimum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %940 = sdy.sharding_constraint %939 <@mesh, []> : tensor<f32>
    %cst_131 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %941 = call @_where_289(%937, %922, %cst_131) : (tensor<4xi1>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    %cst_132 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %942 = stablehlo.reduce(%941 init: %cst_132) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %943 = sdy.sharding_constraint %942 <@mesh, []> : tensor<f32>
    %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %944 = sdy.sharding_constraint %cst_133 <@mesh, []> : tensor<f32>
    %945 = stablehlo.compare GT, %924, %944, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %946 = call @_where_295(%945, %940, %cst_134) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %947 = sdy.sharding_constraint %cst_135 <@mesh, []> : tensor<f32>
    %948 = stablehlo.compare GT, %924, %947, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %949 = call @_where_295(%948, %943, %cst_136) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %950 = stablehlo.iota dim = 0 : tensor<5xf32>
    %951 = sdy.sharding_constraint %950 <@mesh, [{}]> : tensor<5xf32>
    %952 = stablehlo.convert %937 : (tensor<4xi1>) -> tensor<4xi32>
    %953 = sdy.sharding_constraint %952 <@mesh, [{}]> : tensor<4xi32>
    %c_137 = stablehlo.constant dense<0> : tensor<i32>
    %954 = stablehlo.reduce(%953 init: %c_137) applies stablehlo.add across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
    %955 = sdy.sharding_constraint %954 <@mesh, []> : tensor<i32>
    %956 = stablehlo.divide %928, %924 : tensor<f32>
    %957 = sdy.sharding_constraint %956 <@mesh, []> : tensor<f32>
    %958 = stablehlo.divide %934, %924 : tensor<f32>
    %959 = sdy.sharding_constraint %958 <@mesh, []> : tensor<f32>
    %960 = stablehlo.multiply %957, %957 : tensor<f32>
    %961 = sdy.sharding_constraint %960 <@mesh, []> : tensor<f32>
    %962 = stablehlo.subtract %959, %961 : tensor<f32>
    %963 = sdy.sharding_constraint %962 <@mesh, []> : tensor<f32>
    %964 = stablehlo.divide %934, %924 : tensor<f32>
    %965 = sdy.sharding_constraint %964 <@mesh, []> : tensor<f32>
    %966 = stablehlo.sqrt %965 : tensor<f32>
    %967 = sdy.sharding_constraint %966 <@mesh, []> : tensor<f32>
    %968 = stablehlo.reshape %892 : (tensor<1xf32>) -> tensor<f32>
    %969 = sdy.sharding_constraint %968 <@mesh, []> : tensor<f32>
    %970 = stablehlo.reshape %745 : (tensor<1xf32>) -> tensor<f32>
    %971 = sdy.sharding_constraint %970 <@mesh, []> : tensor<f32>
    %cst_138 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %972 = sdy.sharding_constraint %cst_138 <@mesh, []> : tensor<f32>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %973 = sdy.sharding_constraint %cst_139 <@mesh, []> : tensor<f32>
    %974 = stablehlo.multiply %973, %972 : tensor<f32>
    %975 = sdy.sharding_constraint %974 <@mesh, []> : tensor<f32>
    %cst_140 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %976 = sdy.sharding_constraint %cst_140 <@mesh, []> : tensor<f32>
    %977 = stablehlo.divide %975, %976 : tensor<f32>
    %978 = sdy.sharding_constraint %977 <@mesh, []> : tensor<f32>
    %979 = stablehlo.broadcast_in_dim %978, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %980 = sdy.sharding_constraint %979 <@mesh, [{}]> : tensor<1xf32>
    %981 = stablehlo.reshape %873 : (tensor<1xf32>) -> tensor<f32>
    %982 = stablehlo.reshape %874 : (tensor<1xi1>) -> tensor<i1>
    %983 = call @_where_308(%982, %972) : (tensor<i1>, tensor<f32>) -> tensor<f32>
    %984 = stablehlo.divide %983, %981 : tensor<f32>
    %985 = "stablehlo.all_reduce"(%984) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<f32>, %arg86: tensor<f32>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<f32>
      stablehlo.return %3950 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %986 = stablehlo.broadcast_in_dim %985, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %987 = stablehlo.multiply %986, %830 : tensor<8xf32>
    %988 = stablehlo.negate %987 : tensor<8xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %989 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<8x64xf32>
    %990 = "stablehlo.scatter"(%989, %862, %988) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg85: tensor<f32>, %arg86: tensor<f32>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<f32>
      stablehlo.return %3950 : tensor<f32>
    }) : (tensor<8x64xf32>, tensor<8x2xi32>, tensor<8xf32>) -> tensor<8x64xf32>
    %991 = stablehlo.divide %987, %844 : tensor<8xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %992 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %993 = stablehlo.select %846, %992, %991 : tensor<8xi1>, tensor<8xf32>
    %994 = stablehlo.select %846, %991, %992 : tensor<8xi1>, tensor<8xf32>
    %995 = stablehlo.negate %993 : tensor<8xf32>
    %996 = stablehlo.add %994, %995 : tensor<8xf32>
    %997 = stablehlo.broadcast_in_dim %996, dims = [0] : (tensor<8xf32>) -> tensor<8x64xf32>
    %998 = stablehlo.multiply %997, %842 : tensor<8x64xf32>
    %999 = stablehlo.add %990, %998 : tensor<8x64xf32>
    %1000 = stablehlo.convert %999 : (tensor<8x64xf32>) -> tensor<8x64xbf16>
    %1001 = stablehlo.dot_general %1000, %828, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xbf16>, tensor<8x32xbf16>) -> tensor<64x32xbf16>
    %1002 = stablehlo.transpose %1001, dims = [1, 0] : (tensor<64x32xbf16>) -> tensor<32x64xbf16>
    %1003 = stablehlo.dot_general %1000, %825, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x64xbf16>, tensor<32x64xbf16>) -> tensor<8x32xbf16>
    %1004 = stablehlo.reshape %1003 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %1005 = "stablehlo.all_reduce"(%1004) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<2x4x32xbf16>) -> tensor<2x4x32xbf16>
    %1006 = "stablehlo.all_reduce"(%1002) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %1007 = sdy.sharding_constraint %1006 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xbf16>
    %1008 = sdy.sharding_constraint %1005 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1009 = stablehlo.multiply %787, %1008 : tensor<2x4x32xbf16>
    %1010 = sdy.sharding_constraint %1009 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1011 = stablehlo.multiply %1008, %804 : tensor<2x4x32xbf16>
    %1012 = sdy.sharding_constraint %1011 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1013 = stablehlo.multiply %1010, %811 : tensor<2x4x32xbf16>
    %1014 = sdy.sharding_constraint %1013 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1015 = stablehlo.dot_general %1014, %790#0, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %1016 = sdy.sharding_constraint %1015 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1017 = stablehlo.transpose %1016, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %1018 = sdy.sharding_constraint %1017 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1019 = stablehlo.dot_general %1014, %66, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %1020 = sdy.sharding_constraint %1019 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1021 = call @silu_329(%790#1, %790#2, %789, %1020) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %1022 = stablehlo.dot_general %1021, %787, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %1023 = sdy.sharding_constraint %1022 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1024 = stablehlo.transpose %1023, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %1025 = sdy.sharding_constraint %1024 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1026 = stablehlo.dot_general %1021, %64, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %1027 = sdy.sharding_constraint %1026 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1028 = stablehlo.add %1012, %1027 : tensor<2x4x32xbf16>
    %1029 = sdy.sharding_constraint %1028 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1030 = stablehlo.convert %1029 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1031 = sdy.sharding_constraint %1030 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1032 = stablehlo.multiply %777, %1031 : tensor<2x4x32xf32>
    %1033 = sdy.sharding_constraint %1032 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1034 = stablehlo.reduce(%1033 init: %cst_143) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1035 = sdy.sharding_constraint %1034 <@mesh, [{}]> : tensor<32xf32>
    %1036 = stablehlo.reshape %1035 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1037 = sdy.sharding_constraint %1036 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1038 = stablehlo.broadcast_in_dim %781, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1039 = sdy.sharding_constraint %1038 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1040 = stablehlo.multiply %1031, %1039 : tensor<2x4x32xf32>
    %1041 = sdy.sharding_constraint %1040 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1042 = stablehlo.reduce(%1037 init: %cst_144) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1043 = sdy.sharding_constraint %1042 <@mesh, [{}]> : tensor<32xf32>
    %1044 = stablehlo.convert %1043 : (tensor<32xf32>) -> tensor<32xbf16>
    %1045 = sdy.sharding_constraint %1044 <@mesh, [{}]> : tensor<32xbf16>
    %1046 = stablehlo.multiply %748, %1041 : tensor<2x4x32xf32>
    %1047 = sdy.sharding_constraint %1046 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1048 = stablehlo.reduce(%1047 init: %cst_145) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1049 = sdy.sharding_constraint %1048 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1050 = stablehlo.reshape %1049 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1051 = sdy.sharding_constraint %1050 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1052 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1053 = sdy.sharding_constraint %1052 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1054 = stablehlo.multiply %1041, %1053 : tensor<2x4x32xf32>
    %1055 = sdy.sharding_constraint %1054 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1056 = stablehlo.multiply %1051, %773 : tensor<2x4x1xf32>
    %1057 = sdy.sharding_constraint %1056 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_146 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1058 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1059 = sdy.sharding_constraint %1058 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1060 = stablehlo.divide %1057, %1059 : tensor<2x4x1xf32>
    %1061 = sdy.sharding_constraint %1060 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1062 = stablehlo.reduce(%1061 init: %cst_147) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1063 = sdy.sharding_constraint %1062 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1064 = stablehlo.broadcast_in_dim %1063, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %1065 = sdy.sharding_constraint %1064 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1066 = stablehlo.multiply %1065, %753 : tensor<2x4x32xf32>
    %1067 = sdy.sharding_constraint %1066 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1068 = stablehlo.add %1055, %1067 : tensor<2x4x32xf32>
    %1069 = sdy.sharding_constraint %1068 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1070 = stablehlo.convert %1069 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %1071 = sdy.sharding_constraint %1070 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1072 = sdy.sharding_constraint %1045 <@mesh, [{}]> : tensor<32xbf16>
    %1073 = stablehlo.slice %980 [0:1] : (tensor<1xf32>) -> tensor<1xf32>
    %1074 = stablehlo.reshape %1073 : (tensor<1xf32>) -> tensor<f32>
    %1075 = sdy.sharding_constraint %1074 <@mesh, []> : tensor<f32>
    %1076:22 = stablehlo.optimization_barrier %24, %26, %28, %30, %32, %34, %38, %36, %40, %42, %44, %46, %48, %50, %52, %54, %142, %56, %58, %60, %1071, %1075 : tensor<32xbf16>, tensor<32x128xbf16>, tensor<128x32xbf16>, tensor<32x32xbf16>, tensor<32x16xbf16>, tensor<32x16xbf16>, tensor<32x2xbf16>, tensor<32x32xbf16>, tensor<32xbf16>, tensor<32x128xbf16>, tensor<128x32xbf16>, tensor<32x4xbf16>, tensor<4xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<2x4x32xbf16>, tensor<4x32x32xbf16>, tensor<4x32x32xbf16>, tensor<4x32x32xbf16>, tensor<2x4x32xbf16>, tensor<f32>
    %1077 = sdy.sharding_constraint %1076#0 <@mesh, [{}]> : tensor<32xbf16>
    %1078 = stablehlo.convert %1076#16 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1079 = sdy.sharding_constraint %1078 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1080 = chlo.square %1079 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_148 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1081 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<2x4x32xf32>
    %1082 = sdy.sharding_constraint %1081 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1083 = stablehlo.multiply %1082, %1079 : tensor<2x4x32xf32>
    %1084 = sdy.sharding_constraint %1083 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1085 = stablehlo.reduce(%1080 init: %cst_149) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1086 = sdy.sharding_constraint %1085 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1087 = stablehlo.broadcast_in_dim %1086, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1088 = sdy.sharding_constraint %1087 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_150 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1089 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1090 = sdy.sharding_constraint %1089 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1091 = stablehlo.divide %1088, %1090 : tensor<2x4x1xf32>
    %1092 = sdy.sharding_constraint %1091 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_151 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1093 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1094 = sdy.sharding_constraint %1093 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1095 = stablehlo.add %1092, %1094 : tensor<2x4x1xf32>
    %1096 = sdy.sharding_constraint %1095 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1097 = stablehlo.rsqrt %1096 : tensor<2x4x1xf32>
    %1098 = sdy.sharding_constraint %1097 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1099 = stablehlo.divide %1098, %1096 : tensor<2x4x1xf32>
    %1100 = sdy.sharding_constraint %1099 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_152 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1101 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1102 = sdy.sharding_constraint %1101 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1103 = stablehlo.multiply %1102, %1100 : tensor<2x4x1xf32>
    %1104 = sdy.sharding_constraint %1103 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1105 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1106 = sdy.sharding_constraint %1105 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1107 = stablehlo.multiply %1079, %1106 : tensor<2x4x32xf32>
    %1108 = sdy.sharding_constraint %1107 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1109 = stablehlo.convert %1077 : (tensor<32xbf16>) -> tensor<32xf32>
    %1110 = sdy.sharding_constraint %1109 <@mesh, [{}]> : tensor<32xf32>
    %1111 = stablehlo.broadcast_in_dim %1110, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1112 = sdy.sharding_constraint %1111 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1113 = stablehlo.broadcast_in_dim %1112, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1114 = sdy.sharding_constraint %1113 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1115 = stablehlo.multiply %1108, %1114 : tensor<2x4x32xf32>
    %1116 = sdy.sharding_constraint %1115 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1117 = stablehlo.convert %1116 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %1118 = sdy.sharding_constraint %1117 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1119 = stablehlo.dot_general %1118, %1076#1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %1120 = sdy.sharding_constraint %1119 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1121:3 = call @silu_347(%1120) : (tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>)
    %1122 = stablehlo.dot_general %1121#0, %1076#2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %1123 = sdy.sharding_constraint %1122 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1124 = stablehlo.negate %1123 : tensor<2x4x32xbf16>
    %1125 = sdy.sharding_constraint %1124 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1126 = stablehlo.exponential %1125 : tensor<2x4x32xbf16>
    %1127 = sdy.sharding_constraint %1126 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_153 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1128 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1129 = sdy.sharding_constraint %1128 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1130 = stablehlo.add %1129, %1127 : tensor<2x4x32xbf16>
    %1131 = sdy.sharding_constraint %1130 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_154 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1132 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1133 = sdy.sharding_constraint %1132 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1134 = stablehlo.divide %1133, %1131 : tensor<2x4x32xbf16>
    %1135 = sdy.sharding_constraint %1134 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_155 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1136 = sdy.sharding_constraint %cst_155 <@mesh, []> : tensor<bf16>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1138 = sdy.sharding_constraint %1137 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1139 = stablehlo.subtract %1138, %1135 : tensor<2x4x32xbf16>
    %1140 = sdy.sharding_constraint %1139 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1141 = stablehlo.multiply %1135, %1140 : tensor<2x4x32xbf16>
    %1142 = sdy.sharding_constraint %1141 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1143 = stablehlo.multiply %1118, %1135 : tensor<2x4x32xbf16>
    %1144 = sdy.sharding_constraint %1143 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1145 = stablehlo.dot_general %1144, %1076#3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %1146 = sdy.sharding_constraint %1145 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %1147 = stablehlo.reshape %1146 : (tensor<2x4x32xbf16>) -> tensor<2x4x2x16xbf16>
    %1148 = sdy.sharding_constraint %1147 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1149 = stablehlo.dot_general %1144, %1076#4, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %1150 = sdy.sharding_constraint %1149 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %1151 = stablehlo.reshape %1150 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %1152 = sdy.sharding_constraint %1151 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %1153 = stablehlo.dot_general %1144, %1076#5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x16xbf16>) -> tensor<2x4x16xbf16>
    %1154 = sdy.sharding_constraint %1153 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %1155 = stablehlo.reshape %1154 : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %1156 = sdy.sharding_constraint %1155 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %1157 = stablehlo.convert %1148 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1158 = sdy.sharding_constraint %1157 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1159 = chlo.square %1158 : tensor<2x4x2x16xf32> -> tensor<2x4x2x16xf32>
    %cst_156 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1160 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f32>) -> tensor<2x4x2x16xf32>
    %1161 = sdy.sharding_constraint %1160 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1162 = stablehlo.multiply %1161, %1158 : tensor<2x4x2x16xf32>
    %1163 = sdy.sharding_constraint %1162 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1164 = stablehlo.reduce(%1159 init: %cst_157) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %1165 = sdy.sharding_constraint %1164 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %1166 = stablehlo.broadcast_in_dim %1165, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %1167 = sdy.sharding_constraint %1166 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_158 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %1168 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %1169 = sdy.sharding_constraint %1168 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1170 = stablehlo.divide %1167, %1169 : tensor<2x4x2x1xf32>
    %1171 = sdy.sharding_constraint %1170 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_159 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1172 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %1173 = sdy.sharding_constraint %1172 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1174 = stablehlo.add %1171, %1173 : tensor<2x4x2x1xf32>
    %1175 = sdy.sharding_constraint %1174 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1176 = stablehlo.rsqrt %1175 : tensor<2x4x2x1xf32>
    %1177 = sdy.sharding_constraint %1176 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1178 = stablehlo.divide %1177, %1175 : tensor<2x4x2x1xf32>
    %1179 = sdy.sharding_constraint %1178 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_160 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1180 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %1181 = sdy.sharding_constraint %1180 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1182 = stablehlo.multiply %1181, %1179 : tensor<2x4x2x1xf32>
    %1183 = sdy.sharding_constraint %1182 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %1184 = stablehlo.convert %1148 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1185 = sdy.sharding_constraint %1184 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1186 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x16xf32>
    %1187 = sdy.sharding_constraint %1186 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1188 = stablehlo.multiply %1185, %1187 : tensor<2x4x2x16xf32>
    %1189 = sdy.sharding_constraint %1188 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %1190 = stablehlo.convert %1189 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %1191 = sdy.sharding_constraint %1190 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1192 = stablehlo.convert %1152 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %1193 = sdy.sharding_constraint %1192 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1194 = chlo.square %1193 : tensor<2x4x1x16xf32> -> tensor<2x4x1x16xf32>
    %cst_161 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1195 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<2x4x1x16xf32>
    %1196 = sdy.sharding_constraint %1195 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1197 = stablehlo.multiply %1196, %1193 : tensor<2x4x1x16xf32>
    %1198 = sdy.sharding_constraint %1197 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1199 = stablehlo.reduce(%1194 init: %cst_162) applies stablehlo.add across dimensions = [3] : (tensor<2x4x1x16xf32>, tensor<f32>) -> tensor<2x4x1xf32>
    %1200 = sdy.sharding_constraint %1199 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1201 = stablehlo.broadcast_in_dim %1200, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x1x1xf32>
    %1202 = sdy.sharding_constraint %1201 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_163 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %1203 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %1204 = sdy.sharding_constraint %1203 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1205 = stablehlo.divide %1202, %1204 : tensor<2x4x1x1xf32>
    %1206 = sdy.sharding_constraint %1205 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_164 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %1207 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %1208 = sdy.sharding_constraint %1207 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1209 = stablehlo.add %1206, %1208 : tensor<2x4x1x1xf32>
    %1210 = sdy.sharding_constraint %1209 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1211 = stablehlo.rsqrt %1210 : tensor<2x4x1x1xf32>
    %1212 = sdy.sharding_constraint %1211 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1213 = stablehlo.divide %1212, %1210 : tensor<2x4x1x1xf32>
    %1214 = sdy.sharding_constraint %1213 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_165 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1215 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %1216 = sdy.sharding_constraint %1215 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1217 = stablehlo.multiply %1216, %1214 : tensor<2x4x1x1xf32>
    %1218 = sdy.sharding_constraint %1217 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %1219 = stablehlo.convert %1152 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %1220 = sdy.sharding_constraint %1219 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1221 = stablehlo.broadcast_in_dim %1212, dims = [0, 1, 2, 3] : (tensor<2x4x1x1xf32>) -> tensor<2x4x1x16xf32>
    %1222 = sdy.sharding_constraint %1221 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1223 = stablehlo.multiply %1220, %1222 : tensor<2x4x1x16xf32>
    %1224 = sdy.sharding_constraint %1223 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %1225 = stablehlo.convert %1224 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %1226 = sdy.sharding_constraint %1225 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %cst_166 = stablehlo.constant dense<1.296880e+00> : tensor<bf16>
    %1227 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %1228 = sdy.sharding_constraint %1227 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1229 = stablehlo.multiply %1191, %1228 : tensor<2x4x2x16xbf16>
    %1230 = sdy.sharding_constraint %1229 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1231 = stablehlo.broadcast_in_dim %1226, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %1232 = sdy.sharding_constraint %1231 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %1233 = stablehlo.broadcast_in_dim %1232, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %1234 = sdy.sharding_constraint %1233 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %1235 = stablehlo.reshape %1234 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %1236 = sdy.sharding_constraint %1235 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1237 = stablehlo.broadcast_in_dim %1156, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %1238 = sdy.sharding_constraint %1237 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %1239 = stablehlo.broadcast_in_dim %1238, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %1240 = sdy.sharding_constraint %1239 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %1241 = stablehlo.reshape %1240 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %1242 = sdy.sharding_constraint %1241 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_167 = stablehlo.constant dense<2.500000e-01> : tensor<bf16>
    %1243 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %1244 = sdy.sharding_constraint %1243 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1245 = stablehlo.multiply %1230, %1244 : tensor<2x4x2x16xbf16>
    %1246 = sdy.sharding_constraint %1245 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %1247 = stablehlo.dot_general %1246, %1236, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x4xbf16>
    %1248 = sdy.sharding_constraint %1247 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %1249 = stablehlo.iota dim = 0 : tensor<4xi32>
    %1250 = sdy.sharding_constraint %1249 <@mesh, [{}]> : tensor<4xi32>
    %1251 = stablehlo.broadcast_in_dim %1250, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %1252 = sdy.sharding_constraint %1251 <@mesh, [{}, {}]> : tensor<4x1xi32>
    %1253 = stablehlo.iota dim = 0 : tensor<4xi32>
    %1254 = sdy.sharding_constraint %1253 <@mesh, [{}]> : tensor<4xi32>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [1] : (tensor<4xi32>) -> tensor<1x4xi32>
    %1256 = sdy.sharding_constraint %1255 <@mesh, [{}, {}]> : tensor<1x4xi32>
    %1257 = stablehlo.broadcast_in_dim %1256, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<4x4xi32>
    %1258 = sdy.sharding_constraint %1257 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %1259 = stablehlo.broadcast_in_dim %1252, dims = [0, 1] : (tensor<4x1xi32>) -> tensor<4x4xi32>
    %1260 = sdy.sharding_constraint %1259 <@mesh, [{}, {}]> : tensor<4x4xi32>
    %1261 = stablehlo.compare LE, %1258, %1260, SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    %1262 = stablehlo.broadcast_in_dim %1261, dims = [1, 2] : (tensor<4x4xi1>) -> tensor<1x4x4xi1>
    %1263 = sdy.sharding_constraint %1262 <@mesh, [{}, {}, {}]> : tensor<1x4x4xi1>
    %1264 = stablehlo.broadcast_in_dim %1263, dims = [0, 2, 3] : (tensor<1x4x4xi1>) -> tensor<1x1x4x4xi1>
    %1265 = sdy.sharding_constraint %1264 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x4x4xi1>
    %cst_168 = stablehlo.constant dense<-9.982440e+08> : tensor<bf16>
    %1266:2 = call @_where_354(%1265, %1248, %cst_168) : (tensor<1x1x4x4xi1>, tensor<2x2x4x4xbf16>, tensor<bf16>) -> (tensor<2x2x4x4xbf16>, tensor<2x2x4x4xi1>)
    %1267 = stablehlo.convert %1266#0 : (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %1268 = sdy.sharding_constraint %1267 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_169 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1269 = stablehlo.reduce(%1268 init: %cst_169) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %1270 = sdy.sharding_constraint %1269 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %cst_170 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1271 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<2x2x4xf32>
    %1272 = sdy.sharding_constraint %1271 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %1273 = stablehlo.maximum %1272, %1270 : tensor<2x2x4xf32>
    %1274 = sdy.sharding_constraint %1273 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %1275 = stablehlo.broadcast_in_dim %1274, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %1276 = sdy.sharding_constraint %1275 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1277 = stablehlo.broadcast_in_dim %1276, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %1278 = sdy.sharding_constraint %1277 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1279 = stablehlo.subtract %1268, %1278 : tensor<2x2x4x4xf32>
    %1280 = sdy.sharding_constraint %1279 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1281 = stablehlo.exponential %1280 : tensor<2x2x4x4xf32>
    %1282 = sdy.sharding_constraint %1281 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1283 = stablehlo.reduce(%1282 init: %cst_171) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %1284 = sdy.sharding_constraint %1283 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %1285 = stablehlo.broadcast_in_dim %1284, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %1286 = sdy.sharding_constraint %1285 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %1288 = sdy.sharding_constraint %1287 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1289 = stablehlo.divide %1282, %1288 : tensor<2x2x4x4xf32>
    %1290 = sdy.sharding_constraint %1289 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %1291 = stablehlo.multiply %1286, %1286 : tensor<2x2x4x1xf32>
    %1292 = sdy.sharding_constraint %1291 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %cst_172 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1293 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<2x2x4x1xf32>
    %1294 = sdy.sharding_constraint %1293 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1295 = stablehlo.divide %1294, %1292 : tensor<2x2x4x1xf32>
    %1296 = sdy.sharding_constraint %1295 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1297 = sdy.sharding_constraint %1296 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %1298 = stablehlo.convert %1290 : (tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xbf16>
    %1299 = sdy.sharding_constraint %1298 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %1300 = stablehlo.dot_general %1242, %1299, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2x16xbf16>, tensor<2x2x4x4xbf16>) -> tensor<2x2x16x4xbf16>
    %1301 = sdy.sharding_constraint %1300 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %1302 = stablehlo.transpose %1301, dims = [0, 3, 1, 2] : (tensor<2x2x16x4xbf16>) -> tensor<2x4x2x16xbf16>
    %1303 = sdy.sharding_constraint %1302 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1304 = stablehlo.broadcast_in_dim %1156, dims = [0, 1, 2, 4] : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %1305 = sdy.sharding_constraint %1304 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %1306 = stablehlo.broadcast_in_dim %1305, dims = [0, 1, 2, 3, 4] : (tensor<2x4x1x1x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %1307 = sdy.sharding_constraint %1306 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %1308 = stablehlo.reshape %1307 : (tensor<2x4x1x2x16xbf16>) -> tensor<2x4x2x16xbf16>
    %1309 = sdy.sharding_constraint %1308 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1310 = sdy.sharding_constraint %1309 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1311 = stablehlo.multiply %1303, %1310 : tensor<2x4x2x16xbf16>
    %1312 = sdy.sharding_constraint %1311 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1313 = stablehlo.convert %1312 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1314 = sdy.sharding_constraint %1313 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1315 = stablehlo.reduce(%1314 init: %cst_173) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %1316 = sdy.sharding_constraint %1315 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %1317 = stablehlo.broadcast_in_dim %1316, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %1318 = sdy.sharding_constraint %1317 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %1319 = stablehlo.convert %1318 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %1320 = sdy.sharding_constraint %1319 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1321 = stablehlo.multiply %1310, %1310 : tensor<2x4x2x16xbf16>
    %1322 = sdy.sharding_constraint %1321 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1323 = stablehlo.convert %1322 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %1324 = sdy.sharding_constraint %1323 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1325 = stablehlo.reduce(%1324 init: %cst_174) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %1326 = sdy.sharding_constraint %1325 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %1327 = stablehlo.broadcast_in_dim %1326, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %1328 = sdy.sharding_constraint %1327 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %1329 = stablehlo.convert %1328 : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x1xbf16>
    %1330 = sdy.sharding_constraint %1329 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_175 = stablehlo.constant dense<9.983770e-07> : tensor<bf16>
    %1331 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %1332 = sdy.sharding_constraint %1331 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1333 = stablehlo.add %1330, %1332 : tensor<2x4x2x1xbf16>
    %1334 = sdy.sharding_constraint %1333 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1335 = stablehlo.divide %1320, %1334 : tensor<2x4x2x1xbf16>
    %1336 = sdy.sharding_constraint %1335 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1337 = stablehlo.multiply %1334, %1334 : tensor<2x4x2x1xbf16>
    %1338 = sdy.sharding_constraint %1337 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_176 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1339 = stablehlo.broadcast_in_dim %cst_176, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %1340 = sdy.sharding_constraint %1339 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1341 = stablehlo.divide %1340, %1338 : tensor<2x4x2x1xbf16>
    %1342 = sdy.sharding_constraint %1341 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1343 = sdy.sharding_constraint %1342 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1344 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %1345 = sdy.sharding_constraint %1344 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1346 = stablehlo.multiply %1345, %1310 : tensor<2x4x2x16xbf16>
    %1347 = sdy.sharding_constraint %1346 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1348 = stablehlo.subtract %1303, %1347 : tensor<2x4x2x16xbf16>
    %1349 = sdy.sharding_constraint %1348 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1350 = stablehlo.dot_general %1144, %1076#6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x2xbf16>) -> tensor<2x4x2xbf16>
    %1351 = sdy.sharding_constraint %1350 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1352 = stablehlo.negate %1351 : tensor<2x4x2xbf16>
    %1353 = sdy.sharding_constraint %1352 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1354 = stablehlo.exponential %1353 : tensor<2x4x2xbf16>
    %1355 = sdy.sharding_constraint %1354 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_177 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1356 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %1357 = sdy.sharding_constraint %1356 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1358 = stablehlo.add %1357, %1355 : tensor<2x4x2xbf16>
    %1359 = sdy.sharding_constraint %1358 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_178 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1360 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %1361 = sdy.sharding_constraint %1360 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1362 = stablehlo.divide %1361, %1359 : tensor<2x4x2xbf16>
    %1363 = sdy.sharding_constraint %1362 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %cst_179 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1364 = sdy.sharding_constraint %cst_179 <@mesh, []> : tensor<bf16>
    %1365 = stablehlo.broadcast_in_dim %1364, dims = [] : (tensor<bf16>) -> tensor<2x4x2xbf16>
    %1366 = sdy.sharding_constraint %1365 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1367 = stablehlo.subtract %1366, %1363 : tensor<2x4x2xbf16>
    %1368 = sdy.sharding_constraint %1367 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1369 = stablehlo.multiply %1363, %1368 : tensor<2x4x2xbf16>
    %1370 = sdy.sharding_constraint %1369 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1371 = stablehlo.broadcast_in_dim %1363, dims = [0, 1, 2] : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %1372 = sdy.sharding_constraint %1371 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_180 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %1373 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %1374 = sdy.sharding_constraint %1373 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1375 = stablehlo.multiply %1374, %1372 : tensor<2x4x2x1xbf16>
    %1376 = sdy.sharding_constraint %1375 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1377 = stablehlo.broadcast_in_dim %1376, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %1378 = sdy.sharding_constraint %1377 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1379 = stablehlo.multiply %1378, %1349 : tensor<2x4x2x16xbf16>
    %1380 = sdy.sharding_constraint %1379 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1381 = stablehlo.reshape %1380 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x32xbf16>
    %1382 = sdy.sharding_constraint %1381 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %1383 = stablehlo.dot_general %1382, %1076#7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %1384 = sdy.sharding_constraint %1383 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1385 = stablehlo.add %1076#16, %1384 : tensor<2x4x32xbf16>
    %1386 = sdy.sharding_constraint %1385 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1387 = sdy.sharding_constraint %1076#8 <@mesh, [{}]> : tensor<32xbf16>
    %1388 = stablehlo.convert %1386 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1389 = sdy.sharding_constraint %1388 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1390 = chlo.square %1389 : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
    %cst_181 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1391 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<2x4x32xf32>
    %1392 = sdy.sharding_constraint %1391 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1393 = stablehlo.multiply %1392, %1389 : tensor<2x4x32xf32>
    %1394 = sdy.sharding_constraint %1393 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1395 = stablehlo.reduce(%1390 init: %cst_182) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1396 = sdy.sharding_constraint %1395 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1397 = stablehlo.broadcast_in_dim %1396, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1398 = sdy.sharding_constraint %1397 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_183 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1399 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1400 = sdy.sharding_constraint %1399 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1401 = stablehlo.divide %1398, %1400 : tensor<2x4x1xf32>
    %1402 = sdy.sharding_constraint %1401 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_184 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1403 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1404 = sdy.sharding_constraint %1403 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1405 = stablehlo.add %1402, %1404 : tensor<2x4x1xf32>
    %1406 = sdy.sharding_constraint %1405 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1407 = stablehlo.rsqrt %1406 : tensor<2x4x1xf32>
    %1408 = sdy.sharding_constraint %1407 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1409 = stablehlo.divide %1408, %1406 : tensor<2x4x1xf32>
    %1410 = sdy.sharding_constraint %1409 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_185 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %1411 = stablehlo.broadcast_in_dim %cst_185, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1412 = sdy.sharding_constraint %1411 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1413 = stablehlo.multiply %1412, %1410 : tensor<2x4x1xf32>
    %1414 = sdy.sharding_constraint %1413 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1415 = stablehlo.broadcast_in_dim %1408, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1416 = sdy.sharding_constraint %1415 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1417 = stablehlo.multiply %1389, %1416 : tensor<2x4x32xf32>
    %1418 = sdy.sharding_constraint %1417 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1419 = stablehlo.convert %1387 : (tensor<32xbf16>) -> tensor<32xf32>
    %1420 = sdy.sharding_constraint %1419 <@mesh, [{}]> : tensor<32xf32>
    %1421 = stablehlo.broadcast_in_dim %1420, dims = [2] : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1422 = sdy.sharding_constraint %1421 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1424 = sdy.sharding_constraint %1423 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1425 = stablehlo.multiply %1418, %1424 : tensor<2x4x32xf32>
    %1426 = sdy.sharding_constraint %1425 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1427 = stablehlo.convert %1426 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %1428 = sdy.sharding_constraint %1427 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1429 = stablehlo.dot_general %1428, %1076#9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x128xbf16>) -> tensor<2x4x128xbf16>
    %1430 = sdy.sharding_constraint %1429 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1431:3 = call @silu_347(%1430) : (tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>)
    %1432 = stablehlo.dot_general %1431#0, %1076#10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<128x32xbf16>) -> tensor<2x4x32xbf16>
    %1433 = sdy.sharding_constraint %1432 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1434 = stablehlo.negate %1433 : tensor<2x4x32xbf16>
    %1435 = sdy.sharding_constraint %1434 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1436 = stablehlo.exponential %1435 : tensor<2x4x32xbf16>
    %1437 = sdy.sharding_constraint %1436 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_186 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1438 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1439 = sdy.sharding_constraint %1438 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1440 = stablehlo.add %1439, %1437 : tensor<2x4x32xbf16>
    %1441 = sdy.sharding_constraint %1440 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_187 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1442 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1443 = sdy.sharding_constraint %1442 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1444 = stablehlo.divide %1443, %1441 : tensor<2x4x32xbf16>
    %1445 = sdy.sharding_constraint %1444 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %cst_188 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %1446 = sdy.sharding_constraint %cst_188 <@mesh, []> : tensor<bf16>
    %1447 = stablehlo.broadcast_in_dim %1446, dims = [] : (tensor<bf16>) -> tensor<2x4x32xbf16>
    %1448 = sdy.sharding_constraint %1447 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1449 = stablehlo.subtract %1448, %1445 : tensor<2x4x32xbf16>
    %1450 = sdy.sharding_constraint %1449 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1451 = stablehlo.multiply %1445, %1450 : tensor<2x4x32xbf16>
    %1452 = sdy.sharding_constraint %1451 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1453 = stablehlo.multiply %1428, %1445 : tensor<2x4x32xbf16>
    %1454 = sdy.sharding_constraint %1453 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1455 = stablehlo.reshape %1454 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1456 = sdy.sharding_constraint %1455 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1457 = sdy.sharding_constraint %1076#11 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %1458 = stablehlo.dot_general %1456, %1457, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x4xbf16>) -> tensor<8x4xbf16>
    %1459 = sdy.sharding_constraint %1458 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xbf16>
    %1460 = stablehlo.convert %1459 : (tensor<8x4xbf16>) -> tensor<8x4xf32>
    %1461 = sdy.sharding_constraint %1460 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1462 = stablehlo.convert %1076#12 : (tensor<4xbf16>) -> tensor<4xf32>
    %1463 = sdy.sharding_constraint %1462 <@mesh, [{}]> : tensor<4xf32>
    %1464 = stablehlo.broadcast_in_dim %1463, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %1465 = sdy.sharding_constraint %1464 <@mesh, [{}, {}]> : tensor<1x4xf32>
    %1466 = stablehlo.broadcast_in_dim %1465, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<8x4xf32>
    %1467 = sdy.sharding_constraint %1466 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1468 = stablehlo.add %1461, %1467 : tensor<8x4xf32>
    %1469 = sdy.sharding_constraint %1468 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %values_189, %indices_190 = chlo.top_k(%1469, k = 3) : tensor<8x4xf32> -> (tensor<8x3xf32>, tensor<8x3xi32>)
    %1470 = sdy.sharding_constraint %values_189 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xf32>
    %1471 = sdy.sharding_constraint %indices_190 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x3xi32>
    %1472 = stablehlo.slice %1471 [0:8, 0:2] : (tensor<8x3xi32>) -> tensor<8x2xi32>
    %1473 = sdy.sharding_constraint %1472 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %1474:2 = call @take_along_axis_364(%1461, %1473) : (tensor<8x4xf32>, tensor<8x2xi32>) -> (tensor<8x2xf32>, tensor<8x2x1xi32>)
    %1475 = stablehlo.negate %1474#0 : tensor<8x2xf32>
    %1476 = sdy.sharding_constraint %1475 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1477 = stablehlo.exponential %1476 : tensor<8x2xf32>
    %1478 = sdy.sharding_constraint %1477 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_191 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1479 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %1480 = sdy.sharding_constraint %1479 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1481 = stablehlo.add %1480, %1478 : tensor<8x2xf32>
    %1482 = sdy.sharding_constraint %1481 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_192 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1483 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %1484 = sdy.sharding_constraint %1483 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1485 = stablehlo.divide %1484, %1482 : tensor<8x2xf32>
    %1486 = sdy.sharding_constraint %1485 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_193 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1487 = sdy.sharding_constraint %cst_193 <@mesh, []> : tensor<f32>
    %1488 = stablehlo.broadcast_in_dim %1487, dims = [] : (tensor<f32>) -> tensor<8x2xf32>
    %1489 = sdy.sharding_constraint %1488 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1490 = stablehlo.subtract %1489, %1486 : tensor<8x2xf32>
    %1491 = sdy.sharding_constraint %1490 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1492 = stablehlo.multiply %1486, %1491 : tensor<8x2xf32>
    %1493 = sdy.sharding_constraint %1492 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1494 = stablehlo.reduce(%1486 init: %cst_194) applies stablehlo.add across dimensions = [1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8xf32>
    %1495 = sdy.sharding_constraint %1494 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1496 = stablehlo.broadcast_in_dim %1495, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %1497 = sdy.sharding_constraint %1496 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_195 = stablehlo.constant dense<9.99999971E-10> : tensor<f32>
    %1498 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1499 = sdy.sharding_constraint %1498 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1500 = stablehlo.add %1497, %1499 : tensor<8x1xf32>
    %1501 = sdy.sharding_constraint %1500 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_196 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %1502 = stablehlo.broadcast_in_dim %cst_196, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1503 = sdy.sharding_constraint %1502 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1504 = stablehlo.divide %1503, %1501 : tensor<8x1xf32>
    %1505 = sdy.sharding_constraint %1504 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1506 = stablehlo.multiply %1501, %1501 : tensor<8x1xf32>
    %1507 = sdy.sharding_constraint %1506 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_197 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1508 = stablehlo.broadcast_in_dim %cst_197, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1509 = sdy.sharding_constraint %1508 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1510 = stablehlo.divide %1509, %1507 : tensor<8x1xf32>
    %1511 = sdy.sharding_constraint %1510 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1512 = sdy.sharding_constraint %1511 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1513 = stablehlo.broadcast_in_dim %1505, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x2xf32>
    %1514 = sdy.sharding_constraint %1513 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1515 = stablehlo.multiply %1486, %1514 : tensor<8x2xf32>
    %1516 = sdy.sharding_constraint %1515 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1517 = stablehlo.convert %1516 : (tensor<8x2xf32>) -> tensor<8x2xbf16>
    %1518 = sdy.sharding_constraint %1517 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %cst_198 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1519 = stablehlo.reduce(%1461 init: %cst_198) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %1520 = sdy.sharding_constraint %1519 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_199 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1521 = stablehlo.broadcast_in_dim %cst_199, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1522 = sdy.sharding_constraint %1521 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1523 = stablehlo.maximum %1522, %1520 : tensor<8xf32>
    %1524 = sdy.sharding_constraint %1523 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1525 = stablehlo.is_finite %1524 : (tensor<8xf32>) -> tensor<8xi1>
    %1526 = sdy.sharding_constraint %1525 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xi1>
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1527 = stablehlo.broadcast_in_dim %cst_200, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1528 = sdy.sharding_constraint %1527 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1529 = stablehlo.select %1526, %1524, %1528 : tensor<8xi1>, tensor<8xf32>
    %1530 = sdy.sharding_constraint %1529 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1531 = stablehlo.broadcast_in_dim %1530, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %1532 = sdy.sharding_constraint %1531 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1533 = stablehlo.broadcast_in_dim %1532, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x4xf32>
    %1534 = sdy.sharding_constraint %1533 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1535 = stablehlo.subtract %1461, %1534 : tensor<8x4xf32>
    %1536 = sdy.sharding_constraint %1535 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1537 = stablehlo.exponential %1536 : tensor<8x4xf32>
    %1538 = sdy.sharding_constraint %1537 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1539 = stablehlo.reduce(%1538 init: %cst_201) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
    %1540 = sdy.sharding_constraint %1539 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1541 = stablehlo.abs %1540 : tensor<8xf32>
    %1542 = sdy.sharding_constraint %1541 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1543 = sdy.sharding_constraint %cst_202 <@mesh, []> : tensor<f32>
    %1544 = stablehlo.broadcast_in_dim %1543, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1545 = sdy.sharding_constraint %1544 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1546 = stablehlo.compare GE, %1540, %1545, FLOAT : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xi1>
    %1547 = stablehlo.log %1542 : tensor<8xf32>
    %1548 = sdy.sharding_constraint %1547 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1549 = stablehlo.add %1548, %1530 : tensor<8xf32>
    %1550 = sdy.sharding_constraint %1549 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_203 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1551 = stablehlo.broadcast_in_dim %cst_203, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1552 = sdy.sharding_constraint %1551 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1553 = stablehlo.multiply %1552, %1550 : tensor<8xf32>
    %1554 = sdy.sharding_constraint %1553 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1555 = stablehlo.concatenate %1076#17, %1076#18, dim = 2 : (tensor<4x32x32xbf16>, tensor<4x32x32xbf16>) -> tensor<4x32x64xbf16>
    %1556 = sdy.sharding_constraint %1555 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x64xbf16>
    %1557 = sdy.sharding_constraint %1456 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1558 = sdy.sharding_constraint %1473 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xi32>
    %1559 = sdy.sharding_constraint %1518 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %1560 = sdy.sharding_constraint %1556 <@mesh, [{}, {}, {}]> : tensor<4x32x64xbf16>
    %1561 = sdy.sharding_constraint %1076#19 <@mesh, [{}, {}, {}]> : tensor<4x32x32xbf16>
    %1562 = stablehlo.reshape %1558 : (tensor<8x2xi32>) -> tensor<16xi32>
    %1563 = stablehlo.reshape %1559 : (tensor<8x2xbf16>) -> tensor<16xbf16>
    %1564 = call @argsort_374(%1562) : (tensor<16xi32>) -> tensor<16xi32>
    %1565 = stablehlo.iota dim = 0 : tensor<16xi32>
    %c_204 = stablehlo.constant dense<2> : tensor<i32>
    %1566 = call @floor_divide_375(%1565, %c_204) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_205 = stablehlo.constant dense<0> : tensor<i32>
    %1567 = stablehlo.broadcast_in_dim %c_205, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1568 = stablehlo.compare LT, %1564, %1567, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_206 = stablehlo.constant dense<16> : tensor<i32>
    %1569 = stablehlo.broadcast_in_dim %c_206, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1570 = stablehlo.add %1564, %1569 : tensor<16xi32>
    %1571 = stablehlo.select %1568, %1570, %1564 : tensor<16xi1>, tensor<16xi32>
    %1572 = stablehlo.broadcast_in_dim %1571, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1573 = "stablehlo.gather"(%1566, %1572) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xi32>, tensor<16x1xi32>) -> tensor<16xi32>
    %c_207 = stablehlo.constant dense<0> : tensor<i32>
    %1574 = stablehlo.broadcast_in_dim %c_207, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1575 = stablehlo.compare LT, %1573, %1574, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_208 = stablehlo.constant dense<8> : tensor<i32>
    %1576 = stablehlo.broadcast_in_dim %c_208, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1577 = stablehlo.add %1573, %1576 : tensor<16xi32>
    %1578 = stablehlo.select %1575, %1577, %1573 : tensor<16xi1>, tensor<16xi32>
    %1579 = stablehlo.broadcast_in_dim %1578, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1580 = "stablehlo.gather"(%1557, %1579) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<8x32xbf16>, tensor<16x1xi32>) -> tensor<16x32xbf16>
    %c_209 = stablehlo.constant dense<0> : tensor<i32>
    %1581 = stablehlo.broadcast_in_dim %c_209, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1582 = stablehlo.compare LT, %1564, %1581, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_210 = stablehlo.constant dense<16> : tensor<i32>
    %1583 = stablehlo.broadcast_in_dim %c_210, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1584 = stablehlo.add %1564, %1583 : tensor<16xi32>
    %1585 = stablehlo.select %1582, %1584, %1564 : tensor<16xi1>, tensor<16xi32>
    %1586 = stablehlo.broadcast_in_dim %1585, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1587 = "stablehlo.gather"(%1563, %1586) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xbf16>, tensor<16x1xi32>) -> tensor<16xbf16>
    %c_211 = stablehlo.constant dense<0> : tensor<i32>
    %1588 = stablehlo.broadcast_in_dim %c_211, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %c_212 = stablehlo.constant dense<0> : tensor<i32>
    %1589 = call @clip_377(%1562, %c_212) : (tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    %c_213 = stablehlo.constant dense<0> : tensor<i32>
    %1590 = stablehlo.broadcast_in_dim %c_213, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1591 = stablehlo.compare LT, %1589, %1590, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_214 = stablehlo.constant dense<4> : tensor<i32>
    %1592 = stablehlo.broadcast_in_dim %c_214, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1593 = stablehlo.add %1589, %1592 : tensor<16xi32>
    %1594 = stablehlo.select %1591, %1593, %1589 : tensor<16xi1>, tensor<16xi32>
    %1595 = stablehlo.broadcast_in_dim %1594, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %c_215 = stablehlo.constant dense<1> : tensor<i32>
    %1596 = stablehlo.broadcast_in_dim %c_215, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1597 = "stablehlo.scatter"(%1588, %1595, %1596) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg85: tensor<i32>, %arg86: tensor<i32>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<i32>
      stablehlo.return %3950 : tensor<i32>
    }) : (tensor<4xi32>, tensor<16x1xi32>, tensor<16xi32>) -> tensor<4xi32>
    %cst_216 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1598 = stablehlo.pad %1580, %cst_216, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %1599 = stablehlo.broadcast_in_dim %1598, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1600 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1601 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_217 = stablehlo.constant dense<0> : tensor<i32>
    %1602 = stablehlo.broadcast_in_dim %c_217, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1603 = stablehlo.slice %1602 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1604 = stablehlo.slice %1601 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1605 = stablehlo.concatenate %1603, %1604, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1606 = stablehlo.broadcast_in_dim %1601, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1607 = stablehlo.broadcast_in_dim %1605, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1608 = stablehlo.compare LE, %1607, %1600, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1609 = stablehlo.compare LT, %1600, %1606, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1610 = stablehlo.and %1608, %1609 : tensor<4x512x32xi1>
    %cst_218 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1611 = stablehlo.broadcast_in_dim %cst_218, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1612 = stablehlo.select %1610, %1599, %1611 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1613 = stablehlo.dot_general %1612, %1560, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x64xbf16>) -> tensor<512x64xbf16>
    %1614 = stablehlo.slice %1613 [0:16, 0:64] : (tensor<512x64xbf16>) -> tensor<16x64xbf16>
    %1615 = stablehlo.slice %1614 [0:16, 0:32] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %1616 = stablehlo.slice %1614 [0:16, 32:64] : (tensor<16x64xbf16>) -> tensor<16x32xbf16>
    %1617:3 = call @silu_378(%1615) : (tensor<16x32xbf16>) -> (tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>)
    %1618 = stablehlo.multiply %1617#0, %1616 : tensor<16x32xbf16>
    %cst_219 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1619 = stablehlo.pad %1618, %cst_219, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %1620 = stablehlo.broadcast_in_dim %1619, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1621 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1622 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_220 = stablehlo.constant dense<0> : tensor<i32>
    %1623 = stablehlo.broadcast_in_dim %c_220, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1624 = stablehlo.slice %1623 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1625 = stablehlo.slice %1622 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1626 = stablehlo.concatenate %1624, %1625, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1627 = stablehlo.broadcast_in_dim %1622, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1628 = stablehlo.broadcast_in_dim %1626, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1629 = stablehlo.compare LE, %1628, %1621, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1630 = stablehlo.compare LT, %1621, %1627, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1631 = stablehlo.and %1629, %1630 : tensor<4x512x32xi1>
    %cst_221 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1632 = stablehlo.broadcast_in_dim %cst_221, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1633 = stablehlo.select %1631, %1620, %1632 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1634 = stablehlo.dot_general %1633, %1561, contracting_dims = [2, 0] x [1, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x32xbf16>) -> tensor<512x32xbf16>
    %1635 = stablehlo.slice %1634 [0:16, 0:32] : (tensor<512x32xbf16>) -> tensor<16x32xbf16>
    %1636 = stablehlo.broadcast_in_dim %1587, dims = [0] : (tensor<16xbf16>) -> tensor<16x1xbf16>
    %c_222 = stablehlo.constant dense<0> : tensor<i32>
    %1637 = stablehlo.broadcast_in_dim %c_222, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1638 = stablehlo.compare LT, %1573, %1637, SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
    %c_223 = stablehlo.constant dense<8> : tensor<i32>
    %1639 = stablehlo.broadcast_in_dim %c_223, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %1640 = stablehlo.add %1573, %1639 : tensor<16xi32>
    %1641 = stablehlo.select %1638, %1640, %1573 : tensor<16xi1>, tensor<16xi32>
    %1642 = stablehlo.broadcast_in_dim %1641, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1643 = stablehlo.reshape %1454 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1644 = sdy.sharding_constraint %1643 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1645 = stablehlo.dot_general %1644, %1076#13, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1646 = sdy.sharding_constraint %1645 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1647 = stablehlo.dot_general %1644, %1076#14, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1648 = sdy.sharding_constraint %1647 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1649:3 = call @silu_380(%1646) : (tensor<8x32xbf16>) -> (tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>)
    %1650 = stablehlo.multiply %1649#0, %1648 : tensor<8x32xbf16>
    %1651 = sdy.sharding_constraint %1650 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1652 = sdy.sharding_constraint %1076#20 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1653 = stablehlo.reshape %1652 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1654 = sdy.sharding_constraint %1653 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1655 = stablehlo.dot_general %1654, %1651, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<32x32xbf16>
    %1656 = sdy.sharding_constraint %1655 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1657 = stablehlo.transpose %1656, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1658 = sdy.sharding_constraint %1657 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1659 = stablehlo.dot_general %1654, %1076#15, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1660 = sdy.sharding_constraint %1659 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1661 = stablehlo.multiply %1649#0, %1660 : tensor<8x32xbf16>
    %1662 = sdy.sharding_constraint %1661 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1663 = stablehlo.multiply %1660, %1648 : tensor<8x32xbf16>
    %1664 = sdy.sharding_constraint %1663 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}]> : tensor<8x32xbf16>
    %1665 = call @silu_386(%1649#1, %1646, %1649#2, %1664) : (tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<8x32xbf16>
    %1666 = stablehlo.dot_general %1662, %1644, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<32x32xbf16>
    %1667 = sdy.sharding_constraint %1666 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1668 = stablehlo.transpose %1667, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1669 = sdy.sharding_constraint %1668 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1670 = stablehlo.dot_general %1662, %1076#14, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1671 = sdy.sharding_constraint %1670 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1672 = stablehlo.dot_general %1665, %1644, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<8x32xbf16>) -> tensor<32x32xbf16>
    %1673 = sdy.sharding_constraint %1672 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1674 = stablehlo.transpose %1673, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1675 = sdy.sharding_constraint %1674 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1676 = stablehlo.dot_general %1665, %1076#13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x32xbf16>, tensor<32x32xbf16>) -> tensor<8x32xbf16>
    %1677 = sdy.sharding_constraint %1676 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1678 = stablehlo.add %1671, %1677 : tensor<8x32xbf16>
    %1679 = sdy.sharding_constraint %1678 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1680 = stablehlo.reshape %1679 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %1681 = sdy.sharding_constraint %1680 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1682 = sdy.sharding_constraint %1076#20 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1683 = stablehlo.reshape %1682 : (tensor<2x4x32xbf16>) -> tensor<8x32xbf16>
    %1684 = sdy.sharding_constraint %1683 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %c_224 = stablehlo.constant dense<7> : tensor<1xi32>
    %c_225 = stablehlo.constant dense<0> : tensor<i32>
    %1685 = stablehlo.broadcast_in_dim %c_225, dims = [] : (tensor<i32>) -> tensor<16x1xi32>
    %1686 = stablehlo.compare GE, %1642, %1685, SIGNED : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x1xi1>
    %1687 = stablehlo.broadcast_in_dim %c_224, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<16x1xi32>
    %1689 = stablehlo.compare LE, %1642, %1688, SIGNED : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x1xi1>
    %1690 = stablehlo.and %1686, %1689 : tensor<16x1xi1>
    %c_226 = stablehlo.constant dense<true> : tensor<i1>
    %1691 = stablehlo.reduce(%1690 init: %c_226) applies stablehlo.and across dimensions = [1] : (tensor<16x1xi1>, tensor<i1>) -> tensor<16xi1>
    %1692 = "stablehlo.gather"(%1684, %1642) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<8x32xbf16>, tensor<16x1xi32>) -> tensor<16x32xbf16>
    %1693 = stablehlo.broadcast_in_dim %1691, dims = [0] : (tensor<16xi1>) -> tensor<16x32xi1>
    %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1694 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<bf16>) -> tensor<16x32xbf16>
    %1695 = stablehlo.select %1693, %1692, %1694 : tensor<16x32xi1>, tensor<16x32xbf16>
    %1696 = stablehlo.multiply %1635, %1695 : tensor<16x32xbf16>
    %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1697 = stablehlo.reduce(%1696 init: %cst_228) applies stablehlo.add across dimensions = [1] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<16xbf16>
    %1698 = stablehlo.reshape %1697 : (tensor<16xbf16>) -> tensor<16x1xbf16>
    %1699 = stablehlo.broadcast_in_dim %1636, dims = [0, 1] : (tensor<16x1xbf16>) -> tensor<16x32xbf16>
    %1700 = stablehlo.multiply %1695, %1699 : tensor<16x32xbf16>
    %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1701 = stablehlo.reduce(%1698 init: %cst_229) applies stablehlo.add across dimensions = [1] : (tensor<16x1xbf16>, tensor<bf16>) -> tensor<16xbf16>
    %cst_230 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1702 = stablehlo.pad %1700, %cst_230, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x32xbf16>, tensor<bf16>) -> tensor<512x32xbf16>
    %1703 = stablehlo.broadcast_in_dim %1619, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1704 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1705 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_231 = stablehlo.constant dense<0> : tensor<i32>
    %1706 = stablehlo.broadcast_in_dim %c_231, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1707 = stablehlo.slice %1706 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1708 = stablehlo.slice %1705 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1709 = stablehlo.concatenate %1707, %1708, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1710 = stablehlo.broadcast_in_dim %1705, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1711 = stablehlo.broadcast_in_dim %1709, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1712 = stablehlo.compare LE, %1711, %1704, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1713 = stablehlo.compare LT, %1704, %1710, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1714 = stablehlo.and %1712, %1713 : tensor<4x512x32xi1>
    %cst_232 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1715 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1716 = stablehlo.select %1714, %1703, %1715 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1717 = stablehlo.broadcast_in_dim %1702, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1718 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1719 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_233 = stablehlo.constant dense<0> : tensor<i32>
    %1720 = stablehlo.broadcast_in_dim %c_233, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1721 = stablehlo.slice %1720 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1722 = stablehlo.slice %1719 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1723 = stablehlo.concatenate %1721, %1722, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1724 = stablehlo.broadcast_in_dim %1719, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1725 = stablehlo.broadcast_in_dim %1723, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1726 = stablehlo.compare LE, %1725, %1718, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1727 = stablehlo.compare LT, %1718, %1724, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1728 = stablehlo.and %1726, %1727 : tensor<4x512x32xi1>
    %cst_234 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1729 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1730 = stablehlo.select %1728, %1717, %1729 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1731 = stablehlo.dot_general %1716, %1730, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x512x32xbf16>) -> tensor<4x32x32xbf16>
    %1732 = stablehlo.broadcast_in_dim %1702, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1733 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1734 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_235 = stablehlo.constant dense<0> : tensor<i32>
    %1735 = stablehlo.broadcast_in_dim %c_235, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1736 = stablehlo.slice %1735 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1737 = stablehlo.slice %1734 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1738 = stablehlo.concatenate %1736, %1737, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1739 = stablehlo.broadcast_in_dim %1734, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1740 = stablehlo.broadcast_in_dim %1738, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1741 = stablehlo.compare LE, %1740, %1733, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1742 = stablehlo.compare LT, %1733, %1739, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1743 = stablehlo.and %1741, %1742 : tensor<4x512x32xi1>
    %cst_236 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1744 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1745 = stablehlo.select %1743, %1732, %1744 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1746 = stablehlo.dot_general %1745, %1561, contracting_dims = [2, 0] x [2, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x32x32xbf16>) -> tensor<512x32xbf16>
    %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1747 = stablehlo.pad %1746, %cst_237, low = [0, 0], high = [-496, 0], interior = [0, 0] : (tensor<512x32xbf16>, tensor<bf16>) -> tensor<16x32xbf16>
    %1748 = stablehlo.slice %1747 [0:16, 0:32] : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %1749 = stablehlo.multiply %1617#0, %1748 : tensor<16x32xbf16>
    %1750 = stablehlo.multiply %1748, %1616 : tensor<16x32xbf16>
    %1751 = call @silu_413(%1617#1, %1615, %1617#2, %1750) : (tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %1752 = stablehlo.concatenate %1751, %1749, dim = 1 : (tensor<16x32xbf16>, tensor<16x32xbf16>) -> tensor<16x64xbf16>
    %cst_238 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1753 = stablehlo.pad %1752, %cst_238, low = [0, 0], high = [496, 0], interior = [0, 0] : (tensor<16x64xbf16>, tensor<bf16>) -> tensor<512x64xbf16>
    %1754 = stablehlo.broadcast_in_dim %1598, dims = [1, 2] : (tensor<512x32xbf16>) -> tensor<4x512x32xbf16>
    %1755 = stablehlo.iota dim = 1 : tensor<4x512x32xi32>
    %1756 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_239 = stablehlo.constant dense<0> : tensor<i32>
    %1757 = stablehlo.broadcast_in_dim %c_239, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1758 = stablehlo.slice %1757 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1759 = stablehlo.slice %1756 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1760 = stablehlo.concatenate %1758, %1759, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1761 = stablehlo.broadcast_in_dim %1756, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1762 = stablehlo.broadcast_in_dim %1760, dims = [0] : (tensor<4xi32>) -> tensor<4x512x32xi32>
    %1763 = stablehlo.compare LE, %1762, %1755, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1764 = stablehlo.compare LT, %1755, %1761, SIGNED : (tensor<4x512x32xi32>, tensor<4x512x32xi32>) -> tensor<4x512x32xi1>
    %1765 = stablehlo.and %1763, %1764 : tensor<4x512x32xi1>
    %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1766 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<bf16>) -> tensor<4x512x32xbf16>
    %1767 = stablehlo.select %1765, %1754, %1766 : tensor<4x512x32xi1>, tensor<4x512x32xbf16>
    %1768 = stablehlo.broadcast_in_dim %1753, dims = [1, 2] : (tensor<512x64xbf16>) -> tensor<4x512x64xbf16>
    %1769 = stablehlo.iota dim = 1 : tensor<4x512x64xi32>
    %1770 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_241 = stablehlo.constant dense<0> : tensor<i32>
    %1771 = stablehlo.broadcast_in_dim %c_241, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1772 = stablehlo.slice %1771 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1773 = stablehlo.slice %1770 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1774 = stablehlo.concatenate %1772, %1773, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1775 = stablehlo.broadcast_in_dim %1770, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1776 = stablehlo.broadcast_in_dim %1774, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1777 = stablehlo.compare LE, %1776, %1769, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1778 = stablehlo.compare LT, %1769, %1775, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1779 = stablehlo.and %1777, %1778 : tensor<4x512x64xi1>
    %cst_242 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1780 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<bf16>) -> tensor<4x512x64xbf16>
    %1781 = stablehlo.select %1779, %1768, %1780 : tensor<4x512x64xi1>, tensor<4x512x64xbf16>
    %1782 = stablehlo.dot_general %1767, %1781, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x512x32xbf16>, tensor<4x512x64xbf16>) -> tensor<4x32x64xbf16>
    %1783 = stablehlo.broadcast_in_dim %1753, dims = [1, 2] : (tensor<512x64xbf16>) -> tensor<4x512x64xbf16>
    %1784 = stablehlo.iota dim = 1 : tensor<4x512x64xi32>
    %1785 = call @cumsum(%1597) : (tensor<4xi32>) -> tensor<4xi32>
    %c_243 = stablehlo.constant dense<0> : tensor<i32>
    %1786 = stablehlo.broadcast_in_dim %c_243, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %1787 = stablehlo.slice %1786 [0:1] : (tensor<4xi32>) -> tensor<1xi32>
    %1788 = stablehlo.slice %1785 [0:3] : (tensor<4xi32>) -> tensor<3xi32>
    %1789 = stablehlo.concatenate %1787, %1788, dim = 0 : (tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
    %1790 = stablehlo.broadcast_in_dim %1785, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1791 = stablehlo.broadcast_in_dim %1789, dims = [0] : (tensor<4xi32>) -> tensor<4x512x64xi32>
    %1792 = stablehlo.compare LE, %1791, %1784, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1793 = stablehlo.compare LT, %1784, %1790, SIGNED : (tensor<4x512x64xi32>, tensor<4x512x64xi32>) -> tensor<4x512x64xi1>
    %1794 = stablehlo.and %1792, %1793 : tensor<4x512x64xi1>
    %cst_244 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1795 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<bf16>) -> tensor<4x512x64xbf16>
    %1796 = stablehlo.select %1794, %1783, %1795 : tensor<4x512x64xi1>, tensor<4x512x64xbf16>
    %1797 = stablehlo.dot_general %1796, %1560, contracting_dims = [2, 0] x [2, 0], precision = [DEFAULT, DEFAULT] : (tensor<4x512x64xbf16>, tensor<4x32x64xbf16>) -> tensor<512x32xbf16>
    %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1798 = stablehlo.pad %1797, %cst_245, low = [0, 0], high = [-496, 0], interior = [0, 0] : (tensor<512x32xbf16>, tensor<bf16>) -> tensor<16x32xbf16>
    %1799 = stablehlo.slice %1798 [0:16, 0:32] : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1800 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<bf16>) -> tensor<16xbf16>
    %1801 = "stablehlo.scatter"(%1800, %1586, %1701) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<16xbf16>, tensor<16x1xi32>, tensor<16xbf16>) -> tensor<16xbf16>
    %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1802 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<bf16>) -> tensor<8x32xbf16>
    %1803 = "stablehlo.scatter"(%1802, %1579, %1799) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<8x32xbf16>, tensor<16x1xi32>, tensor<16x32xbf16>) -> tensor<8x32xbf16>
    %1804 = stablehlo.reshape %1801 : (tensor<16xbf16>) -> tensor<8x2xbf16>
    %1805 = "stablehlo.all_reduce"(%1803) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<8x32xbf16>) -> tensor<8x32xbf16>
    %1806 = "stablehlo.all_reduce"(%1804) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<8x2xbf16>) -> tensor<8x2xbf16>
    %1807 = "stablehlo.all_reduce"(%1782) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<4x32x64xbf16>) -> tensor<4x32x64xbf16>
    %1808 = "stablehlo.all_reduce"(%1731) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<4x32x32xbf16>) -> tensor<4x32x32xbf16>
    %1809 = sdy.sharding_constraint %1808 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xbf16>
    %1810 = sdy.sharding_constraint %1807 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x64xbf16>
    %1811 = sdy.sharding_constraint %1806 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xbf16>
    %1812 = sdy.sharding_constraint %1805 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1813 = stablehlo.slice %1810 [0:4, 0:32, 0:32] : (tensor<4x32x64xbf16>) -> tensor<4x32x32xbf16>
    %1814 = sdy.sharding_constraint %1813 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %1815 = stablehlo.slice %1810 [0:4, 0:32, 32:64] : (tensor<4x32x64xbf16>) -> tensor<4x32x32xbf16>
    %1816 = sdy.sharding_constraint %1815 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xbf16>
    %cst_248 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %1817 = sdy.sharding_constraint %cst_248 <@mesh, []> : tensor<f32>
    %1818 = stablehlo.divide %1076#21, %1817 : tensor<f32>
    %1819 = sdy.sharding_constraint %1818 <@mesh, []> : tensor<f32>
    %1820 = stablehlo.broadcast_in_dim %1819, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1821 = sdy.sharding_constraint %1820 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1822 = stablehlo.multiply %1821, %1554 : tensor<8xf32>
    %1823 = sdy.sharding_constraint %1822 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1824 = stablehlo.divide %1823, %1542 : tensor<8xf32>
    %1825 = sdy.sharding_constraint %1824 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %cst_249 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1826 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1827 = sdy.sharding_constraint %1826 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1828 = stablehlo.select %1546, %1827, %1825 : tensor<8xi1>, tensor<8xf32>
    %1829 = sdy.sharding_constraint %1828 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1830 = stablehlo.select %1546, %1825, %1827 : tensor<8xi1>, tensor<8xf32>
    %1831 = sdy.sharding_constraint %1830 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1832 = stablehlo.negate %1829 : tensor<8xf32>
    %1833 = sdy.sharding_constraint %1832 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1834 = stablehlo.add %1831, %1833 : tensor<8xf32>
    %1835 = sdy.sharding_constraint %1834 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1836 = stablehlo.broadcast_in_dim %1835, dims = [0] : (tensor<8xf32>) -> tensor<8x4xf32>
    %1837 = sdy.sharding_constraint %1836 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1838 = stablehlo.multiply %1837, %1538 : tensor<8x4xf32>
    %1839 = sdy.sharding_constraint %1838 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1840 = stablehlo.convert %1811 : (tensor<8x2xbf16>) -> tensor<8x2xf32>
    %1841 = sdy.sharding_constraint %1840 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1842 = stablehlo.multiply %1486, %1841 : tensor<8x2xf32>
    %1843 = sdy.sharding_constraint %1842 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1844 = stablehlo.reduce(%1843 init: %cst_250) applies stablehlo.add across dimensions = [1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8xf32>
    %1845 = sdy.sharding_constraint %1844 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1846 = stablehlo.reshape %1845 : (tensor<8xf32>) -> tensor<8x1xf32>
    %1847 = sdy.sharding_constraint %1846 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1848 = stablehlo.broadcast_in_dim %1505, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x2xf32>
    %1849 = sdy.sharding_constraint %1848 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1850 = stablehlo.multiply %1841, %1849 : tensor<8x2xf32>
    %1851 = sdy.sharding_constraint %1850 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1852 = stablehlo.multiply %1847, %1512 : tensor<8x1xf32>
    %1853 = sdy.sharding_constraint %1852 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_251 = stablehlo.constant dense<2.500000e+00> : tensor<f32>
    %1854 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %1855 = sdy.sharding_constraint %1854 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1856 = stablehlo.multiply %1853, %1855 : tensor<8x1xf32>
    %1857 = sdy.sharding_constraint %1856 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %1858 = stablehlo.negate %1857 : tensor<8x1xf32>
    %1859 = sdy.sharding_constraint %1858 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x1xf32>
    %cst_252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1860 = stablehlo.reduce(%1859 init: %cst_252) applies stablehlo.add across dimensions = [1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8xf32>
    %1861 = sdy.sharding_constraint %1860 <@mesh, [{"replica_dcn", "data", "expert"}]> : tensor<8xf32>
    %1862 = stablehlo.broadcast_in_dim %1861, dims = [0] : (tensor<8xf32>) -> tensor<8x2xf32>
    %1863 = sdy.sharding_constraint %1862 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1864 = stablehlo.add %1851, %1863 : tensor<8x2xf32>
    %1865 = sdy.sharding_constraint %1864 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1866 = stablehlo.multiply %1865, %1493 : tensor<8x2xf32>
    %1867 = sdy.sharding_constraint %1866 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x2xf32>
    %1868 = call @take_along_axis_454(%1474#1, %1867) : (tensor<8x2x1xi32>, tensor<8x2xf32>) -> tensor<8x4xf32>
    %1869 = stablehlo.add %1839, %1868 : tensor<8x4xf32>
    %1870 = sdy.sharding_constraint %1869 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xf32>
    %1871 = stablehlo.convert %1870 : (tensor<8x4xf32>) -> tensor<8x4xbf16>
    %1872 = sdy.sharding_constraint %1871 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x4xbf16>
    %1873 = stablehlo.dot_general %1872, %1456, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x4xbf16>, tensor<8x32xbf16>) -> tensor<4x32xbf16>
    %1874 = sdy.sharding_constraint %1873 <@mesh, [{}, {}]> : tensor<4x32xbf16>
    %1875 = stablehlo.transpose %1874, dims = [1, 0] : (tensor<4x32xbf16>) -> tensor<32x4xbf16>
    %1876 = sdy.sharding_constraint %1875 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %1877 = stablehlo.dot_general %1872, %1457, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4xbf16>, tensor<32x4xbf16>) -> tensor<8x32xbf16>
    %1878 = sdy.sharding_constraint %1877 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1879 = stablehlo.add %1812, %1878 : tensor<8x32xbf16>
    %1880 = sdy.sharding_constraint %1879 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<8x32xbf16>
    %1881 = sdy.sharding_constraint %1876 <@mesh, [{}, {}]> : tensor<32x4xbf16>
    %1882 = stablehlo.reshape %1880 : (tensor<8x32xbf16>) -> tensor<2x4x32xbf16>
    %1883 = sdy.sharding_constraint %1882 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1884 = stablehlo.add %1681, %1883 : tensor<2x4x32xbf16>
    %1885 = sdy.sharding_constraint %1884 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1886 = stablehlo.multiply %1428, %1885 : tensor<2x4x32xbf16>
    %1887 = sdy.sharding_constraint %1886 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1888 = stablehlo.multiply %1885, %1445 : tensor<2x4x32xbf16>
    %1889 = sdy.sharding_constraint %1888 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1890 = stablehlo.multiply %1887, %1452 : tensor<2x4x32xbf16>
    %1891 = sdy.sharding_constraint %1890 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1892 = stablehlo.dot_general %1891, %1431#0, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %1893 = sdy.sharding_constraint %1892 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1894 = stablehlo.transpose %1893, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %1895 = sdy.sharding_constraint %1894 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1896 = stablehlo.dot_general %1891, %1076#10, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %1897 = sdy.sharding_constraint %1896 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %1898 = call @silu_463(%1431#1, %1430, %1431#2, %1897) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %1899 = stablehlo.dot_general %1898, %1428, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %1900 = sdy.sharding_constraint %1899 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %1901 = stablehlo.transpose %1900, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %1902 = sdy.sharding_constraint %1901 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %1903 = stablehlo.dot_general %1898, %1076#9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %1904 = sdy.sharding_constraint %1903 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1905 = stablehlo.add %1889, %1904 : tensor<2x4x32xbf16>
    %1906 = sdy.sharding_constraint %1905 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1907 = stablehlo.convert %1906 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %1908 = sdy.sharding_constraint %1907 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1909 = stablehlo.multiply %1418, %1908 : tensor<2x4x32xf32>
    %1910 = sdy.sharding_constraint %1909 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_253 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1911 = stablehlo.reduce(%1910 init: %cst_253) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1912 = sdy.sharding_constraint %1911 <@mesh, [{}]> : tensor<32xf32>
    %1913 = stablehlo.reshape %1912 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1914 = sdy.sharding_constraint %1913 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %1915 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %1916 = sdy.sharding_constraint %1915 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1917 = stablehlo.multiply %1908, %1916 : tensor<2x4x32xf32>
    %1918 = sdy.sharding_constraint %1917 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1919 = stablehlo.reduce(%1914 init: %cst_254) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1920 = sdy.sharding_constraint %1919 <@mesh, [{}]> : tensor<32xf32>
    %1921 = stablehlo.convert %1920 : (tensor<32xf32>) -> tensor<32xbf16>
    %1922 = sdy.sharding_constraint %1921 <@mesh, [{}]> : tensor<32xbf16>
    %1923 = stablehlo.multiply %1389, %1918 : tensor<2x4x32xf32>
    %1924 = sdy.sharding_constraint %1923 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_255 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1925 = stablehlo.reduce(%1924 init: %cst_255) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1926 = sdy.sharding_constraint %1925 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1927 = stablehlo.reshape %1926 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %1928 = sdy.sharding_constraint %1927 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1929 = stablehlo.broadcast_in_dim %1408, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %1930 = sdy.sharding_constraint %1929 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1931 = stablehlo.multiply %1918, %1930 : tensor<2x4x32xf32>
    %1932 = sdy.sharding_constraint %1931 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1933 = stablehlo.multiply %1928, %1414 : tensor<2x4x1xf32>
    %1934 = sdy.sharding_constraint %1933 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_256 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %1935 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %1936 = sdy.sharding_constraint %1935 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %1937 = stablehlo.divide %1934, %1936 : tensor<2x4x1xf32>
    %1938 = sdy.sharding_constraint %1937 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_257 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1939 = stablehlo.reduce(%1938 init: %cst_257) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %1940 = sdy.sharding_constraint %1939 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %1941 = stablehlo.broadcast_in_dim %1940, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %1942 = sdy.sharding_constraint %1941 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1943 = stablehlo.multiply %1942, %1394 : tensor<2x4x32xf32>
    %1944 = sdy.sharding_constraint %1943 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1945 = stablehlo.add %1932, %1944 : tensor<2x4x32xf32>
    %1946 = sdy.sharding_constraint %1945 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %1947 = stablehlo.convert %1946 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %1948 = sdy.sharding_constraint %1947 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1949 = stablehlo.add %1076#20, %1948 : tensor<2x4x32xbf16>
    %1950 = sdy.sharding_constraint %1949 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1951 = sdy.sharding_constraint %1922 <@mesh, [{}]> : tensor<32xbf16>
    %1952 = stablehlo.dot_general %1950, %1382, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x32xbf16>) -> tensor<32x32xbf16>
    %1953 = sdy.sharding_constraint %1952 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %1954 = stablehlo.transpose %1953, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1955 = sdy.sharding_constraint %1954 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %1956 = stablehlo.dot_general %1950, %1076#7, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %1957 = sdy.sharding_constraint %1956 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %1958 = stablehlo.reshape %1957 : (tensor<2x4x32xbf16>) -> tensor<2x4x2x16xbf16>
    %1959 = sdy.sharding_constraint %1958 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1960 = stablehlo.broadcast_in_dim %1376, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %1961 = sdy.sharding_constraint %1960 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1962 = stablehlo.multiply %1961, %1959 : tensor<2x4x2x16xbf16>
    %1963 = sdy.sharding_constraint %1962 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1964 = stablehlo.multiply %1959, %1349 : tensor<2x4x2x16xbf16>
    %1965 = sdy.sharding_constraint %1964 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_258 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1966 = stablehlo.reduce(%1965 init: %cst_258) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xbf16>, tensor<bf16>) -> tensor<2x4x2xbf16>
    %1967 = sdy.sharding_constraint %1966 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1968 = stablehlo.reshape %1967 : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %1969 = sdy.sharding_constraint %1968 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_259 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %1970 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<bf16>) -> tensor<2x4x2x1xbf16>
    %1971 = sdy.sharding_constraint %1970 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1972 = stablehlo.multiply %1971, %1969 : tensor<2x4x2x1xbf16>
    %1973 = sdy.sharding_constraint %1972 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %cst_260 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1974 = stablehlo.reduce(%1973 init: %cst_260) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xbf16>, tensor<bf16>) -> tensor<2x4x2xbf16>
    %1975 = sdy.sharding_constraint %1974 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1976 = stablehlo.multiply %1975, %1370 : tensor<2x4x2xbf16>
    %1977 = sdy.sharding_constraint %1976 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1978 = stablehlo.dot_general %1977, %1144, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2xbf16>, tensor<2x4x32xbf16>) -> tensor<2x32xbf16>
    %1979 = sdy.sharding_constraint %1978 <@mesh, [{}, {}]> : tensor<2x32xbf16>
    %1980 = stablehlo.transpose %1979, dims = [1, 0] : (tensor<2x32xbf16>) -> tensor<32x2xbf16>
    %1981 = sdy.sharding_constraint %1980 <@mesh, [{}, {}]> : tensor<32x2xbf16>
    %1982 = stablehlo.dot_general %1977, %1076#6, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x2xbf16>, tensor<32x2xbf16>) -> tensor<2x4x32xbf16>
    %1983 = sdy.sharding_constraint %1982 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %1984 = stablehlo.negate %1963 : tensor<2x4x2x16xbf16>
    %1985 = sdy.sharding_constraint %1984 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1986 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x16xbf16>
    %1987 = sdy.sharding_constraint %1986 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1988 = stablehlo.multiply %1987, %1985 : tensor<2x4x2x16xbf16>
    %1989 = sdy.sharding_constraint %1988 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %1990 = stablehlo.multiply %1985, %1310 : tensor<2x4x2x16xbf16>
    %1991 = sdy.sharding_constraint %1990 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %cst_261 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1992 = stablehlo.reduce(%1991 init: %cst_261) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xbf16>, tensor<bf16>) -> tensor<2x4x2xbf16>
    %1993 = sdy.sharding_constraint %1992 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xbf16>
    %1994 = stablehlo.reshape %1993 : (tensor<2x4x2xbf16>) -> tensor<2x4x2x1xbf16>
    %1995 = sdy.sharding_constraint %1994 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1996 = stablehlo.multiply %1995, %1343 : tensor<2x4x2x1xbf16>
    %1997 = sdy.sharding_constraint %1996 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %1998 = stablehlo.multiply %1997, %1320 : tensor<2x4x2x1xbf16>
    %1999 = sdy.sharding_constraint %1998 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2000 = stablehlo.negate %1999 : tensor<2x4x2x1xbf16>
    %2001 = sdy.sharding_constraint %2000 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2002 = stablehlo.divide %1995, %1334 : tensor<2x4x2x1xbf16>
    %2003 = sdy.sharding_constraint %2002 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xbf16>
    %2004 = stablehlo.convert %2001 : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x1xf32>
    %2005 = sdy.sharding_constraint %2004 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %cst_262 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2006 = stablehlo.reduce(%2005 init: %cst_262) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2007 = sdy.sharding_constraint %2006 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %2008 = stablehlo.broadcast_in_dim %2007, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x16xf32>
    %2009 = sdy.sharding_constraint %2008 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %2010 = stablehlo.convert %2009 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2011 = sdy.sharding_constraint %2010 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2012 = stablehlo.multiply %1310, %2011 : tensor<2x4x2x16xbf16>
    %2013 = sdy.sharding_constraint %2012 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2014 = stablehlo.add %1989, %2013 : tensor<2x4x2x16xbf16>
    %2015 = sdy.sharding_constraint %2014 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2016 = stablehlo.multiply %2011, %1310 : tensor<2x4x2x16xbf16>
    %2017 = sdy.sharding_constraint %2016 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2018 = stablehlo.add %2015, %2017 : tensor<2x4x2x16xbf16>
    %2019 = sdy.sharding_constraint %2018 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2020 = stablehlo.convert %2003 : (tensor<2x4x2x1xbf16>) -> tensor<2x4x2x1xf32>
    %2021 = sdy.sharding_constraint %2020 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x2x1xf32>
    %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2022 = stablehlo.reduce(%2021 init: %cst_263) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2023 = sdy.sharding_constraint %2022 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x2xf32>
    %2024 = stablehlo.broadcast_in_dim %2023, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x16xf32>
    %2025 = sdy.sharding_constraint %2024 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xf32>
    %2026 = stablehlo.convert %2025 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2027 = sdy.sharding_constraint %2026 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2028 = stablehlo.multiply %1303, %2027 : tensor<2x4x2x16xbf16>
    %2029 = sdy.sharding_constraint %2028 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2030 = stablehlo.add %2019, %2029 : tensor<2x4x2x16xbf16>
    %2031 = sdy.sharding_constraint %2030 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2032 = stablehlo.multiply %2027, %1310 : tensor<2x4x2x16xbf16>
    %2033 = sdy.sharding_constraint %2032 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2034 = stablehlo.add %1963, %2033 : tensor<2x4x2x16xbf16>
    %2035 = sdy.sharding_constraint %2034 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2036 = sdy.sharding_constraint %2031 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2037 = stablehlo.reshape %2036 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %2038 = sdy.sharding_constraint %2037 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %cst_264 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2039 = stablehlo.reduce(%2038 init: %cst_264) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x2x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2040 = sdy.sharding_constraint %2039 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2041 = stablehlo.broadcast_in_dim %2040, dims = [0, 1, 4] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %2042 = sdy.sharding_constraint %2041 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %cst_265 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2043 = stablehlo.reduce(%2042 init: %cst_265) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2044 = sdy.sharding_constraint %2043 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2045 = stablehlo.broadcast_in_dim %2044, dims = [0, 1, 3] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %2046 = sdy.sharding_constraint %2045 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2047 = stablehlo.transpose %2035, dims = [0, 2, 3, 1] : (tensor<2x4x2x16xbf16>) -> tensor<2x2x16x4xbf16>
    %2048 = sdy.sharding_constraint %2047 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %2049 = stablehlo.dot_general %2048, %1242, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x2x16x4xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x4xbf16>
    %2050 = sdy.sharding_constraint %2049 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %2051 = stablehlo.dot_general %2048, %1299, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x16x4xbf16>, tensor<2x2x4x4xbf16>) -> tensor<2x2x16x4xbf16>
    %2052 = sdy.sharding_constraint %2051 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x2x16x4xbf16>
    %2053 = stablehlo.transpose %2052, dims = [0, 3, 1, 2] : (tensor<2x2x16x4xbf16>) -> tensor<2x4x2x16xbf16>
    %2054 = sdy.sharding_constraint %2053 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2055 = stablehlo.convert %2050 : (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %2056 = sdy.sharding_constraint %2055 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2057 = stablehlo.broadcast_in_dim %1297, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %2058 = sdy.sharding_constraint %2057 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2059 = stablehlo.multiply %2056, %2058 : tensor<2x2x4x4xf32>
    %2060 = sdy.sharding_constraint %2059 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2061 = stablehlo.multiply %2060, %1282 : tensor<2x2x4x4xf32>
    %2062 = sdy.sharding_constraint %2061 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_266 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2063 = stablehlo.reduce(%2062 init: %cst_266) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %2064 = sdy.sharding_constraint %2063 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %2065 = stablehlo.reshape %2064 : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %2066 = sdy.sharding_constraint %2065 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %2067 = stablehlo.negate %2066 : tensor<2x2x4x1xf32>
    %2068 = sdy.sharding_constraint %2067 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x1xf32>
    %2069 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<2x2x4x1xf32>) -> tensor<2x2x4x4xf32>
    %2070 = sdy.sharding_constraint %2069 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2071 = stablehlo.divide %2056, %2070 : tensor<2x2x4x4xf32>
    %2072 = sdy.sharding_constraint %2071 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %cst_267 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2073 = stablehlo.reduce(%2068 init: %cst_267) applies stablehlo.add across dimensions = [3] : (tensor<2x2x4x1xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %2074 = sdy.sharding_constraint %2073 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}]> : tensor<2x2x4xf32>
    %2075 = stablehlo.broadcast_in_dim %2074, dims = [0, 1, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4x4xf32>
    %2076 = sdy.sharding_constraint %2075 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2077 = stablehlo.add %2072, %2076 : tensor<2x2x4x4xf32>
    %2078 = sdy.sharding_constraint %2077 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2079 = stablehlo.multiply %2078, %1282 : tensor<2x2x4x4xf32>
    %2080 = sdy.sharding_constraint %2079 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xf32>
    %2081 = stablehlo.convert %2080 : (tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xbf16>
    %2082 = sdy.sharding_constraint %2081 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %2083 = call @_where_497(%1266#1, %2082) : (tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xbf16>
    %2084 = stablehlo.dot_general %2083, %1246, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2x4x4xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x16xbf16>
    %2085 = sdy.sharding_constraint %2084 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x2x4x16xbf16>
    %2086 = stablehlo.transpose %2085, dims = [0, 2, 1, 3] : (tensor<2x2x4x16xbf16>) -> tensor<2x4x2x16xbf16>
    %2087 = sdy.sharding_constraint %2086 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x2x16xbf16>
    %2088 = stablehlo.dot_general %2083, %1236, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2x4x4xbf16>, tensor<2x4x2x16xbf16>) -> tensor<2x2x4x16xbf16>
    %2089 = sdy.sharding_constraint %2088 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x16xbf16>
    %2090 = stablehlo.transpose %2089, dims = [0, 2, 1, 3] : (tensor<2x2x4x16xbf16>) -> tensor<2x4x2x16xbf16>
    %2091 = sdy.sharding_constraint %2090 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %cst_268 = stablehlo.constant dense<2.500000e-01> : tensor<bf16>
    %2092 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %2093 = sdy.sharding_constraint %2092 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2094 = stablehlo.multiply %2091, %2093 : tensor<2x4x2x16xbf16>
    %2095 = sdy.sharding_constraint %2094 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2096 = stablehlo.reshape %2054 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %2097 = sdy.sharding_constraint %2096 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2098 = stablehlo.reduce(%2097 init: %cst_269) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x2x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2099 = sdy.sharding_constraint %2098 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2100 = stablehlo.broadcast_in_dim %2099, dims = [0, 1, 4] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %2101 = sdy.sharding_constraint %2100 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %cst_270 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2102 = stablehlo.reduce(%2101 init: %cst_270) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2103 = sdy.sharding_constraint %2102 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2104 = stablehlo.broadcast_in_dim %2103, dims = [0, 1, 3] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %2105 = sdy.sharding_constraint %2104 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2106 = stablehlo.add %2046, %2105 : tensor<2x4x1x16xbf16>
    %2107 = sdy.sharding_constraint %2106 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2108 = stablehlo.reshape %2087 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x1x2x16xbf16>
    %2109 = sdy.sharding_constraint %2108 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x2x16xbf16>
    %cst_271 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2110 = stablehlo.reduce(%2109 init: %cst_271) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x2x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2111 = sdy.sharding_constraint %2110 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2112 = stablehlo.broadcast_in_dim %2111, dims = [0, 1, 4] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x1x16xbf16>
    %2113 = sdy.sharding_constraint %2112 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}, {"model"}]> : tensor<2x4x1x1x16xbf16>
    %cst_272 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2114 = stablehlo.reduce(%2113 init: %cst_272) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1x16xbf16>, tensor<bf16>) -> tensor<2x4x16xbf16>
    %2115 = sdy.sharding_constraint %2114 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2116 = stablehlo.broadcast_in_dim %2115, dims = [0, 1, 3] : (tensor<2x4x16xbf16>) -> tensor<2x4x1x16xbf16>
    %2117 = sdy.sharding_constraint %2116 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %cst_273 = stablehlo.constant dense<1.296880e+00> : tensor<bf16>
    %2118 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<bf16>) -> tensor<2x4x2x16xbf16>
    %2119 = sdy.sharding_constraint %2118 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2120 = stablehlo.multiply %2095, %2119 : tensor<2x4x2x16xbf16>
    %2121 = sdy.sharding_constraint %2120 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2122 = stablehlo.convert %2117 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x1x16xf32>
    %2123 = sdy.sharding_constraint %2122 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2124 = stablehlo.multiply %1220, %2123 : tensor<2x4x1x16xf32>
    %2125 = sdy.sharding_constraint %2124 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %cst_274 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2126 = stablehlo.reduce(%2125 init: %cst_274) applies stablehlo.add across dimensions = [3] : (tensor<2x4x1x16xf32>, tensor<f32>) -> tensor<2x4x1xf32>
    %2127 = sdy.sharding_constraint %2126 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2128 = stablehlo.reshape %2127 : (tensor<2x4x1xf32>) -> tensor<2x4x1x1xf32>
    %2129 = sdy.sharding_constraint %2128 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %2130 = stablehlo.broadcast_in_dim %1212, dims = [0, 1, 2, 3] : (tensor<2x4x1x1xf32>) -> tensor<2x4x1x16xf32>
    %2131 = sdy.sharding_constraint %2130 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2132 = stablehlo.multiply %2123, %2131 : tensor<2x4x1x16xf32>
    %2133 = sdy.sharding_constraint %2132 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2134 = stablehlo.convert %2133 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %2135 = sdy.sharding_constraint %2134 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2136 = stablehlo.multiply %2129, %1218 : tensor<2x4x1x1xf32>
    %2137 = sdy.sharding_constraint %2136 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_275 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %2138 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f32>) -> tensor<2x4x1x1xf32>
    %2139 = sdy.sharding_constraint %2138 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %2140 = stablehlo.divide %2137, %2139 : tensor<2x4x1x1xf32>
    %2141 = sdy.sharding_constraint %2140 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {}]> : tensor<2x4x1x1xf32>
    %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2142 = stablehlo.reduce(%2141 init: %cst_276) applies stablehlo.add across dimensions = [2, 3] : (tensor<2x4x1x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2143 = sdy.sharding_constraint %2142 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2144 = stablehlo.broadcast_in_dim %2143, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %2145 = sdy.sharding_constraint %2144 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2146 = stablehlo.broadcast_in_dim %2145, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x1x16xf32>
    %2147 = sdy.sharding_constraint %2146 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2148 = stablehlo.multiply %2147, %1198 : tensor<2x4x1x16xf32>
    %2149 = sdy.sharding_constraint %2148 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xf32>
    %2150 = stablehlo.convert %2149 : (tensor<2x4x1x16xf32>) -> tensor<2x4x1x16xbf16>
    %2151 = sdy.sharding_constraint %2150 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2152 = stablehlo.add %2135, %2151 : tensor<2x4x1x16xbf16>
    %2153 = sdy.sharding_constraint %2152 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}, {"model"}]> : tensor<2x4x1x16xbf16>
    %2154 = stablehlo.convert %2121 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x2x16xf32>
    %2155 = sdy.sharding_constraint %2154 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2156 = stablehlo.multiply %1185, %2155 : tensor<2x4x2x16xf32>
    %2157 = sdy.sharding_constraint %2156 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %cst_277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2158 = stablehlo.reduce(%2157 init: %cst_277) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x16xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2159 = sdy.sharding_constraint %2158 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %2160 = stablehlo.reshape %2159 : (tensor<2x4x2xf32>) -> tensor<2x4x2x1xf32>
    %2161 = sdy.sharding_constraint %2160 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %2162 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2, 3] : (tensor<2x4x2x1xf32>) -> tensor<2x4x2x16xf32>
    %2163 = sdy.sharding_constraint %2162 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2164 = stablehlo.multiply %2155, %2163 : tensor<2x4x2x16xf32>
    %2165 = sdy.sharding_constraint %2164 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2166 = stablehlo.convert %2165 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2167 = sdy.sharding_constraint %2166 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2168 = stablehlo.multiply %2161, %1183 : tensor<2x4x2x1xf32>
    %2169 = sdy.sharding_constraint %2168 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_278 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %2170 = stablehlo.broadcast_in_dim %cst_278, dims = [] : (tensor<f32>) -> tensor<2x4x2x1xf32>
    %2171 = sdy.sharding_constraint %2170 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %2172 = stablehlo.divide %2169, %2171 : tensor<2x4x2x1xf32>
    %2173 = sdy.sharding_constraint %2172 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x1xf32>
    %cst_279 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2174 = stablehlo.reduce(%2173 init: %cst_279) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x1xf32>, tensor<f32>) -> tensor<2x4x2xf32>
    %2175 = sdy.sharding_constraint %2174 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x2xf32>
    %2176 = stablehlo.broadcast_in_dim %2175, dims = [0, 1, 2] : (tensor<2x4x2xf32>) -> tensor<2x4x2x16xf32>
    %2177 = sdy.sharding_constraint %2176 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2178 = stablehlo.multiply %2177, %1163 : tensor<2x4x2x16xf32>
    %2179 = sdy.sharding_constraint %2178 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xf32>
    %2180 = stablehlo.convert %2179 : (tensor<2x4x2x16xf32>) -> tensor<2x4x2x16xbf16>
    %2181 = sdy.sharding_constraint %2180 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2182 = stablehlo.add %2167, %2181 : tensor<2x4x2x16xbf16>
    %2183 = sdy.sharding_constraint %2182 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}, {}]> : tensor<2x4x2x16xbf16>
    %2184 = stablehlo.reshape %2107 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x16xbf16>
    %2185 = sdy.sharding_constraint %2184 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2186 = stablehlo.dot_general %2185, %1144, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<2x4x32xbf16>) -> tensor<16x32xbf16>
    %2187 = sdy.sharding_constraint %2186 <@mesh, [{"model"}, {"data"}]> : tensor<16x32xbf16>
    %2188 = stablehlo.transpose %2187, dims = [1, 0] : (tensor<16x32xbf16>) -> tensor<32x16xbf16>
    %2189 = sdy.sharding_constraint %2188 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %2190 = stablehlo.dot_general %2185, %1076#5, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<32x16xbf16>) -> tensor<2x4x32xbf16>
    %2191 = sdy.sharding_constraint %2190 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2192 = stablehlo.add %1983, %2191 : tensor<2x4x32xbf16>
    %2193 = sdy.sharding_constraint %2192 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2194 = stablehlo.reshape %2153 : (tensor<2x4x1x16xbf16>) -> tensor<2x4x16xbf16>
    %2195 = sdy.sharding_constraint %2194 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x16xbf16>
    %2196 = stablehlo.dot_general %2195, %1144, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<2x4x32xbf16>) -> tensor<16x32xbf16>
    %2197 = sdy.sharding_constraint %2196 <@mesh, [{"model"}, {"data"}]> : tensor<16x32xbf16>
    %2198 = stablehlo.transpose %2197, dims = [1, 0] : (tensor<16x32xbf16>) -> tensor<32x16xbf16>
    %2199 = sdy.sharding_constraint %2198 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xbf16>
    %2200 = stablehlo.dot_general %2195, %1076#4, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x16xbf16>, tensor<32x16xbf16>) -> tensor<2x4x32xbf16>
    %2201 = sdy.sharding_constraint %2200 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2202 = stablehlo.add %2193, %2201 : tensor<2x4x32xbf16>
    %2203 = sdy.sharding_constraint %2202 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2204 = stablehlo.reshape %2183 : (tensor<2x4x2x16xbf16>) -> tensor<2x4x32xbf16>
    %2205 = sdy.sharding_constraint %2204 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {"model"}]> : tensor<2x4x32xbf16>
    %2206 = stablehlo.dot_general %2205, %1144, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x32xbf16>) -> tensor<32x32xbf16>
    %2207 = sdy.sharding_constraint %2206 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xbf16>
    %2208 = stablehlo.transpose %2207, dims = [1, 0] : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2209 = sdy.sharding_constraint %2208 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xbf16>
    %2210 = stablehlo.dot_general %2205, %1076#3, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<32x32xbf16>) -> tensor<2x4x32xbf16>
    %2211 = sdy.sharding_constraint %2210 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2212 = stablehlo.add %2203, %2211 : tensor<2x4x32xbf16>
    %2213 = sdy.sharding_constraint %2212 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2214 = stablehlo.multiply %1118, %2213 : tensor<2x4x32xbf16>
    %2215 = sdy.sharding_constraint %2214 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2216 = stablehlo.multiply %2213, %1135 : tensor<2x4x32xbf16>
    %2217 = sdy.sharding_constraint %2216 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2218 = stablehlo.multiply %2215, %1142 : tensor<2x4x32xbf16>
    %2219 = sdy.sharding_constraint %2218 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2220 = stablehlo.dot_general %2219, %1121#0, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %2221 = sdy.sharding_constraint %2220 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2222 = stablehlo.transpose %2221, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %2223 = sdy.sharding_constraint %2222 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2224 = stablehlo.dot_general %2219, %1076#2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %2225 = sdy.sharding_constraint %2224 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2226 = call @silu_463(%1121#1, %1120, %1121#2, %2225) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %2227 = stablehlo.dot_general %2226, %1118, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %2228 = sdy.sharding_constraint %2227 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2229 = stablehlo.transpose %2228, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %2230 = sdy.sharding_constraint %2229 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2231 = stablehlo.dot_general %2226, %1076#1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %2232 = sdy.sharding_constraint %2231 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2233 = stablehlo.add %2217, %2232 : tensor<2x4x32xbf16>
    %2234 = sdy.sharding_constraint %2233 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2235 = stablehlo.convert %2234 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %2236 = sdy.sharding_constraint %2235 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2237 = stablehlo.multiply %1108, %2236 : tensor<2x4x32xf32>
    %2238 = sdy.sharding_constraint %2237 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2239 = stablehlo.reduce(%2238 init: %cst_280) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2240 = sdy.sharding_constraint %2239 <@mesh, [{}]> : tensor<32xf32>
    %2241 = stablehlo.reshape %2240 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %2242 = sdy.sharding_constraint %2241 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %2243 = stablehlo.broadcast_in_dim %1112, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %2244 = sdy.sharding_constraint %2243 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2245 = stablehlo.multiply %2236, %2244 : tensor<2x4x32xf32>
    %2246 = sdy.sharding_constraint %2245 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_281 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2247 = stablehlo.reduce(%2242 init: %cst_281) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2248 = sdy.sharding_constraint %2247 <@mesh, [{}]> : tensor<32xf32>
    %2249 = stablehlo.convert %2248 : (tensor<32xf32>) -> tensor<32xbf16>
    %2250 = sdy.sharding_constraint %2249 <@mesh, [{}]> : tensor<32xbf16>
    %2251 = stablehlo.multiply %1079, %2246 : tensor<2x4x32xf32>
    %2252 = sdy.sharding_constraint %2251 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_282 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2253 = stablehlo.reduce(%2252 init: %cst_282) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2254 = sdy.sharding_constraint %2253 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2255 = stablehlo.reshape %2254 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %2256 = sdy.sharding_constraint %2255 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2257 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %2258 = sdy.sharding_constraint %2257 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2259 = stablehlo.multiply %2246, %2258 : tensor<2x4x32xf32>
    %2260 = sdy.sharding_constraint %2259 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2261 = stablehlo.multiply %2256, %1104 : tensor<2x4x1xf32>
    %2262 = sdy.sharding_constraint %2261 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_283 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %2263 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %2264 = sdy.sharding_constraint %2263 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2265 = stablehlo.divide %2262, %2264 : tensor<2x4x1xf32>
    %2266 = sdy.sharding_constraint %2265 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2267 = stablehlo.reduce(%2266 init: %cst_284) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2268 = sdy.sharding_constraint %2267 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2269 = stablehlo.broadcast_in_dim %2268, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %2270 = sdy.sharding_constraint %2269 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2271 = stablehlo.multiply %2270, %1084 : tensor<2x4x32xf32>
    %2272 = sdy.sharding_constraint %2271 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2273 = stablehlo.add %2260, %2272 : tensor<2x4x32xf32>
    %2274 = sdy.sharding_constraint %2273 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2275 = stablehlo.convert %2274 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %2276 = sdy.sharding_constraint %2275 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2277 = stablehlo.add %1950, %2276 : tensor<2x4x32xbf16>
    %2278 = sdy.sharding_constraint %2277 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2279 = sdy.sharding_constraint %2250 <@mesh, [{}]> : tensor<32xbf16>
    %2280 = stablehlo.multiply %116, %2278 : tensor<2x4x32xbf16>
    %2281 = sdy.sharding_constraint %2280 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2282 = stablehlo.multiply %2278, %133 : tensor<2x4x32xbf16>
    %2283 = sdy.sharding_constraint %2282 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2284 = stablehlo.multiply %2281, %140 : tensor<2x4x32xbf16>
    %2285 = sdy.sharding_constraint %2284 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2286 = stablehlo.dot_general %2285, %119#0, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<2x4x128xbf16>) -> tensor<32x128xbf16>
    %2287 = sdy.sharding_constraint %2286 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2288 = stablehlo.transpose %2287, dims = [1, 0] : (tensor<32x128xbf16>) -> tensor<128x32xbf16>
    %2289 = sdy.sharding_constraint %2288 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2290 = stablehlo.dot_general %2285, %20, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x32xbf16>, tensor<128x32xbf16>) -> tensor<2x4x128xbf16>
    %2291 = sdy.sharding_constraint %2290 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x128xbf16>
    %2292 = call @silu_329(%119#1, %119#2, %118, %2291) : (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16>
    %2293 = stablehlo.dot_general %2292, %116, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<2x4x32xbf16>) -> tensor<128x32xbf16>
    %2294 = sdy.sharding_constraint %2293 <@mesh, [{}, {}]> : tensor<128x32xbf16>
    %2295 = stablehlo.transpose %2294, dims = [1, 0] : (tensor<128x32xbf16>) -> tensor<32x128xbf16>
    %2296 = sdy.sharding_constraint %2295 <@mesh, [{}, {}]> : tensor<32x128xbf16>
    %2297 = stablehlo.dot_general %2292, %18, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x128xbf16>, tensor<32x128xbf16>) -> tensor<2x4x32xbf16>
    %2298 = sdy.sharding_constraint %2297 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2299 = stablehlo.add %2283, %2298 : tensor<2x4x32xbf16>
    %2300 = sdy.sharding_constraint %2299 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2301 = stablehlo.convert %2300 : (tensor<2x4x32xbf16>) -> tensor<2x4x32xf32>
    %2302 = sdy.sharding_constraint %2301 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2303 = stablehlo.multiply %106, %2302 : tensor<2x4x32xf32>
    %2304 = sdy.sharding_constraint %2303 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2305 = stablehlo.reduce(%2304 init: %cst_285) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2306 = sdy.sharding_constraint %2305 <@mesh, [{}]> : tensor<32xf32>
    %2307 = stablehlo.reshape %2306 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %2308 = sdy.sharding_constraint %2307 <@mesh, [{}, {}, {}]> : tensor<1x1x32xf32>
    %2309 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<2x4x32xf32>
    %2310 = sdy.sharding_constraint %2309 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2311 = stablehlo.multiply %2302, %2310 : tensor<2x4x32xf32>
    %2312 = sdy.sharding_constraint %2311 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2313 = stablehlo.reduce(%2308 init: %cst_286) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x32xf32>, tensor<f32>) -> tensor<32xf32>
    %2314 = sdy.sharding_constraint %2313 <@mesh, [{}]> : tensor<32xf32>
    %2315 = stablehlo.convert %2314 : (tensor<32xf32>) -> tensor<32xbf16>
    %2316 = sdy.sharding_constraint %2315 <@mesh, [{}]> : tensor<32xbf16>
    %2317 = stablehlo.multiply %77, %2312 : tensor<2x4x32xf32>
    %2318 = sdy.sharding_constraint %2317 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %cst_287 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2319 = stablehlo.reduce(%2318 init: %cst_287) applies stablehlo.add across dimensions = [2] : (tensor<2x4x32xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2320 = sdy.sharding_constraint %2319 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2321 = stablehlo.reshape %2320 : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %2322 = sdy.sharding_constraint %2321 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2323 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x32xf32>
    %2324 = sdy.sharding_constraint %2323 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2325 = stablehlo.multiply %2312, %2324 : tensor<2x4x32xf32>
    %2326 = sdy.sharding_constraint %2325 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2327 = stablehlo.multiply %2322, %102 : tensor<2x4x1xf32>
    %2328 = sdy.sharding_constraint %2327 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_288 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %2329 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<2x4x1xf32>
    %2330 = sdy.sharding_constraint %2329 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %2331 = stablehlo.divide %2328, %2330 : tensor<2x4x1xf32>
    %2332 = sdy.sharding_constraint %2331 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x1xf32>
    %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2333 = stablehlo.reduce(%2332 init: %cst_289) applies stablehlo.add across dimensions = [2] : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4xf32>
    %2334 = sdy.sharding_constraint %2333 <@mesh, [{"replica_dcn", "data", "expert"}, {}]> : tensor<2x4xf32>
    %2335 = stablehlo.broadcast_in_dim %2334, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x32xf32>
    %2336 = sdy.sharding_constraint %2335 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2337 = stablehlo.multiply %2336, %82 : tensor<2x4x32xf32>
    %2338 = sdy.sharding_constraint %2337 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2339 = stablehlo.add %2326, %2338 : tensor<2x4x32xf32>
    %2340 = sdy.sharding_constraint %2339 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xf32>
    %2341 = stablehlo.convert %2340 : (tensor<2x4x32xf32>) -> tensor<2x4x32xbf16>
    %2342 = sdy.sharding_constraint %2341 <@mesh, [{"replica_dcn", "data", "expert"}, {}, {}]> : tensor<2x4x32xbf16>
    %2343 = sdy.sharding_constraint %2316 <@mesh, [{}]> : tensor<32xbf16>
    %cst_290 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2344 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<bf16>) -> tensor<64x32xbf16>
    %2345 = "stablehlo.scatter"(%2344, %72, %2342) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg85: tensor<bf16>, %arg86: tensor<bf16>):
      %3950 = stablehlo.add %arg85, %arg86 : tensor<bf16>
      stablehlo.return %3950 : tensor<bf16>
    }) : (tensor<64x32xbf16>, tensor<2x4x1xi32>, tensor<2x4x32xbf16>) -> tensor<64x32xbf16>
    %2346 = sdy.sharding_constraint %2345 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xbf16>
    %2347 = stablehlo.convert %1018 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2348 = sdy.sharding_constraint %2347 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2349 = stablehlo.convert %1025 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2350 = sdy.sharding_constraint %2349 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2351 = stablehlo.convert %1072 : (tensor<32xbf16>) -> tensor<32xf32>
    %2352 = sdy.sharding_constraint %2351 <@mesh, [{}]> : tensor<32xf32>
    %2353 = stablehlo.convert %1809 : (tensor<4x32x32xbf16>) -> tensor<4x32x32xf32>
    %2354 = sdy.sharding_constraint %2353 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2355 = stablehlo.convert %1816 : (tensor<4x32x32xbf16>) -> tensor<4x32x32xf32>
    %2356 = sdy.sharding_constraint %2355 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2357 = stablehlo.convert %1814 : (tensor<4x32x32xbf16>) -> tensor<4x32x32xf32>
    %2358 = sdy.sharding_constraint %2357 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2359 = stablehlo.convert %1658 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2360 = sdy.sharding_constraint %2359 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2361 = stablehlo.convert %1669 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2362 = sdy.sharding_constraint %2361 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2363 = stablehlo.convert %1675 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2364 = sdy.sharding_constraint %2363 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2365 = stablehlo.convert %1881 : (tensor<32x4xbf16>) -> tensor<32x4xf32>
    %2366 = sdy.sharding_constraint %2365 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2367 = stablehlo.convert %1895 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2368 = sdy.sharding_constraint %2367 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2369 = stablehlo.convert %1902 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2370 = sdy.sharding_constraint %2369 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2371 = stablehlo.convert %1951 : (tensor<32xbf16>) -> tensor<32xf32>
    %2372 = sdy.sharding_constraint %2371 <@mesh, [{}]> : tensor<32xf32>
    %2373 = stablehlo.convert %1981 : (tensor<32x2xbf16>) -> tensor<32x2xf32>
    %2374 = sdy.sharding_constraint %2373 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2375 = stablehlo.convert %1955 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2376 = sdy.sharding_constraint %2375 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2377 = stablehlo.convert %2189 : (tensor<32x16xbf16>) -> tensor<32x16xf32>
    %2378 = sdy.sharding_constraint %2377 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2379 = stablehlo.convert %2199 : (tensor<32x16xbf16>) -> tensor<32x16xf32>
    %2380 = sdy.sharding_constraint %2379 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2381 = stablehlo.convert %2209 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
    %2382 = sdy.sharding_constraint %2381 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2383 = stablehlo.convert %2223 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2384 = sdy.sharding_constraint %2383 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2385 = stablehlo.convert %2230 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2386 = sdy.sharding_constraint %2385 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2387 = stablehlo.convert %2279 : (tensor<32xbf16>) -> tensor<32xf32>
    %2388 = sdy.sharding_constraint %2387 <@mesh, [{}]> : tensor<32xf32>
    %2389 = stablehlo.convert %1007 : (tensor<32x64xbf16>) -> tensor<32x64xf32>
    %2390 = sdy.sharding_constraint %2389 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2391 = stablehlo.convert %2289 : (tensor<128x32xbf16>) -> tensor<128x32xf32>
    %2392 = sdy.sharding_constraint %2391 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2393 = stablehlo.convert %2296 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %2394 = sdy.sharding_constraint %2393 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2395 = stablehlo.convert %2343 : (tensor<32xbf16>) -> tensor<32xf32>
    %2396 = sdy.sharding_constraint %2395 <@mesh, [{}]> : tensor<32xf32>
    %2397 = stablehlo.convert %2346 : (tensor<64x32xbf16>) -> tensor<64x32xf32>
    %2398 = sdy.sharding_constraint %2397 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_291 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2399 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2400 = sdy.sharding_constraint %2399 <@mesh, [{}]> : tensor<4xf32>
    %2401 = sdy.sharding_constraint %2398 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_292 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2402 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2403 = sdy.sharding_constraint %2402 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2404 = stablehlo.multiply %2403, %2401 : tensor<64x32xf32>
    %2405 = sdy.sharding_constraint %2404 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_293 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2406 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2407 = sdy.sharding_constraint %2406 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2408 = stablehlo.multiply %2407, %arg28 : tensor<64x32xf32>
    %2409 = sdy.sharding_constraint %2408 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2410 = stablehlo.add %2405, %2409 : tensor<64x32xf32>
    %2411 = sdy.sharding_constraint %2410 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2412 = sdy.sharding_constraint %2396 <@mesh, [{}]> : tensor<32xf32>
    %cst_294 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2413 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2414 = sdy.sharding_constraint %2413 <@mesh, [{}]> : tensor<32xf32>
    %2415 = stablehlo.multiply %2414, %2412 : tensor<32xf32>
    %2416 = sdy.sharding_constraint %2415 <@mesh, [{}]> : tensor<32xf32>
    %cst_295 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2417 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2418 = sdy.sharding_constraint %2417 <@mesh, [{}]> : tensor<32xf32>
    %2419 = stablehlo.multiply %2418, %arg29 : tensor<32xf32>
    %2420 = sdy.sharding_constraint %2419 <@mesh, [{}]> : tensor<32xf32>
    %2421 = stablehlo.add %2416, %2420 : tensor<32xf32>
    %2422 = sdy.sharding_constraint %2421 <@mesh, [{}]> : tensor<32xf32>
    %2423 = sdy.sharding_constraint %2394 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_296 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2424 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2425 = sdy.sharding_constraint %2424 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2426 = stablehlo.multiply %2425, %2423 : tensor<32x128xf32>
    %2427 = sdy.sharding_constraint %2426 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_297 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2428 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2429 = sdy.sharding_constraint %2428 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2430 = stablehlo.multiply %2429, %arg30 : tensor<32x128xf32>
    %2431 = sdy.sharding_constraint %2430 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2432 = stablehlo.add %2427, %2431 : tensor<32x128xf32>
    %2433 = sdy.sharding_constraint %2432 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2434 = sdy.sharding_constraint %2392 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_298 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2435 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2436 = sdy.sharding_constraint %2435 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2437 = stablehlo.multiply %2436, %2434 : tensor<128x32xf32>
    %2438 = sdy.sharding_constraint %2437 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_299 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2439 = stablehlo.broadcast_in_dim %cst_299, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2440 = sdy.sharding_constraint %2439 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2441 = stablehlo.multiply %2440, %arg31 : tensor<128x32xf32>
    %2442 = sdy.sharding_constraint %2441 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2443 = stablehlo.add %2438, %2442 : tensor<128x32xf32>
    %2444 = sdy.sharding_constraint %2443 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2445 = sdy.sharding_constraint %2390 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_300 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2446 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %2447 = sdy.sharding_constraint %2446 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2448 = stablehlo.multiply %2447, %2445 : tensor<32x64xf32>
    %2449 = sdy.sharding_constraint %2448 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_301 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2450 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %2451 = sdy.sharding_constraint %2450 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2452 = stablehlo.multiply %2451, %arg32 : tensor<32x64xf32>
    %2453 = sdy.sharding_constraint %2452 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2454 = stablehlo.add %2449, %2453 : tensor<32x64xf32>
    %2455 = sdy.sharding_constraint %2454 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2456 = sdy.sharding_constraint %2388 <@mesh, [{}]> : tensor<32xf32>
    %cst_302 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2457 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2458 = sdy.sharding_constraint %2457 <@mesh, [{}]> : tensor<32xf32>
    %2459 = stablehlo.multiply %2458, %2456 : tensor<32xf32>
    %2460 = sdy.sharding_constraint %2459 <@mesh, [{}]> : tensor<32xf32>
    %cst_303 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2461 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2462 = sdy.sharding_constraint %2461 <@mesh, [{}]> : tensor<32xf32>
    %2463 = stablehlo.multiply %2462, %arg33 : tensor<32xf32>
    %2464 = sdy.sharding_constraint %2463 <@mesh, [{}]> : tensor<32xf32>
    %2465 = stablehlo.add %2460, %2464 : tensor<32xf32>
    %2466 = sdy.sharding_constraint %2465 <@mesh, [{}]> : tensor<32xf32>
    %2467 = sdy.sharding_constraint %2386 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_304 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2468 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2469 = sdy.sharding_constraint %2468 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2470 = stablehlo.multiply %2469, %2467 : tensor<32x128xf32>
    %2471 = sdy.sharding_constraint %2470 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_305 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2472 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2473 = sdy.sharding_constraint %2472 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2474 = stablehlo.multiply %2473, %arg34 : tensor<32x128xf32>
    %2475 = sdy.sharding_constraint %2474 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2476 = stablehlo.add %2471, %2475 : tensor<32x128xf32>
    %2477 = sdy.sharding_constraint %2476 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2478 = sdy.sharding_constraint %2384 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_306 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2479 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2480 = sdy.sharding_constraint %2479 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2481 = stablehlo.multiply %2480, %2478 : tensor<128x32xf32>
    %2482 = sdy.sharding_constraint %2481 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_307 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2483 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2484 = sdy.sharding_constraint %2483 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2485 = stablehlo.multiply %2484, %arg35 : tensor<128x32xf32>
    %2486 = sdy.sharding_constraint %2485 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2487 = stablehlo.add %2482, %2486 : tensor<128x32xf32>
    %2488 = sdy.sharding_constraint %2487 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2489 = sdy.sharding_constraint %2382 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_308 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2490 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2491 = sdy.sharding_constraint %2490 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2492 = stablehlo.multiply %2491, %2489 : tensor<32x32xf32>
    %2493 = sdy.sharding_constraint %2492 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_309 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2494 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2495 = sdy.sharding_constraint %2494 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2496 = stablehlo.multiply %2495, %arg36 : tensor<32x32xf32>
    %2497 = sdy.sharding_constraint %2496 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2498 = stablehlo.add %2493, %2497 : tensor<32x32xf32>
    %2499 = sdy.sharding_constraint %2498 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2500 = sdy.sharding_constraint %2380 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_310 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2501 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2502 = sdy.sharding_constraint %2501 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2503 = stablehlo.multiply %2502, %2500 : tensor<32x16xf32>
    %2504 = sdy.sharding_constraint %2503 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_311 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2505 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2506 = sdy.sharding_constraint %2505 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2507 = stablehlo.multiply %2506, %arg37 : tensor<32x16xf32>
    %2508 = sdy.sharding_constraint %2507 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2509 = stablehlo.add %2504, %2508 : tensor<32x16xf32>
    %2510 = sdy.sharding_constraint %2509 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2511 = sdy.sharding_constraint %2378 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_312 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2512 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2513 = sdy.sharding_constraint %2512 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2514 = stablehlo.multiply %2513, %2511 : tensor<32x16xf32>
    %2515 = sdy.sharding_constraint %2514 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_313 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2516 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2517 = sdy.sharding_constraint %2516 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2518 = stablehlo.multiply %2517, %arg38 : tensor<32x16xf32>
    %2519 = sdy.sharding_constraint %2518 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2520 = stablehlo.add %2515, %2519 : tensor<32x16xf32>
    %2521 = sdy.sharding_constraint %2520 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2522 = sdy.sharding_constraint %2376 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_314 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2523 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2524 = sdy.sharding_constraint %2523 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2525 = stablehlo.multiply %2524, %2522 : tensor<32x32xf32>
    %2526 = sdy.sharding_constraint %2525 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_315 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2527 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2528 = sdy.sharding_constraint %2527 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2529 = stablehlo.multiply %2528, %arg39 : tensor<32x32xf32>
    %2530 = sdy.sharding_constraint %2529 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2531 = stablehlo.add %2526, %2530 : tensor<32x32xf32>
    %2532 = sdy.sharding_constraint %2531 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2533 = sdy.sharding_constraint %2374 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_316 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2534 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %2535 = sdy.sharding_constraint %2534 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2536 = stablehlo.multiply %2535, %2533 : tensor<32x2xf32>
    %2537 = sdy.sharding_constraint %2536 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_317 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2538 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %2539 = sdy.sharding_constraint %2538 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2540 = stablehlo.multiply %2539, %arg40 : tensor<32x2xf32>
    %2541 = sdy.sharding_constraint %2540 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2542 = stablehlo.add %2537, %2541 : tensor<32x2xf32>
    %2543 = sdy.sharding_constraint %2542 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2544 = sdy.sharding_constraint %2372 <@mesh, [{}]> : tensor<32xf32>
    %cst_318 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2545 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2546 = sdy.sharding_constraint %2545 <@mesh, [{}]> : tensor<32xf32>
    %2547 = stablehlo.multiply %2546, %2544 : tensor<32xf32>
    %2548 = sdy.sharding_constraint %2547 <@mesh, [{}]> : tensor<32xf32>
    %cst_319 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2549 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2550 = sdy.sharding_constraint %2549 <@mesh, [{}]> : tensor<32xf32>
    %2551 = stablehlo.multiply %2550, %arg41 : tensor<32xf32>
    %2552 = sdy.sharding_constraint %2551 <@mesh, [{}]> : tensor<32xf32>
    %2553 = stablehlo.add %2548, %2552 : tensor<32xf32>
    %2554 = sdy.sharding_constraint %2553 <@mesh, [{}]> : tensor<32xf32>
    %2555 = sdy.sharding_constraint %2370 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_320 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2556 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2557 = sdy.sharding_constraint %2556 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2558 = stablehlo.multiply %2557, %2555 : tensor<32x128xf32>
    %2559 = sdy.sharding_constraint %2558 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_321 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2560 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2561 = sdy.sharding_constraint %2560 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2562 = stablehlo.multiply %2561, %arg42 : tensor<32x128xf32>
    %2563 = sdy.sharding_constraint %2562 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2564 = stablehlo.add %2559, %2563 : tensor<32x128xf32>
    %2565 = sdy.sharding_constraint %2564 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2566 = sdy.sharding_constraint %2368 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_322 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2567 = stablehlo.broadcast_in_dim %cst_322, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2568 = sdy.sharding_constraint %2567 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2569 = stablehlo.multiply %2568, %2566 : tensor<128x32xf32>
    %2570 = sdy.sharding_constraint %2569 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_323 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2571 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2572 = sdy.sharding_constraint %2571 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2573 = stablehlo.multiply %2572, %arg43 : tensor<128x32xf32>
    %2574 = sdy.sharding_constraint %2573 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2575 = stablehlo.add %2570, %2574 : tensor<128x32xf32>
    %2576 = sdy.sharding_constraint %2575 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2577 = sdy.sharding_constraint %2366 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_324 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2578 = stablehlo.broadcast_in_dim %cst_324, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %2579 = sdy.sharding_constraint %2578 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2580 = stablehlo.multiply %2579, %2577 : tensor<32x4xf32>
    %2581 = sdy.sharding_constraint %2580 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_325 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2582 = stablehlo.broadcast_in_dim %cst_325, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %2583 = sdy.sharding_constraint %2582 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2584 = stablehlo.multiply %2583, %arg44 : tensor<32x4xf32>
    %2585 = sdy.sharding_constraint %2584 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2586 = stablehlo.add %2581, %2585 : tensor<32x4xf32>
    %2587 = sdy.sharding_constraint %2586 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2588 = sdy.sharding_constraint %2400 <@mesh, [{}]> : tensor<4xf32>
    %cst_326 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2589 = stablehlo.broadcast_in_dim %cst_326, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2590 = sdy.sharding_constraint %2589 <@mesh, [{}]> : tensor<4xf32>
    %2591 = stablehlo.multiply %2590, %2588 : tensor<4xf32>
    %2592 = sdy.sharding_constraint %2591 <@mesh, [{}]> : tensor<4xf32>
    %cst_327 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2593 = stablehlo.broadcast_in_dim %cst_327, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2594 = sdy.sharding_constraint %2593 <@mesh, [{}]> : tensor<4xf32>
    %2595 = stablehlo.multiply %2594, %arg45 : tensor<4xf32>
    %2596 = sdy.sharding_constraint %2595 <@mesh, [{}]> : tensor<4xf32>
    %2597 = stablehlo.add %2592, %2596 : tensor<4xf32>
    %2598 = sdy.sharding_constraint %2597 <@mesh, [{}]> : tensor<4xf32>
    %2599 = sdy.sharding_constraint %2364 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_328 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2600 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2601 = sdy.sharding_constraint %2600 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2602 = stablehlo.multiply %2601, %2599 : tensor<32x32xf32>
    %2603 = sdy.sharding_constraint %2602 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_329 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2604 = stablehlo.broadcast_in_dim %cst_329, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2605 = sdy.sharding_constraint %2604 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2606 = stablehlo.multiply %2605, %arg46 : tensor<32x32xf32>
    %2607 = sdy.sharding_constraint %2606 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2608 = stablehlo.add %2603, %2607 : tensor<32x32xf32>
    %2609 = sdy.sharding_constraint %2608 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2610 = sdy.sharding_constraint %2362 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_330 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2611 = stablehlo.broadcast_in_dim %cst_330, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2612 = sdy.sharding_constraint %2611 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2613 = stablehlo.multiply %2612, %2610 : tensor<32x32xf32>
    %2614 = sdy.sharding_constraint %2613 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_331 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2615 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2616 = sdy.sharding_constraint %2615 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2617 = stablehlo.multiply %2616, %arg47 : tensor<32x32xf32>
    %2618 = sdy.sharding_constraint %2617 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2619 = stablehlo.add %2614, %2618 : tensor<32x32xf32>
    %2620 = sdy.sharding_constraint %2619 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2621 = sdy.sharding_constraint %2360 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_332 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2622 = stablehlo.broadcast_in_dim %cst_332, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2623 = sdy.sharding_constraint %2622 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2624 = stablehlo.multiply %2623, %2621 : tensor<32x32xf32>
    %2625 = sdy.sharding_constraint %2624 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_333 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2626 = stablehlo.broadcast_in_dim %cst_333, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2627 = sdy.sharding_constraint %2626 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2628 = stablehlo.multiply %2627, %arg48 : tensor<32x32xf32>
    %2629 = sdy.sharding_constraint %2628 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2630 = stablehlo.add %2625, %2629 : tensor<32x32xf32>
    %2631 = sdy.sharding_constraint %2630 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2632 = sdy.sharding_constraint %2358 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_334 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2633 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2634 = sdy.sharding_constraint %2633 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2635 = stablehlo.multiply %2634, %2632 : tensor<4x32x32xf32>
    %2636 = sdy.sharding_constraint %2635 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_335 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2637 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2638 = sdy.sharding_constraint %2637 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2639 = stablehlo.multiply %2638, %arg49 : tensor<4x32x32xf32>
    %2640 = sdy.sharding_constraint %2639 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2641 = stablehlo.add %2636, %2640 : tensor<4x32x32xf32>
    %2642 = sdy.sharding_constraint %2641 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2643 = sdy.sharding_constraint %2356 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_336 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2644 = stablehlo.broadcast_in_dim %cst_336, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2645 = sdy.sharding_constraint %2644 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2646 = stablehlo.multiply %2645, %2643 : tensor<4x32x32xf32>
    %2647 = sdy.sharding_constraint %2646 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_337 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2648 = stablehlo.broadcast_in_dim %cst_337, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2649 = sdy.sharding_constraint %2648 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2650 = stablehlo.multiply %2649, %arg50 : tensor<4x32x32xf32>
    %2651 = sdy.sharding_constraint %2650 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2652 = stablehlo.add %2647, %2651 : tensor<4x32x32xf32>
    %2653 = sdy.sharding_constraint %2652 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2654 = sdy.sharding_constraint %2354 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_338 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2655 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2656 = sdy.sharding_constraint %2655 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2657 = stablehlo.multiply %2656, %2654 : tensor<4x32x32xf32>
    %2658 = sdy.sharding_constraint %2657 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_339 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2659 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2660 = sdy.sharding_constraint %2659 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2661 = stablehlo.multiply %2660, %arg51 : tensor<4x32x32xf32>
    %2662 = sdy.sharding_constraint %2661 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2663 = stablehlo.add %2658, %2662 : tensor<4x32x32xf32>
    %2664 = sdy.sharding_constraint %2663 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2665 = sdy.sharding_constraint %2352 <@mesh, [{}]> : tensor<32xf32>
    %cst_340 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2666 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2667 = sdy.sharding_constraint %2666 <@mesh, [{}]> : tensor<32xf32>
    %2668 = stablehlo.multiply %2667, %2665 : tensor<32xf32>
    %2669 = sdy.sharding_constraint %2668 <@mesh, [{}]> : tensor<32xf32>
    %cst_341 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2670 = stablehlo.broadcast_in_dim %cst_341, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2671 = sdy.sharding_constraint %2670 <@mesh, [{}]> : tensor<32xf32>
    %2672 = stablehlo.multiply %2671, %arg52 : tensor<32xf32>
    %2673 = sdy.sharding_constraint %2672 <@mesh, [{}]> : tensor<32xf32>
    %2674 = stablehlo.add %2669, %2673 : tensor<32xf32>
    %2675 = sdy.sharding_constraint %2674 <@mesh, [{}]> : tensor<32xf32>
    %2676 = sdy.sharding_constraint %2350 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_342 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2677 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2678 = sdy.sharding_constraint %2677 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2679 = stablehlo.multiply %2678, %2676 : tensor<32x128xf32>
    %2680 = sdy.sharding_constraint %2679 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_343 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2681 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2682 = sdy.sharding_constraint %2681 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2683 = stablehlo.multiply %2682, %arg53 : tensor<32x128xf32>
    %2684 = sdy.sharding_constraint %2683 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2685 = stablehlo.add %2680, %2684 : tensor<32x128xf32>
    %2686 = sdy.sharding_constraint %2685 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2687 = sdy.sharding_constraint %2348 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_344 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %2688 = stablehlo.broadcast_in_dim %cst_344, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2689 = sdy.sharding_constraint %2688 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2690 = stablehlo.multiply %2689, %2687 : tensor<128x32xf32>
    %2691 = sdy.sharding_constraint %2690 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_345 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %2692 = stablehlo.broadcast_in_dim %cst_345, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2693 = sdy.sharding_constraint %2692 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2694 = stablehlo.multiply %2693, %arg54 : tensor<128x32xf32>
    %2695 = sdy.sharding_constraint %2694 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2696 = stablehlo.add %2691, %2695 : tensor<128x32xf32>
    %2697 = sdy.sharding_constraint %2696 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2698 = stablehlo.multiply %2398, %2398 : tensor<64x32xf32>
    %2699 = sdy.sharding_constraint %2698 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_346 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2700 = stablehlo.broadcast_in_dim %cst_346, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2701 = sdy.sharding_constraint %2700 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2702 = stablehlo.multiply %2701, %2699 : tensor<64x32xf32>
    %2703 = sdy.sharding_constraint %2702 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_347 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2704 = stablehlo.broadcast_in_dim %cst_347, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %2705 = sdy.sharding_constraint %2704 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2706 = stablehlo.multiply %2705, %arg55 : tensor<64x32xf32>
    %2707 = sdy.sharding_constraint %2706 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2708 = stablehlo.add %2703, %2707 : tensor<64x32xf32>
    %2709 = sdy.sharding_constraint %2708 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %2710 = stablehlo.multiply %2396, %2396 : tensor<32xf32>
    %2711 = sdy.sharding_constraint %2710 <@mesh, [{}]> : tensor<32xf32>
    %cst_348 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2712 = stablehlo.broadcast_in_dim %cst_348, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2713 = sdy.sharding_constraint %2712 <@mesh, [{}]> : tensor<32xf32>
    %2714 = stablehlo.multiply %2713, %2711 : tensor<32xf32>
    %2715 = sdy.sharding_constraint %2714 <@mesh, [{}]> : tensor<32xf32>
    %cst_349 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2716 = stablehlo.broadcast_in_dim %cst_349, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2717 = sdy.sharding_constraint %2716 <@mesh, [{}]> : tensor<32xf32>
    %2718 = stablehlo.multiply %2717, %arg56 : tensor<32xf32>
    %2719 = sdy.sharding_constraint %2718 <@mesh, [{}]> : tensor<32xf32>
    %2720 = stablehlo.add %2715, %2719 : tensor<32xf32>
    %2721 = sdy.sharding_constraint %2720 <@mesh, [{}]> : tensor<32xf32>
    %2722 = stablehlo.multiply %2394, %2394 : tensor<32x128xf32>
    %2723 = sdy.sharding_constraint %2722 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_350 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2724 = stablehlo.broadcast_in_dim %cst_350, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2725 = sdy.sharding_constraint %2724 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2726 = stablehlo.multiply %2725, %2723 : tensor<32x128xf32>
    %2727 = sdy.sharding_constraint %2726 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_351 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2728 = stablehlo.broadcast_in_dim %cst_351, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2729 = sdy.sharding_constraint %2728 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2730 = stablehlo.multiply %2729, %arg57 : tensor<32x128xf32>
    %2731 = sdy.sharding_constraint %2730 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2732 = stablehlo.add %2727, %2731 : tensor<32x128xf32>
    %2733 = sdy.sharding_constraint %2732 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2734 = stablehlo.multiply %2392, %2392 : tensor<128x32xf32>
    %2735 = sdy.sharding_constraint %2734 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_352 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2736 = stablehlo.broadcast_in_dim %cst_352, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2737 = sdy.sharding_constraint %2736 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2738 = stablehlo.multiply %2737, %2735 : tensor<128x32xf32>
    %2739 = sdy.sharding_constraint %2738 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_353 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2740 = stablehlo.broadcast_in_dim %cst_353, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2741 = sdy.sharding_constraint %2740 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2742 = stablehlo.multiply %2741, %arg58 : tensor<128x32xf32>
    %2743 = sdy.sharding_constraint %2742 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2744 = stablehlo.add %2739, %2743 : tensor<128x32xf32>
    %2745 = sdy.sharding_constraint %2744 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2746 = stablehlo.multiply %2390, %2390 : tensor<32x64xf32>
    %2747 = sdy.sharding_constraint %2746 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_354 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2748 = stablehlo.broadcast_in_dim %cst_354, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %2749 = sdy.sharding_constraint %2748 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2750 = stablehlo.multiply %2749, %2747 : tensor<32x64xf32>
    %2751 = sdy.sharding_constraint %2750 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_355 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2752 = stablehlo.broadcast_in_dim %cst_355, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %2753 = sdy.sharding_constraint %2752 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2754 = stablehlo.multiply %2753, %arg59 : tensor<32x64xf32>
    %2755 = sdy.sharding_constraint %2754 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2756 = stablehlo.add %2751, %2755 : tensor<32x64xf32>
    %2757 = sdy.sharding_constraint %2756 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %2758 = stablehlo.multiply %2388, %2388 : tensor<32xf32>
    %2759 = sdy.sharding_constraint %2758 <@mesh, [{}]> : tensor<32xf32>
    %cst_356 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2760 = stablehlo.broadcast_in_dim %cst_356, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2761 = sdy.sharding_constraint %2760 <@mesh, [{}]> : tensor<32xf32>
    %2762 = stablehlo.multiply %2761, %2759 : tensor<32xf32>
    %2763 = sdy.sharding_constraint %2762 <@mesh, [{}]> : tensor<32xf32>
    %cst_357 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2764 = stablehlo.broadcast_in_dim %cst_357, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2765 = sdy.sharding_constraint %2764 <@mesh, [{}]> : tensor<32xf32>
    %2766 = stablehlo.multiply %2765, %arg60 : tensor<32xf32>
    %2767 = sdy.sharding_constraint %2766 <@mesh, [{}]> : tensor<32xf32>
    %2768 = stablehlo.add %2763, %2767 : tensor<32xf32>
    %2769 = sdy.sharding_constraint %2768 <@mesh, [{}]> : tensor<32xf32>
    %2770 = stablehlo.multiply %2386, %2386 : tensor<32x128xf32>
    %2771 = sdy.sharding_constraint %2770 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_358 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2772 = stablehlo.broadcast_in_dim %cst_358, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2773 = sdy.sharding_constraint %2772 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2774 = stablehlo.multiply %2773, %2771 : tensor<32x128xf32>
    %2775 = sdy.sharding_constraint %2774 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_359 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2776 = stablehlo.broadcast_in_dim %cst_359, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2777 = sdy.sharding_constraint %2776 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2778 = stablehlo.multiply %2777, %arg61 : tensor<32x128xf32>
    %2779 = sdy.sharding_constraint %2778 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2780 = stablehlo.add %2775, %2779 : tensor<32x128xf32>
    %2781 = sdy.sharding_constraint %2780 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2782 = stablehlo.multiply %2384, %2384 : tensor<128x32xf32>
    %2783 = sdy.sharding_constraint %2782 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_360 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2784 = stablehlo.broadcast_in_dim %cst_360, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2785 = sdy.sharding_constraint %2784 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2786 = stablehlo.multiply %2785, %2783 : tensor<128x32xf32>
    %2787 = sdy.sharding_constraint %2786 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_361 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2788 = stablehlo.broadcast_in_dim %cst_361, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2789 = sdy.sharding_constraint %2788 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2790 = stablehlo.multiply %2789, %arg62 : tensor<128x32xf32>
    %2791 = sdy.sharding_constraint %2790 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2792 = stablehlo.add %2787, %2791 : tensor<128x32xf32>
    %2793 = sdy.sharding_constraint %2792 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2794 = stablehlo.multiply %2382, %2382 : tensor<32x32xf32>
    %2795 = sdy.sharding_constraint %2794 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_362 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2796 = stablehlo.broadcast_in_dim %cst_362, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2797 = sdy.sharding_constraint %2796 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2798 = stablehlo.multiply %2797, %2795 : tensor<32x32xf32>
    %2799 = sdy.sharding_constraint %2798 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_363 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2800 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2801 = sdy.sharding_constraint %2800 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2802 = stablehlo.multiply %2801, %arg63 : tensor<32x32xf32>
    %2803 = sdy.sharding_constraint %2802 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2804 = stablehlo.add %2799, %2803 : tensor<32x32xf32>
    %2805 = sdy.sharding_constraint %2804 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2806 = stablehlo.multiply %2380, %2380 : tensor<32x16xf32>
    %2807 = sdy.sharding_constraint %2806 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_364 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2808 = stablehlo.broadcast_in_dim %cst_364, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2809 = sdy.sharding_constraint %2808 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2810 = stablehlo.multiply %2809, %2807 : tensor<32x16xf32>
    %2811 = sdy.sharding_constraint %2810 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_365 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2812 = stablehlo.broadcast_in_dim %cst_365, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2813 = sdy.sharding_constraint %2812 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2814 = stablehlo.multiply %2813, %arg64 : tensor<32x16xf32>
    %2815 = sdy.sharding_constraint %2814 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2816 = stablehlo.add %2811, %2815 : tensor<32x16xf32>
    %2817 = sdy.sharding_constraint %2816 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2818 = stablehlo.multiply %2378, %2378 : tensor<32x16xf32>
    %2819 = sdy.sharding_constraint %2818 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_366 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2820 = stablehlo.broadcast_in_dim %cst_366, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2821 = sdy.sharding_constraint %2820 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2822 = stablehlo.multiply %2821, %2819 : tensor<32x16xf32>
    %2823 = sdy.sharding_constraint %2822 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_367 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2824 = stablehlo.broadcast_in_dim %cst_367, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %2825 = sdy.sharding_constraint %2824 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2826 = stablehlo.multiply %2825, %arg65 : tensor<32x16xf32>
    %2827 = sdy.sharding_constraint %2826 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2828 = stablehlo.add %2823, %2827 : tensor<32x16xf32>
    %2829 = sdy.sharding_constraint %2828 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %2830 = stablehlo.multiply %2376, %2376 : tensor<32x32xf32>
    %2831 = sdy.sharding_constraint %2830 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_368 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2832 = stablehlo.broadcast_in_dim %cst_368, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2833 = sdy.sharding_constraint %2832 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2834 = stablehlo.multiply %2833, %2831 : tensor<32x32xf32>
    %2835 = sdy.sharding_constraint %2834 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_369 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2836 = stablehlo.broadcast_in_dim %cst_369, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2837 = sdy.sharding_constraint %2836 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2838 = stablehlo.multiply %2837, %arg66 : tensor<32x32xf32>
    %2839 = sdy.sharding_constraint %2838 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2840 = stablehlo.add %2835, %2839 : tensor<32x32xf32>
    %2841 = sdy.sharding_constraint %2840 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2842 = stablehlo.multiply %2374, %2374 : tensor<32x2xf32>
    %2843 = sdy.sharding_constraint %2842 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_370 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2844 = stablehlo.broadcast_in_dim %cst_370, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %2845 = sdy.sharding_constraint %2844 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2846 = stablehlo.multiply %2845, %2843 : tensor<32x2xf32>
    %2847 = sdy.sharding_constraint %2846 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_371 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2848 = stablehlo.broadcast_in_dim %cst_371, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %2849 = sdy.sharding_constraint %2848 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2850 = stablehlo.multiply %2849, %arg67 : tensor<32x2xf32>
    %2851 = sdy.sharding_constraint %2850 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2852 = stablehlo.add %2847, %2851 : tensor<32x2xf32>
    %2853 = sdy.sharding_constraint %2852 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %2854 = stablehlo.multiply %2372, %2372 : tensor<32xf32>
    %2855 = sdy.sharding_constraint %2854 <@mesh, [{}]> : tensor<32xf32>
    %cst_372 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2856 = stablehlo.broadcast_in_dim %cst_372, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2857 = sdy.sharding_constraint %2856 <@mesh, [{}]> : tensor<32xf32>
    %2858 = stablehlo.multiply %2857, %2855 : tensor<32xf32>
    %2859 = sdy.sharding_constraint %2858 <@mesh, [{}]> : tensor<32xf32>
    %cst_373 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2860 = stablehlo.broadcast_in_dim %cst_373, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2861 = sdy.sharding_constraint %2860 <@mesh, [{}]> : tensor<32xf32>
    %2862 = stablehlo.multiply %2861, %arg68 : tensor<32xf32>
    %2863 = sdy.sharding_constraint %2862 <@mesh, [{}]> : tensor<32xf32>
    %2864 = stablehlo.add %2859, %2863 : tensor<32xf32>
    %2865 = sdy.sharding_constraint %2864 <@mesh, [{}]> : tensor<32xf32>
    %2866 = stablehlo.multiply %2370, %2370 : tensor<32x128xf32>
    %2867 = sdy.sharding_constraint %2866 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_374 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2868 = stablehlo.broadcast_in_dim %cst_374, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2869 = sdy.sharding_constraint %2868 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2870 = stablehlo.multiply %2869, %2867 : tensor<32x128xf32>
    %2871 = sdy.sharding_constraint %2870 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_375 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2872 = stablehlo.broadcast_in_dim %cst_375, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %2873 = sdy.sharding_constraint %2872 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2874 = stablehlo.multiply %2873, %arg69 : tensor<32x128xf32>
    %2875 = sdy.sharding_constraint %2874 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2876 = stablehlo.add %2871, %2875 : tensor<32x128xf32>
    %2877 = sdy.sharding_constraint %2876 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %2878 = stablehlo.multiply %2368, %2368 : tensor<128x32xf32>
    %2879 = sdy.sharding_constraint %2878 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_376 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2880 = stablehlo.broadcast_in_dim %cst_376, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2881 = sdy.sharding_constraint %2880 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2882 = stablehlo.multiply %2881, %2879 : tensor<128x32xf32>
    %2883 = sdy.sharding_constraint %2882 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_377 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2884 = stablehlo.broadcast_in_dim %cst_377, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %2885 = sdy.sharding_constraint %2884 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2886 = stablehlo.multiply %2885, %arg70 : tensor<128x32xf32>
    %2887 = sdy.sharding_constraint %2886 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2888 = stablehlo.add %2883, %2887 : tensor<128x32xf32>
    %2889 = sdy.sharding_constraint %2888 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %2890 = stablehlo.multiply %2366, %2366 : tensor<32x4xf32>
    %2891 = sdy.sharding_constraint %2890 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_378 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2892 = stablehlo.broadcast_in_dim %cst_378, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %2893 = sdy.sharding_constraint %2892 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2894 = stablehlo.multiply %2893, %2891 : tensor<32x4xf32>
    %2895 = sdy.sharding_constraint %2894 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_379 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2896 = stablehlo.broadcast_in_dim %cst_379, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %2897 = sdy.sharding_constraint %2896 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2898 = stablehlo.multiply %2897, %arg71 : tensor<32x4xf32>
    %2899 = sdy.sharding_constraint %2898 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2900 = stablehlo.add %2895, %2899 : tensor<32x4xf32>
    %2901 = sdy.sharding_constraint %2900 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %2902 = stablehlo.multiply %2400, %2400 : tensor<4xf32>
    %2903 = sdy.sharding_constraint %2902 <@mesh, [{}]> : tensor<4xf32>
    %cst_380 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2904 = stablehlo.broadcast_in_dim %cst_380, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2905 = sdy.sharding_constraint %2904 <@mesh, [{}]> : tensor<4xf32>
    %2906 = stablehlo.multiply %2905, %2903 : tensor<4xf32>
    %2907 = sdy.sharding_constraint %2906 <@mesh, [{}]> : tensor<4xf32>
    %cst_381 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2908 = stablehlo.broadcast_in_dim %cst_381, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2909 = sdy.sharding_constraint %2908 <@mesh, [{}]> : tensor<4xf32>
    %2910 = stablehlo.multiply %2909, %arg72 : tensor<4xf32>
    %2911 = sdy.sharding_constraint %2910 <@mesh, [{}]> : tensor<4xf32>
    %2912 = stablehlo.add %2907, %2911 : tensor<4xf32>
    %2913 = sdy.sharding_constraint %2912 <@mesh, [{}]> : tensor<4xf32>
    %2914 = stablehlo.multiply %2364, %2364 : tensor<32x32xf32>
    %2915 = sdy.sharding_constraint %2914 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_382 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2916 = stablehlo.broadcast_in_dim %cst_382, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2917 = sdy.sharding_constraint %2916 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2918 = stablehlo.multiply %2917, %2915 : tensor<32x32xf32>
    %2919 = sdy.sharding_constraint %2918 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_383 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2920 = stablehlo.broadcast_in_dim %cst_383, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2921 = sdy.sharding_constraint %2920 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2922 = stablehlo.multiply %2921, %arg73 : tensor<32x32xf32>
    %2923 = sdy.sharding_constraint %2922 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2924 = stablehlo.add %2919, %2923 : tensor<32x32xf32>
    %2925 = sdy.sharding_constraint %2924 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2926 = stablehlo.multiply %2362, %2362 : tensor<32x32xf32>
    %2927 = sdy.sharding_constraint %2926 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_384 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2928 = stablehlo.broadcast_in_dim %cst_384, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2929 = sdy.sharding_constraint %2928 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2930 = stablehlo.multiply %2929, %2927 : tensor<32x32xf32>
    %2931 = sdy.sharding_constraint %2930 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_385 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2932 = stablehlo.broadcast_in_dim %cst_385, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2933 = sdy.sharding_constraint %2932 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2934 = stablehlo.multiply %2933, %arg74 : tensor<32x32xf32>
    %2935 = sdy.sharding_constraint %2934 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2936 = stablehlo.add %2931, %2935 : tensor<32x32xf32>
    %2937 = sdy.sharding_constraint %2936 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %2938 = stablehlo.multiply %2360, %2360 : tensor<32x32xf32>
    %2939 = sdy.sharding_constraint %2938 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_386 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2940 = stablehlo.broadcast_in_dim %cst_386, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2941 = sdy.sharding_constraint %2940 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2942 = stablehlo.multiply %2941, %2939 : tensor<32x32xf32>
    %2943 = sdy.sharding_constraint %2942 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_387 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2944 = stablehlo.broadcast_in_dim %cst_387, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %2945 = sdy.sharding_constraint %2944 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2946 = stablehlo.multiply %2945, %arg75 : tensor<32x32xf32>
    %2947 = sdy.sharding_constraint %2946 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2948 = stablehlo.add %2943, %2947 : tensor<32x32xf32>
    %2949 = sdy.sharding_constraint %2948 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %2950 = stablehlo.multiply %2358, %2358 : tensor<4x32x32xf32>
    %2951 = sdy.sharding_constraint %2950 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_388 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2952 = stablehlo.broadcast_in_dim %cst_388, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2953 = sdy.sharding_constraint %2952 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2954 = stablehlo.multiply %2953, %2951 : tensor<4x32x32xf32>
    %2955 = sdy.sharding_constraint %2954 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_389 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2956 = stablehlo.broadcast_in_dim %cst_389, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2957 = sdy.sharding_constraint %2956 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2958 = stablehlo.multiply %2957, %arg76 : tensor<4x32x32xf32>
    %2959 = sdy.sharding_constraint %2958 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2960 = stablehlo.add %2955, %2959 : tensor<4x32x32xf32>
    %2961 = sdy.sharding_constraint %2960 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2962 = stablehlo.multiply %2356, %2356 : tensor<4x32x32xf32>
    %2963 = sdy.sharding_constraint %2962 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_390 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2964 = stablehlo.broadcast_in_dim %cst_390, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2965 = sdy.sharding_constraint %2964 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2966 = stablehlo.multiply %2965, %2963 : tensor<4x32x32xf32>
    %2967 = sdy.sharding_constraint %2966 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_391 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2968 = stablehlo.broadcast_in_dim %cst_391, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2969 = sdy.sharding_constraint %2968 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2970 = stablehlo.multiply %2969, %arg77 : tensor<4x32x32xf32>
    %2971 = sdy.sharding_constraint %2970 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2972 = stablehlo.add %2967, %2971 : tensor<4x32x32xf32>
    %2973 = sdy.sharding_constraint %2972 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %2974 = stablehlo.multiply %2354, %2354 : tensor<4x32x32xf32>
    %2975 = sdy.sharding_constraint %2974 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_392 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2976 = stablehlo.broadcast_in_dim %cst_392, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2977 = sdy.sharding_constraint %2976 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2978 = stablehlo.multiply %2977, %2975 : tensor<4x32x32xf32>
    %2979 = sdy.sharding_constraint %2978 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_393 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2980 = stablehlo.broadcast_in_dim %cst_393, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %2981 = sdy.sharding_constraint %2980 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2982 = stablehlo.multiply %2981, %arg78 : tensor<4x32x32xf32>
    %2983 = sdy.sharding_constraint %2982 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2984 = stablehlo.add %2979, %2983 : tensor<4x32x32xf32>
    %2985 = sdy.sharding_constraint %2984 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %2986 = stablehlo.multiply %2352, %2352 : tensor<32xf32>
    %2987 = sdy.sharding_constraint %2986 <@mesh, [{}]> : tensor<32xf32>
    %cst_394 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %2988 = stablehlo.broadcast_in_dim %cst_394, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2989 = sdy.sharding_constraint %2988 <@mesh, [{}]> : tensor<32xf32>
    %2990 = stablehlo.multiply %2989, %2987 : tensor<32xf32>
    %2991 = sdy.sharding_constraint %2990 <@mesh, [{}]> : tensor<32xf32>
    %cst_395 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %2992 = stablehlo.broadcast_in_dim %cst_395, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %2993 = sdy.sharding_constraint %2992 <@mesh, [{}]> : tensor<32xf32>
    %2994 = stablehlo.multiply %2993, %arg79 : tensor<32xf32>
    %2995 = sdy.sharding_constraint %2994 <@mesh, [{}]> : tensor<32xf32>
    %2996 = stablehlo.add %2991, %2995 : tensor<32xf32>
    %2997 = sdy.sharding_constraint %2996 <@mesh, [{}]> : tensor<32xf32>
    %2998 = stablehlo.multiply %2350, %2350 : tensor<32x128xf32>
    %2999 = sdy.sharding_constraint %2998 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_396 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %3000 = stablehlo.broadcast_in_dim %cst_396, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3001 = sdy.sharding_constraint %3000 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3002 = stablehlo.multiply %3001, %2999 : tensor<32x128xf32>
    %3003 = sdy.sharding_constraint %3002 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_397 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %3004 = stablehlo.broadcast_in_dim %cst_397, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3005 = sdy.sharding_constraint %3004 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3006 = stablehlo.multiply %3005, %arg80 : tensor<32x128xf32>
    %3007 = sdy.sharding_constraint %3006 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3008 = stablehlo.add %3003, %3007 : tensor<32x128xf32>
    %3009 = sdy.sharding_constraint %3008 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3010 = stablehlo.multiply %2348, %2348 : tensor<128x32xf32>
    %3011 = sdy.sharding_constraint %3010 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_398 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %3012 = stablehlo.broadcast_in_dim %cst_398, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3013 = sdy.sharding_constraint %3012 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3014 = stablehlo.multiply %3013, %3011 : tensor<128x32xf32>
    %3015 = sdy.sharding_constraint %3014 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_399 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %3016 = stablehlo.broadcast_in_dim %cst_399, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3017 = sdy.sharding_constraint %3016 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3018 = stablehlo.multiply %3017, %arg81 : tensor<128x32xf32>
    %3019 = sdy.sharding_constraint %3018 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3020 = stablehlo.add %3015, %3019 : tensor<128x32xf32>
    %3021 = sdy.sharding_constraint %3020 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %c_400 = stablehlo.constant dense<2147483647> : tensor<i32>
    %3022 = sdy.sharding_constraint %c_400 <@mesh, []> : tensor<i32>
    %3023 = stablehlo.compare LT, %arg27, %3022, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_401 = stablehlo.constant dense<1> : tensor<i32>
    %3024 = sdy.sharding_constraint %c_401 <@mesh, []> : tensor<i32>
    %3025 = stablehlo.add %arg27, %3024 : tensor<i32>
    %3026 = sdy.sharding_constraint %3025 <@mesh, []> : tensor<i32>
    %c_402 = stablehlo.constant dense<2147483647> : tensor<i32>
    %3027 = call @_where_596(%3023, %3026, %c_402) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %cst_403 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %3028 = stablehlo.convert %3027 : (tensor<i32>) -> tensor<f32>
    %3029 = sdy.sharding_constraint %cst_403 <@mesh, []> : tensor<f32>
    %3030 = stablehlo.power %3029, %3028 : tensor<f32>
    %3031 = sdy.sharding_constraint %3030 <@mesh, []> : tensor<f32>
    %cst_404 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3032 = sdy.sharding_constraint %cst_404 <@mesh, []> : tensor<f32>
    %3033 = stablehlo.subtract %3032, %3031 : tensor<f32>
    %3034 = sdy.sharding_constraint %3033 <@mesh, []> : tensor<f32>
    %3035 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %3036 = sdy.sharding_constraint %3035 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3037 = stablehlo.divide %2411, %3036 : tensor<64x32xf32>
    %3038 = sdy.sharding_constraint %3037 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3039 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3040 = sdy.sharding_constraint %3039 <@mesh, [{}]> : tensor<32xf32>
    %3041 = stablehlo.divide %2422, %3040 : tensor<32xf32>
    %3042 = sdy.sharding_constraint %3041 <@mesh, [{}]> : tensor<32xf32>
    %3043 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3044 = sdy.sharding_constraint %3043 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3045 = stablehlo.divide %2433, %3044 : tensor<32x128xf32>
    %3046 = sdy.sharding_constraint %3045 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3047 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3048 = sdy.sharding_constraint %3047 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3049 = stablehlo.divide %2444, %3048 : tensor<128x32xf32>
    %3050 = sdy.sharding_constraint %3049 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3051 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %3052 = sdy.sharding_constraint %3051 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3053 = stablehlo.divide %2455, %3052 : tensor<32x64xf32>
    %3054 = sdy.sharding_constraint %3053 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3055 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3056 = sdy.sharding_constraint %3055 <@mesh, [{}]> : tensor<32xf32>
    %3057 = stablehlo.divide %2466, %3056 : tensor<32xf32>
    %3058 = sdy.sharding_constraint %3057 <@mesh, [{}]> : tensor<32xf32>
    %3059 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3060 = sdy.sharding_constraint %3059 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3061 = stablehlo.divide %2477, %3060 : tensor<32x128xf32>
    %3062 = sdy.sharding_constraint %3061 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3063 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3064 = sdy.sharding_constraint %3063 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3065 = stablehlo.divide %2488, %3064 : tensor<128x32xf32>
    %3066 = sdy.sharding_constraint %3065 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3067 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3068 = sdy.sharding_constraint %3067 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3069 = stablehlo.divide %2499, %3068 : tensor<32x32xf32>
    %3070 = sdy.sharding_constraint %3069 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3071 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3072 = sdy.sharding_constraint %3071 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3073 = stablehlo.divide %2510, %3072 : tensor<32x16xf32>
    %3074 = sdy.sharding_constraint %3073 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3075 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3076 = sdy.sharding_constraint %3075 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3077 = stablehlo.divide %2521, %3076 : tensor<32x16xf32>
    %3078 = sdy.sharding_constraint %3077 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3079 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3080 = sdy.sharding_constraint %3079 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3081 = stablehlo.divide %2532, %3080 : tensor<32x32xf32>
    %3082 = sdy.sharding_constraint %3081 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3083 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %3084 = sdy.sharding_constraint %3083 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3085 = stablehlo.divide %2543, %3084 : tensor<32x2xf32>
    %3086 = sdy.sharding_constraint %3085 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3087 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3088 = sdy.sharding_constraint %3087 <@mesh, [{}]> : tensor<32xf32>
    %3089 = stablehlo.divide %2554, %3088 : tensor<32xf32>
    %3090 = sdy.sharding_constraint %3089 <@mesh, [{}]> : tensor<32xf32>
    %3091 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3092 = sdy.sharding_constraint %3091 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3093 = stablehlo.divide %2565, %3092 : tensor<32x128xf32>
    %3094 = sdy.sharding_constraint %3093 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3095 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3096 = sdy.sharding_constraint %3095 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3097 = stablehlo.divide %2576, %3096 : tensor<128x32xf32>
    %3098 = sdy.sharding_constraint %3097 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3099 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %3100 = sdy.sharding_constraint %3099 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3101 = stablehlo.divide %2587, %3100 : tensor<32x4xf32>
    %3102 = sdy.sharding_constraint %3101 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3103 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %3104 = sdy.sharding_constraint %3103 <@mesh, [{}]> : tensor<4xf32>
    %3105 = stablehlo.divide %2598, %3104 : tensor<4xf32>
    %3106 = sdy.sharding_constraint %3105 <@mesh, [{}]> : tensor<4xf32>
    %3107 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3108 = sdy.sharding_constraint %3107 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3109 = stablehlo.divide %2609, %3108 : tensor<32x32xf32>
    %3110 = sdy.sharding_constraint %3109 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3111 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3112 = sdy.sharding_constraint %3111 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3113 = stablehlo.divide %2620, %3112 : tensor<32x32xf32>
    %3114 = sdy.sharding_constraint %3113 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3115 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3116 = sdy.sharding_constraint %3115 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3117 = stablehlo.divide %2631, %3116 : tensor<32x32xf32>
    %3118 = sdy.sharding_constraint %3117 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3119 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3120 = sdy.sharding_constraint %3119 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3121 = stablehlo.divide %2642, %3120 : tensor<4x32x32xf32>
    %3122 = sdy.sharding_constraint %3121 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3123 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3124 = sdy.sharding_constraint %3123 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3125 = stablehlo.divide %2653, %3124 : tensor<4x32x32xf32>
    %3126 = sdy.sharding_constraint %3125 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3127 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3128 = sdy.sharding_constraint %3127 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3129 = stablehlo.divide %2664, %3128 : tensor<4x32x32xf32>
    %3130 = sdy.sharding_constraint %3129 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3131 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3132 = sdy.sharding_constraint %3131 <@mesh, [{}]> : tensor<32xf32>
    %3133 = stablehlo.divide %2675, %3132 : tensor<32xf32>
    %3134 = sdy.sharding_constraint %3133 <@mesh, [{}]> : tensor<32xf32>
    %3135 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3136 = sdy.sharding_constraint %3135 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3137 = stablehlo.divide %2686, %3136 : tensor<32x128xf32>
    %3138 = sdy.sharding_constraint %3137 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3139 = stablehlo.broadcast_in_dim %3034, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3140 = sdy.sharding_constraint %3139 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3141 = stablehlo.divide %2697, %3140 : tensor<128x32xf32>
    %3142 = sdy.sharding_constraint %3141 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_405 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %3143 = stablehlo.convert %3027 : (tensor<i32>) -> tensor<f32>
    %3144 = sdy.sharding_constraint %cst_405 <@mesh, []> : tensor<f32>
    %3145 = stablehlo.power %3144, %3143 : tensor<f32>
    %3146 = sdy.sharding_constraint %3145 <@mesh, []> : tensor<f32>
    %cst_406 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3147 = sdy.sharding_constraint %cst_406 <@mesh, []> : tensor<f32>
    %3148 = stablehlo.subtract %3147, %3146 : tensor<f32>
    %3149 = sdy.sharding_constraint %3148 <@mesh, []> : tensor<f32>
    %3150 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %3151 = sdy.sharding_constraint %3150 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3152 = stablehlo.divide %2709, %3151 : tensor<64x32xf32>
    %3153 = sdy.sharding_constraint %3152 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3154 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3155 = sdy.sharding_constraint %3154 <@mesh, [{}]> : tensor<32xf32>
    %3156 = stablehlo.divide %2721, %3155 : tensor<32xf32>
    %3157 = sdy.sharding_constraint %3156 <@mesh, [{}]> : tensor<32xf32>
    %3158 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3159 = sdy.sharding_constraint %3158 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3160 = stablehlo.divide %2733, %3159 : tensor<32x128xf32>
    %3161 = sdy.sharding_constraint %3160 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3162 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3163 = sdy.sharding_constraint %3162 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3164 = stablehlo.divide %2745, %3163 : tensor<128x32xf32>
    %3165 = sdy.sharding_constraint %3164 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3166 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %3167 = sdy.sharding_constraint %3166 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3168 = stablehlo.divide %2757, %3167 : tensor<32x64xf32>
    %3169 = sdy.sharding_constraint %3168 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3170 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3171 = sdy.sharding_constraint %3170 <@mesh, [{}]> : tensor<32xf32>
    %3172 = stablehlo.divide %2769, %3171 : tensor<32xf32>
    %3173 = sdy.sharding_constraint %3172 <@mesh, [{}]> : tensor<32xf32>
    %3174 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3175 = sdy.sharding_constraint %3174 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3176 = stablehlo.divide %2781, %3175 : tensor<32x128xf32>
    %3177 = sdy.sharding_constraint %3176 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3178 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3179 = sdy.sharding_constraint %3178 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3180 = stablehlo.divide %2793, %3179 : tensor<128x32xf32>
    %3181 = sdy.sharding_constraint %3180 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3182 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3183 = sdy.sharding_constraint %3182 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3184 = stablehlo.divide %2805, %3183 : tensor<32x32xf32>
    %3185 = sdy.sharding_constraint %3184 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3186 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3187 = sdy.sharding_constraint %3186 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3188 = stablehlo.divide %2817, %3187 : tensor<32x16xf32>
    %3189 = sdy.sharding_constraint %3188 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3190 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3191 = sdy.sharding_constraint %3190 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3192 = stablehlo.divide %2829, %3191 : tensor<32x16xf32>
    %3193 = sdy.sharding_constraint %3192 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3194 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3195 = sdy.sharding_constraint %3194 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3196 = stablehlo.divide %2841, %3195 : tensor<32x32xf32>
    %3197 = sdy.sharding_constraint %3196 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3198 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %3199 = sdy.sharding_constraint %3198 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3200 = stablehlo.divide %2853, %3199 : tensor<32x2xf32>
    %3201 = sdy.sharding_constraint %3200 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3202 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3203 = sdy.sharding_constraint %3202 <@mesh, [{}]> : tensor<32xf32>
    %3204 = stablehlo.divide %2865, %3203 : tensor<32xf32>
    %3205 = sdy.sharding_constraint %3204 <@mesh, [{}]> : tensor<32xf32>
    %3206 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3207 = sdy.sharding_constraint %3206 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3208 = stablehlo.divide %2877, %3207 : tensor<32x128xf32>
    %3209 = sdy.sharding_constraint %3208 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3210 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3211 = sdy.sharding_constraint %3210 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3212 = stablehlo.divide %2889, %3211 : tensor<128x32xf32>
    %3213 = sdy.sharding_constraint %3212 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3214 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %3215 = sdy.sharding_constraint %3214 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3216 = stablehlo.divide %2901, %3215 : tensor<32x4xf32>
    %3217 = sdy.sharding_constraint %3216 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3218 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %3219 = sdy.sharding_constraint %3218 <@mesh, [{}]> : tensor<4xf32>
    %3220 = stablehlo.divide %2913, %3219 : tensor<4xf32>
    %3221 = sdy.sharding_constraint %3220 <@mesh, [{}]> : tensor<4xf32>
    %3222 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3223 = sdy.sharding_constraint %3222 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3224 = stablehlo.divide %2925, %3223 : tensor<32x32xf32>
    %3225 = sdy.sharding_constraint %3224 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3226 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3227 = sdy.sharding_constraint %3226 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3228 = stablehlo.divide %2937, %3227 : tensor<32x32xf32>
    %3229 = sdy.sharding_constraint %3228 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3230 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3231 = sdy.sharding_constraint %3230 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3232 = stablehlo.divide %2949, %3231 : tensor<32x32xf32>
    %3233 = sdy.sharding_constraint %3232 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3234 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3235 = sdy.sharding_constraint %3234 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3236 = stablehlo.divide %2961, %3235 : tensor<4x32x32xf32>
    %3237 = sdy.sharding_constraint %3236 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3238 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3239 = sdy.sharding_constraint %3238 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3240 = stablehlo.divide %2973, %3239 : tensor<4x32x32xf32>
    %3241 = sdy.sharding_constraint %3240 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3242 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3243 = sdy.sharding_constraint %3242 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3244 = stablehlo.divide %2985, %3243 : tensor<4x32x32xf32>
    %3245 = sdy.sharding_constraint %3244 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3246 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3247 = sdy.sharding_constraint %3246 <@mesh, [{}]> : tensor<32xf32>
    %3248 = stablehlo.divide %2997, %3247 : tensor<32xf32>
    %3249 = sdy.sharding_constraint %3248 <@mesh, [{}]> : tensor<32xf32>
    %3250 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3251 = sdy.sharding_constraint %3250 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3252 = stablehlo.divide %3009, %3251 : tensor<32x128xf32>
    %3253 = sdy.sharding_constraint %3252 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3254 = stablehlo.broadcast_in_dim %3149, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3255 = sdy.sharding_constraint %3254 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3256 = stablehlo.divide %3021, %3255 : tensor<128x32xf32>
    %3257 = sdy.sharding_constraint %3256 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_407 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3258 = stablehlo.broadcast_in_dim %cst_407, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %3259 = sdy.sharding_constraint %3258 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3260 = stablehlo.add %3153, %3259 : tensor<64x32xf32>
    %3261 = sdy.sharding_constraint %3260 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3262 = stablehlo.sqrt %3261 : tensor<64x32xf32>
    %3263 = sdy.sharding_constraint %3262 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_408 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3264 = stablehlo.broadcast_in_dim %cst_408, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %3265 = sdy.sharding_constraint %3264 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3266 = stablehlo.add %3263, %3265 : tensor<64x32xf32>
    %3267 = sdy.sharding_constraint %3266 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3268 = stablehlo.divide %3038, %3267 : tensor<64x32xf32>
    %3269 = sdy.sharding_constraint %3268 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_409 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3270 = stablehlo.broadcast_in_dim %cst_409, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3271 = sdy.sharding_constraint %3270 <@mesh, [{}]> : tensor<32xf32>
    %3272 = stablehlo.add %3157, %3271 : tensor<32xf32>
    %3273 = sdy.sharding_constraint %3272 <@mesh, [{}]> : tensor<32xf32>
    %3274 = stablehlo.sqrt %3273 : tensor<32xf32>
    %3275 = sdy.sharding_constraint %3274 <@mesh, [{}]> : tensor<32xf32>
    %cst_410 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3276 = stablehlo.broadcast_in_dim %cst_410, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3277 = sdy.sharding_constraint %3276 <@mesh, [{}]> : tensor<32xf32>
    %3278 = stablehlo.add %3275, %3277 : tensor<32xf32>
    %3279 = sdy.sharding_constraint %3278 <@mesh, [{}]> : tensor<32xf32>
    %3280 = stablehlo.divide %3042, %3279 : tensor<32xf32>
    %3281 = sdy.sharding_constraint %3280 <@mesh, [{}]> : tensor<32xf32>
    %cst_411 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3282 = stablehlo.broadcast_in_dim %cst_411, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3283 = sdy.sharding_constraint %3282 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3284 = stablehlo.add %3161, %3283 : tensor<32x128xf32>
    %3285 = sdy.sharding_constraint %3284 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3286 = stablehlo.sqrt %3285 : tensor<32x128xf32>
    %3287 = sdy.sharding_constraint %3286 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_412 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3288 = stablehlo.broadcast_in_dim %cst_412, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3289 = sdy.sharding_constraint %3288 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3290 = stablehlo.add %3287, %3289 : tensor<32x128xf32>
    %3291 = sdy.sharding_constraint %3290 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3292 = stablehlo.divide %3046, %3291 : tensor<32x128xf32>
    %3293 = sdy.sharding_constraint %3292 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_413 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3294 = stablehlo.broadcast_in_dim %cst_413, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3295 = sdy.sharding_constraint %3294 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3296 = stablehlo.add %3165, %3295 : tensor<128x32xf32>
    %3297 = sdy.sharding_constraint %3296 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3298 = stablehlo.sqrt %3297 : tensor<128x32xf32>
    %3299 = sdy.sharding_constraint %3298 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_414 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3300 = stablehlo.broadcast_in_dim %cst_414, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3301 = sdy.sharding_constraint %3300 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3302 = stablehlo.add %3299, %3301 : tensor<128x32xf32>
    %3303 = sdy.sharding_constraint %3302 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3304 = stablehlo.divide %3050, %3303 : tensor<128x32xf32>
    %3305 = sdy.sharding_constraint %3304 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_415 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3306 = stablehlo.broadcast_in_dim %cst_415, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %3307 = sdy.sharding_constraint %3306 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3308 = stablehlo.add %3169, %3307 : tensor<32x64xf32>
    %3309 = sdy.sharding_constraint %3308 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3310 = stablehlo.sqrt %3309 : tensor<32x64xf32>
    %3311 = sdy.sharding_constraint %3310 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_416 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3312 = stablehlo.broadcast_in_dim %cst_416, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %3313 = sdy.sharding_constraint %3312 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3314 = stablehlo.add %3311, %3313 : tensor<32x64xf32>
    %3315 = sdy.sharding_constraint %3314 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3316 = stablehlo.divide %3054, %3315 : tensor<32x64xf32>
    %3317 = sdy.sharding_constraint %3316 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_417 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3318 = stablehlo.broadcast_in_dim %cst_417, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3319 = sdy.sharding_constraint %3318 <@mesh, [{}]> : tensor<32xf32>
    %3320 = stablehlo.add %3173, %3319 : tensor<32xf32>
    %3321 = sdy.sharding_constraint %3320 <@mesh, [{}]> : tensor<32xf32>
    %3322 = stablehlo.sqrt %3321 : tensor<32xf32>
    %3323 = sdy.sharding_constraint %3322 <@mesh, [{}]> : tensor<32xf32>
    %cst_418 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3324 = stablehlo.broadcast_in_dim %cst_418, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3325 = sdy.sharding_constraint %3324 <@mesh, [{}]> : tensor<32xf32>
    %3326 = stablehlo.add %3323, %3325 : tensor<32xf32>
    %3327 = sdy.sharding_constraint %3326 <@mesh, [{}]> : tensor<32xf32>
    %3328 = stablehlo.divide %3058, %3327 : tensor<32xf32>
    %3329 = sdy.sharding_constraint %3328 <@mesh, [{}]> : tensor<32xf32>
    %cst_419 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3330 = stablehlo.broadcast_in_dim %cst_419, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3331 = sdy.sharding_constraint %3330 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3332 = stablehlo.add %3177, %3331 : tensor<32x128xf32>
    %3333 = sdy.sharding_constraint %3332 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3334 = stablehlo.sqrt %3333 : tensor<32x128xf32>
    %3335 = sdy.sharding_constraint %3334 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_420 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3336 = stablehlo.broadcast_in_dim %cst_420, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3337 = sdy.sharding_constraint %3336 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3338 = stablehlo.add %3335, %3337 : tensor<32x128xf32>
    %3339 = sdy.sharding_constraint %3338 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3340 = stablehlo.divide %3062, %3339 : tensor<32x128xf32>
    %3341 = sdy.sharding_constraint %3340 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_421 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3342 = stablehlo.broadcast_in_dim %cst_421, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3343 = sdy.sharding_constraint %3342 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3344 = stablehlo.add %3181, %3343 : tensor<128x32xf32>
    %3345 = sdy.sharding_constraint %3344 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3346 = stablehlo.sqrt %3345 : tensor<128x32xf32>
    %3347 = sdy.sharding_constraint %3346 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_422 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3348 = stablehlo.broadcast_in_dim %cst_422, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3349 = sdy.sharding_constraint %3348 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3350 = stablehlo.add %3347, %3349 : tensor<128x32xf32>
    %3351 = sdy.sharding_constraint %3350 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3352 = stablehlo.divide %3066, %3351 : tensor<128x32xf32>
    %3353 = sdy.sharding_constraint %3352 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_423 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3354 = stablehlo.broadcast_in_dim %cst_423, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3355 = sdy.sharding_constraint %3354 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3356 = stablehlo.add %3185, %3355 : tensor<32x32xf32>
    %3357 = sdy.sharding_constraint %3356 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3358 = stablehlo.sqrt %3357 : tensor<32x32xf32>
    %3359 = sdy.sharding_constraint %3358 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_424 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3360 = stablehlo.broadcast_in_dim %cst_424, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3361 = sdy.sharding_constraint %3360 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3362 = stablehlo.add %3359, %3361 : tensor<32x32xf32>
    %3363 = sdy.sharding_constraint %3362 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3364 = stablehlo.divide %3070, %3363 : tensor<32x32xf32>
    %3365 = sdy.sharding_constraint %3364 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_425 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3366 = stablehlo.broadcast_in_dim %cst_425, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3367 = sdy.sharding_constraint %3366 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3368 = stablehlo.add %3189, %3367 : tensor<32x16xf32>
    %3369 = sdy.sharding_constraint %3368 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3370 = stablehlo.sqrt %3369 : tensor<32x16xf32>
    %3371 = sdy.sharding_constraint %3370 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_426 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3372 = stablehlo.broadcast_in_dim %cst_426, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3373 = sdy.sharding_constraint %3372 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3374 = stablehlo.add %3371, %3373 : tensor<32x16xf32>
    %3375 = sdy.sharding_constraint %3374 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3376 = stablehlo.divide %3074, %3375 : tensor<32x16xf32>
    %3377 = sdy.sharding_constraint %3376 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_427 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3378 = stablehlo.broadcast_in_dim %cst_427, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3379 = sdy.sharding_constraint %3378 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3380 = stablehlo.add %3193, %3379 : tensor<32x16xf32>
    %3381 = sdy.sharding_constraint %3380 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3382 = stablehlo.sqrt %3381 : tensor<32x16xf32>
    %3383 = sdy.sharding_constraint %3382 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_428 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3384 = stablehlo.broadcast_in_dim %cst_428, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3385 = sdy.sharding_constraint %3384 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3386 = stablehlo.add %3383, %3385 : tensor<32x16xf32>
    %3387 = sdy.sharding_constraint %3386 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3388 = stablehlo.divide %3078, %3387 : tensor<32x16xf32>
    %3389 = sdy.sharding_constraint %3388 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_429 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3390 = stablehlo.broadcast_in_dim %cst_429, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3391 = sdy.sharding_constraint %3390 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3392 = stablehlo.add %3197, %3391 : tensor<32x32xf32>
    %3393 = sdy.sharding_constraint %3392 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3394 = stablehlo.sqrt %3393 : tensor<32x32xf32>
    %3395 = sdy.sharding_constraint %3394 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_430 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3396 = stablehlo.broadcast_in_dim %cst_430, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3397 = sdy.sharding_constraint %3396 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3398 = stablehlo.add %3395, %3397 : tensor<32x32xf32>
    %3399 = sdy.sharding_constraint %3398 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3400 = stablehlo.divide %3082, %3399 : tensor<32x32xf32>
    %3401 = sdy.sharding_constraint %3400 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_431 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3402 = stablehlo.broadcast_in_dim %cst_431, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %3403 = sdy.sharding_constraint %3402 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3404 = stablehlo.add %3201, %3403 : tensor<32x2xf32>
    %3405 = sdy.sharding_constraint %3404 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3406 = stablehlo.sqrt %3405 : tensor<32x2xf32>
    %3407 = sdy.sharding_constraint %3406 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_432 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3408 = stablehlo.broadcast_in_dim %cst_432, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %3409 = sdy.sharding_constraint %3408 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3410 = stablehlo.add %3407, %3409 : tensor<32x2xf32>
    %3411 = sdy.sharding_constraint %3410 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3412 = stablehlo.divide %3086, %3411 : tensor<32x2xf32>
    %3413 = sdy.sharding_constraint %3412 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_433 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3414 = stablehlo.broadcast_in_dim %cst_433, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3415 = sdy.sharding_constraint %3414 <@mesh, [{}]> : tensor<32xf32>
    %3416 = stablehlo.add %3205, %3415 : tensor<32xf32>
    %3417 = sdy.sharding_constraint %3416 <@mesh, [{}]> : tensor<32xf32>
    %3418 = stablehlo.sqrt %3417 : tensor<32xf32>
    %3419 = sdy.sharding_constraint %3418 <@mesh, [{}]> : tensor<32xf32>
    %cst_434 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3420 = stablehlo.broadcast_in_dim %cst_434, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3421 = sdy.sharding_constraint %3420 <@mesh, [{}]> : tensor<32xf32>
    %3422 = stablehlo.add %3419, %3421 : tensor<32xf32>
    %3423 = sdy.sharding_constraint %3422 <@mesh, [{}]> : tensor<32xf32>
    %3424 = stablehlo.divide %3090, %3423 : tensor<32xf32>
    %3425 = sdy.sharding_constraint %3424 <@mesh, [{}]> : tensor<32xf32>
    %cst_435 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3426 = stablehlo.broadcast_in_dim %cst_435, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3427 = sdy.sharding_constraint %3426 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3428 = stablehlo.add %3209, %3427 : tensor<32x128xf32>
    %3429 = sdy.sharding_constraint %3428 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3430 = stablehlo.sqrt %3429 : tensor<32x128xf32>
    %3431 = sdy.sharding_constraint %3430 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_436 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3432 = stablehlo.broadcast_in_dim %cst_436, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3433 = sdy.sharding_constraint %3432 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3434 = stablehlo.add %3431, %3433 : tensor<32x128xf32>
    %3435 = sdy.sharding_constraint %3434 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3436 = stablehlo.divide %3094, %3435 : tensor<32x128xf32>
    %3437 = sdy.sharding_constraint %3436 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_437 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3438 = stablehlo.broadcast_in_dim %cst_437, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3439 = sdy.sharding_constraint %3438 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3440 = stablehlo.add %3213, %3439 : tensor<128x32xf32>
    %3441 = sdy.sharding_constraint %3440 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3442 = stablehlo.sqrt %3441 : tensor<128x32xf32>
    %3443 = sdy.sharding_constraint %3442 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_438 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3444 = stablehlo.broadcast_in_dim %cst_438, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3445 = sdy.sharding_constraint %3444 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3446 = stablehlo.add %3443, %3445 : tensor<128x32xf32>
    %3447 = sdy.sharding_constraint %3446 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3448 = stablehlo.divide %3098, %3447 : tensor<128x32xf32>
    %3449 = sdy.sharding_constraint %3448 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_439 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3450 = stablehlo.broadcast_in_dim %cst_439, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %3451 = sdy.sharding_constraint %3450 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3452 = stablehlo.add %3217, %3451 : tensor<32x4xf32>
    %3453 = sdy.sharding_constraint %3452 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3454 = stablehlo.sqrt %3453 : tensor<32x4xf32>
    %3455 = sdy.sharding_constraint %3454 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_440 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3456 = stablehlo.broadcast_in_dim %cst_440, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %3457 = sdy.sharding_constraint %3456 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3458 = stablehlo.add %3455, %3457 : tensor<32x4xf32>
    %3459 = sdy.sharding_constraint %3458 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3460 = stablehlo.divide %3102, %3459 : tensor<32x4xf32>
    %3461 = sdy.sharding_constraint %3460 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_441 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3462 = stablehlo.broadcast_in_dim %cst_441, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %3463 = sdy.sharding_constraint %3462 <@mesh, [{}]> : tensor<4xf32>
    %3464 = stablehlo.add %3221, %3463 : tensor<4xf32>
    %3465 = sdy.sharding_constraint %3464 <@mesh, [{}]> : tensor<4xf32>
    %3466 = stablehlo.sqrt %3465 : tensor<4xf32>
    %3467 = sdy.sharding_constraint %3466 <@mesh, [{}]> : tensor<4xf32>
    %cst_442 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3468 = stablehlo.broadcast_in_dim %cst_442, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %3469 = sdy.sharding_constraint %3468 <@mesh, [{}]> : tensor<4xf32>
    %3470 = stablehlo.add %3467, %3469 : tensor<4xf32>
    %3471 = sdy.sharding_constraint %3470 <@mesh, [{}]> : tensor<4xf32>
    %3472 = stablehlo.divide %3106, %3471 : tensor<4xf32>
    %3473 = sdy.sharding_constraint %3472 <@mesh, [{}]> : tensor<4xf32>
    %cst_443 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3474 = stablehlo.broadcast_in_dim %cst_443, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3475 = sdy.sharding_constraint %3474 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3476 = stablehlo.add %3225, %3475 : tensor<32x32xf32>
    %3477 = sdy.sharding_constraint %3476 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3478 = stablehlo.sqrt %3477 : tensor<32x32xf32>
    %3479 = sdy.sharding_constraint %3478 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_444 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3480 = stablehlo.broadcast_in_dim %cst_444, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3481 = sdy.sharding_constraint %3480 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3482 = stablehlo.add %3479, %3481 : tensor<32x32xf32>
    %3483 = sdy.sharding_constraint %3482 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3484 = stablehlo.divide %3110, %3483 : tensor<32x32xf32>
    %3485 = sdy.sharding_constraint %3484 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_445 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3486 = stablehlo.broadcast_in_dim %cst_445, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3487 = sdy.sharding_constraint %3486 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3488 = stablehlo.add %3229, %3487 : tensor<32x32xf32>
    %3489 = sdy.sharding_constraint %3488 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3490 = stablehlo.sqrt %3489 : tensor<32x32xf32>
    %3491 = sdy.sharding_constraint %3490 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_446 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3492 = stablehlo.broadcast_in_dim %cst_446, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3493 = sdy.sharding_constraint %3492 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3494 = stablehlo.add %3491, %3493 : tensor<32x32xf32>
    %3495 = sdy.sharding_constraint %3494 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3496 = stablehlo.divide %3114, %3495 : tensor<32x32xf32>
    %3497 = sdy.sharding_constraint %3496 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_447 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3498 = stablehlo.broadcast_in_dim %cst_447, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3499 = sdy.sharding_constraint %3498 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3500 = stablehlo.add %3233, %3499 : tensor<32x32xf32>
    %3501 = sdy.sharding_constraint %3500 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3502 = stablehlo.sqrt %3501 : tensor<32x32xf32>
    %3503 = sdy.sharding_constraint %3502 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_448 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3504 = stablehlo.broadcast_in_dim %cst_448, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3505 = sdy.sharding_constraint %3504 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3506 = stablehlo.add %3503, %3505 : tensor<32x32xf32>
    %3507 = sdy.sharding_constraint %3506 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3508 = stablehlo.divide %3118, %3507 : tensor<32x32xf32>
    %3509 = sdy.sharding_constraint %3508 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_449 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3510 = stablehlo.broadcast_in_dim %cst_449, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3511 = sdy.sharding_constraint %3510 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3512 = stablehlo.add %3237, %3511 : tensor<4x32x32xf32>
    %3513 = sdy.sharding_constraint %3512 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3514 = stablehlo.sqrt %3513 : tensor<4x32x32xf32>
    %3515 = sdy.sharding_constraint %3514 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_450 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3516 = stablehlo.broadcast_in_dim %cst_450, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3517 = sdy.sharding_constraint %3516 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3518 = stablehlo.add %3515, %3517 : tensor<4x32x32xf32>
    %3519 = sdy.sharding_constraint %3518 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3520 = stablehlo.divide %3122, %3519 : tensor<4x32x32xf32>
    %3521 = sdy.sharding_constraint %3520 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_451 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3522 = stablehlo.broadcast_in_dim %cst_451, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3523 = sdy.sharding_constraint %3522 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3524 = stablehlo.add %3241, %3523 : tensor<4x32x32xf32>
    %3525 = sdy.sharding_constraint %3524 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3526 = stablehlo.sqrt %3525 : tensor<4x32x32xf32>
    %3527 = sdy.sharding_constraint %3526 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_452 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3528 = stablehlo.broadcast_in_dim %cst_452, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3529 = sdy.sharding_constraint %3528 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3530 = stablehlo.add %3527, %3529 : tensor<4x32x32xf32>
    %3531 = sdy.sharding_constraint %3530 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3532 = stablehlo.divide %3126, %3531 : tensor<4x32x32xf32>
    %3533 = sdy.sharding_constraint %3532 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_453 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3534 = stablehlo.broadcast_in_dim %cst_453, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3535 = sdy.sharding_constraint %3534 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3536 = stablehlo.add %3245, %3535 : tensor<4x32x32xf32>
    %3537 = sdy.sharding_constraint %3536 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3538 = stablehlo.sqrt %3537 : tensor<4x32x32xf32>
    %3539 = sdy.sharding_constraint %3538 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_454 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3540 = stablehlo.broadcast_in_dim %cst_454, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3541 = sdy.sharding_constraint %3540 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3542 = stablehlo.add %3539, %3541 : tensor<4x32x32xf32>
    %3543 = sdy.sharding_constraint %3542 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3544 = stablehlo.divide %3130, %3543 : tensor<4x32x32xf32>
    %3545 = sdy.sharding_constraint %3544 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_455 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3546 = stablehlo.broadcast_in_dim %cst_455, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3547 = sdy.sharding_constraint %3546 <@mesh, [{}]> : tensor<32xf32>
    %3548 = stablehlo.add %3249, %3547 : tensor<32xf32>
    %3549 = sdy.sharding_constraint %3548 <@mesh, [{}]> : tensor<32xf32>
    %3550 = stablehlo.sqrt %3549 : tensor<32xf32>
    %3551 = sdy.sharding_constraint %3550 <@mesh, [{}]> : tensor<32xf32>
    %cst_456 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3552 = stablehlo.broadcast_in_dim %cst_456, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3553 = sdy.sharding_constraint %3552 <@mesh, [{}]> : tensor<32xf32>
    %3554 = stablehlo.add %3551, %3553 : tensor<32xf32>
    %3555 = sdy.sharding_constraint %3554 <@mesh, [{}]> : tensor<32xf32>
    %3556 = stablehlo.divide %3134, %3555 : tensor<32xf32>
    %3557 = sdy.sharding_constraint %3556 <@mesh, [{}]> : tensor<32xf32>
    %cst_457 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3558 = stablehlo.broadcast_in_dim %cst_457, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3559 = sdy.sharding_constraint %3558 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3560 = stablehlo.add %3253, %3559 : tensor<32x128xf32>
    %3561 = sdy.sharding_constraint %3560 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3562 = stablehlo.sqrt %3561 : tensor<32x128xf32>
    %3563 = sdy.sharding_constraint %3562 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_458 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3564 = stablehlo.broadcast_in_dim %cst_458, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3565 = sdy.sharding_constraint %3564 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3566 = stablehlo.add %3563, %3565 : tensor<32x128xf32>
    %3567 = sdy.sharding_constraint %3566 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3568 = stablehlo.divide %3138, %3567 : tensor<32x128xf32>
    %3569 = sdy.sharding_constraint %3568 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_459 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3570 = stablehlo.broadcast_in_dim %cst_459, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3571 = sdy.sharding_constraint %3570 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3572 = stablehlo.add %3257, %3571 : tensor<128x32xf32>
    %3573 = sdy.sharding_constraint %3572 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3574 = stablehlo.sqrt %3573 : tensor<128x32xf32>
    %3575 = sdy.sharding_constraint %3574 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_460 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %3576 = stablehlo.broadcast_in_dim %cst_460, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3577 = sdy.sharding_constraint %3576 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3578 = stablehlo.add %3575, %3577 : tensor<128x32xf32>
    %3579 = sdy.sharding_constraint %3578 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3580 = stablehlo.divide %3142, %3579 : tensor<128x32xf32>
    %3581 = sdy.sharding_constraint %3580 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_461 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3582 = stablehlo.broadcast_in_dim %cst_461, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %3583 = sdy.sharding_constraint %3582 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3584 = stablehlo.multiply %3583, %arg1 : tensor<64x32xf32>
    %3585 = sdy.sharding_constraint %3584 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3586 = stablehlo.add %3269, %3585 : tensor<64x32xf32>
    %3587 = sdy.sharding_constraint %3586 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_462 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3588 = stablehlo.broadcast_in_dim %cst_462, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3589 = sdy.sharding_constraint %3588 <@mesh, [{}]> : tensor<32xf32>
    %3590 = stablehlo.multiply %3589, %arg2 : tensor<32xf32>
    %3591 = sdy.sharding_constraint %3590 <@mesh, [{}]> : tensor<32xf32>
    %3592 = stablehlo.add %3281, %3591 : tensor<32xf32>
    %3593 = sdy.sharding_constraint %3592 <@mesh, [{}]> : tensor<32xf32>
    %cst_463 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3594 = stablehlo.broadcast_in_dim %cst_463, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3595 = sdy.sharding_constraint %3594 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3596 = stablehlo.multiply %3595, %arg3 : tensor<32x128xf32>
    %3597 = sdy.sharding_constraint %3596 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3598 = stablehlo.add %3293, %3597 : tensor<32x128xf32>
    %3599 = sdy.sharding_constraint %3598 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_464 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3600 = stablehlo.broadcast_in_dim %cst_464, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3601 = sdy.sharding_constraint %3600 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3602 = stablehlo.multiply %3601, %arg4 : tensor<128x32xf32>
    %3603 = sdy.sharding_constraint %3602 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3604 = stablehlo.add %3305, %3603 : tensor<128x32xf32>
    %3605 = sdy.sharding_constraint %3604 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_465 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3606 = stablehlo.broadcast_in_dim %cst_465, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %3607 = sdy.sharding_constraint %3606 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3608 = stablehlo.multiply %3607, %arg5 : tensor<32x64xf32>
    %3609 = sdy.sharding_constraint %3608 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3610 = stablehlo.add %3317, %3609 : tensor<32x64xf32>
    %3611 = sdy.sharding_constraint %3610 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_466 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3612 = stablehlo.broadcast_in_dim %cst_466, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3613 = sdy.sharding_constraint %3612 <@mesh, [{}]> : tensor<32xf32>
    %3614 = stablehlo.multiply %3613, %arg6 : tensor<32xf32>
    %3615 = sdy.sharding_constraint %3614 <@mesh, [{}]> : tensor<32xf32>
    %3616 = stablehlo.add %3329, %3615 : tensor<32xf32>
    %3617 = sdy.sharding_constraint %3616 <@mesh, [{}]> : tensor<32xf32>
    %cst_467 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3618 = stablehlo.broadcast_in_dim %cst_467, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3619 = sdy.sharding_constraint %3618 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3620 = stablehlo.multiply %3619, %arg7 : tensor<32x128xf32>
    %3621 = sdy.sharding_constraint %3620 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3622 = stablehlo.add %3341, %3621 : tensor<32x128xf32>
    %3623 = sdy.sharding_constraint %3622 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_468 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3624 = stablehlo.broadcast_in_dim %cst_468, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3625 = sdy.sharding_constraint %3624 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3626 = stablehlo.multiply %3625, %arg8 : tensor<128x32xf32>
    %3627 = sdy.sharding_constraint %3626 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3628 = stablehlo.add %3353, %3627 : tensor<128x32xf32>
    %3629 = sdy.sharding_constraint %3628 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_469 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3630 = stablehlo.broadcast_in_dim %cst_469, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3631 = sdy.sharding_constraint %3630 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3632 = stablehlo.multiply %3631, %arg9 : tensor<32x32xf32>
    %3633 = sdy.sharding_constraint %3632 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3634 = stablehlo.add %3365, %3633 : tensor<32x32xf32>
    %3635 = sdy.sharding_constraint %3634 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_470 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3636 = stablehlo.broadcast_in_dim %cst_470, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3637 = sdy.sharding_constraint %3636 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3638 = stablehlo.multiply %3637, %arg10 : tensor<32x16xf32>
    %3639 = sdy.sharding_constraint %3638 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3640 = stablehlo.add %3377, %3639 : tensor<32x16xf32>
    %3641 = sdy.sharding_constraint %3640 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_471 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3642 = stablehlo.broadcast_in_dim %cst_471, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3643 = sdy.sharding_constraint %3642 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3644 = stablehlo.multiply %3643, %arg11 : tensor<32x16xf32>
    %3645 = sdy.sharding_constraint %3644 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3646 = stablehlo.add %3389, %3645 : tensor<32x16xf32>
    %3647 = sdy.sharding_constraint %3646 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_472 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3648 = stablehlo.broadcast_in_dim %cst_472, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3649 = sdy.sharding_constraint %3648 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3650 = stablehlo.multiply %3649, %arg12 : tensor<32x32xf32>
    %3651 = sdy.sharding_constraint %3650 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3652 = stablehlo.add %3401, %3651 : tensor<32x32xf32>
    %3653 = sdy.sharding_constraint %3652 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_473 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3654 = stablehlo.broadcast_in_dim %cst_473, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %3655 = sdy.sharding_constraint %3654 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3656 = stablehlo.multiply %3655, %arg13 : tensor<32x2xf32>
    %3657 = sdy.sharding_constraint %3656 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3658 = stablehlo.add %3413, %3657 : tensor<32x2xf32>
    %3659 = sdy.sharding_constraint %3658 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_474 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3660 = stablehlo.broadcast_in_dim %cst_474, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3661 = sdy.sharding_constraint %3660 <@mesh, [{}]> : tensor<32xf32>
    %3662 = stablehlo.multiply %3661, %arg14 : tensor<32xf32>
    %3663 = sdy.sharding_constraint %3662 <@mesh, [{}]> : tensor<32xf32>
    %3664 = stablehlo.add %3425, %3663 : tensor<32xf32>
    %3665 = sdy.sharding_constraint %3664 <@mesh, [{}]> : tensor<32xf32>
    %cst_475 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3666 = stablehlo.broadcast_in_dim %cst_475, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3667 = sdy.sharding_constraint %3666 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3668 = stablehlo.multiply %3667, %arg15 : tensor<32x128xf32>
    %3669 = sdy.sharding_constraint %3668 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3670 = stablehlo.add %3437, %3669 : tensor<32x128xf32>
    %3671 = sdy.sharding_constraint %3670 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_476 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3672 = stablehlo.broadcast_in_dim %cst_476, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3673 = sdy.sharding_constraint %3672 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3674 = stablehlo.multiply %3673, %arg16 : tensor<128x32xf32>
    %3675 = sdy.sharding_constraint %3674 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3676 = stablehlo.add %3449, %3675 : tensor<128x32xf32>
    %3677 = sdy.sharding_constraint %3676 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_477 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3678 = stablehlo.broadcast_in_dim %cst_477, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %3679 = sdy.sharding_constraint %3678 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3680 = stablehlo.multiply %3679, %arg17 : tensor<32x4xf32>
    %3681 = sdy.sharding_constraint %3680 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3682 = stablehlo.add %3461, %3681 : tensor<32x4xf32>
    %3683 = sdy.sharding_constraint %3682 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_478 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3684 = stablehlo.broadcast_in_dim %cst_478, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %3685 = sdy.sharding_constraint %3684 <@mesh, [{}]> : tensor<4xf32>
    %3686 = stablehlo.multiply %3685, %12 : tensor<4xf32>
    %3687 = sdy.sharding_constraint %3686 <@mesh, [{}]> : tensor<4xf32>
    %3688 = stablehlo.add %3473, %3687 : tensor<4xf32>
    %3689 = sdy.sharding_constraint %3688 <@mesh, [{}]> : tensor<4xf32>
    %cst_479 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3690 = stablehlo.broadcast_in_dim %cst_479, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3691 = sdy.sharding_constraint %3690 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3692 = stablehlo.multiply %3691, %arg18 : tensor<32x32xf32>
    %3693 = sdy.sharding_constraint %3692 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3694 = stablehlo.add %3485, %3693 : tensor<32x32xf32>
    %3695 = sdy.sharding_constraint %3694 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_480 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3696 = stablehlo.broadcast_in_dim %cst_480, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3697 = sdy.sharding_constraint %3696 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3698 = stablehlo.multiply %3697, %arg19 : tensor<32x32xf32>
    %3699 = sdy.sharding_constraint %3698 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3700 = stablehlo.add %3497, %3699 : tensor<32x32xf32>
    %3701 = sdy.sharding_constraint %3700 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_481 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3702 = stablehlo.broadcast_in_dim %cst_481, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3703 = sdy.sharding_constraint %3702 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3704 = stablehlo.multiply %3703, %arg20 : tensor<32x32xf32>
    %3705 = sdy.sharding_constraint %3704 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3706 = stablehlo.add %3509, %3705 : tensor<32x32xf32>
    %3707 = sdy.sharding_constraint %3706 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_482 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3708 = stablehlo.broadcast_in_dim %cst_482, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3709 = sdy.sharding_constraint %3708 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3710 = stablehlo.multiply %3709, %arg21 : tensor<4x32x32xf32>
    %3711 = sdy.sharding_constraint %3710 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3712 = stablehlo.add %3521, %3711 : tensor<4x32x32xf32>
    %3713 = sdy.sharding_constraint %3712 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_483 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3714 = stablehlo.broadcast_in_dim %cst_483, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3715 = sdy.sharding_constraint %3714 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3716 = stablehlo.multiply %3715, %arg22 : tensor<4x32x32xf32>
    %3717 = sdy.sharding_constraint %3716 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3718 = stablehlo.add %3533, %3717 : tensor<4x32x32xf32>
    %3719 = sdy.sharding_constraint %3718 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_484 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3720 = stablehlo.broadcast_in_dim %cst_484, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3721 = sdy.sharding_constraint %3720 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3722 = stablehlo.multiply %3721, %arg23 : tensor<4x32x32xf32>
    %3723 = sdy.sharding_constraint %3722 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3724 = stablehlo.add %3545, %3723 : tensor<4x32x32xf32>
    %3725 = sdy.sharding_constraint %3724 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_485 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3726 = stablehlo.broadcast_in_dim %cst_485, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3727 = sdy.sharding_constraint %3726 <@mesh, [{}]> : tensor<32xf32>
    %3728 = stablehlo.multiply %3727, %arg24 : tensor<32xf32>
    %3729 = sdy.sharding_constraint %3728 <@mesh, [{}]> : tensor<32xf32>
    %3730 = stablehlo.add %3557, %3729 : tensor<32xf32>
    %3731 = sdy.sharding_constraint %3730 <@mesh, [{}]> : tensor<32xf32>
    %cst_486 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3732 = stablehlo.broadcast_in_dim %cst_486, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3733 = sdy.sharding_constraint %3732 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3734 = stablehlo.multiply %3733, %arg25 : tensor<32x128xf32>
    %3735 = sdy.sharding_constraint %3734 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3736 = stablehlo.add %3569, %3735 : tensor<32x128xf32>
    %3737 = sdy.sharding_constraint %3736 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_487 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3738 = stablehlo.broadcast_in_dim %cst_487, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3739 = sdy.sharding_constraint %3738 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3740 = stablehlo.multiply %3739, %arg26 : tensor<128x32xf32>
    %3741 = sdy.sharding_constraint %3740 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3742 = stablehlo.add %3581, %3741 : tensor<128x32xf32>
    %3743 = sdy.sharding_constraint %3742 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_488 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3744 = stablehlo.broadcast_in_dim %cst_488, dims = [] : (tensor<f32>) -> tensor<64x32xf32>
    %3745 = sdy.sharding_constraint %3744 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3746 = stablehlo.multiply %3745, %3587 : tensor<64x32xf32>
    %3747 = sdy.sharding_constraint %3746 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %cst_489 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3748 = stablehlo.broadcast_in_dim %cst_489, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3749 = sdy.sharding_constraint %3748 <@mesh, [{}]> : tensor<32xf32>
    %3750 = stablehlo.multiply %3749, %3593 : tensor<32xf32>
    %3751 = sdy.sharding_constraint %3750 <@mesh, [{}]> : tensor<32xf32>
    %cst_490 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3752 = stablehlo.broadcast_in_dim %cst_490, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3753 = sdy.sharding_constraint %3752 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3754 = stablehlo.multiply %3753, %3599 : tensor<32x128xf32>
    %3755 = sdy.sharding_constraint %3754 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_491 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3756 = stablehlo.broadcast_in_dim %cst_491, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3757 = sdy.sharding_constraint %3756 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3758 = stablehlo.multiply %3757, %3605 : tensor<128x32xf32>
    %3759 = sdy.sharding_constraint %3758 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_492 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3760 = stablehlo.broadcast_in_dim %cst_492, dims = [] : (tensor<f32>) -> tensor<32x64xf32>
    %3761 = sdy.sharding_constraint %3760 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3762 = stablehlo.multiply %3761, %3611 : tensor<32x64xf32>
    %3763 = sdy.sharding_constraint %3762 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %cst_493 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3764 = stablehlo.broadcast_in_dim %cst_493, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3765 = sdy.sharding_constraint %3764 <@mesh, [{}]> : tensor<32xf32>
    %3766 = stablehlo.multiply %3765, %3617 : tensor<32xf32>
    %3767 = sdy.sharding_constraint %3766 <@mesh, [{}]> : tensor<32xf32>
    %cst_494 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3768 = stablehlo.broadcast_in_dim %cst_494, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3769 = sdy.sharding_constraint %3768 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3770 = stablehlo.multiply %3769, %3623 : tensor<32x128xf32>
    %3771 = sdy.sharding_constraint %3770 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_495 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3772 = stablehlo.broadcast_in_dim %cst_495, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3773 = sdy.sharding_constraint %3772 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3774 = stablehlo.multiply %3773, %3629 : tensor<128x32xf32>
    %3775 = sdy.sharding_constraint %3774 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_496 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3776 = stablehlo.broadcast_in_dim %cst_496, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3777 = sdy.sharding_constraint %3776 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3778 = stablehlo.multiply %3777, %3635 : tensor<32x32xf32>
    %3779 = sdy.sharding_constraint %3778 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_497 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3780 = stablehlo.broadcast_in_dim %cst_497, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3781 = sdy.sharding_constraint %3780 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3782 = stablehlo.multiply %3781, %3641 : tensor<32x16xf32>
    %3783 = sdy.sharding_constraint %3782 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_498 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3784 = stablehlo.broadcast_in_dim %cst_498, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %3785 = sdy.sharding_constraint %3784 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3786 = stablehlo.multiply %3785, %3647 : tensor<32x16xf32>
    %3787 = sdy.sharding_constraint %3786 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %cst_499 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3788 = stablehlo.broadcast_in_dim %cst_499, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3789 = sdy.sharding_constraint %3788 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3790 = stablehlo.multiply %3789, %3653 : tensor<32x32xf32>
    %3791 = sdy.sharding_constraint %3790 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_500 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3792 = stablehlo.broadcast_in_dim %cst_500, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %3793 = sdy.sharding_constraint %3792 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3794 = stablehlo.multiply %3793, %3659 : tensor<32x2xf32>
    %3795 = sdy.sharding_constraint %3794 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %cst_501 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3796 = stablehlo.broadcast_in_dim %cst_501, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3797 = sdy.sharding_constraint %3796 <@mesh, [{}]> : tensor<32xf32>
    %3798 = stablehlo.multiply %3797, %3665 : tensor<32xf32>
    %3799 = sdy.sharding_constraint %3798 <@mesh, [{}]> : tensor<32xf32>
    %cst_502 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3800 = stablehlo.broadcast_in_dim %cst_502, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3801 = sdy.sharding_constraint %3800 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3802 = stablehlo.multiply %3801, %3671 : tensor<32x128xf32>
    %3803 = sdy.sharding_constraint %3802 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_503 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3804 = stablehlo.broadcast_in_dim %cst_503, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3805 = sdy.sharding_constraint %3804 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3806 = stablehlo.multiply %3805, %3677 : tensor<128x32xf32>
    %3807 = sdy.sharding_constraint %3806 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %cst_504 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3808 = stablehlo.broadcast_in_dim %cst_504, dims = [] : (tensor<f32>) -> tensor<32x4xf32>
    %3809 = sdy.sharding_constraint %3808 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3810 = stablehlo.multiply %3809, %3683 : tensor<32x4xf32>
    %3811 = sdy.sharding_constraint %3810 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %cst_505 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3812 = stablehlo.broadcast_in_dim %cst_505, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %3813 = sdy.sharding_constraint %3812 <@mesh, [{}]> : tensor<4xf32>
    %3814 = stablehlo.multiply %3813, %3689 : tensor<4xf32>
    %3815 = sdy.sharding_constraint %3814 <@mesh, [{}]> : tensor<4xf32>
    %cst_506 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3816 = stablehlo.broadcast_in_dim %cst_506, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3817 = sdy.sharding_constraint %3816 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3818 = stablehlo.multiply %3817, %3695 : tensor<32x32xf32>
    %3819 = sdy.sharding_constraint %3818 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_507 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3820 = stablehlo.broadcast_in_dim %cst_507, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3821 = sdy.sharding_constraint %3820 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3822 = stablehlo.multiply %3821, %3701 : tensor<32x32xf32>
    %3823 = sdy.sharding_constraint %3822 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %cst_508 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3824 = stablehlo.broadcast_in_dim %cst_508, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %3825 = sdy.sharding_constraint %3824 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3826 = stablehlo.multiply %3825, %3707 : tensor<32x32xf32>
    %3827 = sdy.sharding_constraint %3826 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %cst_509 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3828 = stablehlo.broadcast_in_dim %cst_509, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3829 = sdy.sharding_constraint %3828 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3830 = stablehlo.multiply %3829, %3713 : tensor<4x32x32xf32>
    %3831 = sdy.sharding_constraint %3830 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_510 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3832 = stablehlo.broadcast_in_dim %cst_510, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3833 = sdy.sharding_constraint %3832 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3834 = stablehlo.multiply %3833, %3719 : tensor<4x32x32xf32>
    %3835 = sdy.sharding_constraint %3834 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_511 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3836 = stablehlo.broadcast_in_dim %cst_511, dims = [] : (tensor<f32>) -> tensor<4x32x32xf32>
    %3837 = sdy.sharding_constraint %3836 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3838 = stablehlo.multiply %3837, %3725 : tensor<4x32x32xf32>
    %3839 = sdy.sharding_constraint %3838 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_512 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3840 = stablehlo.broadcast_in_dim %cst_512, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3841 = sdy.sharding_constraint %3840 <@mesh, [{}]> : tensor<32xf32>
    %3842 = stablehlo.multiply %3841, %3731 : tensor<32xf32>
    %3843 = sdy.sharding_constraint %3842 <@mesh, [{}]> : tensor<32xf32>
    %cst_513 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3844 = stablehlo.broadcast_in_dim %cst_513, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %3845 = sdy.sharding_constraint %3844 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3846 = stablehlo.multiply %3845, %3737 : tensor<32x128xf32>
    %3847 = sdy.sharding_constraint %3846 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %cst_514 = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %3848 = stablehlo.broadcast_in_dim %cst_514, dims = [] : (tensor<f32>) -> tensor<128x32xf32>
    %3849 = sdy.sharding_constraint %3848 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3850 = stablehlo.multiply %3849, %3743 : tensor<128x32xf32>
    %3851 = sdy.sharding_constraint %3850 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3852 = stablehlo.multiply %2358, %2358 : tensor<4x32x32xf32>
    %3853 = sdy.sharding_constraint %3852 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_515 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3854 = stablehlo.reduce(%3853 init: %cst_515) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %3855 = sdy.sharding_constraint %3854 <@mesh, []> : tensor<f32>
    %cst_516 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3856 = sdy.sharding_constraint %cst_516 <@mesh, []> : tensor<f32>
    %3857 = stablehlo.add %3856, %3855 : tensor<f32>
    %3858 = sdy.sharding_constraint %3857 <@mesh, []> : tensor<f32>
    %3859 = stablehlo.multiply %2356, %2356 : tensor<4x32x32xf32>
    %3860 = sdy.sharding_constraint %3859 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_517 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3861 = stablehlo.reduce(%3860 init: %cst_517) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %3862 = sdy.sharding_constraint %3861 <@mesh, []> : tensor<f32>
    %3863 = stablehlo.add %3858, %3862 : tensor<f32>
    %3864 = sdy.sharding_constraint %3863 <@mesh, []> : tensor<f32>
    %3865 = stablehlo.multiply %2354, %2354 : tensor<4x32x32xf32>
    %3866 = sdy.sharding_constraint %3865 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_518 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3867 = stablehlo.reduce(%3866 init: %cst_518) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %3868 = sdy.sharding_constraint %3867 <@mesh, []> : tensor<f32>
    %3869 = stablehlo.add %3864, %3868 : tensor<f32>
    %3870 = sdy.sharding_constraint %3869 <@mesh, []> : tensor<f32>
    %3871 = stablehlo.sqrt %3870 : tensor<f32>
    %3872 = sdy.sharding_constraint %3871 <@mesh, []> : tensor<f32>
    %3873 = stablehlo.multiply %3831, %3831 : tensor<4x32x32xf32>
    %3874 = sdy.sharding_constraint %3873 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_519 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3875 = stablehlo.reduce(%3874 init: %cst_519) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %3876 = sdy.sharding_constraint %3875 <@mesh, []> : tensor<f32>
    %cst_520 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3877 = sdy.sharding_constraint %cst_520 <@mesh, []> : tensor<f32>
    %3878 = stablehlo.add %3877, %3876 : tensor<f32>
    %3879 = sdy.sharding_constraint %3878 <@mesh, []> : tensor<f32>
    %3880 = stablehlo.multiply %3835, %3835 : tensor<4x32x32xf32>
    %3881 = sdy.sharding_constraint %3880 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %cst_521 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3882 = stablehlo.reduce(%3881 init: %cst_521) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %3883 = sdy.sharding_constraint %3882 <@mesh, []> : tensor<f32>
    %3884 = stablehlo.add %3879, %3883 : tensor<f32>
    %3885 = sdy.sharding_constraint %3884 <@mesh, []> : tensor<f32>
    %3886 = stablehlo.multiply %3839, %3839 : tensor<4x32x32xf32>
    %3887 = sdy.sharding_constraint %3886 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %cst_522 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3888 = stablehlo.reduce(%3887 init: %cst_522) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<4x32x32xf32>, tensor<f32>) -> tensor<f32>
    %3889 = sdy.sharding_constraint %3888 <@mesh, []> : tensor<f32>
    %3890 = stablehlo.add %3885, %3889 : tensor<f32>
    %3891 = sdy.sharding_constraint %3890 <@mesh, []> : tensor<f32>
    %3892 = stablehlo.sqrt %3891 : tensor<f32>
    %3893 = sdy.sharding_constraint %3892 <@mesh, []> : tensor<f32>
    %3894 = stablehlo.add %arg1, %3747 : tensor<64x32xf32>
    %3895 = sdy.sharding_constraint %3894 <@mesh, [{"model"}, {"replica_dcn", "data"}]> : tensor<64x32xf32>
    %3896 = stablehlo.add %arg2, %3751 : tensor<32xf32>
    %3897 = sdy.sharding_constraint %3896 <@mesh, [{}]> : tensor<32xf32>
    %3898 = stablehlo.add %arg3, %3755 : tensor<32x128xf32>
    %3899 = sdy.sharding_constraint %3898 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3900 = stablehlo.add %arg4, %3759 : tensor<128x32xf32>
    %3901 = sdy.sharding_constraint %3900 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3902 = stablehlo.add %arg5, %3763 : tensor<32x64xf32>
    %3903 = sdy.sharding_constraint %3902 <@mesh, [{"replica_dcn", "data"}, {"model"}]> : tensor<32x64xf32>
    %3904 = stablehlo.add %arg6, %3767 : tensor<32xf32>
    %3905 = sdy.sharding_constraint %3904 <@mesh, [{}]> : tensor<32xf32>
    %3906 = stablehlo.add %arg7, %3771 : tensor<32x128xf32>
    %3907 = sdy.sharding_constraint %3906 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3908 = stablehlo.add %arg8, %3775 : tensor<128x32xf32>
    %3909 = sdy.sharding_constraint %3908 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3910 = stablehlo.add %arg9, %3779 : tensor<32x32xf32>
    %3911 = sdy.sharding_constraint %3910 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3912 = stablehlo.add %arg10, %3783 : tensor<32x16xf32>
    %3913 = sdy.sharding_constraint %3912 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3914 = stablehlo.add %arg11, %3787 : tensor<32x16xf32>
    %3915 = sdy.sharding_constraint %3914 <@mesh, [{"data"}, {"model"}]> : tensor<32x16xf32>
    %3916 = stablehlo.add %arg12, %3791 : tensor<32x32xf32>
    %3917 = sdy.sharding_constraint %3916 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3918 = stablehlo.add %arg13, %3795 : tensor<32x2xf32>
    %3919 = sdy.sharding_constraint %3918 <@mesh, [{}, {}]> : tensor<32x2xf32>
    %3920 = stablehlo.add %arg14, %3799 : tensor<32xf32>
    %3921 = sdy.sharding_constraint %3920 <@mesh, [{}]> : tensor<32xf32>
    %3922 = stablehlo.add %arg15, %3803 : tensor<32x128xf32>
    %3923 = sdy.sharding_constraint %3922 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3924 = stablehlo.add %arg16, %3807 : tensor<128x32xf32>
    %3925 = sdy.sharding_constraint %3924 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3926 = stablehlo.add %arg17, %3811 : tensor<32x4xf32>
    %3927 = sdy.sharding_constraint %3926 <@mesh, [{}, {}]> : tensor<32x4xf32>
    %3928 = stablehlo.add %12, %3815 : tensor<4xf32>
    %3929 = sdy.sharding_constraint %3928 <@mesh, [{}]> : tensor<4xf32>
    %3930 = stablehlo.add %arg18, %3819 : tensor<32x32xf32>
    %3931 = sdy.sharding_constraint %3930 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3932 = stablehlo.add %arg19, %3823 : tensor<32x32xf32>
    %3933 = sdy.sharding_constraint %3932 <@mesh, [{"data"}, {"model"}]> : tensor<32x32xf32>
    %3934 = stablehlo.add %arg20, %3827 : tensor<32x32xf32>
    %3935 = sdy.sharding_constraint %3934 <@mesh, [{"model"}, {"data"}]> : tensor<32x32xf32>
    %3936 = stablehlo.add %arg21, %3831 : tensor<4x32x32xf32>
    %3937 = sdy.sharding_constraint %3936 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3938 = stablehlo.add %arg22, %3835 : tensor<4x32x32xf32>
    %3939 = sdy.sharding_constraint %3938 <@mesh, [{"expert"}, {"data"}, {"model"}]> : tensor<4x32x32xf32>
    %3940 = stablehlo.add %arg23, %3839 : tensor<4x32x32xf32>
    %3941 = sdy.sharding_constraint %3940 <@mesh, [{"expert"}, {"model"}, {"data"}]> : tensor<4x32x32xf32>
    %3942 = stablehlo.add %arg24, %3843 : tensor<32xf32>
    %3943 = sdy.sharding_constraint %3942 <@mesh, [{}]> : tensor<32xf32>
    %3944 = stablehlo.add %arg25, %3847 : tensor<32x128xf32>
    %3945 = sdy.sharding_constraint %3944 <@mesh, [{}, {}]> : tensor<32x128xf32>
    %3946 = stablehlo.add %arg26, %3851 : tensor<128x32xf32>
    %3947 = sdy.sharding_constraint %3946 <@mesh, [{}, {}]> : tensor<128x32xf32>
    %3948 = stablehlo.add %arg0, %c : tensor<i32>
    %3949 = sdy.sharding_constraint %3948 <@mesh, []> : tensor<i32>
    %cst_523 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %cst_524 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    return %3949, %3895, %3897, %3899, %3901, %3903, %3905, %3907, %3909, %3911, %3913, %3915, %3917, %3919, %3921, %3923, %3925, %3927, %3929, %3931, %3933, %3935, %3937, %3939, %3941, %3943, %3945, %3947, %3027, %2411, %2422, %2433, %2444, %2455, %2466, %2477, %2488, %2499, %2510, %2521, %2532, %2543, %2554, %2565, %2576, %2587, %2598, %2609, %2620, %2631, %2642, %2653, %2664, %2675, %2686, %2697, %2709, %2721, %2733, %2745, %2757, %2769, %2781, %2793, %2805, %2817, %2829, %2841, %2853, %2865, %2877, %2889, %2901, %2913, %2925, %2937, %2949, %2961, %2973, %2985, %2997, %3009, %3021, %741, %741, %872, %884, %882, %912, %969, %916, %918, %914, %946, %949, %924, %955, %928, %934, %957, %963, %967, %951, %920, %902, %907, %735, %897, %971, %3872, %cst_523, %cst_524, %3893 : tensor<i32>, tensor<64x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x64xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x32xf32>, tensor<32x16xf32>, tensor<32x16xf32>, tensor<32x32xf32>, tensor<32x2xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x4xf32>, tensor<4xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<i32>, tensor<64x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x64xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x32xf32>, tensor<32x16xf32>, tensor<32x16xf32>, tensor<32x32xf32>, tensor<32x2xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x4xf32>, tensor<4xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<64x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x64xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x32xf32>, tensor<32x16xf32>, tensor<32x16xf32>, tensor<32x32xf32>, tensor<32x2xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x4xf32>, tensor<4xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<4x32x32xf32>, tensor<32xf32>, tensor<32x128xf32>, tensor<128x32xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<5xf32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
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
  func.func private @silu_41(%arg0: tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16> {
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
    %14 = call @_where_182(%11, %13, %2) : (tensor<16xi1>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %14 : tensor<16xi32>
  }
  func.func private @_where_182(%arg0: tensor<16xi1>, %arg1: tensor<16xi32>, %arg2: tensor<16xi32>) -> tensor<16xi32> {
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
  func.func private @silu_209(%arg0: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
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
  func.func private @silu_229(%arg0: tensor<8x32xbf16>) -> tensor<8x32xbf16> {
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
  func.func private @_where_278(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_289(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<f32>) -> tensor<4xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = sdy.sharding_constraint %1 <@mesh, [{}]> : tensor<4xf32>
    %3 = stablehlo.select %arg0, %arg1, %2 : tensor<4xi1>, tensor<4xf32>
    %4 = sdy.sharding_constraint %3 <@mesh, [{}]> : tensor<4xf32>
    return %4 : tensor<4xf32>
  }
  func.func private @_where_295(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    %2 = sdy.sharding_constraint %1 <@mesh, []> : tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @_where_308(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.select %arg0, %arg1, %cst : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @silu_329(%arg0: tensor<2x4x128xbf16>, %arg1: tensor<2x4x128xbf16>, %arg2: tensor<2x4x128xbf16>, %arg3: tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16> {
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
  func.func private @silu_347(%arg0: tensor<2x4x128xbf16>) -> (tensor<2x4x128xbf16>, tensor<2x4x128xbf16>, tensor<2x4x128xbf16>) {
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
  func.func private @_where_354(%arg0: tensor<1x1x4x4xi1>, %arg1: tensor<2x2x4x4xbf16>, %arg2: tensor<bf16>) -> (tensor<2x2x4x4xbf16>, tensor<2x2x4x4xi1>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x4x4xi1>) -> tensor<2x2x4x4xi1>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xi1>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<bf16>) -> tensor<2x2x4x4xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %4 = stablehlo.select %1, %arg1, %3 : tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %5 = sdy.sharding_constraint %4 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    return %5, %1 : tensor<2x2x4x4xbf16>, tensor<2x2x4x4xi1>
  }
  func.func private @take_along_axis_364(%arg0: tensor<8x4xf32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xf32>, tensor<8x2x1xi32>) {
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
  func.func private @argsort_374(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<16xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare LT, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) : (tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>)
    return %1#1 : tensor<16xi32>
  }
  func.func private @floor_divide_375(%arg0: tensor<16xi32>, %arg1: tensor<i32>) -> tensor<16xi32> {
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
    %14 = call @_where_376(%11, %13, %2) : (tensor<16xi1>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %14 : tensor<16xi32>
  }
  func.func private @_where_376(%arg0: tensor<16xi1>, %arg1: tensor<16xi32>, %arg2: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<16xi1>, tensor<16xi32>
    return %0 : tensor<16xi32>
  }
  func.func private @clip_377(%arg0: tensor<16xi32>, %arg1: tensor<i32>) -> tensor<16xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16xi32>
    return %2 : tensor<16xi32>
  }
  func.func private @silu_378(%arg0: tensor<16x32xbf16>) -> (tensor<16x32xbf16>, tensor<16x32xbf16>, tensor<16x32xbf16>) {
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
  func.func private @silu_380(%arg0: tensor<8x32xbf16>) -> (tensor<8x32xbf16>, tensor<8x32xbf16>, tensor<8x32xbf16>) {
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
  func.func private @silu_386(%arg0: tensor<8x32xbf16>, %arg1: tensor<8x32xbf16>, %arg2: tensor<8x32xbf16>, %arg3: tensor<8x32xbf16>) -> tensor<8x32xbf16> {
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
  func.func private @silu_413(%arg0: tensor<16x32xbf16>, %arg1: tensor<16x32xbf16>, %arg2: tensor<16x32xbf16>, %arg3: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
    %0 = stablehlo.multiply %arg1, %arg3 : tensor<16x32xbf16>
    %1 = stablehlo.multiply %arg3, %arg0 : tensor<16x32xbf16>
    %2 = stablehlo.multiply %0, %arg2 : tensor<16x32xbf16>
    %3 = stablehlo.add %1, %2 : tensor<16x32xbf16>
    return %3 : tensor<16x32xbf16>
  }
  func.func private @take_along_axis_454(%arg0: tensor<8x2x1xi32>, %arg1: tensor<8x2xf32>) -> tensor<8x4xf32> {
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
  func.func private @silu_463(%arg0: tensor<2x4x128xbf16>, %arg1: tensor<2x4x128xbf16>, %arg2: tensor<2x4x128xbf16>, %arg3: tensor<2x4x128xbf16>) -> tensor<2x4x128xbf16> {
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
  func.func private @_where_497(%arg0: tensor<2x2x4x4xi1>, %arg1: tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x2x4x4xbf16>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %3 = sdy.sharding_constraint %2 <@mesh, [{"replica_dcn", "data", "expert"}, {"model"}, {}, {}]> : tensor<2x2x4x4xbf16>
    return %3 : tensor<2x2x4x4xbf16>
  }
  func.func private @_where_596(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    %1 = sdy.sharding_constraint %0 <@mesh, []> : tensor<i32>
    return %1 : tensor<i32>
  }
}
