# fast_track config inventory (marin_belay_v1_w2026-10-19 @ 87dfa66f24)

Columns: the field's default; the values Belay Ckpt 1 sets at the H100 rungs d512-d1280 (`belay`) and at
`gb200-d6144`; the values the other live launchers set where they differ from the default (`launch.py` default
d512 baseline, `--dense` d512); and how many of the last six days' Iris launches set the field
(`H100 runs`: the
221 H100 research launches on cw-us-east-02a, `all`: 1029 launches including GB200/MFU). `belay_tracks.py`
(Vanilla / Basic MoE) is read from its source. Decisions: keep, keep (research) = only today's d512 studies
use it, delete (dead) = off by default and set by no recipe or launcher on the branch, delete (GB200/MFU) =
only the GB200 recipe or MFU probes reach it.

## GrugModelConfig (147 fields)

| field | default | belay (H100) | gb200-d6144 | baseline / dense | H100 runs | all | decision |
|---|---|---|---|---|---|---|---|
| `vocab_size` | `16384` |  |  |  | 0 | 0 | keep |
| `hidden_dim` | `512` | 1024, 1280, 768 | 6144 |  | 0 | 0 | keep |
| `intermediate_dim` | `256` | 384, 576, 768, 960 | 4608 | 1536 | 400 | 996 | keep |
| `shared_expert_intermediate_dim` | `256` | 384, 576, 768, 960 | 4608 |  | 400 | 996 | keep |
| `num_shared_experts` | `2` | 1 | 1 |  | 193 | 789 | keep |
| `num_experts` | `384` |  |  |  | 190 | 190 | keep |
| `num_experts_per_token` | `8` | 10 | 10 |  | 193 | 789 | keep |
| `moe_dynamic` | `<MoeDynamicK.NONE: 'none'>` |  |  |  | 0 | 0 | delete (dead) |
| `moe_dynamic_layers` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `moe_k_static` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `moe_k_low` | `5` |  |  |  | 0 | 0 | delete (dead) |
| `attn_head_skip` | `0.0` |  |  |  | 0 | 0 | delete (dead) |
| `moe_small_expert_width` | `0.0` |  |  |  | 0 | 0 | delete (dead) |
| `moe_dynamic_position` | `512` |  |  |  | 0 | 0 | delete (dead) |
| `moe_dynamic_frac` | `0.5` |  |  |  | 0 | 0 | delete (dead) |
| `moe_relu_l1` | `0.0` |  |  |  | 0 | 0 | delete (dead) |
| `moe_relu_offset` | `0.0` |  |  |  | 0 | 0 | delete (dead) |
| `moe_eps` | `0.0` |  |  |  | 0 | 0 | delete (dead) |
| `moe_top_p` | `0.0` |  |  |  | 0 | 0 | delete (dead) |
| `moe_k_layers` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `moe_null_layers` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `moe_null_renorm_scale` | `1.0` |  |  |  | 0 | 0 | delete (dead) |
| `moe_null_learned_gain` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `moe_null_target_frac` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `latent_dim` | `256` | 384, 512, 640 | 3072 | None | 0 | 596 | keep |
| `latent_out_dim` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `latent_out_full_layers` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `num_layers` | `6` | 12, 14, 8 | 48 |  | 11 | 11 | keep |
| `num_heads` | `4` | 10, 6, 8 | 48 |  | 0 | 0 | keep |
| `mla_num_heads` | `None` | 10, 4, 6, 8 | 48 |  | 400 | 577 | keep |
| `num_kv_heads` | `1` | 2 |  |  | 0 | 908 | keep |
| `local_kv_heads` | `1` | 2 |  |  | 0 | 908 | keep |
| `global_kv_heads` | `1` |  |  |  | 0 | 601 | keep |
| `head_dim` | `128` |  |  |  | 0 | 0 | keep |
| `gpu_attention` | `'gpu_fa4_cute'` |  | 'gpu_pallas_fa4' |  | 212 | 1109 | keep (research): kernel name passed to levanter attention; 28 recent H100 runs set gpu_pallas_fa4 |
| `max_seq_len` | `4096` |  |  |  | 0 | 0 | keep |
| `sliding_window` | `2048` |  |  |  | 0 | 0 | keep |
| `global_every` | `4` |  |  |  | 0 | 0 | keep |
| `capacity_factor` | `1.05` |  |  |  | 196 | 808 | keep |
| `layer_norm_eps` | `1e-05` |  |  |  | 0 | 0 | keep |
| `initializer_std` | `0.02` | 0.013975424859373685, 0.015625, 0.018042195912175808, 0.022097086912079608 | 0.0063788795384978605 | 0.022097086912079608 | 0 | 0 | keep |
| `qk_mult` | `1.3` | 4.0 | 4.0 |  | 190 | 190 | keep |
| `qk_norm` | `True` | False | False |  | 210 | 387 | keep |
| `rope_global_layers` | `False` |  |  |  | 0 | 0 | keep |
| `rope_full_head` | `False` |  |  |  | 0 | 0 | keep |
| `attn_xsa` | `True` |  |  |  | 0 | 0 | keep |
| `attn_output_gate` | `True` |  |  |  | 0 | 0 | keep |
| `sconv` | `True` |  |  |  | 0 | 0 | keep |
| `sconv_kernel` | `4` |  |  |  | 0 | 0 | keep |
| `sconv_impl` | `None` |  |  |  | 210 | 676 | keep (research) |
| `sconv_sites` | `('k', 'attn', 'mlp')` | ('attn', 'mlp') | ('attn', 'mlp') |  | 23 | 23 | keep |
| `pooled_transport_capacity_factor` | `1.05` |  |  |  | 0 | 0 | keep |
| `rope` | `RotaryConfig(theta=10000.0, scaling_f...` |  |  |  | 0 | 0 | keep |
| `dense_mlp` | `False` |  |  | True | 0 | 0 | keep |
| `inkling_relpos` | `False` | True | True |  | 210 | 388 | keep |
| `rel_dim` | `16` |  |  |  | 0 | 0 | keep |
| `rel_extent` | `1024` |  |  |  | 210 | 388 | keep |
| `mla` | `False` | True | True |  | 210 | 387 | keep |
| `mla_kv_latent_dim` | `512` |  |  |  | 210 | 387 | keep |
| `local_mixer` | `<LocalMixer.SLIDING_WINDOW: 'sliding_...` | <LocalMixer.KDA: 'kda'> | <LocalMixer.KDA: 'kda'> |  | 210 | 519 | keep |
| `kda_dt_range` | `(0.02, 0.5)` |  |  |  | 0 | 0 | keep |
| `kda_save_chunk_states` | `False` | True |  | True | 0 | 7 | keep |
| `kda_chunk_size` | `16` | 32 | 32 |  | 400 | 711 | keep |
| `kda_onchip_decay_gate` | `False` | True | True |  | 400 | 709 | keep |
| `kda_onchip_erase_gate` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `kda_state_grads` | `<StateGrads.XLA: 'xla'>` | <StateGrads.PALLAS: 'pallas'> |  |  | 400 | 725 | keep |
| `kda_state_block_v` | `32` |  | 64 |  | 210 | 519 | keep (research): kernel tile; 28 recent H100 runs set 64 |
| `moe_shared_placement` | `<SharedExpertPlacement.FUSED: 'fused'>` |  | <SharedExpertPlacement.OVERLAP: 'over... |  | 0 | 619 | delete (GB200/MFU): FUSED (default) kept as the only path; OVERLAP / UNFUSED were GB200 scheduling |
| `attn_res` | `False` | True | True |  | 210 | 808 | keep |
| `attn_res_num_blocks` | `8` |  |  |  | 214 | 781 | keep |
| `attn_res_block_starts` | `()` |  |  |  | 1 | 1 | keep (research) |
| `attn_res_two_phase` | `True` |  |  |  | 0 | 0 | delete (GB200/MFU): scan_layers only |
| `attn_res_plain_sum` | `False` |  |  |  | 214 | 810 | keep (research) |
| `attn_res_shared_query` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `attn_res_kv_layers` | `()` |  |  |  | 76 | 76 | keep (research) |
| `extra_mlp_layers` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `extra_mlp_dim` | `0` |  |  |  | 0 | 0 | delete (dead) |
| `entropy_skip_layers` | `()` |  |  |  | 18 | 18 | keep (research) |
| `skip_adapter_dim` | `0` |  |  |  | 50 | 50 | keep (research) |
| `skip_router_layers` | `()` |  |  |  | 32 | 32 | keep (research) |
| `skip_router_frac` | `0.2` |  |  |  | 53 | 53 | keep (research) |
| `mid_head_layer` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `mid_head_loss_weight` | `1.0` |  |  |  | 0 | 0 | delete (dead) |
| `mid_head_trunk_grad` | `True` |  |  |  | 0 | 0 | delete (dead) |
| `mid_confidence_dim` | `256` |  |  |  | 0 | 0 | delete (dead) |
| `final_conf_layers` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `skip_probe_layer` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `skip_probe_target` | `<SkipProbeTarget.CORRECT: 'correct'>` |  |  |  | 0 | 0 | delete (dead) |
| `mid_skip_layers` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `mid_skip_threshold` | `0.9` |  |  |  | 0 | 0 | delete (dead) |
| `entropy_head_input_layer` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `extra_mlp_select` | `<ExtraMlpSelect.HIGH: 'high'>` |  |  |  | 18 | 18 | keep (research) |
| `extra_mlp_frac` | `0.25` |  |  |  | 30 | 30 | keep (research) |
| `attn_res_kv_source_layers` | `None` |  |  |  | 38 | 38 | keep (research) |
| `attn_res_partial_scale` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `sublayer_gated_norm` | `True` |  |  |  | 0 | 0 | keep |
| `attn_res_layer_backward` | `<AttnResLayerBackward.RECOMPUTE: 'rec...` | <AttnResLayerBackward.SAVE: 'save'> |  | <AttnResLayerBackward.SAVE: 'save'> | 1 | 1 | keep |
| `scan_offload_partials` | `False` |  | True |  | 0 | 598 | delete (GB200/MFU) |
| `scan_save_routed_output` | `False` |  | True |  | 0 | 598 | delete (GB200/MFU) |
| `attn_res_fused_source_grads` | `False` |  | True |  | 0 | 600 | delete (GB200/MFU) |
| `scan_layers` | `False` |  | True |  | 0 | 598 | delete (GB200/MFU) |
| `attn_res_remat_attention` | `False` | True |  |  | 0 | 0 | keep |
| `value_embeds` | `<ValueEmbeds.NONE: 'none'>` | <ValueEmbeds.GATED: 'gated'> | <ValueEmbeds.GATED: 'gated'> |  | 199 | 199 | keep |
| `value_embed_fsdp` | `False` |  | True |  | 0 | 0 | delete (GB200/MFU) |
| `router_combine` | `<RouterCombine.SIGMOID_RENORM: 'sigmo...` | <RouterCombine.SQRT_SOFTPLUS_RENORM: ... | <RouterCombine.SQRT_SOFTPLUS_RENORM: ... |  | 191 | 191 | keep |
| `routing_renorm_sum` | `2.5` |  |  |  | 0 | 0 | keep |
| `embed_gated_norm` | `True` | False | False |  | 190 | 190 | keep |
| `final_gated_norm` | `True` | False | False |  | 190 | 190 | keep |
| `mla_k_norm` | `False` | True | True |  | 400 | 577 | keep |
| `mla_q_norm` | `False` | True | True |  | 400 | 577 | keep |
| `mla_key_offset` | `False` | True | True |  | 190 | 190 | keep |
| `logit_soft_cap` | `None` | 10.0 | 10.0 |  | 196 | 196 | keep |
| `lm_head_grad_probe` | `False` |  |  |  | 14 | 14 | keep (research) |
| `lm_head_grad_probe_site` | `<GradProbeSite.POST_NORM: 'post_norm'>` |  |  |  | 10 | 10 | keep (research) |
| `lm_head_grad_dump_every` | `0` |  |  |  | 5 | 5 | keep (research) |
| `token_embed_stats` | `False` |  |  |  | 8 | 8 | keep (research) |
| `lm_head_grad_clip` | `None` |  |  |  | 7 | 7 | keep (research) |
| `init_std_mult_gates` | `1.0` | 2.0 | 2.0 |  | 190 | 190 | keep |
| `kda_write_gate` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `kda_erase_gate` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `init_std_mult_beta` | `None` | 1.0 | 1.0 |  | 190 | 190 | keep |
| `shared_ungated_relu2` | `False` | True | True |  | 400 | 996 | keep |
| `moe_expert_waves` | `1` |  |  |  | 0 | 0 | delete (dead) |
| `moe_expert_remat` | `True` |  |  |  | 0 | 0 | delete (dead) |
| `fp8_recipe` | `<Fp8Recipe.NONE: 'none'>` |  | <Fp8Recipe.MXFP8: 'mxfp8'> |  | 127 | 726 | delete (GB200/MFU) |
| `mxfp8_fused_grad_quant` | `False` |  | True |  | 127 | 726 | delete (GB200/MFU) |
| `moe_ragged_chunks` | `2` |  | 1 |  | 6 | 517 | keep (research): H100 rung default 2; 6 recent H100 runs set 1 |
| `moe_overlap_weight_grads` | `False` |  |  |  | 0 | 4 | delete (GB200/MFU) |
| `kda_fused_projections` | `False` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `moe_mxfp8_dispatch` | `False` |  | True |  | 127 | 792 | delete (GB200/MFU) |
| `moe_serial_pair` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `moe_implementation` | `'fixed_pooled_wave_all_to_all'` | 'ragged_all_to_all' | 'ragged_all_to_all' |  | 401 | 1002 | keep |
| `moe_ungated_kernel` | `False` | True | True |  | 400 | 996 | keep |
| `moe_ungated_relu2` | `False` | True | True |  | 400 | 996 | keep |
| `expert_activation` | `'silu'` |  |  |  | 0 | 0 | keep |
| `moe_drop_renorm` | `False` |  |  |  | 190 | 190 | delete (dead): Ckpt 1 sets it to its default (false); entry dropped from MODEL_SETTINGS |
| `learnable_qk_mult` | `False` | True | True |  | 190 | 190 | keep |
| `second_embed` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `embed2_rows` | `0` |  |  |  | 0 | 0 | delete (dead) |
| `bigram_gate` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `bigram_gate_rank` | `0` |  |  |  | 0 | 0 | delete (dead) |
| `embed2_fsdp` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `embed2_row_sharded_gather` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `embed2_ngram` | `2` |  |  |  | 0 | 0 | delete (dead) |
| `embed2_dim` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `embed2_lambda_init` | `0.1` |  |  |  | 0 | 0 | delete (dead) |
| `second_embed_bigram` | `False` |  |  |  | 0 | 0 | delete (dead) |

## GrugMoeMuonHConfig (35 fields)

| field | default | belay (H100) | gb200-d6144 | baseline / dense | H100 runs | all | decision |
|---|---|---|---|---|---|---|---|
| `learning_rate` | `0.0006` | 0.006121557526161884, 0.008073453011564312, 0.008759003425502152, 0.014445221852839635 |  | 0.022715338080280003, 0.03489331765714976 | 0 | 333 | keep |
| `weight_decay` | `0.1` |  |  |  | 0 | 0 | keep |
| `min_lr_ratio` | `0.1` | 0.05 |  | 0.05 | 0 | 0 | keep |
| `warmup` | `0.01` |  |  |  | 0 | 333 | keep |
| `decay` | `None` |  |  |  | 0 | 0 | keep |
| `rewarmup` | `0.0` |  |  |  | 0 | 0 | keep |
| `cooldown` | `None` |  |  |  | 0 | 0 | keep |
| `cycle_length` | `None` |  |  |  | 0 | 0 | keep |
| `cycles` | `None` |  |  |  | 0 | 0 | keep |
| `lr_schedule` | `'cosine'` |  |  | 'linear' | 195 | 195 | keep |
| `haps` | `None` |  |  |  | 0 | 0 | keep |
| `weight_decay_modules` | `None` |  |  |  | 0 | 0 | keep |
| `default_weight_decay_mask` | `None` |  |  |  | 0 | 0 | keep |
| `adam_lr` | `0.0006` | 0.0014126671214219732, 0.0018631045411302259, 0.002021308482808189, 0.0033335127352706853 |  | 0.0052420010954492315, 0.008052304074726869 | 0 | 333 | keep |
| `momentum` | `0.95` |  |  |  | 0 | 0 | keep |
| `nesterov` | `True` |  |  |  | 0 | 0 | keep |
| `backend_steps` | `5` |  |  |  | 0 | 0 | keep |
| `beta1` | `0.9` |  |  |  | 0 | 0 | keep |
| `beta2` | `0.95` | 0.992027944069944, 0.996005996001 |  | 0.996005996001 | 0 | 0 | keep |
| `epsilon` | `1e-08` | 1.5297569333459482e-15, 1.739609180030963e-15, 2.3217964743120788e-15, 9.087716507990333e-16 |  | 2.541677269835807e-16, 4.725416042635823e-16 | 0 | 333 | keep |
| `muon_epsilon` | `1e-08` |  |  |  | 0 | 0 | keep |
| `max_grad_norm` | `None` |  |  |  | 0 | 0 | delete (dead): never set (MuonH path) |
| `coefficient_type` | `'quintic'` |  |  |  | 0 | 0 | keep |
| `gate_router_weight_decay` | `0.02` |  |  |  | 0 | 0 | keep |
| `kda_beta_group` | `'kda_beta'` | 'adam' | 'adam' |  | 190 | 190 | keep |
| `embed2_lr_mult` | `1.0` |  |  |  | 0 | 0 | delete (dead) |
| `embed2_beta2` | `0.95` |  |  |  | 0 | 0 | delete (dead) |
| `muon_pre_norm` | `'none'` | 'in' | 'in' |  | 191 | 191 | keep |
| `muonh_decay_power` | `None` | 0.7 | 0.7 |  | 195 | 195 | keep |
| `muon_bimaxwell` | `False` |  |  |  | 0 | 0 | delete (dead) |
| `adam_ademamix_alpha` | `0.0` | 5.0 | 5.0 |  | 196 | 196 | keep |
| `adam_ademamix_beta3` | `0.999` | 0.98 | 0.98 |  | 190 | 190 | keep |
| `router_lr_mult` | `1.0` | 0.5 | 0.5 |  | 190 | 190 | keep |
| `embed_optimizer` | `<EmbedOptimizer.ADAM_GROUP: 'adam_gro...` |  |  |  | 6 | 6 | keep (research) |
| `embed_weight_decay` | `0.0` |  |  |  | 4 | 4 | keep (research) |

## GrugTrainerConfig (30 fields)

| field | default | belay (H100) | gb200-d6144 | baseline / dense | H100 runs | all | decision |
|---|---|---|---|---|---|---|---|
| `trainer` | `TrainerConfig(seed=0, mp=Policy(param...` | TrainerConfig(seed=0, mp=Policy(param... |  | TrainerConfig(seed=0, mp=Policy(param... | 0 | 0 | keep |
| `data_seed` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `log_every` | `1` |  |  |  | 0 | 0 | keep |
| `z_loss_weight` | `0.0001` |  |  |  | 0 | 0 | keep |
| `watch_mode` | `<WatchMode.INLINE: 'inline'>` |  |  |  | 0 | 0 | delete (dead): DIAGNOSTIC watch never used; inline watch is the only mode |
| `save_checkpoints` | `False` | True |  |  | 0 | 0 | keep |
| `hlo_dump_path` | `None` |  |  |  | 0 | 0 | keep |
| `token_dump_path` | `None` |  |  |  | 0 | 0 | keep |
| `token_dump_skip_layers` | `()` |  |  |  | 0 | 0 | keep |
| `lm_head_grad_dump_dir` | `None` |  |  |  | 0 | 0 | keep |
| `routing_dump_dir` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `routing_dump_steps` | `()` |  |  |  | 0 | 0 | delete (dead) |
| `xla_autotune_cache_path` | `None` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `xla_autotune_seed_topology` | `None` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `preload_library_kernels` | `False` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `first_step_timeout` | `None` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `shared_executable_dir` | `None` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `layer_skip` | `None` |  |  |  | 0 | 0 | keep |
| `extra_mlp_start_frac` | `0.5` |  |  |  | 0 | 0 | keep |
| `pipeline_depth` | `0` |  |  |  | 0 | 0 | delete (dead): opt-in 1 crashed near step 515; only 15 H100 MFU probes used it |
| `loader_buffer_batches` | `512` |  |  |  | 0 | 0 | keep |
| `xla_memory_report_path` | `None` |  |  |  | 0 | 0 | keep |
| `ema_beta` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `ema_last_steps` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `completion_marker_path` | `None` | '/tmp/hlo_hash_out/final_eval_done.js... |  | '/tmp/hlo_hash_out/final_eval_done.js... | 0 | 0 | keep |
| `grad_accum_microbatches` | `1` |  |  |  | 0 | 0 | keep |
| `expert_axis_size` | `1` | 8 |  | 8 | 0 | 0 | keep |
| `replica_axis_size` | `None` | 1 |  | 1 | 0 | 0 | keep |
| `offload_opt_state` | `False` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `master_param_mode` | `<MasterParamMode.DEVICE: 'device'>` |  |  |  | 0 | 0 | delete (GB200/MFU): fp32 pinned-host master, GB200 hero memory |

## GrugEvalConfig (5 fields)

| field | default | belay (H100) | gb200-d6144 | baseline / dense | H100 runs | all | decision |
|---|---|---|---|---|---|---|---|
| `eval_batch_size` | `512` | 8 |  | 8 | 0 | 0 | keep |
| `steps_per_eval` | `1000` | 24995, 32323, 57578, 8821 |  | 2385, 690 | 0 | 0 | keep |
| `compute_bpb` | `True` |  |  |  | 0 | 0 | keep |
| `dropless_eval` | `False` | True |  | True | 0 | 0 | keep |
| `dropless_eval_moe_implementation` | `'sonic_cute'` | 'sonic' |  | 'sonic' | 0 | 0 | keep |

## GrugRunConfig (13 fields)

| field | default | belay (H100) | gb200-d6144 | baseline / dense | H100 runs | all | decision |
|---|---|---|---|---|---|---|---|
| `model` | `'(required)'` | GrugModelConfig(vocab_size=16384, hid... |  | GrugModelConfig(vocab_size=16384, hid... | 0 | 0 | keep |
| `data` | `'(required)'` | LmDataConfig(tokenizer='hero-bpe-v163... |  | LmDataConfig(tokenizer='hero-bpe-v163... | 0 | 0 | keep |
| `resources` | `'(required)'` | None |  | None | 0 | 0 | keep |
| `tensorstore_cache_bytes` | `None` | 1000000000 |  | 1000000000 | 0 | 0 | keep |
| `optimizer` | `AdamConfig(learning_rate=0.0006, weig...` | GrugMoeMuonHConfig(learning_rate=0.00..., GrugMoeMuonHConfig(learning_rate=0.01... |  | GrugMoeMuonHConfig(learning_rate=0.02..., GrugMoeMuonHConfig(learning_rate=0.03... | 0 | 0 | keep |
| `trainer` | `GrugTrainerConfig(trainer=TrainerConf...` | GrugTrainerConfig(trainer=TrainerConf... |  | GrugTrainerConfig(trainer=TrainerConf... | 0 | 0 | keep |
| `eval` | `GrugEvalConfig(eval_batch_size=512, s...` | GrugEvalConfig(eval_batch_size=8, ste... |  | GrugEvalConfig(eval_batch_size=8, ste... | 0 | 0 | keep |
| `stop_after_steps` | `None` | 24995, 32323, 57578, 8821 |  | 2385, 690 | 0 | 0 | keep |
| `processes_per_task` | `1` |  |  | 8 | 0 | 0 | keep |
| `ragged_transport` | `<RaggedTransport.DEVICE: 'device'>` | <RaggedTransport.ONE_SHOT: 'one_shot'> |  |  | 0 | 0 | keep: DEVICE (the GB200 NCCL GIN kernel, old default) removed; ONE_SHOT is the default |
| `collective_overlap_limit` | `None` |  |  |  | 0 | 0 | delete (dead) |
| `max_retries_failure` | `0` | 3 |  | 3 | 0 | 0 | keep |
| `max_task_failures` | `10` | 3 |  | 3 | 0 | 0 | keep |

## H100LadderRung (10 fields)

| field | default | belay (H100) | gb200-d6144 | baseline / dense | H100 runs | all | decision |
|---|---|---|---|---|---|---|---|
| `shape` | `'(required)'` |  |  |  | 0 | 0 | keep |
| `gpus_per_task` | `'(required)'` |  |  |  | 0 | 0 | keep |
| `baseline_batch` | `'(required)'` |  |  |  | 0 | 0 | keep |
| `attn_res_layer_backward` | `<AttnResLayerBackward.RECOMPUTE: 'rec...` |  |  |  | 0 | 0 | keep |
| `kda_save_chunk_states` | `False` |  |  |  | 0 | 0 | keep |
| `attn_res_remat_attention` | `False` |  |  |  | 0 | 0 | keep |
| `device` | `'H100'` |  |  |  | 0 | 0 | delete (GB200/MFU) |
| `nodes` | `1` |  |  |  | 0 | 0 | keep |
| `moe_ragged_chunks` | `2` |  |  |  | 0 | 0 | keep: H100 rung default 2; 6 recent H100 runs set 1 |
| `hero_memory` | `False` |  |  |  | 0 | 0 | delete (GB200/MFU) |

## launch.py options

| option | decision | note |
|---|---|---|
| `--run-id` | keep |  |
| `--size` | keep | gb200-d6144 and the unused d1536 rung removed |
| `--match` | keep |  |
| `--batch-size` | keep |  |
| `--num-steps` | keep |  |
| `--no-eval` | keep |  |
| `--dense` | keep | belay_tracks Vanilla |
| `--save-checkpoints` | keep | Ckpt 1 ladder |
| `--temporary-checkpoints-only` | keep | 76 recent runs |
| `--optimizer` | keep | belay_tracks Vanilla uses adamw |
| `--recipe` | keep |  |
| `--attn-res-remat-attention` | delete (dead) | the d1280 rung sets it; flag unused |
| `--seed` | keep |  |
| `--profile-start-step` | keep | profiling |
| `--dump-hlo` | keep | profiling |
| `--extra-mlp-start-frac` | keep (research) | entropy-skip studies |
| `--layer-skip-layers` | keep (research) |  |
| `--layer-skip-rate` | keep (research) |  |
| `--layer-skip-positions` | keep (research) |  |
| `--layer-skip-token-ids-file` | keep (research) |  |
| `--token-dump` | keep (research) |  |
| `--token-dump-skip-layers` | keep (research) |  |
| `--seed-xla-autotune-cache` | delete (GB200/MFU) |  |
| `--watch-interval` | keep |  |
| `--xla-autotune` | delete (GB200/MFU) |  |
| `--routing-dump-steps` | delete (dead) | no recent run |
| `--resume-from` | keep | 47 recent runs |
| `--master-param-mode` | delete (GB200/MFU) |  |
| `--xla-memory-report` | keep | OOM diagnosis |
| `--pgle-runs` | delete (GB200/MFU) | PGLE plumbing |
| `--pipeline-depth` | delete (dead) | see pipeline_depth |
| `--loader-buffer-batches` | keep |  |
| `--ema-beta` | delete (dead) |  |
| `--ema-last-steps` | delete (dead) |  |
| `--max-retries` | keep |  |
| `--z-loss-weight` | keep |  |
| `--ragged-transport` | keep | one_shot / nccl |
| `--collective-overlap-limit` | delete (dead) |  |
| `--single-process` | keep | Ckpt 1 ladder |
| `--checkpoint-keep` | keep |  |
| `--stop-after-steps` | keep |  |
| `--nodes` | keep | multi-node H100 |
| `--expert-axis-size` | keep | multi-node H100 |
| `--grad-accum` | keep |  |
| `--model-set` | keep |  |
| `--opt-set` | keep |  |
| `--priority` | keep |  |
| `--job-env` | keep |  |
| `--submit` | keep |  |
| `--target-cluster` | keep |  |
| `--version/--override/--run/--max-concurrent` | keep | build_options |

## belay_ckpt1.py

| option | decision | note |
|---|---|---|
| `--size d512..d1280` | keep | H100 ladder (LADDER_RUNGS) |
| `--size gb200-d6144` | delete (GB200/MFU) | GB200_ARGS / GB200_MODEL_SETTINGS / GB200_XLA_FLAGS removed |
| `MODEL_SETTINGS moe_drop_renorm=false` | delete (dead) | field deleted |
| `LADDER_ARGS --ragged-transport one_shot` | keep | now the default; kept explicit |

## Totals (config fields)

- delete (GB200/MFU): 21
- delete (dead): 62
- keep: 136
- keep (research): 21

## Outcome (marin_belay_v1_clean)

GB200/MFU machinery was removed from the research branch rather than split into a `PerfConfig`: the MXFP8 sites, the
scanned layer stack (offloaded partials, saved routed outputs, fused source grads, two-phase AttnRes), the
shared-expert overlap placement and value-embed FSDP thread through `_proj`, the MoE dispatch, the KDA projections and
the AttnRes loop, so a split would have left a hook at every site. The launcher/trainer side (autotune caches and
seeding, shared executables, library-kernel preload, first-step timeout, pinned-host master params, offloaded optimizer
state, PGLE, the pipelined loop, the device ragged transport, the GB200 rung and target config) went with it. All of
it stays on `marin_belay_v1_ckpt1` (the accepted Ckpt 1 code, incl. `belay_ckpt1 --size gb200-d6144`) and the
`marin_belay_v1_mfuladder_*` branches.

Deleted modules: `attention_rows_probe.py`, `router_soft_select.py`, `trace_capture.py`, `attn_res_source_grads.py`,
`gating.py`, `sparse_attention.py`, `library_kernels.py`, `gb200_target_config.txtpb`; tests `test_scan_layers.py`,
`test_latent_out_tail.py`, `test_collective_overlap_limit.py`. `key_offset.py` keeps only the partial key offset.
