# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.testing.cpu_devices import run_on_cpu_devices


def test_runtime_flag_survives_hf_reload_and_loader_on_32_cpu_mesh():
    run_on_cpu_devices(
        """
        from dataclasses import replace
        from types import SimpleNamespace
        import haliax as hax
        import jax
        import jax.numpy as jnp
        from levanter.adaptor import FrozenParameterAdaptorConfig
        from levanter.compat.hf_checkpoints import HFCheckpointConverter
        from levanter.compute_dtype import cast_model_for_compute
        from levanter.main.train_lm import preserve_hf_runtime_settings, _load_lm_model_from_configured_source
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel, GrugMoeHfConfig
        from experiments.interp.snowball_identity.levanter_positions import router_dot_dtypes, router_dot_is_fp32
        from experiments.post_training.identity_context_distill.sft import identity_sft_config
        from experiments.post_training.identity_context_distill.test_sft import source_arguments

        base = identity_sft_config(hosts=1, dataset_uri='synthetic', cache_uri='cache', output_uri='output',
                                   run_id='reload', seed=310, **source_arguments())
        enabled = replace(base, model=replace(base.model, router_dot_fp32=True),
                          trainer=replace(base.trainer, compute_dtype_overrides={'router_bias': 'float32'}))
        tc = replace(enabled.trainer, tracker=(), require_accelerator=False)
        tiny = SnowballConfig(vocab_size=64, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8,
                             intermediate_dim=16, shared_expert_intermediate_dim=16,
                             num_experts=8, num_experts_per_token=4, num_layers=1, max_seq_len=8192, qk_mult=1.75)
        requested = replace(tiny, router_dot_fp32=True)
        converter = HFCheckpointConverter(SnowballConfig, HfConfigClass=GrugMoeHfConfig)
        hf = requested.to_hf_config(64)
        assert 'router_dot_fp32' not in hf.to_dict()
        reloaded = converter.config_from_hf_config(hf)
        assert reloaded.router_dot_fp32 is False
        restored = preserve_hf_runtime_settings(reloaded, requested, tc.compute_dtype_overrides)
        assert restored.router_dot_fp32 is True
        assert restored.qk_mult == reloaded.qk_mult == 1.75
        cfg = replace(enabled, model=restored, trainer=tc,
                      adapter=FrozenParameterAdaptorConfig(frozen_field_names=('router_bias',)))
        seen = []
        def fixture_load(model_type, **kwargs):
            # Tiny synthetic parameters only. The real loader must receive the POST-reload config explicitly.
            assert kwargs['config'].router_dot_fp32 is True
            assert kwargs['dtype'] is None
            seen.append(kwargs['config'])
            return SnowballLMHeadModel.init(hax.Axis('vocab', 64), kwargs['config'], key=jax.random.key(1))
        with jax.set_mesh(tc.device_mesh), hax.axis_mapping(tc.parameter_axis_mapping):
            model = _load_lm_model_from_configured_source(
                config=cfg, converter=SimpleNamespace(load_pretrained=fixture_load), Vocab=hax.Axis('vocab', 64),
                model_key=jax.random.key(0), adapter_key=jax.random.key(2),
                parameter_axis_mapping=tc.parameter_axis_mapping, trainer=SimpleNamespace(mp=tc.mp))
            assert seen and model.config.router_dot_fp32 is True
            view = cast_model_for_compute(model, tc.mp, tc.compute_dtype_overrides)
            witness = router_dot_dtypes(view, tc.mp.compute_dtype)
            assert router_dot_is_fp32(witness), witness
            assert all(row == {'lhs': 'float32', 'rhs': 'float32', 'out': 'float32'} for row in witness)
            assert view.transformer.blocks[0].mlp.router.dtype == jnp.bfloat16
            assert view.transformer.blocks[0].mlp.router_bias.dtype == jnp.float32
        """,
        device_count=32,
    )
