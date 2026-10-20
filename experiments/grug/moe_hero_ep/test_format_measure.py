# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import textwrap
from unittest.mock import patch

import numpy as np
from levanter.data.passthrough_tokenizer import PassthroughTokenizer
from levanter.data.text.formats import PrebuiltLmDatasetFormat, preprocessor_for_format
from levanter.store.cache import write_levanter_cache
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.moe_hero_ep.format_measure import PtMaterializeRequest, materialize_pt_panel
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest


def test_pt_materializer_selects_existing_validation_cache_without_loading_training_sources(tmp_path):
    tokenizer = PassthroughTokenizer(32)
    data_format = PrebuiltLmDatasetFormat(loss_weights_key="loss_weights")
    cache_path = tmp_path / "heldout-cache"
    records = [
        {"input_ids": np.array([2, 3, 4, 5], np.int32), "loss_weights": np.array([1, 0.5, 2, 1], np.float32)},
        {"input_ids": np.array([6, 7, 8, 9], np.int32), "loss_weights": np.array([0.5, 1, 0.25, 1], np.float32)},
    ]
    write_levanter_cache(
        records, str(cache_path / "validation"), metadata=preprocessor_for_format(data_format, tokenizer).metadata
    )
    encoded = {
        "tokenizer": "pinned-tokenizer",
        "cache_dir": None,
        "train_weights": {"train": 1.0},
        "components": {
            "train": {"cache_dir": str(tmp_path / "absent-training-cache")},
            "pt": {
                "cache_dir": str(cache_path),
                "tags": ["heldout"],
                "format": {"type": "prebuilt", "loss_weights_key": "loss_weights"},
            },
        },
    }
    request = PtMaterializeRequest(
        data=encoded,
        data_digest=digest(encoded),
        component="pt",
        tag="heldout",
        row_indices=(1, 0),
        sequence_length=4,
        tokenizer="pinned-tokenizer",
        tokenizer_revision="a" * 40,
    )
    # Only tokenizer acquisition is replaced; cache loading, tag selection, named
    # LM conversion and causal target weighting run through the actual libraries.
    with patch("levanter.data.text.datasets.load_marin_tokenizer", return_value=tokenizer):
        panel = asyncio.run(materialize_pt_panel(request))
    assert [row.tokens for row in panel.rows] == [(6, 7, 8, 9), (2, 3, 4, 5)]
    assert [row.loss_weight for row in panel.rows] == [(0.5, 1.0, 0.25, 0.0), (1.0, 0.5, 2.0, 0.0)]
    assert panel.tags == ("heldout", "pt")
    assert not (tmp_path / "absent-training-cache").exists()


def test_pt_panel_materialization_loss_routing_and_padding_match_direct_model(tmp_path):
    script = r"""
        import asyncio
        import hashlib
        from pathlib import Path
        import draccus
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from haliax import Axis
        from jax.sharding import AxisType, Mesh
        from levanter.data.dataset import ListAsyncDataset
        from levanter.data.text.datasets import NamedLmDataset
        from levanter.data.text.examples import GrugLmExample
        from levanter.grug.attention import AttentionMask
        from experiments.grug.moe_hero_ep.format_measure import (
            PtMaterializeRequest, NativePtRequest, PtRow, panel_from_dataset,
            measure_pt_model, measure_native, read_pt_panel, token_bytes_digest,
        )
        from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
        from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest, Checkpoint
        from experiments.grug.moe_hero_ep.weights import cast_weights

        data = {'tokenizer': 'pinned-tokenizer', 'heldout-cache': 'immutable-cache-ledger'}
        materialize = PtMaterializeRequest(data=data, data_digest=digest(data), component='pt',
            tag='heldout', row_indices=(2,0,1), sequence_length=8, tokenizer='pinned-tokenizer',
            tokenizer_revision='a'*40)
        segments = jnp.array([0,0,0,1,1,1,-1,-1], jnp.int32)
        examples = [GrugLmExample.causal(jnp.array(ids,jnp.int32),
            loss_weight=jnp.array([1,.5,0,1,2,0,0,0],jnp.float32), segment_ids=segments)
            for ids in ([2,3,4,5,6,7,0,0],[7,6,5,4,3,2,23,19],[3,4,5,6,7,8,11,17])]
        dataset = NamedLmDataset(ListAsyncDataset(examples), Axis('position',8))
        panel = asyncio.run(panel_from_dataset(dataset,materialize,('heldout','pt')))
        assert [row.row_index for row in panel.rows] == [2,0,1]
        assert panel.rows[0].loss_weight == (1.,.5,0.,1.,2.,0.,0.,0.)
        panel_path = Path(ROOT)/'panel.json'
        panel_path.write_text(panel.model_dump_json(indent=2)+'\n')
        checkpoint = Checkpoint(uri=str(Path(ROOT)/'deliberately-absent-weights'), run_id='tiny',
                                step=0,timestamp='2026-10-20T00:00:00Z',metadata_digest='unread')
        cfg = GrugModelConfig(vocab_size=32, hidden_dim=16, intermediate_dim=8,
            shared_expert_intermediate_dim=8,num_experts=4,num_experts_per_token=2,
            num_layers=2,num_heads=2,num_kv_heads=1,head_dim=8,max_seq_len=8,
            sliding_window=4,global_every=2,attention_implementation='reference',
            moe_implementation='scatter',expert_chunks=1)
        request = NativePtRequest(panel_path=str(panel_path),
            panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(), checkpoint=checkpoint,
            model=draccus.encode(cfg),tokenizer=panel.tokenizer,tokenizer_revision=panel.tokenizer_revision,
            source_revision='b'*40,expert_axis_size=1,replica_axis_size=1)
        assert read_pt_panel(request)==panel
        names=('replica_dcn','data','context','expert','model')
        for devices in (1,2):
            mesh=Mesh(np.asarray(jax.devices()[:devices]).reshape(1,devices,1,1,1),names,
                      axis_types=(AxisType.Explicit,)*5)
            with jax.set_mesh(mesh):
                model=cast_weights(Transformer.init(cfg,key=jax.random.key(3)),jnp.bfloat16)
                measured=measure_pt_model(model,panel,mesh)
                @eqx.filter_jit
                def direct(current,tokens,segment):
                    hidden,routing=current(tokens,mask=AttentionMask.causal().with_segment_ids(segment))
                    logits=jnp.einsum('bsh,hv->bsv',hidden.astype(jnp.float32),
                                      current.output_proj.astype(jnp.float32))
                    return logits,routing
                reference=[]
                routing_reference=[]
                for start in range(0,len(panel.rows),2):
                    real=panel.rows[start:start+2]
                    # A partial reference duplicates the real row rather than
                    # adding filler; direct routing counts then divide by two.
                    repeat=2//len(real)
                    batch=real*repeat
                    logits,routing=direct(model,jnp.asarray([row.tokens for row in batch]),
                                          jnp.asarray([row.segment_ids for row in batch]))
                    logits=np.asarray(logits,dtype=np.float64)
                    for index,row in enumerate(real):
                        values=logits[index]
                        labels=np.array(row.tokens[1:]+(0,))
                        maximum=values.max(axis=-1)
                        losses=maximum+np.log(np.exp(values-maximum[:,None]).sum(axis=-1))
                        losses-=values[np.arange(len(labels)),labels]
                        reference.append(float((losses*np.asarray(row.loss_weight)).sum()))
                    routing_reference.append((np.asarray(routing['routing_counts_per_layer'])//repeat,
                                              float(jnp.mean(routing['routing_entropy_per_layer'])),
                                              float(jnp.mean(routing['router_z_loss_per_layer']))))
                for row,expected in zip(measured['rows'],reference,strict=True):
                    np.testing.assert_allclose(row['loss_sum'],expected,rtol=2e-5,atol=2e-5)
                    assert row['loss_weight']==4.5 and row['valid_tokens']==6
                for batch,(counts,entropy,z_loss) in zip(measured['routing_batches'],routing_reference,strict=True):
                    np.testing.assert_array_equal(batch['routing_counts_per_layer'],counts)
                    np.testing.assert_allclose(batch['routing_entropy_mean'],entropy,rtol=1e-6,atol=1e-6)
                    np.testing.assert_allclose(batch['router_z_loss'],z_loss,rtol=1e-6,atol=1e-6)
                assert measured['forward_passes']==2 and measured['batch_size']==2
                assert [batch['filler_rows'] for batch in measured['routing_batches']]==[0,1]
                assert measured['valid_tokens']==18 and measured['loss_weight']==13.5
                np.testing.assert_allclose(measured['loss'],sum(reference)/13.5,
                                           rtol=2e-5,atol=2e-5)
                changed=[]
                for row in panel.rows:
                    tokens=row.tokens[:-2]+(31,30)
                    changed.append(PtRow(row_index=row.row_index,tokens=tokens,loss_weight=row.loss_weight,
                        segment_ids=row.segment_ids,
                        arrays_sha256=token_bytes_digest(tokens,row.loss_weight,row.segment_ids)))
                changed_panel=panel.model_copy(update={'rows':tuple(changed)})
                changed_result=measure_pt_model(model,changed_panel,mesh)
                for key in ('loss_sum','loss_weight','valid_tokens','routing_counts_per_layer',
                            'routing_entropy_mean','router_z_loss'):
                    np.testing.assert_array_equal(changed_result[key],measured[key])
        # Replacing bytes must fail before even the absent checkpoint metadata is accessed.
        panel_path.write_text(panel_path.read_text()+' ')
        try:
            measure_native(request,str(Path(ROOT)/'never-created-result'))
        except ValueError as error:
            assert str(error)=='PT panel bytes changed'
        else:
            raise AssertionError('Changed panel accepted before weight I/O')
        assert not (Path(ROOT)/'never-created-result').exists()
    """
    run_on_cpu_devices("ROOT = " + repr(str(tmp_path)) + "\n" + textwrap.dedent(script), device_count=2)
