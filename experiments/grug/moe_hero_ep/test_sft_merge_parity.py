# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import textwrap

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices


@pytest.mark.timeout(120)
def test_native_merge_parity_measures_compact_export_and_rejects_wrong_weights_or_ancestry(tmp_path):
    script = r"""
        import dataclasses
        import hashlib
        import json
        from pathlib import Path
        from unittest.mock import patch
        import draccus
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import yaml
        from levanter.checkpoint import save_checkpoint
        from levanter.grug.sharding import compact_grug_mesh
        from experiments.grug.moe_hero_ep.averaging import (
            AverageConfig,CheckpointPin,TokenizerPin,average_checkpoints,_DONORS)
        from experiments.grug.moe_hero_ep.format_measure import PtPanel,PtRow,token_bytes_digest
        from experiments.grug.moe_hero_ep.model import GrugModelConfig,Transformer
        from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
        from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint,digest
        from experiments.grug.moe_hero_ep.sft import (
            BaseReference,SftConfig,initialize_from_checkpoint,make_sft_step,
            save_sft_checkpoint,export_sft)
        from experiments.grug.moe_hero_ep.sft_merge_parity import (
            MergeParityRequest,MergeProbe,run_merge_parity)
        from experiments.grug.moe_hero_ep.weights import restore_weights

        root=Path(ROOT)
        config=GrugModelConfig(vocab_size=128010,hidden_dim=8,intermediate_dim=4,
            shared_expert_intermediate_dim=4,num_experts=2,num_experts_per_token=1,
            num_layers=1,num_heads=2,num_kv_heads=1,head_dim=4,max_seq_len=8,
            sliding_window=4,global_every=2,attention_implementation='reference',
            moe_implementation='scatter',expert_chunks=1)
        mesh=compact_grug_mesh(expert_axis_size=1,replica_axis_size=1)
        with jax.set_mesh(mesh):
            base=Transformer.init(config,key=jax.random.key(3))
            pending=jnp.array([[.0031,-.0047]],jnp.float32)
            source=str(root/'pt')
            save_checkpoint({'master_params':base,'pending_qb_betas':pending},123,source,is_temporary=False)
            metadata=json.loads((Path(source)/'metadata.json').read_text())
            manifest=json.loads((Path(source)/'manifest.json').read_text())
            run={'model':draccus.encode(config),
                 'optimizer':draccus.encode(GrugMoeMuonHConfig(learning_rate=.1,adam_lr=.02)),
                 'trainer':{'trainer':{'num_train_steps':1000}},'data':{'tokenizer':'test-tokenizer'}}
            run=json.loads(json.dumps(run))
            config_path=root/'source.yaml';config_path.write_text(yaml.safe_dump(run))
            pin=CheckpointPin(source,digest(metadata),str(config_path),digest(run),'a'*40,
                              digest(manifest),'local tokenizer fixture')
            class LocalProvider:
                worker_identity={'provider':'local-cpu-fixture'}
                def worker_region(self): return 'local-fixture'
                def storage_region(self,path): return 'local-fixture'
            class LocalTokenizer:
                def encode(self,text,*,add_special_tokens):
                    mapping={text:token for row in _DONORS for text,token in (row[:2],row[2:])}
                    return [mapping[text]]
            candidate_c=str(root/'c')
            preparation=AverageConfig((pin,),candidate_c,str(root/'scratch'),
                TokenizerPin('test-tokenizer','a'*40),(),'b'*40,8,
                max_chunk_bytes=8*1024*1024,initialize_donors=True)
            with patch('experiments.grug.moe_hero_ep.averaging.AutoTokenizer.from_pretrained',
                       return_value=LocalTokenizer()):
                c_metadata=average_checkpoints(preparation,region_probe=LocalProvider())
            reference=BaseReference(candidate_c,digest(c_metadata),source,123,'A','test-tokenizer','a'*40)
            sft=SftConfig(rank=2,alpha=2.,token_ids=tuple(row[1] for row in _DONORS),
                adapter_learning_rate=1e-3,vocab_learning_rate=1e-3,num_steps=10,warmup_steps=0,
                seed=7,vocab_block_size=4096,token_block_size=4)
            frozen,state=initialize_from_checkpoint(reference,config,mesh,sft)
            tokens=jnp.array([[128006,3,128007,5,128009,4,128002,2],
                              [128006,5,128007,3,128003,7,128009,4]],jnp.int32)
            weights=jnp.ones_like(tokens,dtype=jnp.float32).at[:,-1].set(0)
            learned,_=make_sft_step(sft)(state,frozen,tokens,weights,jnp.ones_like(tokens,dtype=bool))
            assert np.count_nonzero(np.asarray(learned.params.output_delta))>0
            assert np.count_nonzero(np.asarray(learned.params.adapters.stacked_blocks.stacked.attn.lora_q.b))>0
            compact=str(root/'compact');ordinary=str(root/'ordinary')
            save_sft_checkpoint(compact,learned,reference,sft)
            export_sft(ordinary,frozen,learned,reference,sft)
            compact_meta=json.loads((Path(compact)/'metadata.json').read_text())
            ordinary_meta=json.loads((Path(ordinary)/'metadata.json').read_text())
            rows=[]
            for index,values in enumerate(np.asarray(tokens)):
                ids=tuple(int(value) for value in values)
                loss_weight=(1.,)*7+(0.,);segments=(0,)*8
                rows.append(PtRow(row_index=index,tokens=ids,loss_weight=loss_weight,segment_ids=segments,
                                  arrays_sha256=token_bytes_digest(ids,loss_weight,segments)))
            data={'tokenizer':'test-tokenizer','cache':'pinned-real-token-row-fixture'}
            panel=PtPanel(source_data_config=data,source_data_config_digest=digest(data),component='pt',
                tags=('pt',),tokenizer='test-tokenizer',tokenizer_revision='a'*40,sequence_length=8,rows=tuple(rows))
            panel_path=root/'panel.json';panel_path.write_text(panel.model_dump_json(indent=2)+'\n')
            checkpoint=Checkpoint(uri=ordinary,run_id='tiny-d',step=1,
                timestamp=ordinary_meta['timestamp'],metadata_digest=digest(ordinary_meta))
            request=MergeParityRequest(reference=reference,sft=sft,model=draccus.encode(config),
                compact_checkpoint=compact,compact_metadata_digest=digest(compact_meta),merged_checkpoint=checkpoint,
                panel_path=str(panel_path),panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                probes=(MergeProbe(row_index=0,prediction_position=6),MergeProbe(row_index=1,prediction_position=6)),
                source_revision='b'*40,expert_axis_size=1,replica_axis_size=1,
                max_absolute_logit_error=.03,max_absolute_probability_error=.0001,max_greedy_disagreements=2)
            result=run_merge_parity(request,str(root/'good-report'))
            assert result['passed']
            retained=json.loads((root/'good-report/report.json').read_text())
            body={key:value for key,value in retained.items() if key!='report_digest'}
            assert retained['report_digest']==digest(body)
            assert retained['request']['compact_metadata_digest']==digest(compact_meta)
            assert retained['request']['merged_checkpoint']['metadata_digest']==digest(ordinary_meta)
            assert len(retained['probes'])==2 and retained['compact_counters']['sft_step']==1
            assert retained['selected_input_token_counts']['128006']==2
            assert retained['selected_input_token_counts']['128002']==1
            assert retained['trained_input_token_ids']==list(sft.token_ids)
            assert retained['missing_trained_input_token_ids']==[]
            assert retained['compared_output_columns']==128010
            assert all(isinstance(row['compact_greedy_id'],int) and isinstance(row['merged_greedy_id'],int)
                       for row in retained['probes'])

            # Matching numbers alone cannot certify input-row merges when real
            # prefixes fail to exercise trained special-token embeddings.
            uncovered=request.model_copy(update={'probes':(
                MergeProbe(row_index=0,prediction_position=3),MergeProbe(row_index=1,prediction_position=5))})
            coverage=run_merge_parity(uncovered,str(root/'missing-coverage-report'))
            assert coverage['numerical_parity_passed'] and not coverage['passed']
            assert coverage['missing_trained_input_token_ids']==[128002,128009]

            # Tokens in an earlier packed document cannot count as exercising
            # the input path for the later document's selected prediction.
            split_segments=(0,)*6+(1,)*2
            split_rows=tuple(PtRow(row_index=row.row_index,tokens=row.tokens,loss_weight=row.loss_weight,
                segment_ids=split_segments,
                arrays_sha256=token_bytes_digest(row.tokens,row.loss_weight,split_segments)) for row in panel.rows)
            split_panel=panel.model_copy(update={'rows':split_rows})
            split_path=root/'split-panel.json';split_path.write_text(split_panel.model_dump_json(indent=2)+'\n')
            split_request=request.model_copy(update={'panel_path':str(split_path),
                'panel_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest()})
            split_result=run_merge_parity(split_request,str(root/'split-coverage-report'))
            assert split_result['numerical_parity_passed'] and not split_result['passed']
            assert split_result['missing_trained_input_token_ids']==[128006,128007,128003]
            assert split_result['selected_input_token_counts']['128006']==0

            # A same-lineage checkpoint with corrupted actual weights must fail
            # numerical measurement, even though all metadata gates pass.
            normal=restore_weights(ordinary,digest(ordinary_meta),config,mesh)
            corrupted=eqx.tree_at(lambda model:model.output_proj,normal,normal.output_proj*128.)
            wrong=str(root/'wrong-weights')
            merge_metadata={key:ordinary_meta[key] for key in
                            ('hero_sft','merge_base','merge_accumulator_dtype','serving_dtype')}
            save_checkpoint({'params':corrupted,'pending_qb_betas':pending},1,wrong,is_temporary=False,
                            metadata=merge_metadata)
            wrong_meta=json.loads((Path(wrong)/'metadata.json').read_text())
            changed=request.model_copy(update={'merged_checkpoint':checkpoint.model_copy(
                update={'uri':wrong,'metadata_digest':digest(wrong_meta)})})
            failure=run_merge_parity(changed,str(root/'bad-report'))
            assert not failure['passed'] and failure['max_absolute_logit_error']>.03
            assert json.loads((root/'bad-report/report.json').read_text())['passed'] is False

            # This checkpoint has only metadata, so accepting its wrong C
            # ancestry would fail later with a missing-array error instead.
            unrelated=json.loads(json.dumps(ordinary_meta))
            unrelated['hero_sft']['base']['metadata_digest']='0'*64
            absent=root/'metadata-only-wrong-lineage';absent.mkdir()
            (absent/'metadata.json').write_text(json.dumps(unrelated))
            changed=request.model_copy(update={'merged_checkpoint':checkpoint.model_copy(
                update={'uri':str(absent),'metadata_digest':digest(unrelated)})})
            try:
                run_merge_parity(changed,str(root/'not-created'))
            except ValueError as error:
                assert 'ancestry' in str(error)
            else:
                raise AssertionError('Wrong C/D ancestry accepted')
            assert not (root/'not-created').exists()
    """
    run_on_cpu_devices("ROOT = " + repr(str(tmp_path)) + "\n" + textwrap.dedent(script), device_count=1)
