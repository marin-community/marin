# Shared text and science model-call rendering

Both calls use one token embedding table, Grug transformer stack, and output projection. Text uses physical rotary positions and causal attention. Science uses a scientific-position adapter, zero rotary positions, full attention, and aligned targets. The compiler keeps these incompatible attention layouts in separate dense batches.

## Shared parameter boundary

| Component | Sharing |
| --- | --- |
| Token embedding | one table with vocabulary size 64 |
| Transformer blocks | same parameters for both calls |
| Output projection | same parameters and vocabulary for both calls |
| Scientific-position embedding | added only where scientific_position_id >= 0 |

## Data-dependent call strategies

| Task | Dense shape | Scientific positions | Rotary positions | Attention | Targets |
| --- | --- | --- | --- | --- | --- |
| synthetic_text | 4 x 7 | none | 0..6 | causal_segment | shifted next-token labels |
| synthetic_advection | 1 x 32 | field, time, and mesh coordinates | all 0 | full_segment | aligned scientific-value labels |

### Text call, first document

| Physical position | Input token | Scientific position | Rotary position | Target | Loss |
| --- | --- | --- | --- | --- | --- |
| 0 | <text:bos> | - | 0 | <text:the> | 1 |
| 1 | <text:the> | - | 1 | <text:ocean> | 1 |
| 2 | <text:ocean> | - | 2 | <text:field> | 1 |
| 3 | <text:field> | - | 3 | <text:changes> | 1 |
| 4 | <text:changes> | - | 4 | <text:slowly> | 1 |
| 5 | <text:slowly> | - | 5 | <text:eos> | 1 |
| 6 | <text:eos> | - | 6 | - | 0 |

### Scientific call, first document

| Physical position | Input token | Scientific position | Rotary position | Target | Loss |
| --- | --- | --- | --- | --- | --- |
| 0 | value:13 | synthetic_advection.initial[cell=(0.0,)] | 0 | - | 0 |
| 1 | value:10 | synthetic_advection.initial[cell=(0.25,)] | 0 | - | 0 |
| 2 | value:8 | synthetic_advection.initial[cell=(0.5,)] | 0 | - | 0 |
| 3 | value:4 | synthetic_advection.initial[cell=(0.75,)] | 0 | - | 0 |
| 4 | value:1 | synthetic_advection.forcing[time=0,cell=(0.0,)] | 0 | - | 0 |
| 5 | value:0 | synthetic_advection.forcing[time=0,cell=(0.25,)] | 0 | - | 0 |
| 6 | value:0 | synthetic_advection.forcing[time=0,cell=(0.5,)] | 0 | - | 0 |
| 7 | value:0 | synthetic_advection.forcing[time=0,cell=(0.75,)] | 0 | - | 0 |
| 8 | value:0 | synthetic_advection.forcing[time=1,cell=(0.0,)] | 0 | - | 0 |
| 9 | value:3 | synthetic_advection.forcing[time=1,cell=(0.25,)] | 0 | - | 0 |
