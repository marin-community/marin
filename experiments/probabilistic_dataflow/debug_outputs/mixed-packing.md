# Mixed-domain packing debug rendering

Advection and contact documents share dense rows. Segment IDs keep full attention inside each document. Record order affects packing only; scientific position IDs travel with records and loss weights select target-query records.

## Packed heterogeneous batch

| Property | Value |
| --- | --- |
| shape | 1 rows x 48 tokens |
| documents | 2 |
| supervised tokens | 18 |
| rotary positions | all 0; RoPE is the identity |
| attention | full_segment within each segment |
| attention boundary | segment_id; records cannot attend across documents |
| padding | segment_id=-1 and loss_weight=0 |

| Row | Document spans | First eight physical records |
| --- | --- | --- |
| 0 | seg=0 0:28 advection-0/call0/doc0 losses=12<br>seg=1 28:38 contacts-0/call0/doc0 losses=6 | synthetic_advection.initial[cell=(0.0,)] <= value:13, synthetic_advection.initial[cell=(0.25,)] <= value:10, synthetic_advection.initial[cell=(0.5,)] <= value:8, synthetic_advection.initial[cell=(0.75,)] <= value:4, synthetic_advection.forcing[time=0,cell=(0.0,)] <= value:1, synthetic_advection.forcing[time=0,cell=(0.25,)] <= value:0, synthetic_advection.forcing[time=0,cell=(0.5,)] <= value:0, synthetic_advection.forcing[time=0,cell=(0.75,)] <= value:0 |
