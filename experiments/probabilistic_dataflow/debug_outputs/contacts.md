# Synthetic contacts debug rendering

The set axis has four residues. Its unordered-pair axis identifies the six `{left, right}` pairs. The `contacts_program()` creates one unordered document with four observed residue records and six target-query records.

## 1. Inference Program Values

```mermaid
flowchart LR
  n0["%0 sequence<br/>input<br/>residue_class[residue:set=4] bins=8 tokens=4"]
  n1["%1 contacts<br/>sample<br/>contact[unordered_pair(residue):unordered_pair=6] bins=2 tokens=6"]
  n0 --> n1
```

| ID | Value | Kind | Type | Inputs | Operation/factor | Factor ID | FlowInfo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| %0 | sequence | input | residue_class[residue:set=4] bins=8 tokens=4 | - | - | - | provenance={synthetic.sequence}<br>split_keys={synthetic-train}<br>random_ancestors={} |
| %1 | contacts | sample | contact[unordered_pair(residue):unordered_pair=6] bins=2 tokens=6 | %0 sequence | contact_map | synthetic_contacts:contacts:1 | provenance={synthetic.sequence}<br>split_keys={synthetic-train}<br>random_ancestors={synthetic_contacts:contacts:1} |

## 2. Inference Plan IR

| Property | Value |
| --- | --- |
| program | synthetic_contacts |
| external inputs | %0 sequence |
| outputs | %1 contacts |
| factors | synthetic_contacts:contacts:1 |
| budget | model_calls=8, generated_tokens=64 |

```mermaid
flowchart LR
  c0["call 0<br/>generate<br/>contacts"]
```

| Call | Operator | Iteration | Context | Targets | Depends on | Attention | Positions | Approximation/notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | generate | 0 | %0 sequence | %1 contacts | - | full_segment | scientific | factor synthetic_contacts:contacts:1 is approximated as 6 parallel token marginals |

## 3. Transformer Execution IR

| Call | Operator | Depends on | Documents | Supervised tokens | Attention layout | Position mode |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | generate | - | 1 | 6 | full_segment | scientific |

### Call 0 document inventory

| Document | Predicted semantic values | Records | Loss positions |
| --- | --- | --- | --- |
| 0 | synthetic_contacts.contacts[residue={0,1}], synthetic_contacts.contacts[residue={0,2}], synthetic_contacts.contacts[residue={0,3}], synthetic_contacts.contacts[residue={1,2}], synthetic_contacts.contacts[residue={1,3}], synthetic_contacts.contacts[residue={2,3}] | 10 | 6 |

### Call 0, document 0

- Example: `contacts-0`
- Physical rotary positions: all 0; RoPE is the identity
- Attention: full_segment within this document's segment; no cross-document attention
- Serialization: complete records may be permuted; outputs follow the same permutation
- Loss: 6 aligned target records

| Physical position | Rotary position | Component | Scientific position embedding | Content token | Model treatment | Predicts | Loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | context record | synthetic_contacts.sequence[residue=0] | 35 value:3 | value token + position policy; no direct loss | - | 0 |
| 1 | 0 | context record | synthetic_contacts.sequence[residue=1] | 34 value:2 | value token + position policy; no direct loss | - | 0 |
| 2 | 0 | context record | synthetic_contacts.sequence[residue=2] | 34 value:2 | value token + position policy; no direct loss | - | 0 |
| 3 | 0 | context record | synthetic_contacts.sequence[residue=3] | 33 value:1 | value token + position policy; no direct loss | - | 0 |
| 4 | 0 | target record | synthetic_contacts.contacts[residue={0,1}] | 1 <query> | query token + position policy; target value is a label, not an input | value:0 | 1 |
| 5 | 0 | target record | synthetic_contacts.contacts[residue={0,2}] | 1 <query> | query token + position policy; target value is a label, not an input | value:0 | 1 |
| 6 | 0 | target record | synthetic_contacts.contacts[residue={0,3}] | 1 <query> | query token + position policy; target value is a label, not an input | value:0 | 1 |
| 7 | 0 | target record | synthetic_contacts.contacts[residue={1,2}] | 1 <query> | query token + position policy; target value is a label, not an input | value:1 | 1 |
| 8 | 0 | target record | synthetic_contacts.contacts[residue={1,3}] | 1 <query> | query token + position policy; target value is a label, not an input | value:0 | 1 |
| 9 | 0 | target record | synthetic_contacts.contacts[residue={2,3}] | 1 <query> | query token + position policy; target value is a label, not an input | value:0 | 1 |
