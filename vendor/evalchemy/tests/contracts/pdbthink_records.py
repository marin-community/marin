"""Invented short records; no original PDBThink prompts or gold answers."""


def record(identity="one", *, family="P01", schema="integer", gold=None, representation="minimal_pdb", rotated=False):
    return {
        "render_id": identity,
        "messages": [{"role": "system", "content": "Return a FINAL answer."}, {"role": "user", "content": identity}],
        "answer_schema": schema,
        "gold_answer": gold if gold is not None else {"value": 2},
        "parameters": {},
        "report_metadata": {
            "render_id": identity, "semantic_instance_id": identity, "question_family": family,
            "protein_group_id": "invented-protein", "cluster": "invented-protein", "representation": representation,
            "is_rotation_variant": rotated, "rotation_seed": 1 if rotated else 0, "state_order_seed": None,
            "answer_schema": schema, "is_mechanistic": False, "input_token_count": 10,
        },
    }
