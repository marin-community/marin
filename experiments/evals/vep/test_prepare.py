# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Selection invariants for complete groups and disjoint autosomal panels."""

from experiments.evals.vep.prepare import eligible_groups, select_panels


def test_autosomal_panels_preserve_groups_and_exclude_overlap():
    rows = []
    for chrom in ("1", "3", "X", "Y", "MT"):
        for group in range(35):
            for index in range(10):
                rows.append(
                    {
                        "chrom": chrom,
                        "pos": group * 10000 + index * 300,
                        "ref": "A",
                        "alt": "G",
                        "label": index == 0,
                        "subset": "missense_variant",
                        "match_group": f"{chrom}-{group}",
                    }
                )
    groups = eligible_groups(rows)
    selected = select_panels(groups, 11181, 20)
    assert {row["chrom"] for rows in selected.values() for row in rows} == {"1", "3"}
    assert {row["chrom"] for row in selected["demonstrations"]} == {"1"}
    assert sorted(row["label"] for row in selected["demonstrations"]) == [False, True]
    pilot = selected["pilot"]
    validation = selected["validation"]
    assert len(pilot) == 100 and sum(row["label"] for row in pilot) == 10
    assert len(validation) == 200 and sum(row["label"] for row in validation) == 20
    assert not {row["match_group"] for row in pilot} & {row["match_group"] for row in validation}
    assert all(abs(a["pos"] - b["pos"]) > 254 for a in pilot for b in validation)
    assert selected == select_panels(eligible_groups(list(reversed(rows))), 11181, 20)
