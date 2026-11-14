# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser gates shared by local-package and immutable hosted release checks."""

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from playwright.sync_api import Page, expect

from .release_evidence import admission_text, task_totals

LAYOUTS = [(theme, width) for theme in ("light", "dark") for width in (1440, 390)]

TABLE_BOUNDS = """table => {
 const heads=[...table.tHead.rows[0].cells].map(c=>c.getBoundingClientRect()), failures=[];
 for(const row of table.tBodies[0].rows)for(const [i,cell] of [...row.cells].entries()){
  const b=cell.getBoundingClientRect(),next=row.cells[i+1]?.getBoundingClientRect();
  if((next&&b.right>next.left+1)||Math.abs(b.left-heads[i].left)>1||Math.abs(b.right-heads[i].right)>1)
   failures.push([row.dataset.sourceId,i,'cell/header alignment']);
  for(const chip of cell.querySelectorAll('.chip,.difficulty-model')){
   const c=chip.getBoundingClientRect();
   if(c.left<b.left||c.right>b.right||chip.scrollWidth>chip.clientWidth||getComputedStyle(chip).whiteSpace!=='nowrap')
    failures.push([row.dataset.sourceId,i,'chip or difficulty bounds']);
  }
 }
 return failures;
}"""

MAP_PARTITIONS = """map => {
 const n=e=>Number(e.textContent.replaceAll(',','')), failures=[];let parents=0;
 const partition=(key,total,children)=>{
  const sum=[...children.children].reduce((s,c)=>s+n(c.matches('.map-remainder')?
   c.querySelector('.map-area-count'):c.querySelector(':scope>.map-area>.map-area-count')),0);
  parents++;if(n(total)!==sum)failures.push({key,total:n(total),children:sum});
 };
 for(const card of map.querySelectorAll('.domain-card'))partition(card.dataset.subject,
  card.querySelector('.domain-count'),card.querySelector('.map-topics'));
 for(const branch of map.querySelectorAll('.map-branch')){
  const children=branch.querySelector(':scope>.map-children');
  if(children)partition(branch.dataset.nodeId,branch.querySelector(':scope>.map-area>.map-area-count'),children);
 }
 for(const row of map.querySelectorAll('.map-area')){
  const name=row.querySelector('.map-area-name').getBoundingClientRect();
  const count=row.querySelector('.map-area-count').getBoundingClientRect();
  if(name.right>count.left+1)failures.push({key:row.textContent,overlap:true});
 }
 return {parents,failures,cards:map.querySelectorAll('.domain-card').length};
}"""


def integers(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", text.replace(",", ""))]


def verify_rendered_partition(total: int, admitted: int, shown: int, shown_admitted: int, omitted: list[int]) -> None:
    """Reject missing, overlapping or double-counted population/admission partitions."""
    if len(omitted) % 2 or shown + sum(omitted[::2]) != total:
        raise ValueError("Shown + hidden + filtered + folded tasks do not equal the catalog total")
    if shown_admitted + sum(omitted[1::2]) != admitted:
        raise ValueError("Shown + hidden + filtered + folded admissions do not equal total admission")


def check_view(
    page: Page,
    sources: list[dict[str, Any]],
    *,
    include_hidden: bool = False,
    use: str = "all",
    expanded: bool = False,
) -> dict[str, Any]:
    """Check every rendered row and both independent task partitions."""
    rows = {row["id"]: row for row in sources}
    hidden = [row for row in sources if not include_hidden and (row["quality"] == "QUICK" or row["result"] == "BAD")]
    hidden_ids = {row["id"] for row in hidden}
    eligible = [row for row in sources if row["id"] not in hidden_ids and (use == "all" or row["intended_use"] == use)]
    eligible_ids = {row["id"] for row in eligible}
    groups = {row["duplicate_group"]["group_id"] for row in eligible if row.get("duplicate_group")}
    count = len(eligible) if expanded else len(groups) + sum(not row.get("duplicate_group") for row in eligible)
    expect(page.locator("#rows tr")).to_have_count(count)
    rendered = page.locator("#rows tr").evaluate_all(
        "rows=>rows.map(r=>({id:r.dataset.sourceId,examples:r.cells[1].textContent}))"
    )
    displayed_ids = [row["id"] for row in rendered]
    if len(set(displayed_ids)) != len(displayed_ids) or not set(displayed_ids) <= eligible_ids:
        raise ValueError("Displayed rows are duplicated or violate active filters")
    if expanded and set(displayed_ids) != eligible_ids:
        raise ValueError("Show everything omitted eligible sources")
    if not expanded:
        displayed_groups = [
            rows[key]["duplicate_group"]["group_id"] for key in displayed_ids if rows[key].get("duplicate_group")
        ]
        if set(displayed_groups) != groups or len(displayed_groups) != len(groups):
            raise ValueError("Collapsed rows do not represent every eligible duplicate group exactly once")
    for row in rendered:
        counts = rows[row["id"]]["filtration"]
        expected = (
            f"{counts['example_count']:,}"
            if counts["example_count"] is not None
            else "Pending oracle filtration" if counts["state"] == "pending_oracle" else "Pending"
        )
        if row["examples"] != expected:
            raise ValueError(f"{row['id']}: rendered Examples differs from current-pin count evidence")
    displayed = [rows[key] for key in displayed_ids]
    total, shown = task_totals(sources), task_totals(displayed)
    expect(page.locator("#metric-sources")).to_have_text(f"{total.population:,} / {shown.population:,}")
    expect(page.locator("#metric-examples")).to_have_text(admission_text(total))
    expect(page.locator("#metric-examples-shown")).to_have_text("Shown " + admission_text(shown))
    for selector, items, prefix in [("#metric-quality", sources, ""), ("#metric-quality-shown", displayed, "Shown ")]:
        quality = [
            task_totals([row for row in items if row["quality"] == level]).population
            for level in ("QUICK", "SAMPLE", "FULL")
        ]
        expect(page.locator(selector)).to_have_text(prefix + " / ".join(f"{value:,}" for value in quality))
    filtered = [row for row in sources if row["id"] not in hidden_ids | eligible_ids]
    folded = [row for row in eligible if row["id"] not in displayed_ids]
    expected_omitted = [
        value
        for items in (hidden, filtered, folded)
        if items
        for value in (task_totals(items).population, task_totals(items).admitted)
    ]
    omitted = integers(page.locator("#task-reconciliation").inner_text())
    if omitted != expected_omitted:
        raise ValueError("Reconciliation note does not account for the actual hidden/filter/fold populations")
    verify_rendered_partition(total.population, total.admitted, shown.population, shown.admitted, omitted)
    if not page.locator(".metrics > div").evaluate_all(
        "cards=>cards.every(c=>[...c.children].every(x=>{"
        "const a=c.getBoundingClientRect(),b=x.getBoundingClientRect();"
        "return b.left>=a.left&&b.right<=a.right&&x.scrollWidth<=x.clientWidth}))"
    ):
        raise ValueError("Header content overflows a card")
    expect(page.get_by_role("columnheader", name=re.compile(r"^Oracle$", re.I))).to_have_count(0)
    for right in (False, True):
        page.locator(".table-scroller").evaluate("(e,r)=>e.scrollLeft=r?e.scrollWidth:0", right)
        failures = page.locator(".source-table").evaluate(TABLE_BOUNDS)
        if failures:
            raise ValueError(f"Table overlap: {failures[:10]}")
        if not page.evaluate("document.documentElement.scrollWidth<=innerWidth"):
            raise ValueError("The page overflows horizontally")
    if not page.locator("#rows td:has(>.chip)").evaluate_all(
        "cells=>cells.every(c=>c.children.length===1&&c.textContent===c.firstElementChild.textContent)"
    ):
        raise ValueError("Enum chips have subtitle text")
    page.locator(".table-scroller").evaluate("e=>e.scrollLeft=0")
    return {"total": asdict(total), "shown": asdict(shown), "omitted": omitted, "displayed_ids": displayed_ids}


def source_sweep(page: Page, root: str, sources: list[dict[str, Any]], shots: Path) -> list[dict[str, Any]]:
    """Exercise defaults, filters, Show everything and persistence at all layouts."""
    results = []
    for theme, width in LAYOUTS:
        page.emulate_media(color_scheme=theme)
        page.set_viewport_size({"width": width, "height": 1000})
        page.goto(root)
        expect(page.locator("#difficulty-model")).to_have_value("all")
        default = check_view(page, sources)
        headers = page.locator(".metrics > div").evaluate_all(
            "cards=>cards.map(c=>({label:c.querySelector('.metric-label').textContent,value:c.querySelector('strong').textContent,shown:c.querySelector('.metric-detail')?.textContent||null}))"
        )
        page.screenshot(path=str(shots / f"default-{theme}-{width}.png"))
        page.locator("#use").select_option("train")
        check_view(page, sources, use="train")
        page.locator("#show-everything").click()
        expect(page.locator("#use")).to_have_value("all")
        expect(page.locator("#difficulty-model")).to_have_value("all")
        everything = check_view(page, sources, include_hidden=True, expanded=True)
        page.reload()
        check_view(page, sources, include_hidden=True, expanded=True)
        page.locator(".table-scroller").evaluate("e=>e.scrollLeft=0")
        page.screenshot(path=str(shots / f"everything-{theme}-{width}.png"))
        results.append(
            {"theme": theme, "width": width, "headers": headers, "default": default, "everything": everything}
        )
    return results


def check_map(page: Page) -> dict[str, Any]:
    expect(page.locator("#provenance")).to_contain_text('"release_id"')
    expect(page.locator("#curriculum-map")).not_to_have_attribute("aria-busy", "true")
    result = page.locator("#curriculum-map").evaluate(MAP_PARTITIONS)
    if result["failures"] or not page.evaluate("document.documentElement.scrollWidth<=innerWidth"):
        raise ValueError(f"Curriculum overlap or parent/child mismatch: {result}")
    return result


def curriculum_sweep(page: Page, root: str, releases: list[dict[str, Any]], shots: Path) -> list[dict[str, Any]]:
    """Check available releases, uses and every live domain without fixed populations."""
    results = []
    for theme, width in LAYOUTS:
        page.emulate_media(color_scheme=theme)
        page.set_viewport_size({"width": width, "height": 1000})
        for release in releases:
            for use in ("all", "train", "eval"):
                page.goto(root + "curriculum.html#" + urlencode({"release": release["release_id"], "intended_use": use}))
                result = check_map(page)
                results.append({"theme": theme, "width": width, "release": release["release_id"], "use": use, **result})
        page.goto(root + "curriculum.html")
        check_map(page)
        domains = page.locator("#domain option").evaluate_all("opts=>opts.map(o=>o.value).filter(Boolean)")
        for domain in domains:
            page.locator("#domain").select_option(domain)
            expect(page.locator(".domain-card")).to_have_count(1)
            expect(page.locator(".domain-card")).to_have_attribute("data-subject", domain)
            results.append({"theme": theme, "width": width, "domain": domain, **check_map(page)})
            page.locator("#search").fill("__release_check_unmatched_search__")
            check_map(page)
            page.locator("#search").fill("")
        page.goto(root + "curriculum.html")
        check_map(page)
        page.screenshot(path=str(shots / f"curriculum-{theme}-{width}.png"), full_page=True)
    return results
