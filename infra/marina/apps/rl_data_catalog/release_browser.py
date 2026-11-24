# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser gates shared by local-package and immutable hosted release checks."""

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from playwright.sync_api import Error, Page, Request, Response, expect

from .release_curriculum import count_projection, representative_domain
from .release_evidence import admission_text, task_totals

LOGGER = logging.getLogger(__name__)
BODY_PREFIX_BYTES = 256

LAYOUTS = tuple((theme, width) for theme in ("light", "dark") for width in (1440, 390))

TABLE_BOUNDS = """table => {
 const heads=[...table.tHead.rows[0].cells].map(c=>c.getBoundingClientRect()), failures=[];
 for(const label of table.querySelectorAll('thead button,thead .header-label,tbody .use-cell')){
  const range=document.createRange();range.selectNodeContents(label);
  const tops=[...range.getClientRects()].filter(r=>r.width&&r.height).map(r=>r.top);
  if(tops.length&&Math.max(...tops)-Math.min(...tops)>parseFloat(getComputedStyle(label).lineHeight)/2)
   failures.push([label.closest('thead')?'header':'use',label.textContent,'label text wraps']);
 }
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

    def hidden_status(row: dict[str, Any]) -> bool:
        return row.get("status") in {"Excluded", "Not selectable"}

    def pending_status(row: dict[str, Any]) -> bool:
        return not hidden_status(row) and row.get("status") == "Pending"

    hidden = [
        row
        for row in sources
        if not include_hidden
        and (row["quality"] == "QUICK" or row["result"] == "BAD" or hidden_status(row) or pending_status(row))
    ]
    hidden_excluded = [row for row in hidden if hidden_status(row)]
    hidden_pending = [row for row in hidden if pending_status(row)]
    hidden_quality = [row for row in hidden if not hidden_status(row) and not pending_status(row)]
    hidden_ids = {row["id"] for row in hidden}
    eligible = [row for row in sources if row["id"] not in hidden_ids and (use == "all" or row["intended_use"] == use)]
    eligible_ids = {row["id"] for row in eligible}
    groups = {
        row["duplicate_group"]["group_id"]: row["duplicate_group"] for row in eligible if row.get("duplicate_group")
    }
    foldable = {key for key, group in groups.items() if group["canonical_source_id"] in eligible_ids}
    standalone = {
        row["id"]
        for row in eligible
        if not row.get("duplicate_group") or row["duplicate_group"]["group_id"] not in foldable
    }
    expected_ids = eligible_ids if expanded else standalone | {groups[key]["canonical_source_id"] for key in foldable}
    expect(page.locator("#rows tr")).to_have_count(len(expected_ids))
    rendered = page.locator("#rows tr").evaluate_all(
        "rows=>rows.map(r=>({id:r.dataset.sourceId,examples:r.cells[1].textContent,grad"
        "er:r.querySelector('.grader-cell .chip')?.textContent}))"
    )
    displayed_ids = [row["id"] for row in rendered]
    if not page.locator("#rows tr td:first-child").evaluate_all(
        """cells=>cells.every(cell=>
      cell.querySelectorAll(':scope > .source-name').length===1 &&
      [...cell.children].every(child=>child.matches('button.source-name,button.duplicate-toggle')) &&
      [...cell.childNodes].filter(child=>child.nodeType===Node.TEXT_NODE).every(child=>!child.textContent.trim()))"""
    ):
        raise ValueError(
            "Source cells must contain only the source name and fold toggle; secondary content belongs in Evidence"
        )
    if len(set(displayed_ids)) != len(displayed_ids) or set(displayed_ids) != expected_ids:
        raise ValueError("Displayed sources violate filters or fold under an absent/non-canonical representative")
    for row in rendered:
        grader_labels = {
            "llm-judge": "LLM judge",
            "exact-match": "Exact match",
            "math-verify": "Math verify",
            "unit-tests": "Unit tests",
            "script": "Script",
            "reward-model": "Reward model",
            "none": "None",
            "mixed": "Mixed",
        }
        if row["grader"] != grader_labels.get(rows[row["id"]].get("grader_type"), "Pending"):
            raise ValueError(f"Grader label differs from the bound census: {row['id']}")
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
        for items in (hidden_excluded, hidden_pending, hidden_quality, filtered, folded)
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
            "cards=>cards.map(c=>({label:c.querySelector('.metric-label').textContent,value"
            ":c.querySelector('strong').textContent,shown:c.querySelector('.metric-detail')"
            "?.textContent||null}))"
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
        page.get_by_role("separator", name="Resize Source column", exact=True).press("ArrowRight")
        check_view(page, sources, include_hidden=True, expanded=True)
        page.locator("#reset-column-widths").click()
        page.locator(".table-scroller").evaluate("e=>e.scrollLeft=0")
        page.screenshot(path=str(shots / f"everything-{theme}-{width}.png"))
        results.append(
            {"theme": theme, "width": width, "headers": headers, "default": default, "everything": everything}
        )
    return results


EVIDENCE_BOUNDS = """() => {
 const failures=[];
 for(const field of document.querySelectorAll('.field')){
  const label=field.querySelector('dt').getBoundingClientRect(),value=field.querySelector('dd').getBoundingClientRect();
  if(label.right>value.left+1)failures.push({field:field.textContent,overlap:true});
 }
 for(const table of document.querySelectorAll('table'))for(const row of table.tBodies[0]?.rows||[]){
  const cells=[...row.cells];
  for(let i=0;i<cells.length-1;i++)if(cells[i].getBoundingClientRect().right>cells[i+1].getBoundingClientRect().left+1)
   failures.push({table:table.textContent,overlap:true});
 }
 if(document.documentElement.scrollWidth>innerWidth)failures.push({pageOverflow:true});
 return failures;
}"""


def check_evidence(page: Page) -> None:
    failures = page.evaluate(EVIDENCE_BOUNDS)
    if failures:
        raise ValueError(f"Evidence overlap at {page.url}: {failures}")


def evidence_cases(sources: list[dict[str, Any]], pending_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Select each observed admission, quality and execution presentation."""
    cases = {}
    for row in sources:
        key = (
            row["quality"],
            row["result"],
            row["result_provisional"],
            row["filtration"]["state"],
            row["daytona"]["state"],
            bool(row["daytona"].get("exempt")),
            row["intended_use"],
            tuple(sorted(row.get("method_tags", []))),
            (row["backend_capabilities"] or {}).get("basis"),
            bool(row.get("quality_result_waiver")),
            bool(row.get("quality_review_status")),
            tuple(model["status"] for model in row["difficulty"]["models"]),
        )
        cases.setdefault(key, row)
    selected = {row["id"]: row for row in cases.values()}
    selected.update({row["id"]: row for row in sources if row["id"] in (pending_ids or set())})
    return list(selected.values())


def report_failed_response(response: Response) -> None:
    """Retain the request and bounded error body without changing the gate result."""
    if response.status < 400:
        return
    try:
        prefix = response.body()[:BODY_PREFIX_BYTES].decode("utf-8", errors="replace")
    except Error as error:
        prefix = f"unavailable ({error})"
    LOGGER.error("Audit HTTP %s %s; body prefix %r", response.status, response.url, prefix)


def report_failed_request(request: Request) -> None:
    """Identify requests that failed before an HTTP response was available."""
    LOGGER.error("Audit HTTP unavailable %s; body prefix unavailable; %s", request.url, request.failure)


def browser_json(page: Page, url: str) -> dict[str, Any]:
    """Fetch audit JSON with request/status/body context on HTTP or parse failure."""
    return page.evaluate(
        """async ({path, prefixBytes}) => {
          const url = new URL(path, location.href).href;
          let status = 'unavailable', prefix = 'unavailable';
          try {
            const response = await fetch(url);
            status = response.status;
            const bytes = new Uint8Array(await response.arrayBuffer());
            prefix = new TextDecoder().decode(bytes.slice(0, prefixBytes));
            if (!response.ok) throw new Error('HTTP request failed');
            return JSON.parse(new TextDecoder().decode(bytes));
          } catch (error) {
            throw new Error(`Audit JSON ${url}: HTTP ${status}; ` +
                `body prefix ${JSON.stringify(prefix)}; ${error.message}`);
          }
        }""",
        {"path": url, "prefixBytes": BODY_PREFIX_BYTES},
    )


def pending_release_ids(page: Page, root: str) -> set[str]:
    """Read candidate presentations from this package without changing catalog inputs."""
    manifest = browser_json(page, root + "measurements/pending-filtered-releases.json")
    return {row["source_id"] for row in manifest["sources"]}


def evidence_sweep(page: Page, root: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exercise source/backend evidence for every observed admission/QA/actor state."""
    results = []
    candidates = evidence_cases(sources, pending_release_ids(page, root))
    for theme, width in LAYOUTS:
        page.emulate_media(color_scheme=theme)
        page.set_viewport_size({"width": width, "height": 1000})
        for row in candidates:
            for view in ("source", "backends"):
                page.goto(root + view + ".html#" + urlencode({"source_id": row["id"]}))
                expect(page.locator("#title")).to_have_text(row["display_name"], timeout=15000)
                expect(
                    page.locator("#sections .assessment-card")
                    if view == "source"
                    else page.locator("#daytona-evidence .field")
                ).not_to_have_count(0)
                if view == "source":
                    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false", timeout=15000)
                    expect(page.locator("#sections")).not_to_contain_text("Loading supplementary", timeout=10000)
                    expect(page.locator("#sections")).not_to_contain_text("Loading overlap", timeout=10000)
                check_evidence(page)
                if view == "backends":
                    safety = row["daytona"]
                    verdict = (
                        "—"
                        if safety.get("exempt")
                        else {"na": "N/A", "unmeasured": "Pending census", "safe": "SAFE", "unsafe": "UNSAFE"}[
                            safety["state"]
                        ]
                    )
                    fields = page.locator("#daytona-evidence .field")
                    expect(
                        fields.filter(has=page.get_by_text("Actor verdict", exact=True)).locator("dd")
                    ).to_contain_text(verdict)
                    expect(page.get_by_role("link", name="How to make it safe: Daytona operations ↗")).to_have_count(1)
                results.append({"source_id": row["id"], "view": view, "theme": theme, "width": width})
    return results


def check_map(page: Page) -> dict[str, Any]:
    expect(page.locator("#provenance")).to_contain_text('"release_id"')
    expect(page.locator("#curriculum-map")).not_to_have_attribute("aria-busy", "true")
    result = page.locator("#curriculum-map").evaluate(MAP_PARTITIONS)
    if result["failures"] or not page.evaluate("document.documentElement.scrollWidth<=innerWidth"):
        raise ValueError(f"Curriculum overlap or parent/child mismatch: {result}")
    return result


def curriculum_full_sweep(page: Page, root: str, releases: list[dict[str, Any]], shots: Path) -> list[dict[str, Any]]:
    """Check available releases, uses and every live domain without fixed populations."""
    results = []
    for theme, width in LAYOUTS:
        page.emulate_media(color_scheme=theme)
        page.set_viewport_size({"width": width, "height": 1000})
        for release in releases:
            for use in ("all", "train", "eval"):
                page.goto("about:blank")
                page.goto(root + "curriculum.html#" + urlencode({"release": release["release_id"], "intended_use": use}))
                result = check_map(page)
                confidence = browser_json(
                    page,
                    "api/curriculum?"
                    + urlencode(
                        {
                            "release": page.locator("#release").input_value(),
                            "intended_use": page.locator("#use").input_value(),
                        }
                    ),
                )
                levels = confidence.get("confidence", {}).get("levels", []) if confidence.get("confidence") else []
                for index, level in enumerate(levels):
                    page.locator("#confidence").select_option(str(index))
                    vectors = {
                        row["metric"]: row["instances"][index]
                        for row in confidence["confidence"]["statistics"]
                        if not row["subject_id"] and not row["node_id"]
                    }
                    assert (
                        sum(
                            vectors.get(metric, 0)
                            for metric in ("assigned", "subject_only", "out_of_scope", "insufficient_context", "skipped")
                        )
                        == vectors["denominator"]
                    )
                    expect(page.locator("#task-count")).to_have_text(f"{vectors['denominator']:,}")
                    expect(page.locator("#capability-count")).to_have_text(f"{vectors.get('assigned', 0):,}")
                    assert (
                        sum(integers(value)[0] for value in page.locator("#outcomes b").all_inner_texts())
                        == vectors["denominator"]
                    )
                    results.append(
                        {
                            "theme": theme,
                            "width": width,
                            "release": release["release_id"],
                            "use": use,
                            "confidence": level,
                            **check_map(page),
                        }
                    )
                results.append({"theme": theme, "width": width, "release": release["release_id"], "use": use, **result})
        page.goto("about:blank")
        page.goto(root + "curriculum.html")
        check_map(page)
        coverage = browser_json(page, "api/curriculum")
        subjects = {node["subject_id"]: node["name"] for node in coverage["hierarchy"] if node["kind"] == "subject"}

        def normalize(label: str) -> str:
            return re.sub(r"[^a-z0-9]", "", label.casefold().replace("&", "and"))

        assert all(
            node["kind"] == "subject" or normalize(node["name"]) != normalize(subjects[node["subject_id"]])
            for node in coverage["hierarchy"]
        )
        domains = page.locator("#domain option").evaluate_all("opts=>opts.map(o=>o.value).filter(Boolean)")
        for domain in domains:
            page.locator("#domain").select_option(domain)
            expect(page.locator(".domain-card")).to_have_count(1)
            expect(page.locator(".domain-card")).to_have_attribute("data-subject", domain)
            results.append({"theme": theme, "width": width, "domain": domain, **check_map(page)})
            expected_nodes = {
                node["node_id"]
                for node in coverage["hierarchy"]
                if node["subject_id"] == domain and node["kind"] != "subject"
            }
            rendered_nodes = set(page.locator(".map-branch").evaluate_all("nodes=>nodes.map(n=>n.dataset.nodeId)"))
            assert rendered_nodes == expected_nodes, "A taxonomy node (including an empty capability) is unreachable"
            count = integers(page.locator(".domain-count").inner_text())[0]
            expect(page.locator("#area-summary")).to_contain_text(f"{count:,} task instances")
            page.locator("#source-breakdown summary").click()
            breakdown = page.locator("#area-sources b").all_inner_texts()
            assert sum(integers(value)[0] for value in breakdown) == count
            page.locator("#source-breakdown summary").click()
            expect(page.locator("#task-list")).not_to_have_attribute("aria-busy", "true", timeout=60000)
            expect(page.locator("#task-message")).to_contain_text(
                re.compile("sampled examples|No task examples|No stored sample examples")
            )

            page.locator("#search").fill("__release_check_unmatched_search__")
            check_map(page)
            page.locator("#search").fill("")
        page.goto("about:blank")
        page.goto(root + "curriculum.html")
        check_map(page)
        page.screenshot(path=str(shots / f"curriculum-{theme}-{width}.png"), full_page=True)
    return results


def curriculum_sweep(page: Page, root: str, releases: list[dict[str, Any]], shots: Path) -> list[dict[str, Any]]:
    """Check one real release, Use, domain, confidence control and sample list per layout."""
    release = next((row for row in releases if row["selected"]), releases[0])
    results = []
    for theme, width in LAYOUTS:
        page.emulate_media(color_scheme=theme)
        page.set_viewport_size({"width": width, "height": 1000})
        page.goto(root + "curriculum.html#" + urlencode({"release": release["release_id"], "intended_use": "all"}))
        expect(page.locator("#provenance")).to_contain_text('"release_id"', timeout=60000)
        check_map(page)
        data = browser_json(page, "api/curriculum?" + urlencode({"release": page.locator("#release").input_value()}))
        levels = data["confidence"]["levels"] if data.get("confidence") else [None]
        for level in sorted({0, len(levels) - 1}, reverse=True):
            page.locator("#confidence").select_option(str(level))
            projected = count_projection(data, level)
            expect(page.locator("#task-count")).to_have_text(f"{projected.totals['denominator']:,}")
            expect(page.locator("#capability-count")).to_have_text(f"{projected.totals['assigned']:,}")
            check_map(page)
        domain = representative_domain(data)
        if domain:
            page.locator("#domain").select_option(domain)
            expect(page.locator(".domain-card")).to_have_count(1)
            expect(page.locator("#task-list")).not_to_have_attribute("aria-busy", "true", timeout=60000)
            expect(page.locator("#task-message")).to_contain_text(
                re.compile("sampled examples|No task examples|No stored sample examples")
            )
            subject = next(row for row in data["hierarchy"] if row["subject_id"] == domain and row["kind"] == "subject")
            count = count_projection(data, 0).areas.get((domain, subject["node_id"]), 0)
            expect(page.locator(".domain-count")).to_have_text(f"{count:,}")
            page.locator("#source-breakdown").evaluate("e=>e.open=true")
            assert sum(integers(value)[0] for value in page.locator("#area-sources b").all_inner_texts()) == count
        result = check_map(page)
        page.screenshot(path=str(shots / f"curriculum-{theme}-{width}.png"), full_page=True)
        results.append(
            {"theme": theme, "width": width, "release": release["release_id"], "use": "all", "domain": domain, **result}
        )
    return results


def evidence_smoke(page: Page, root: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one Evidence and Backends presentation in the blocking geometry pass."""
    source = next((row for row in sources if (row.get("assessment") or {}).get("pipeline_key")), sources[0])
    return evidence_sweep(page, root, [source])
