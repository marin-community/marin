# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rendered WCAG/house readability checks shared by installed journeys and release checks."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from playwright.sync_api import Page, expect

from .release_browser import LAYOUTS as RELEASE_LAYOUTS
from .release_browser import browser_json, evidence_cases, pending_release_ids
from .release_curriculum import representative_domain
from .release_receipts import BrowserScope

AUDIT = Path(__file__).with_suffix(".js").read_text()
CONTROLS = 'a[href],button,input,select,textarea,summary,[tabindex]:not([tabindex="-1"])'
LAYOUTS = (*RELEASE_LAYOUTS, ("light", 320), ("dark", 320))


def inspect_state(page: Page, *, focus: bool = False, focus_limit: int | None = None) -> dict[str, Any]:
    """Measure every rendered text node, including content below the fold."""
    result = page.evaluate(AUDIT, {})
    if focus:
        # Tab, rather than programmatic focus, so :focus-visible and browser scrolling are real.
        page.evaluate(
            "() => {document.body.tabIndex=-1; document.body.focus(); document.body.removeAttribute('tabindex');}"
        )
        visited = set()
        for _ in range(focus_limit if focus_limit is not None else page.locator(CONTROLS).count() + 2):
            page.keyboard.press("Tab")
            identity = page.evaluate(
                """selector => {const e=document.activeElement;
                return e === document.body ? null : Array.from(document.querySelectorAll(selector)).indexOf(e);}""",
                CONTROLS,
            )
            if identity is None:
                continue
            if identity in visited:
                break
            visited.add(identity)
            focused = page.evaluate(AUDIT, {"mode": "focus"})
            if any(row["rule"] == "focus-obscured" for row in focused["findings"]):
                page.evaluate("() => new Promise(resolve => requestAnimationFrame(resolve))")
                focused = page.evaluate(AUDIT, {"mode": "focus"})
            for key in ("findings", "measurements"):
                result[key].extend(focused[key])
    return result


def inspect_status_focus(page: Page) -> dict[str, Any]:
    """Check every Status reason/evidence target, including rows below the fold."""
    result: dict[str, Any] = {"findings": [], "measurements": []}
    targets = page.locator("#rows summary, #rows .status-evidence a")
    for target in targets.all():
        # Tab from the preceding control so native scrolling and :focus-visible
        # are exercised. The representative gate must reach deep table rows too.
        target.evaluate(
            """(e, selector) => {
            const controls=[...document.querySelectorAll(selector)]
                .filter(node => node.checkVisibility() && !node.disabled);
            const previous=controls[controls.indexOf(e)-1];
            previous.focus({preventScroll:true});
        }""",
            CONTROLS,
        )
        page.keyboard.press("Tab")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(resolve))")
        if not target.evaluate("e => e === document.activeElement"):
            raise ValueError("Status keyboard target is not reachable with Tab")
        focused = page.evaluate(AUDIT, {"mode": "focus"})
        for key in result:
            result[key].extend(focused[key])
    return result


def readability_layout(
    page: Page, root: str, shots: Path, theme: str, width: int, scope: BrowserScope = BrowserScope.FULL
) -> list[dict[str, Any]]:
    """Capture content and interaction states at one viewport and system theme."""
    views = []
    page.emulate_media(color_scheme=theme, reduced_motion="reduce")
    page.set_viewport_size({"width": width, "height": 1000})
    prefix = f"{theme}-{width}"

    def record(state: str, result: dict[str, Any], *, screenshot: bool = False) -> None:
        view = {
            "theme": theme,
            "width": width,
            "page": urlsplit(page.url).path.rsplit("/", 1)[-1] or "index.html",
            "url": page.url,
            "state": state,
            "measurement_count": len(result["measurements"]),
            "findings": result["findings"],
        }
        views.append(view)
        with (shots / "states.jsonl").open("a") as stream:
            stream.write(json.dumps(view) + "\n")
        if screenshot:
            page.evaluate(
                '() => {window.scrollTo(0,0); document.querySelectorAll(".table-scroller").forEach(e=>e.scrollLeft=0);}'
            )
            page.screenshot(
                path=str(shots / f"{prefix}-{state}.png"),
                full_page=page.evaluate("document.documentElement.scrollHeight < 30000"),
            )
        print(f'readability {prefix} {state}: {len(view["findings"])} findings', flush=True)

    def capture(state: str, *, focus: bool = False, screenshot: bool = True) -> None:
        record(
            state,
            inspect_state(page, focus=focus, focus_limit=6 if scope is BrowserScope.REPRESENTATIVE else None),
            screenshot=screenshot,
        )
        if not focus:
            return
        # Pointer pseudo states are grouped by rendered style and element classes. Each
        # visible node is still measured in every full content pass above.
        cdp = page.context.new_cdp_session(page)
        cdp.send("DOM.enable")
        cdp.send("CSS.enable")
        selectors = page.evaluate(
            """controls => {
            const seen = new Set(), selectors = [];
            for (const e of document.querySelectorAll(controls)) {
                if (!e.checkVisibility() || e.closest('.sr-only,[hidden]') || e.matches(':disabled')) continue;
                const s=getComputedStyle(e), key=[e.tagName,e.getAttribute('class'),
                    e.parentElement.getAttribute('class'),s.color,s.backgroundColor,s.borderColor].join('|');
                if (seen.has(key)) continue;
                seen.add(key); e.dataset.readabilityTarget=String(selectors.length);
                selectors.push('[data-readability-target="'+selectors.length+'"]');
            }
            return selectors;
        }""",
            CONTROLS,
        )
        if scope is BrowserScope.REPRESENTATIVE:
            selectors = selectors[:6]
        document = cdp.send("DOM.getDocument")["root"]["nodeId"]
        for selector in selectors:
            node_id = cdp.send("DOM.querySelector", {"nodeId": document, "selector": selector})["nodeId"]
            for pseudo in [["hover"], ["hover", "active"]]:
                cdp.send("CSS.forcePseudoState", {"nodeId": node_id, "forcedPseudoClasses": pseudo})
                record(state + "-" + "-".join(pseudo) + "-" + str(node_id), page.evaluate(AUDIT, {"target": selector}))
            cdp.send("CSS.forcePseudoState", {"nodeId": node_id, "forcedPseudoClasses": []})
        cdp.detach()
        spacing = page.add_style_tag(
            content="""
            * {line-height:1.5 !important; letter-spacing:.12em !important; word-spacing:.16em !important;}
            p {margin-bottom:2em !important;}
        """
        )
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        record(state + "-text-spacing", page.evaluate(AUDIT, {}), screenshot=True)
        spacing.evaluate("e=>e.remove()")
        page.evaluate(
            """() => Array.from(document.querySelectorAll('body *'))
            .filter(e=>!(e instanceof SVGElement) || e.matches('svg'))
            .map(e=>[e,parseFloat(getComputedStyle(e).fontSize)]).forEach(([e,size])=>{
                e.dataset.readabilitySize=e.style.fontSize; e.style.fontSize=(size*2)+'px';})"""
        )
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        record(state + "-text-200", page.evaluate(AUDIT, {}), screenshot=True)
        page.evaluate(
            """() => document.querySelectorAll('[data-readability-size]').forEach(e=>{
            e.style.fontSize=e.dataset.readabilitySize; delete e.dataset.readabilitySize;})"""
        )

    def visit(path: str, ready: str) -> None:
        page.goto("about:blank")
        page.goto(root + path)
        expect(page.locator(ready).first).to_be_visible(timeout=60000)
        page.evaluate("() => document.fonts.ready")

    visit("", "#rows tr")
    root = page.url.split("#")[0].split("?")[0]
    sources = browser_json(page, "api/sources")["sources"]
    pending_ids = pending_release_ids(page, root)
    capture("sources", focus=True)
    # Exercise every filter value; captures include selected/empty states.
    for field in ["#use", "#quality", "#result", "#grader", "#difficulty-model"]:
        if not page.locator(field).count():
            continue
        values = page.locator(field + " option").evaluate_all("es => es.map(e=>e.value)")
        for value in values if scope is BrowserScope.FULL else list(dict.fromkeys((values[0], values[-1]))):
            page.locator(field).select_option(value)
            capture("filter-" + field[1:] + "-" + (value or "all"), screenshot=False)
        page.locator(field).select_option(values[0])
    for toggle in page.locator(".duplicate-toggle").all():
        toggle.click()
    capture("expanded-groups")
    if page.locator(".source-name").count():
        page.locator(".source-name").first.click()
        expect(page.locator("dialog[open]")).to_be_visible()
        capture("source-dialog", focus=True)
        page.keyboard.press("Escape")
    page.locator("#show-everything").click()
    capture("all-sources", focus=True)
    for details in page.locator(".data-notes details,details.data-notes").all():
        details.evaluate("e => e.open = true")
    capture("source-notes")
    page.locator("#search").fill("__readability_no_match__")
    capture("source-empty")
    source = next((s for s in sources if (s.get("assessment") or {}).get("pipeline_key")), sources[0])
    for path, ready, label in [
        ("source.html#" + urlencode({"source_id": source["id"]}), "#message", "evidence"),
        ("backends.html#" + urlencode({"source_id": source["id"]}), "#message", "backends"),
        ("assessment.html?" + urlencode({"source": source["id"]}), "#message", "assessment"),
        ("guide.html", ".guide-body", "guide"),
    ]:
        visit(path, ready)
        if label in {"assessment", "evidence", "backends"}:
            expect(page.locator("#message")).not_to_be_empty(timeout=60000)
            expect(page.locator("#message")).not_to_contain_text("Loading", timeout=60000)
        if label == "evidence":
            expect(page.locator("#sections")).to_have_attribute("aria-busy", "false", timeout=15000)
            expect(page.locator("#sections")).not_to_contain_text("Loading supplementary", timeout=10000)
            expect(page.locator("#sections")).not_to_contain_text("Loading overlap", timeout=10000)
        page.locator("details").evaluate_all("es => es.forEach(e => e.open = true)")
        capture(label, focus=True)
        if label == "assessment":
            raw_link = page.locator('#reviews a[href^="review.html"]').first
            if raw_link.count():
                raw_path = raw_link.get_attribute("href")
                assert raw_path is not None
                visit(raw_path, "#message")
                expect(page.locator("#message")).not_to_contain_text("Loading", timeout=60000)
                expect(page.locator("#content h2").first).to_be_visible()
                capture("raw-qc-review", focus=True)
                visit(path, "#filter-count")
            visit("defects.html", "#content h2")
            capture("defect-legend", focus=True)
            visit(path, "#filter-count")
            boxes = page.locator(".review-defect input:not(:disabled)")
            if boxes.count():
                boxes.first.check()
                capture("assessment-defect-checked", screenshot=False)
                boxes.first.uncheck()
        if label == "guide":
            counts = page.locator("button.count:not(:disabled)").all()
            for count in counts if scope is BrowserScope.FULL else counts[:1]:
                count.click()
                capture("guide-leaf-" + str(count.get_attribute("data-leaf")), screenshot=False)
    measured = next(
        (
            row
            for row in sources
            if any(
                model.get("measurement_id") and model["status"] == "measured" for model in row["difficulty"]["models"]
            )
        ),
        None,
    )
    if measured is not None:
        model = next(
            model
            for model in measured["difficulty"]["models"]
            if model.get("measurement_id") and model["status"] == "measured"
        )
        visit(
            "attempts.html?" + urlencode({"source": measured["id"], "measurement": model["measurement_id"]}), "#message"
        )
        expect(page.locator("#message")).not_to_contain_text("Loading", timeout=60000)
        expect(page.locator("#message")).not_to_contain_text("not been loaded", timeout=60000)
        expect(page.locator("#message")).not_to_have_class("error")
        capture("difficulty-attempt-list", focus=True)
        rollout_link = page.locator('a[href^="rollout.html"]').first
        if rollout_link.count():
            rollout_path = rollout_link.get_attribute("href")
            assert rollout_path is not None
            visit(rollout_path, "#content h2")
            capture("difficulty-rollout", focus=True)
            # The transcript's folds and raw-bytes disclosure are controls a reader uses; audit
            # them open as well.
            fold = page.locator(".tr-toggle").first
            if fold.count():
                fold.click()
                capture("difficulty-rollout-expanded", screenshot=False)
                fold.click()
            raw = page.locator(".raw-json summary").first
            if raw.count():
                raw.click()
                capture("difficulty-rollout-raw", screenshot=False)
                raw.click()
    cases = evidence_cases(sources, pending_ids) if scope is BrowserScope.FULL else []
    for index, candidate in enumerate(cases):
        if candidate["id"] == source["id"]:
            continue
        for view in ("source", "backends"):
            visit(view + ".html#" + urlencode({"source_id": candidate["id"]}), "#title")
            expect(page.locator("#title")).to_have_text(candidate["display_name"], timeout=60000)
            page.locator("details").evaluate_all("es => es.forEach(e=>e.open=true)")
            capture(view + "-state-" + str(index), screenshot=False)
    visit("curriculum.html", ".domain-card")
    capture("curriculum", focus=True)
    for option in page.locator("#confidence option").all():
        page.locator("#confidence").select_option(option.get_attribute("value") or "")
        capture("confidence-" + (option.get_attribute("value") or "default"), screenshot=False)
    page.locator(".data-notes").evaluate("e => e.open = true")
    page.locator(".data-notes details").evaluate_all("es => es.forEach(e => e.open = true)")
    capture("curriculum-notes")
    # Every domain exposes its nested heatmap nodes, source breakdown and pooled examples.
    domains = page.locator("#domain option").evaluate_all("es=>es.map(e=>e.value).filter(Boolean)")
    if scope is BrowserScope.REPRESENTATIVE:
        page.locator("#confidence").select_option("0")
        data = browser_json(page, "api/curriculum")
        selected = representative_domain(data)
        domains = [selected] if selected else []
    opened_task = False
    for domain in domains:
        page.locator("#domain").select_option(domain)
        expect(page.locator("#task-list")).not_to_have_attribute("aria-busy", "true", timeout=60000)
        page.locator("#source-breakdown").evaluate("e => e.open = true")
        capture("area-" + domain, focus=not opened_task, screenshot=not opened_task)
        if not opened_task and page.locator(".task-row").count():
            page.locator(".task-row").first.click()
            expect(page.locator("#task-detail")).to_be_visible()
            expect(page.locator("#detail-description")).not_to_have_text("Loading…", timeout=60000)
            page.locator("#task-detail details").evaluate_all("es=>es.forEach(e=>e.open=true)")
            capture("public-instructions", focus=True)
            page.keyboard.press("Escape")
            opened_task = True
    page.locator("#search").fill("__readability_no_match__")
    capture("curriculum-empty")
    # Real error rendering, with only the network boundary intercepted.
    page.route(
        "**/api/curriculum?**",
        lambda route: route.fulfill(status=503, json={"detail": "Readability audit error state"}),
    )
    page.goto(root + "curriculum.html")
    expect(page.locator("#message")).not_to_contain_text("Loading", timeout=60000)
    capture("curriculum-error")
    page.unroute("**/api/curriculum?**")
    # The status page: closed totals, then every reason explanation opened, in both
    # themes and at every width, so its controls and panels stay readable.
    visit("status.html", "#status-counts button")
    capture("status", focus=True)
    record("status-table-keyboard", inspect_status_focus(page))
    page.locator("#rows details > summary").evaluate_all("es => es.forEach(e => e.parentElement.open = true)")
    capture("status-reason-open", focus=True)
    record("status-open-table-keyboard", inspect_status_focus(page))
    # The difficulty coverage page: computed counts plus the needs-measurement,
    # blocked and measured tables, in both themes and at every width.
    visit("difficulty.html", "#coverage-counts div")
    capture("difficulty-coverage", focus=True)
    return views


def readability_sweep(
    page: Page, root: str, shots: Path, *, layouts: Sequence[tuple[str, int]] = LAYOUTS
) -> dict[str, Any]:
    """Walk all page families and the interactive states that change rendered paint."""
    return readability_report(page, root, shots, layouts, BrowserScope.FULL)


def readability_smoke(
    page: Page, root: str, shots: Path, *, layouts: Sequence[tuple[str, int]] = LAYOUTS
) -> dict[str, Any]:
    """Keep authored page families and a representative domain in each blocking layout."""
    return readability_report(page, root, shots, layouts, BrowserScope.REPRESENTATIVE)


def readability_report(
    page: Page, root: str, shots: Path, layouts: Sequence[tuple[str, int]], scope: BrowserScope
) -> dict[str, Any]:
    shots.mkdir(parents=True, exist_ok=True)
    views = []
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    for theme, width in layouts:
        views.extend(readability_layout(page, root, shots, theme, width, scope))
    report = {
        "standard": "WCAG 2.2 AA + Atlas house rules",
        "scope": scope.value,
        "views": views,
        "violations": sum(len(v["findings"]) for v in views),
        "errors": errors,
    }
    (shots / "readability.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise ValueError(f"Browser errors during readability sweep: {errors}")
    return report
