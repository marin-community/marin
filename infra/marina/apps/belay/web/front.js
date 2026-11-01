// Marin Belay front page: the project's public home. Charts, timeline, starter kit, tracks and the searchable
// results. The results table and the speedrun chart's test markers come from the registry (results.json, built with
// `experiments.belay.registry build`); stage 4's fits, points and hero budget come from scaling.json; the accepted
// records and tracks are measured values kept here. Loaded after common.js.

function startFront(RD, SC) {
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const NS = "http://www.w3.org/2000/svg";
  const WB = (run) => `https://wandb.ai/marin-community/marin_moe/runs/${encodeURIComponent(run)}`;
  const el = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };
  const txt = (parent, x, y, s, attrs = {}) => { const t = el("text", { x, y, fill: css("--faint"), "font-size": 11, "font-family": css("--mono"), ...attrs }, parent); t.textContent = s; return t; };
  const sci = (x, d = 1) => { const e = Math.floor(Math.log10(x)); return `${(x / 10 ** e).toFixed(d)}e${e}`; };
  const human = (x) => x >= 1e12 ? `${(x / 1e12).toFixed(2)}T` : x >= 1e9 ? `${(x / 1e9).toFixed(1)}B` : x >= 1e6 ? `${(x / 1e6).toFixed(1)}M` : x >= 1e3 ? `${(x / 1e3).toFixed(0)}k` : x.toFixed(0);
  function tipper(boxId, tipId, svg) {
    const show = (n) => { const tip = $(tipId), box = $(boxId).getBoundingClientRect(), r = n.getBoundingClientRect(); tip.innerHTML = n.dataset.tip; tip.style.left = `${Math.min(Math.max(r.left - box.left + r.width / 2, 110), box.width - 110)}px`; tip.style.top = `${r.top - box.top}px`; tip.hidden = false; };
    svg.querySelectorAll("[data-tip]").forEach((n) => {
      n.addEventListener("mouseenter", () => show(n)); n.addEventListener("focus", () => show(n));
      n.addEventListener("mouseleave", () => { $(tipId).hidden = true; }); n.addEventListener("blur", () => { $(tipId).hidden = true; });
    });
  }

  // ---------- Record over time: d512 8-minute Competitive Track (#9451) ----------
  // [batch, day (days after 2026-09-01; same-day records spread across the day), BPB, kind, label, detail]
  const RECORDS = [
    [0, 24.333, 1.2959, "base", "Clean KMA baseline", "cr1-base-d512"],
    [3, 24.667, 1.2852, "speed", "fp32 embedding-gather backward", "+12% examples/s"],
    [17, 25.5, 1.2786, "ml", "Learnable qk_mult, Adam β1 0.8, no embed / lm_head GatedNorm", "lc1-mfu1"],
    [32, 26.143, 1.2686, "ml", "lm_head tanh soft-cap 10", "−0.011 on 2 seeds"],
    [51, 26.286, 1.2576, "ml", "Candidate 1: bigram table, weight EMA, key offset, MuonEq", "batch 51"],
    [63, 26.429, 1.2523, "ml", "Candidate 2: ReLU² 1.5× experts, Bi-Maxwell", "batch 63"],
    [66, 26.571, 1.2499, "ml", "Candidate 3: MLA per-head key norm", "3 seeds"],
    [71, 26.714, 1.2490, "ml", "Candidate 4: bigram content gate", "batch 71"],
    [75, 26.857, 1.2484, "ml", "Candidate 5: MLA query norm, qk_mult init 4", "batch 75"],
    [78, 27.167, 1.2461, "ml", "AdEMAMix on the Adam groups", "batch 78"],
    [86, 27.333, 1.2424, "ml", "512 experts (sparsity 64)", "batch 86"],
    [89, 27.5, 1.2385, "ml", "Candidate 7: gated value embeddings", "batch 89"],
    [90, 27.667, 1.2355, "ml", "Candidate 8: Adam β2 0.98, GDN-2 gates, wider shared expert", "batch 90"],
    [102, 27.833, 1.2337, "ml", "Candidate 9: router LR ×0.5, sqrt-softplus combine", "batch 102"],
    [157, 30.5, 1.2347, "retime", "Re-timed on cw-us-east-02a without PGLE", "candidate 10 at its own 480 s budget; candidate 9 measured 1.2356 there"],
    [174, 31.5, 1.2323, "speed", "Candidate 11: KDA speed flags", "−3.5% step time"],
    [195, 34.5, 1.2322, "ml", "Candidate 12: ragged all-to-all, 6 MLA heads", "fixed-time confirmation"],
  ];
  // Accepted recipes at the 8-minute d512 budget (480 s of training on 8×H100), measured Paloma macro BPB (2026-10-14).
  // [name, BPB, color, dashed, steps, W&B run]; the last is the current accepted recipe.
  const ACCEPTED = [
    ["Vanilla dense", 1.4275, "--amber", false, 6789, "belay-8min-vanilla"],
    ["Marin Aug Hero Arch", 1.2980, "--slate", false, 2769, "belay-8min-hero"],
    ["Belay Ckpt 1", 1.2562, "--rope", false, 2759, "belay-8min-ckpt1"],
  ];

  // Successful d512 tests: each idea's best d512 trial with a mean delta at least NOISE better, in the order the ideas
  // were first tested. [name, mean delta (placeholder BPB), in Belay Ckpt 1, W&B run, batch, registry id]
  // Successful d512 tests, from the registry: each idea's best d512 trial whose mean delta is at least NOISE better,
  // in the order the ideas were first tested. [name, mean delta (placeholder BPB), in Belay Ckpt 1, W&B run, batch, id]
  const WINS = (() => {
    const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const wins = [];
    RD.ideas.forEach((idea, order) => {
      const d512 = idea.trials.filter((t) => t.rung === "d512" && t.delta && t.delta.length && !t.derived);
      if (!d512.length) return;
      const best = d512.reduce((a, b) => (avg(b.delta) < avg(a.delta) ? b : a));
      const m = avg(best.delta);
      if (m > -0.002) return;
      const first = idea.trials.map((t) => t.date).filter(Boolean).sort()[0] || "";
      wins.push({ first, order, row: [idea.name, m, idea.status === "adopted", (best.wandb || [])[0] || null, String(best.batch), idea.xid] });
    });
    wins.sort((a, b) => a.first.localeCompare(b.first) || a.order - b.order);
    return wins.map((w) => w.row);
  })();
  function drawProgress() {
    const svg = $("svg-progress"); svg.innerHTML = "";
    // Height in viewBox units follows the box, so the chart fills its card at any width.
    const box = $("box-progress");
    const W = 600, H = Math.max(340, Math.round((W * box.clientHeight) / Math.max(box.clientWidth, 1)));
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const M = { l: 48, r: 14, t: 14, b: 22 };
    // Broken y axis: a short top band for the dense baseline, the main band for the MoE recipes and the test deltas.
    const TOP = { lo: 1.40, hi: 1.44, y0: M.t, y1: M.t + 34 }, MAIN = { lo: 1.25, hi: 1.302, y0: M.t + 54, y1: H - M.b };
    const Y = (l) => {
      const band = l >= TOP.lo ? TOP : MAIN;
      return band.y0 + ((band.hi - l) / (band.hi - band.lo)) * (band.y1 - band.y0);
    };
    // x is ordinal: a dense segment, the Aug Hero Arch segment holding the tests evenly spaced, the Ckpt 1 segment.
    const n = WINS.length, x0 = M.l, x1 = W - M.r;
    const xd = x0 + 0.08 * (x1 - x0), xc = x1 - 0.1 * (x1 - x0);
    const xw = (i) => xd + ((i + 1) / (n + 1)) * (xc - xd);
    [[MAIN, [1.26, 1.28, 1.30]], [TOP, [1.42]]].forEach(([band, ticks]) => ticks.forEach((l) => {
      el("line", { x1: x0, x2: x1, y1: Y(l), y2: Y(l), stroke: css("--line") }, svg);
      txt(svg, x0 - 8, Y(l) + 4, l.toFixed(2), { "text-anchor": "end" });
    }));
    const yb = (TOP.y1 + MAIN.y0) / 2;
    [-4, 4].forEach((dy) => el("line", { x1: x0 - 6, x2: x0 + 6, y1: yb + dy + 3, y2: yb + dy - 3, stroke: css("--faint"), "stroke-width": 1.4 }, svg));
    el("line", { x1: x0, x2: x1, y1: H - M.b, y2: H - M.b, stroke: css("--faint") }, svg);
    const [dense, hero, ckpt] = ACCEPTED;
    const segs = [[dense, x0, xd], [hero, xd, xc], [ckpt, xc, x1]];
    // Lollipops: from the Aug Hero Arch line down by each test's improvement.
    const yh = Y(hero[1]);
    WINS.forEach(([name, delta, inCkpt, run, batch, xid], i) => {
      const cx = xw(i), cy = Y(hero[1] + delta), col = inCkpt ? css("--rope") : css("--faint");
      el("line", { x1: cx, x2: cx, y1: yh, y2: cy, stroke: col, "stroke-width": inCkpt ? 1.6 : 1.2, opacity: inCkpt ? 0.8 : 0.7 }, svg);
      el("circle", { cx, cy, r: inCkpt ? 3.6 : 3, fill: col, opacity: inCkpt ? 0.95 : 0.85 }, svg);
      const link = run ? el("a", { href: WB(run), target: "_blank", rel: "noopener", style: "cursor:pointer" }, svg) : svg;
      const hit = el("rect", { x: cx - 5, y: yh, width: 10, height: Math.max(cy - yh + 6, 10), fill: "transparent", tabindex: run ? -1 : 0 }, link);
      hit.dataset.tip = `<b>${delta.toFixed(4)}</b> · ${esc(name)}${xid ? ` · ${xid}` : ""}<br>${inCkpt ? "In Belay Ckpt 1" : "Not in Belay Ckpt 1"} · best d512 trial, batch ${esc(batch)}<br><span style="opacity:.6">mean Δ vs its baseline, placeholder BPB${run ? ` · ${run} · click to open in W&amp;B` : ""}</span>`;
    });
    // The accepted staircase: dense -> Aug Hero Arch -> Belay Ckpt 1.
    const stair = segs.map(([[, l], a, b], i) => (i ? `V${Y(l)}` : `M${a},${Y(l)}`) + `H${b}`).join("");
    el("path", { d: stair, fill: "none", stroke: css("--ink"), "stroke-width": 1.4, opacity: 0.35, "stroke-linejoin": "round" }, svg);
    segs.forEach(([[name, l, color, , steps, run], a, b], i) => {
      const cur = i === 2, col = css(color), y = Y(l);
      el("line", { x1: a, x2: b, y1: y, y2: y, stroke: col, "stroke-width": cur ? 3.4 : 2.4 }, svg);
      el("circle", { cx: a, cy: y, r: cur ? 7 : 5.5, fill: col, stroke: css("--surface"), "stroke-width": 2 }, svg);
      const label = `${name} ${l.toFixed(3)}`;
      if (cur) txt(svg, a - 12, y + 4, label, { "text-anchor": "end", fill: col, "font-weight": 700, "font-size": 13 });
      else txt(svg, a + 8, y - 8, label, { fill: css("--slate"), "font-weight": 600, "font-size": 11.5 });
      const link = el("a", { href: WB(run), target: "_blank", rel: "noopener", style: "cursor:pointer" }, svg);
      const hit = el("rect", { x: a - 6, y: y - 8, width: b - a + 12, height: 16, fill: "transparent", tabindex: -1 }, link);
      hit.dataset.tip = `<b>${l.toFixed(4)}</b> · ${name}${cur ? " · current accepted recipe" : " · accepted"}<br>d512 · 8 min on 8×H100 (${steps.toLocaleString()} steps) · measured Paloma macro BPB<br><span style="opacity:.6">${run} · click to open in W&amp;B</span>`;
    });
    tipper("box-progress", "tip-progress", svg);
  }

  // ---------- Scaling laws (log y) ----------
  const SEQ = 4096, RUNGS = ["d512", "d768", "d1024", "d1280"];
  const MOE = { d512: [20840448, 1133066059776], d768: [60555264, 2575067774976], d1024: [162004992, 5929672114176], d1280: [291307520, 9975229317120] };
  const DENSE = { d512: [18087936, 1065420324864], d768: [53477376, 2401121599488], d1024: [144703488, 5504470351872], d1280: [261488640, 9242400522240] };
  const rungC = (t, tpp) => Object.fromEntries(RUNGS.map((r) => [r, (t[r][1] / SEQ) * tpp * t[r][0]]));
  const C_MOE = rungC(MOE, 60);
  // Stage 4 comes from scaling.json: Paloma macro loss L(C) = floor + A·C^−α on analytic training FLOPs, converted to
  // BPB with each ladder's loss/BPB ratio (linear in log10 C). The x axis is hardware FLOPs = model FLOPs ÷ 11-rack MFU.
  const CURVES = SC.recipes.map((r) => ({ ...r, pts: r.points.map((q) => [q.rung, q.bpb, q.C, q.run]) }));
  const curve = (id) => CURVES.find((c) => c.id === id);
  const lossAt = (c, C) => SC.floor + c.A * Math.pow(C, -c.alpha);
  const Lf = (c, C) => lossAt(c, C) / (c.ratio_icpt + c.ratio_slope * Math.log10(C));
  const Lhw = (c, Chw) => Lf(c, Chw * c.mfu);
  const C_HW_BUDGET = SC.gb200_hours * 3600 * SC.peak_flops;
  {
    const h = SC.hero, hc = curve(h.recipe), C_HERO = C_HW_BUDGET * hc.mfu, D_HERO = C_HERO / h.flops_per_token;
    $("h-tps").textContent = human(h.rack_tps); $("h-flops").textContent = sci(C_HERO, 2);
    $("h-n").textContent = human(h.active_params); $("h-d").textContent = human(D_HERO);
    $("hc-tps").title = `Measured: ${hc.name} on one GB200 rack (64 GPUs, ${h.rack_step} s/step)`;
    $("hc-flops").title = `300k GB200-hours = ${sci(C_HW_BUDGET, 1)} hardware FLOPs × ${(hc.mfu * 100).toFixed(1)}% MFU (${hc.mfu_note})`;
    $("hc-n").title = h.active_params_note;
    $("hc-d").title = `D = model FLOPs ÷ ${sci(h.flops_per_token, 3)} training FLOPs per token (${h.flops_per_token_note}) = ${(D_HERO / h.active_params).toFixed(0)} tokens per active param`;
    $("scaling-caveat").textContent = SC.caveat;
  }
  // Timeline table: each recipe's four-rung fit at 300k GB200-hours and its own 11-rack MFU; speedup is the hardware
  // FLOPs the reference recipe needs for the row's projected BPB over the row's hardware FLOPs.
  {
    const hwFor = (c, target) => {
      let lo = 17, hi = 27;
      for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; if (Lhw(c, 10 ** mid) > target) lo = mid; else hi = mid; }
      return 10 ** hi;
    };
    const ref = curve(SC.xlate.reference);
    $("xlate-body").innerHTML = SC.xlate.rows.map((id) => {
      const c = curve(id), model = C_HW_BUDGET * c.mfu, bpb = Lf(c, model), spd = hwFor(ref, bpb) / C_HW_BUDGET;
      return `<tr><td>${SC.xlate.names[id]}</td><td class="r">${human(SC.gb200_hours)}</td><td class="r">${sci(C_HW_BUDGET, 2)}</td>`
        + `<td class="r">${c.mfu_1rack ? (c.mfu_1rack * 100).toFixed(1) + "%" : "–"}</td><td class="r">${(c.mfu * 100).toFixed(1)}%</td><td class="r">${sci(model, 2)}</td>`
        + `<td class="r law">${SC.floor} + ${c.A.toFixed(1)}·C<sup>−${c.alpha.toFixed(4)}</sup></td><td class="r">${bpb.toFixed(3)}</td><td class="r${c === ref ? "" : " spd"}">${spd.toFixed(2)}×</td></tr>`;
    }).join("");
  }
  const lg = $("legend-scaling");
  {
    // Rows double as the plot legend: each recipe's projected BPB at the hero budget, best first.
    const ranked = [...CURVES].sort((p, q) => Lhw(p, C_HW_BUDGET) - Lhw(q, C_HW_BUDGET));
    lg.innerHTML = ranked.map((c, i) => {
      const l = Lhw(c, C_HW_BUDGET);
      return `<li class="${i ? "" : "top"}" title="${c.name}: ${c.mfu_note}"><i class="sw${c.dash ? " dash" : ""}" style="border-color:${css(c.color)}"></i>`
        + `<span class="nm">${c.name}<small>${(c.mfu * 100).toFixed(1)}% MFU</small></span><span class="bp">${l.toFixed(3)}</span></li>`;
    }).join("") + `<li class="budget"><i class="sw"></i><span>Hero budget: 300k GB200-hours</span></li>`;
  }

  function drawScaling() {
    const svg = $("svg-scaling"); svg.innerHTML = "";
    const W = 600, H = 360, M = { l: 48, r: 16, t: 14, b: 34 };
    const xmin = 17.0, xmax = 24.8, ymin = Math.log10(0.66), ymax = Math.log10(1.7);
    const X = (C) => M.l + ((Math.log10(C) - xmin) / (xmax - xmin)) * (W - M.l - M.r);
    const Y = (l) => M.t + ((ymax - Math.log10(l)) / (ymax - ymin)) * (H - M.t - M.b);
    [0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6].forEach((l) => {
      el("line", { x1: M.l, x2: W - M.r, y1: Y(l), y2: Y(l), stroke: css("--line") }, svg);
      txt(svg, M.l - 8, Y(l) + 4, l.toFixed(1), { "text-anchor": "end" });
    });
    for (let e = 17; e <= 24; e++) {
      el("line", { x1: X(10 ** e), x2: X(10 ** e), y1: H - M.b, y2: H - M.b + 4, stroke: css("--faint") }, svg);
      txt(svg, X(10 ** e), H - M.b + 17, `1e${e}`, { "text-anchor": "middle" });
    }
    el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: css("--faint") }, svg);
    const xl = txt(svg, W - M.r, H - 2, "", { "text-anchor": "end", fill: css("--slate") });
    el("tspan", { fill: css("--faint") }, xl).textContent = "Measured at Hero MFU · ";
    el("tspan", {}, xl).textContent = "Hardware FLOPs →";
    txt(svg, M.l, M.t - 2, "BPB (log)", { fill: css("--slate") });
    // hero budget line (hardware FLOPs)
    const hx = X(C_HW_BUDGET);
    el("rect", { x: hx - 1, y: M.t, width: 2, height: H - M.t - M.b, fill: css("--rope"), opacity: 0.9 }, svg);
    const bh = el("rect", { x: hx - 8, y: M.t, width: 16, height: H - M.t - M.b, fill: "transparent", tabindex: 0 }, svg);
    bh.dataset.tip = `<b>Hero budget</b> · ${sci(C_HW_BUDGET, 1)} hardware FLOPs<br>300k GB200-hours at 2.5 PF/s per GB200`;
    CURVES.forEach((c) => {
      const col = css(c.color);
      let d = "";
      for (let i = 0; i <= 140; i++) {
        const C = 10 ** (xmin + (i / 140) * (xmax - xmin)), l = Lhw(c, C);
        if (l > 1.7) continue;
        d += `${d ? "L" : "M"}${X(C).toFixed(1)},${Y(l).toFixed(1)}`;
      }
      el("path", { d, fill: "none", stroke: col, "stroke-width": c.width, "stroke-dasharray": c.dash ? "6 5" : "none" }, svg);
      c.pts.forEach(([r, l, Cm, run]) => {
        const cx = X(Cm / c.mfu), cy = Y(l);
        el("circle", { cx, cy, r: 4.8, fill: col, stroke: css("--surface"), "stroke-width": 1.8 }, svg);
        const link = run ? el("a", { href: WB(run), target: "_blank", rel: "noopener", style: "cursor:pointer" }, svg) : svg;
        const hit = el("circle", { cx, cy, r: 12, fill: "transparent", tabindex: run ? -1 : 0 }, link);
        hit.dataset.tip = `<b>${l.toFixed(4)}</b> · ${r}<br>${c.name}<br><span style="opacity:.75">${sci(Cm, 2)} model FLOPs · ${sci(Cm / c.mfu, 2)} hardware at ${(c.mfu * 100).toFixed(1)}% MFU</span>${run ? `<br><span style="opacity:.6">${run} · click to open in W&amp;B</span>` : ""}`;
      });
      const pl = Lhw(c, C_HW_BUDGET);
      el("circle", { cx: hx, cy: Y(pl), r: 5, fill: css("--surface"), stroke: col, "stroke-width": 2.4 }, svg);
      const hit = el("circle", { cx: hx, cy: Y(pl), r: 11, fill: "transparent", tabindex: 0 }, svg);
      hit.dataset.tip = `<b>${pl.toFixed(3)}</b> projected BPB at the hero budget (loss ${lossAt(c, C_HW_BUDGET * c.mfu).toFixed(4)})<br>${c.name}<br><span style="opacity:.75">loss = ${SC.floor} + ${c.A}·C^−${c.alpha} on model FLOPs; ${c.mfu_note}</span>`;
    });
    // equal-loss compute multiplier in hardware FLOPs: where latest best reaches the Aug Hero Arch's projected BPB
    // Hardware FLOPs at which the new recipe reaches the reference's projected BPB (bisection on the monotone curve).
    const bc = curve(SC.compare.to), nc = curve(SC.compare.from), Lt = Lhw(bc, C_HW_BUDGET);
    let lo = 17, hi = Math.log10(C_HW_BUDGET);
    for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (Lhw(nc, 10 ** mid) > Lt) lo = mid; else hi = mid; }
    const C_EQ = 10 ** hi, NX = C_HW_BUDGET / C_EQ;
    const ex = X(C_EQ), ey = Y(Lt);
    el("line", { x1: ex, x2: hx, y1: ey, y2: ey, stroke: css("--ink"), "stroke-dasharray": "3 3", "stroke-width": 1.6 }, svg);
    el("circle", { cx: ex, cy: ey, r: 4.2, fill: css("--rope"), stroke: css("--surface"), "stroke-width": 1.6 }, svg);
    txt(svg, ex - 6, ey + 18, `${NX.toFixed(1)}× less compute*`, { "text-anchor": "end", fill: css("--ink"), "font-weight": 600, "font-size": 11.5 });
    const eh = el("circle", { cx: ex, cy: ey, r: 11, fill: "transparent", tabindex: 0 }, svg);
    eh.dataset.tip = `<b>${NX.toFixed(2)}×</b> less compute to the same BPB (${Lt.toFixed(3)})<br>${nc.name} at ${sci(C_EQ, 2)} vs ${bc.name} at ${sci(C_HW_BUDGET, 2)} hardware FLOPs<br><span style="opacity:.75">from the four-rung fits; each at its own 11-rack MFU<br>*extrapolated ~4,000× past the largest rung, so expect large error</span>`;
    tipper("box-scaling", "tip-scaling", svg);
  }

  // ---------- Timeline disclosures (one panel open at a time) ----------
  const tbtns = [...document.querySelectorAll(".tbtn")];
  tbtns.forEach((b) => b.addEventListener("click", () => {
    const opening = b.getAttribute("aria-expanded") !== "true";
    tbtns.forEach((o) => { o.setAttribute("aria-expanded", "false"); $(o.getAttribute("aria-controls")).hidden = true; });
    if (opening) { b.setAttribute("aria-expanded", "true"); $(b.getAttribute("aria-controls")).hidden = false; }
  }));

  // ---------- Tracks ----------
  const TRACKS = [
    // runs: [active params, tokens, model FLOPs, Paloma macro BPB (measured), wall-clock minutes on 8×H100]
    { id: "vanilla", name: "Vanilla Track", tag: "dense · 20 tokens / param", desc: "A barebones Qwen3-style dense transformer (full causal GQA, RoPE, QK-norm, SwiGLU) trained with AdamW at a compute-scaled LR. For testing ideas with minimal confounders.", tpp: 20, runs: {"d512": [18881536, 377487360, 9.025e+16, 1.6205, 2.0], "d768": [56636928, 1132986368, 6.416e+17, 1.4146, 6.5], "d1024": [151021568, 3020947456, 3.954e+18, 1.3016, 28.3], "d1280": [275289600, 5506072576, 1.221e+19, 1.2326, 80.4]}, src: "belay_tracks.py --track vanilla", runPrefix: "belay-vanilla-" },
    { id: "moe", name: "Basic MoE", tag: "MoE · 60 tokens / active param", desc: "QB routing, top 4 of 128, 2 shared experts, MuonH. For testing MoE ideas with minimal confounders.", tpp: 60, runs: {"d512": [17219840, 1033371648, 2.237e+17, 1.3514, 6.9], "d768": [48473600, 2908225536, 1.422e+18, 1.2323, 26.1], "d1024": [128169472, 7690256384, 8.574e+18, 1.1336, 112.5], "d1280": [228123648, 13687062528, 2.54e+19, 1.0772, 294.8]}, src: "belay_tracks.py --track moe", runPrefix: "belay-moe-" },
    { id: "comp", name: "Competitive Track", tag: "MoE · 60 tokens / active param", desc: "For building on the best active recipe.", table: MOE, tpp: 60,
      best: { d512: [1.2322, "8-minute record"], d768: [1.1289, "candidate 10"], d1024: [1.0428, "candidate 9"], d1280: [0.9939, "candidate 9"] } },
  ];
  const tc = $("track-cards");
  const mins = (m) => (m < 90 ? `${m.toFixed(0)} min` : `${(m / 60).toFixed(1)} h`);
  TRACKS.forEach((t) => {
    if (t.runs) {
      const rows = RUNGS.map((r) => {
        const [p, tok, f, bpb, m] = t.runs[r];
        return `<tr><td class="mono"><a class="wb" href="${WB(t.runPrefix + r)}" target="_blank" rel="noopener" title="${t.runPrefix + r} in W&amp;B">${r}</a></td><td class="r">${human(p)}</td><td class="r">${human(tok)}</td><td class="r">${sci(f)}</td><td class="r">${mins(m)}</td><td class="r"><b>${bpb.toFixed(4)}</b></td></tr>`;
      }).join("");
      const card = document.createElement("article"); card.className = "track";
      card.innerHTML = `<header><div><div class="tag">${t.tag}</div><h3>${t.name}</h3></div></header>
      <p>${t.desc}</p>
      <div style="overflow-x:auto"><table class="rungs"><thead><tr><th>Rung</th><th class="r">Active</th><th class="r">Tokens</th><th class="r">FLOPs</th><th class="r">8×H100</th><th class="r">Baseline</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note" style="margin:0">Measured Paloma macro BPB; wall clock includes compile and eval. <a href="https://github.com/marin-community/marin/blob/marin_belay_v1_ckpt1/experiments/grug/fast_track/belay_tracks.py" target="_blank" rel="noopener">${t.src}</a></p>`;
      tc.appendChild(card);
      return;
    }
    const C = rungC(t.table, t.tpp);
    const rows = RUNGS.map((r) => {
      const b = t.best[r], p = t.table[r][0];
      return `<tr><td class="mono">${r}</td><td class="r">${human(p)}</td><td class="r">${human(t.tpp * p)}</td><td class="r">${sci(C[r])}</td>
        <td class="r">${b ? `<b>${b[0].toFixed(4)}</b><br><span style="color:var(--faint);font-size:11px;font-family:var(--body)">${b[1]}</span>` : '<span class="open">open</span>'}</td></tr>`;
    }).join("");
    const card = document.createElement("article"); card.className = `track${t.id === "comp" ? " comp" : ""}`;
    card.innerHTML = `<header><div><div class="tag">${t.tag}</div><h3>${t.name}</h3></div>${t.id === "vanilla" ? '<span class="pill est">no runs yet</span>' : ""}</header>
      <p>${t.desc}</p>
      <div style="overflow-x:auto"><table class="rungs"><thead><tr><th>Rung</th><th class="r">Params</th><th class="r">Tokens</th><th class="r">FLOPs</th><th class="r">${t.id === "comp" ? "Record" : "Baseline"}</th></tr></thead><tbody>${rows}</tbody></table></div>
`;
    tc.appendChild(card);
  });

  // ---------- Results (built from the #9451 batch reports) ----------
  const RUNG_IX = { d512: 0, d768: 1, d1024: 2, d1280: 3 };
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const NOISE = 0.002; // BPB
  const fmtD = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(4);
  const cls = (v) => (Math.abs(v) < NOISE ? "neutral" : v < 0 ? "better" : "worse");
  const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const STATUS = {
    in: ["In the recipe", "var(--teal)"], won: ["Won, not adopted", "var(--teal)"], noise: ["Within noise", "var(--slate)"],
    lost: ["Lost", "var(--worse)"], removed: ["Removed", "var(--amber)"], tested: ["Tested, no Δ", "var(--faint)"],
  };
  RD.ideas.forEach((i) => {
    const deltas = i.trials.filter((t) => t.delta && t.delta.length).map((t) => ({ t, m: mean(t.delta) }));
    i.best = deltas.length ? deltas.reduce((a, b) => (b.m < a.m ? b : a)) : null;
    let state = null;
    i.events.forEach((e) => {
      if (e.type === "adopted" && !/proposal/i.test(e.candidate || "")) state = "in";
      if ((e.type === "removed" || e.type === "superseded") && state === "in") state = "removed";
    });
    if (!state) state = !i.best ? "tested" : i.best.m <= -NOISE ? "won" : i.best.m < NOISE ? "noise" : "lost";
    i.status = state;
    i.closed = i.events.some((e) => e.type === "closed");
    i.strip = [null, null, null, null];
    deltas.forEach(({ t, m }) => { const k = RUNG_IX[t.rung]; if (k === undefined) return; if (i.strip[k] === null || m < i.strip[k]) i.strip[k] = m; });
    i.maxRung = Math.max(-1, ...i.trials.map((t) => RUNG_IX[t.rung] ?? -1));
    i.last = i.trials.reduce((a, t) => (t.date > a ? t.date : a), "");
    i.hay = [i.name, i.id, i.summary, i.category, ...(i.tags || []), ...(i.details || []), ...(i.knobs || []),
      ...(i.papers || []).flatMap((p) => [p.arxiv, p.title]),
      ...i.trials.flatMap((t) => [t.variant, t.baseline, t.note, t.tuned, ...(t.wandb || []), ...Object.entries(t.settings || {}).map(([k, v]) => `${k} ${v}`)]),
    ].filter(Boolean).join(" ").toLowerCase();
  });
  const counts = (f) => RD.ideas.reduce((c, i) => { const k = f(i); c[k] = (c[k] || 0) + 1; return c; }, {});
  const nTrials = RD.ideas.reduce((n, i) => n + i.trials.length, 0);
  const sc = counts((i) => i.status);
  $("res-summary").innerHTML = `<span><b>${RD.ideas.length}</b> ideas</span><span><b>${nTrials}</b> trials</span>` +
    `<span><b>${sc.in || 0}</b> in the recipe</span><span><b>${sc.won || 0}</b> won, not adopted</span><span><b>${(sc.noise || 0) + (sc.lost || 0)}</b> within noise or lost</span>` +
    `<span style="color:var(--faint)">from the ${esc(RD.generated_from)}</span>`;

  const F = { status: null, cat: null, reach: null, q: "", sort: "best" };
  try { Object.assign(F, JSON.parse(localStorage.getItem("belay-res") || "{}"), { q: "" }); } catch (e) {}
  const saveF = () => { try { localStorage.setItem("belay-res", JSON.stringify({ status: F.status, cat: F.cat, reach: F.reach, sort: F.sort })); } catch (e) {} };
  const REACH = { d512: ["d512 only", (i) => i.maxRung <= 0], scale: ["Tested at d768+", (i) => i.maxRung >= 1] };
  const open = new Set();

  function chipRow(id, items, key) {
    const box = $(id); box.querySelectorAll(".chip").forEach((n) => n.remove());
    items.forEach(([val, label, n, color]) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.setAttribute("aria-pressed", String(F[key] === val));
      b.innerHTML = `${color ? `<span class="dot" style="background:${color}"></span>` : ""}${esc(label)}<span class="n">${n}</span>`;
      b.addEventListener("click", () => { F[key] = F[key] === val ? null : val; saveF(); drawResults(); });
      box.appendChild(b);
    });
  }
  function matches(i, skip) {
    if (skip !== "status" && F.status && i.status !== F.status) return false;
    if (skip !== "cat" && F.cat && i.category !== F.cat) return false;
    if (skip !== "reach" && F.reach && !REACH[F.reach][1](i)) return false;
    if (F.q) { const terms = F.q.toLowerCase().split(/\s+/).filter(Boolean); if (!terms.every((w) => i.hay.includes(w))) return false; }
    return true;
  }
  const sorters = {
    best: (a, b) => (a.status === "in") !== (b.status === "in") ? (a.status === "in" ? -1 : 1) : (a.best ? a.best.m : 9) - (b.best ? b.best.m : 9),
    recent: (a, b) => (b.last || "").localeCompare(a.last || ""),
    trials: (a, b) => b.trials.length - a.trials.length,
    name: (a, b) => a.name.localeCompare(b.name),
  };
  function stripHtml(i) {
    return `<span class="strip" aria-label="Best result per rung">${i.strip.map((v, k) => {
      if (v === null) return `<i title="${["d512", "d768", "d1024", "d1280"][k]}: not run"></i>`;
      return `<i class="${cls(v)}" title="${["d512", "d768", "d1024", "d1280"][k]}: best Δ ${fmtD(v)}"></i>`;
    }).join("")}</span>`;
  }
  function rowHtml(i) {
    const [sl] = STATUS[i.status];
    const papers = (i.papers || []).map((p) => p.arxiv ? `arXiv:${p.arxiv}` : p.title).join(" · ");
    const sub = papers || i.summary || "";
    const bd = i.best ? `${fmtD(i.best.m)}<small>${esc((i.best.t.variant || "").slice(0, 26))}</small>` : '<span style="color:var(--faint)">–</span>';
    return `<button type="button" aria-expanded="${open.has(i.id)}" aria-controls="det-${i.id}">
      <span class="nm"><b>${esc(i.name)}</b>${i.xid ? `<span class="xid">${esc(i.xid)}</span>` : ""}<span class="sm">${esc(sub)}</span></span>
      <span class="st ${i.status}">${sl}${i.closed && i.status !== "in" ? " · closed" : ""}</span>
      <span class="bd">${bd}</span>
      ${stripHtml(i)}
      <span class="nt">${i.trials.length}</span>
      <span class="dt">${i.last ? i.last.slice(5) : "–"}</span>
    </button><div class="detail" id="det-${i.id}" ${open.has(i.id) ? "" : "hidden"}>${open.has(i.id) ? detailHtml(i) : ""}</div>`;
  }
  function detailHtml(i) {
    const papers = (i.papers || []).length ? `<div class="papers" style="margin-top:8px">${i.papers.map((p) => p.arxiv ? `<a href="https://arxiv.org/abs/${esc(p.arxiv)}" target="_blank" rel="noopener">arXiv:${esc(p.arxiv)}</a>${p.title ? ` <span style="color:var(--slate)">${esc(p.title)}</span>` : ""}` : `<span style="color:var(--slate)">${esc(p.title)}</span>`).join("<br>")}</div>` : "";
    const spec = (i.details || []).length ? `<ul class="spec">${i.details.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<div style="color:var(--faint);font-size:13px">No spec recorded.</div>';
    const knobs = (i.knobs || []).length ? `<div class="knobs">${i.knobs.map((k) => `<code>${esc(k)}</code>`).join("")}<button type="button" class="copy" data-copy="${esc(i.knobs.join(" "))}">Copy</button></div>` : "";
    const src = i.source ? `proposed by ${esc(i.source)}${i.first_batch ? `, batch ${esc(i.first_batch)}` : ""}` : "";
    const ev = [`<li><span class="when">batch ${esc(i.first_batch ?? "?")}</span> <span class="what">Proposed</span><span class="why">${src}</span></li>`,
      ...i.events.map((e) => `<li class="${e.type}"><span class="when">${esc(e.date || "")} · ${e.comment_url ? `<a href="${esc(e.comment_url)}" target="_blank" rel="noopener">batch ${esc(e.batch || "")}</a>` : `batch ${esc(e.batch || "")}`}</span>
        <span class="what"> ${esc(e.type[0].toUpperCase() + e.type.slice(1))}${e.candidate ? ` · ${esc(e.candidate)}` : ""}</span>
        ${e.note ? `<span class="why">${esc(e.note)}</span>` : ""}${e.as_adopted ? `<span class="why">as adopted: ${esc(e.as_adopted.join("; "))}</span>` : ""}</li>`)].join("");
    const rows = i.trials.slice().reverse().map((t) => {
      const m = t.delta && t.delta.length ? mean(t.delta) : null;
      const set = Object.entries(t.settings || {}).map(([k, v]) => `${esc(k)}=${esc(v)}`).join(" · ");
      const wb = (t.wandb || []).map((w) => `<a href="https://wandb.ai/marin-community/marin_moe/runs/${encodeURIComponent(w)}" target="_blank" rel="noopener">${esc(w)}</a>`).join("<br>");
      return `<tr>
        <td class="d">${esc(t.date || "")}<br>${t.comment_url ? `<a href="${esc(t.comment_url)}" target="_blank" rel="noopener">b${esc(t.batch || "")}</a>` : esc(t.batch || "")}</td>
        <td>${esc(t.variant || "")}${set ? `<span class="set" style="display:block">${set}</span>` : ""}${t.tuned ? `<span class="tuned">tuned: ${esc(t.tuned)}</span>` : ""}${t.note ? `<span class="nt2">${esc(t.note)}</span>` : ""}</td>
        <td class="d">${esc(t.rung || "")}<br><span style="color:var(--faint)">${esc((t.protocol || "").replace("_", " "))}${t.steps ? ` · ${t.steps}` : ""}</span></td>
        <td style="color:var(--slate)">${esc(t.baseline || "")}</td>
        <td class="d ${m === null ? "" : cls(m)}">${t.delta && t.delta.length ? t.delta.map(fmtD).join(" / ") : "–"}${t.derived ? ' <span title="computed from reported values" style="color:var(--faint)">*</span>' : ""}</td>
        <td class="d">${t.loss && t.loss.length ? t.loss.map((x) => x.toFixed(4)).join(" / ") : "–"}${t.step_time_change ? `<br><span style="color:var(--amber)">${esc(t.step_time_change)}</span>` : ""}</td>
        <td class="d" style="font-size:11.5px">${wb || "–"}</td></tr>`;
    }).join("");
    return `<p>${esc(i.summary || "")}</p>
      <div class="dgrid">
        <div class="dsec"><h4>Spec${i.xid ? ` · <a class="xlink" href="https://github.com/marin-community/marin/blob/marin_belay_v1_registry/experiments/belay/registry/${esc(i.xid)}-${esc(i.id)}.yaml" target="_blank" rel="noopener">${esc(i.xid)} in the registry</a>` : ""}</h4>${spec}${knobs}${papers}</div>
        <div class="dsec"><h4>Timeline</h4><ol class="tl">${ev}</ol></div>
      </div>
      <div class="dsec"><h4>Trials · newest first · Δ per seed vs the named baseline, negative is better</h4>
        <div class="trials-wrap"><table class="trials"><thead><tr><th>Date</th><th>Variant and settings</th><th>Rung</th><th>Baseline</th><th>Δ BPB</th><th>BPB</th><th>W&amp;B</th></tr></thead><tbody>${rows || '<tr><td colspan="7" style="color:var(--faint)">No trials with numbers.</td></tr>'}</tbody></table></div>
      </div>`;
  }
  function drawResults() {
    const st = ["in", "won", "noise", "lost", "removed", "tested"];
    chipRow("f-status", st.map((k) => [k, STATUS[k][0], RD.ideas.filter((i) => i.status === k && matches(i, "status")).length, STATUS[k][1]]).filter((x) => x[2] || F.status === x[0]), "status");
    chipRow("f-cat", RD.categories.map((c) => [c, c, RD.ideas.filter((i) => i.category === c && matches(i, "cat")).length]).filter((x) => x[2] || F.cat === x[0]), "cat");
    chipRow("f-reach", Object.entries(REACH).map(([k, [lab, fn]]) => [k, lab, RD.ideas.filter((i) => fn(i) && matches(i, "reach")).length]), "reach");
    $("sort").value = F.sort;
    const hits = RD.ideas.filter((i) => matches(i));
    $("res-count").textContent = `${hits.length} of ${RD.ideas.length}`;
    const box = $("groups"); box.innerHTML = "";
    if (!hits.length) { box.innerHTML = `<div class="empty">No ideas match. Clear a filter or try another search term.</div>`; return; }
    RD.categories.forEach((c) => {
      const g = hits.filter((i) => i.category === c).sort(sorters[F.sort]);
      if (!g.length) return;
      const sec = document.createElement("div"); sec.className = "group";
      sec.innerHTML = `<h3>${esc(c)} <span>${g.length}</span></h3><div class="list"><div class="list-head"><span>Idea</span><span>Status</span><span>Best Δ</span><span class="rh"><span>512</span><span>768</span><span>1024</span><span>1280</span></span><span>Trials</span><span>Last</span></div>${g.map((i) => `<div class="row" id="idea-${esc(i.id)}">${rowHtml(i)}</div>`).join("")}</div>`;
      box.appendChild(sec);
    });
  }
  $("groups").addEventListener("click", (ev) => {
    const cp = ev.target.closest(".copy");
    if (cp) { const t = cp.dataset.copy; (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => { cp.textContent = "Copied"; }).catch(() => { cp.textContent = t; }); return; }
    const btn = ev.target.closest(".row > button"); if (!btn) return;
    const row = btn.parentElement, id = row.id.slice(5), i = RD.ideas.find((x) => x.id === id);
    if (open.has(id)) open.delete(id); else open.add(id);
    row.innerHTML = rowHtml(i);
    row.querySelector("button").focus();
  });
  let qt; $("q").addEventListener("input", (e) => { clearTimeout(qt); qt = setTimeout(() => { F.q = e.target.value.trim(); drawResults(); }, 120); });
  $("sort").addEventListener("change", (e) => { F.sort = e.target.value; saveF(); drawResults(); });
  function openFromHash() {
    const h = location.hash.slice(1); if (!h.startsWith("idea-")) return;
    const id = h.slice(5); if (!RD.ideas.some((i) => i.id === id)) return;
    F.status = F.cat = F.reach = null; F.q = ""; $("q").value = ""; open.add(id); drawResults();
    const n = document.getElementById(h); if (n) n.scrollIntoView({ block: "center" });
  }
  window.addEventListener("hashchange", openFromHash);

  const draw = () => { drawProgress(); drawScaling(); };
  let progressFrame = 0;
  new ResizeObserver(() => { cancelAnimationFrame(progressFrame); progressFrame = requestAnimationFrame(drawProgress); }).observe($("box-progress"));
  drawResults(); draw(); openFromHash();
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener && mq.addEventListener("change", draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}

function renderLatest(status) {
  const recent = status.ideas.find((i) => i.result)
  const live = status.runs.filter((r) => !TERMINAL.has(r.state)).length
  const parts = []
  if (recent) {
    const row = (recent.result.rows || []).find((r) => !r.control)
    parts.push(el('b', {}, recent.registry_id ?? ''), ` ${recent.title}`, row ? [' ', delta(row.delta)] : null)
  }
  parts.push(el('span', { class: 'latest-n' }, `${live} running`))
  parts.push(el('span', { class: 'latest-n' }, `${status.pipeline.pending} waiting`))
  setChildren('latest-body', ...parts)
  document.getElementById('latest').hidden = false
}

async function front() {
  showView('front')
  // The section links sit in the top bar, as on the original page, before the page links.
  const sections = document.querySelector('.secnav')
  sections.className = 'nav'
  document.querySelector('.topbar .wrap').insertBefore(sections, document.getElementById('pages'))
  const scalingUrl = STATIC ? 'scaling.json' : `${BELAY_BASE}scaling.json`
  const [results, status, scaling] = await Promise.all([
    loadJson('results'),
    loadJson('state').catch(() => null),
    fetch(scalingUrl).then((r) => r.json()),
  ])
  startFront(results, scaling)
  if (status) {
    state = status
    renderLatest(status)
  }
}

front()
