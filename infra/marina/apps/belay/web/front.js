// Marin Belay front page: the project's public home. Charts, timeline, starter kit and tracks. The
// speedrun chart's test markers come from the registry (results.json, built with
// `experiments.belay.registry build`); stage 4's fits, points and hero budget come from scaling.json; the accepted
// records and tracks are measured values kept here. Loaded after common.js.

function startFront(RD, SC, SP, TK, RECIPE_TAG) {
  const LOSS_NOISE = TK.find((t) => t.track === "speedrun").noise;
  const SIGN = (v) => `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(4)}`;
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

  // ---------- Stage 2: the 8-minute d512 speedrun, on Paloma macro loss ----------
  // The accepted staircase comes from speedrun.json. Each successful d512 test is a drop below the Aug Hero Arch line by
  // its best mean macro-loss Δ (from the registry, ``loss_delta``), in the order the ideas were first tested, marked
  // by whether the live recipe holds it (its latest event for the recipe's checkpoint tag).
  // [name, mean Δ loss, in the live recipe, W&B run, batch, id]
  const inLive = (idea) => (idea.events || []).filter((e) => e.checkpoint === RECIPE_TAG && ["adopted", "removed"].includes(e.type)).at(-1)?.type === "adopted";
  const WINS = (() => {
    const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const wins = [];
    RD.ideas.forEach((idea, order) => {
      const d512 = idea.trials.filter((t) => t.rung === "d512" && t.loss_delta && t.loss_delta.length && !t.derived);
      if (!d512.length) return;
      const best = d512.reduce((a, b) => (avg(b.loss_delta) < avg(a.loss_delta) ? b : a));
      const m = avg(best.loss_delta);
      if (m > -LOSS_NOISE) return;
      const first = idea.trials.map((t) => t.date).filter(Boolean).sort()[0] || "";
      wins.push({ first, order, row: [idea.name, m, inLive(idea), (best.wandb || [])[0] || null, String(best.batch), idea.xid] });
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
    const [hero, current] = SP.accepted;
    const lows = [current.loss, ...WINS.map(([, d]) => hero.loss + d)];
    const lo = Math.floor((Math.min(...lows) - 0.01) * 50) / 50, hi = Math.ceil((hero.loss + 0.01) * 50) / 50;
    const Y = (l) => M.t + ((hi - l) / (hi - lo)) * (H - M.b - M.t);
    // x is ordinal: the Aug Hero Arch segment holds the tests evenly spaced, then the current recipe's segment.
    const n = WINS.length, x0 = M.l, x1 = W - M.r, xc = x1 - 0.14 * (x1 - x0);
    const xw = (i) => x0 + ((i + 1) / (n + 1)) * (xc - x0);
    for (let l = lo; l <= hi + 1e-9; l += 0.02) {
      el("line", { x1: x0, x2: x1, y1: Y(l), y2: Y(l), stroke: css("--line") }, svg);
      txt(svg, x0 - 8, Y(l) + 4, l.toFixed(2), { "text-anchor": "end" });
    }
    el("line", { x1: x0, x2: x1, y1: H - M.b, y2: H - M.b, stroke: css("--faint") }, svg);
    // Lollipops: from the Aug Hero Arch line down by each test's improvement.
    const yh = Y(hero.loss);
    WINS.forEach(([name, delta, inLiveRecipe, run, batch, xid], i) => {
      const cx = xw(i), cy = Y(hero.loss + delta), col = inLiveRecipe ? css("--rope") : css("--faint");
      el("line", { x1: cx, x2: cx, y1: yh, y2: cy, stroke: col, "stroke-width": inLiveRecipe ? 1.6 : 1.2, opacity: inLiveRecipe ? 0.8 : 0.7 }, svg);
      el("circle", { cx, cy, r: inLiveRecipe ? 3.6 : 3, fill: col, opacity: inLiveRecipe ? 0.95 : 0.85 }, svg);
      const link = run ? el("a", { href: WB(run), target: "_blank", rel: "noopener", style: "cursor:pointer" }, svg) : svg;
      const hit = el("rect", { x: cx - 5, y: yh, width: 10, height: Math.max(cy - yh + 6, 10), fill: "transparent", tabindex: run ? -1 : 0 }, link);
      hit.dataset.tip = `<b>${SIGN(delta)}</b> · ${esc(name)}${xid ? ` · ${xid}` : ""}<br>${inLiveRecipe ? "In" : "Not in"} the current recipe · best d512 trial, batch ${esc(batch)}<br><span style="opacity:.6">mean Δ Paloma macro loss vs its baseline${run ? ` · ${run} · click to open in W&amp;B` : ""}</span>`;
    });
    // The accepted staircase: Aug Hero Arch -> the current recipe.
    const segs = [[hero, x0, xc], [current, xc, x1]];
    el("path", { d: `M${x0},${Y(hero.loss)}H${xc}V${Y(current.loss)}H${x1}`, fill: "none", stroke: css("--ink"), "stroke-width": 1.4, opacity: 0.35, "stroke-linejoin": "round" }, svg);
    segs.forEach(([r, a, b], i) => {
      const cur = i === 1, col = css(r.color), y = Y(r.loss);
      el("line", { x1: a, x2: b, y1: y, y2: y, stroke: col, "stroke-width": cur ? 3.4 : 2.4 }, svg);
      el("circle", { cx: a, cy: y, r: cur ? 7 : 5.5, fill: col, stroke: css("--surface"), "stroke-width": 2 }, svg);
      const label = `${r.name} ${r.loss.toFixed(3)}`;
      if (cur) txt(svg, a - 12, y + 4, label, { "text-anchor": "end", fill: col, "font-weight": 700, "font-size": 13 });
      else txt(svg, a + 8, y - 8, label, { fill: css("--slate"), "font-weight": 600, "font-size": 11.5 });
      const link = el("a", { href: WB(r.run), target: "_blank", rel: "noopener", style: "cursor:pointer" }, svg);
      const hit = el("rect", { x: a - 6, y: y - 8, width: b - a + 12, height: 16, fill: "transparent", tabindex: -1 }, link);
      hit.dataset.tip = `<b>${r.loss.toFixed(4)}</b> Paloma macro loss · ${esc(r.name)}<br>d512 · 8 min on 8×H100 (${r.steps.toLocaleString()} steps) · BPB ${r.bpb.toFixed(4)} (${r.bpb_kind})<br><span style="opacity:.6">${r.run} · click to open in W&amp;B</span>`;
    });
    tipper("box-progress", "tip-progress", svg);
  }

  // ---------- Scaling laws (log y) ----------
  const RUNGS = ["d512", "d768", "d1024", "d1280"];
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
    eh.dataset.tip = `<b>${NX.toFixed(2)}×</b> less compute to the same BPB (${Lt.toFixed(3)})<br>${nc.name} at ${sci(C_EQ, 2)} vs ${bc.name} at ${sci(C_HW_BUDGET, 2)} hardware FLOPs<br><span style="opacity:.75">from the four-rung fits; each at its own 11-rack MFU<br>*~4,000× extrapolation, beyond the ~300× expected for a reliable forecast</span>`;
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
    { id: "vanilla", name: "Vanilla Track", tag: "dense · 20 tokens / param", desc: "A barebones Qwen3-style dense transformer (full causal GQA, RoPE, QK-norm, SwiGLU) trained with AdamW at a compute-scaled LR. For testing ideas with minimal confounders.", tpp: 20, runs: {"d512": [18881536, 377487360, 9.025e+16, 1.6205, 2.0], "d768": [56636928, 1132986368, 6.416e+17, 1.4146, 6.5], "d1024": [151021568, 3020947456, 3.954e+18, 1.3016, 28.3], "d1280": [275289600, 5506072576, 1.221e+19, 1.2326, 80.4]}, launchPrefix: "vanilla-", runPrefix: "belay-vanilla-" },
    { id: "moe", name: "Basic MoE", tag: "MoE · 60 tokens / active param", desc: "QB routing, top 4 of 128, 2 shared experts, MuonH. For testing MoE ideas with minimal confounders.", tpp: 60, runs: {"d512": [17219840, 1033371648, 2.237e+17, 1.3514, 6.9], "d768": [48473600, 2908225536, 1.422e+18, 1.2323, 26.1], "d1024": [128169472, 7690256384, 8.574e+18, 1.1336, 112.5], "d1280": [228123648, 13687062528, 2.54e+19, 1.0772, 294.8]}, launchPrefix: "moe-", runPrefix: "belay-moe-" },
    { id: "comp", launchPrefix: "competitive-", ...SC.competitive_track },
  ];
  const tc = $("track-cards");
  const mins = (m) => (m < 90 ? `${m.toFixed(0)} min` : `${(m / 60).toFixed(1)} h`);
  TRACKS.forEach((t) => {
    // Measured tracks: vanilla and basic MoE keep their rows here; the competitive track's come from scaling.json.
    const rowsData = t.rows ?? RUNGS.map((r) => {
      const [active, tokens, flops, bpb, minutes] = t.runs[r];
      return { rung: r, label: r, run: t.runPrefix + r, active, tokens, flops, bpb, minutes, gpus: 8 };
    });
    // Launch commands come from tracks.json, generated from config.TRACKS (`python -m experiments.belay tracks --json`).
    const launches = TK.filter((k) => k.track.startsWith(t.launchPrefix) && k.command)
      .map((k) => `<b>${k.track}</b> · ${k.code}\n${k.command}`).join("\n\n");
    const rows = rowsData.map((x) =>
      `<tr><td class="mono"><a class="wb" href="${WB(x.run)}" target="_blank" rel="noopener" title="${x.run} in W&amp;B">${x.label}</a>${x.note ? `<small>${x.note}</small>` : ""}</td><td class="r">${human(x.active)}</td><td class="r">${human(x.tokens)}</td><td class="r">${sci(x.flops)}</td><td class="r">${mins(x.minutes)}<small>on ${x.gpus}×H100</small></td><td class="r"><b>${x.bpb.toFixed(4)}</b></td></tr>`).join("");
    const card = document.createElement("article"); card.className = `track${t.id === "comp" ? " comp" : ""}`;
    card.innerHTML = `<header><div><div class="tag">${t.tag}</div><h3>${t.name}</h3></div></header>
      <p>${t.desc}</p>
      <div style="overflow-x:auto"><table class="rungs"><thead><tr><th>Rung</th><th class="r">Active</th><th class="r">Tokens</th><th class="r">FLOPs</th><th class="r">${t.rows ? "Wall clock" : "8×H100"}</th><th class="r">${t.rows ? "BPB" : "Baseline"}</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note" style="margin:0">${t.note ?? "Measured Paloma macro BPB; wall clock includes compile and eval."}</p>
      <details class="launch"><summary>Launch commands</summary><pre>${launches}</pre></details>`;
    tc.appendChild(card);
  });

  const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const draw = () => { drawProgress(); drawScaling(); };
  let progressFrame = 0;
  new ResizeObserver(() => { cancelAnimationFrame(progressFrame); progressFrame = requestAnimationFrame(drawProgress); }).observe($("box-progress"));
  draw();
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
  // The section links sit in the top bar, before the page links.
  const sections = document.querySelector('.secnav')
  sections.className = 'nav'
  document.querySelector('.topbar .wrap').insertBefore(sections, document.getElementById('pages'))
  const [results, status, scaling, speedrun, tracks] = await Promise.all([
    loadJson('results'),
    loadJson('state').catch((error) => {
      console.warn('Belay state unavailable; the latest-result strip stays hidden', error)
      return null
    }),
    fetch(asset('scaling.json')).then((r) => r.json()),
    fetch(asset('speedrun.json')).then((r) => r.json()),
    fetch(asset('tracks.json')).then((r) => r.json()),
  ])
  startFront(results, scaling, speedrun, tracks.tracks, tracks.recipe_tag)
  if (status) {
    renderLatest(status)
  }
}

front()
