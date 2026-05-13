# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML viewer for the human-apply pilot.

Implements the §7.3.3 SYNTHESIZED PROTOCOL: present anonymized judge cards on
DisagreeMine-selected contested cells (plus easy anchor cells) to a human
reviewer. The reviewer picks the best judge per cell, optionally marks the call
as clear/close, and writes a 1-sentence reason. Picks persist to browser
localStorage and can be exported as JSONL for the analyzer.

Usage:
    python e9_human_apply_viewer.py \\
        --statements no_agenda protect_privileged_messages avoid_abuse \\
        --output dart_run10/human_apply_viewer.html

Or to include all 15 canonical Bucket D statements:
    python e9_human_apply_viewer.py --bucket-d \\
        --output dart_run10/human_apply_viewer.html

The output HTML is opened in a browser. State persists across reloads via
localStorage keyed on the file's session_id (regenerated each `--output` run
unless --session-id is set explicitly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

# Canonical 3-judge ensemble.
JUDGES = ("gpt", "gemini-pro", "claude")
JUDGE_DISPLAY = {"gpt": "GPT-5.1", "gemini-pro": "Gemini-3.1-Pro", "claude": "Claude Sonnet 4.6"}

# Canonical Bucket D (Run 8 80-cell ensemble) — §1.1 of dart.md.
BUCKET_D_15 = [
    "avoid_abuse",
    "comply_with_laws",
    "do_not_lie",
    "no_agenda",
    "be_clear",
    "sexual_content_involving_minors",
    "no_erotica_or_gore",
    "assume_objective_pov",
    "no_topic_off_limits",
    "formatting",
    "protect_privileged_messages",
    "prevent_imminent_harm",
    "avoid_hateful_content",
    "assume_best_intentions",
    "highlight_misalignments",
]

# Generator string → response source file + field.
GEN_TO_SOURCE = {
    "gpt-5.1": ("e8_responses.jsonl", "response_gpt"),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ("e8_responses.jsonl", "response_weak"),
    "gemini-3-flash-preview": ("e8_responses.jsonl", "response_gemini"),
    "grok-4-1-fast-non-reasoning-opposite": ("e9_opposite_mode_responses.jsonl", "response"),
}

CONDITION = "rubric_plus_spec"  # Canonical phase_4 condition.
N_CONTESTED = 15
N_EASY = 5


def load_judgments(judgments_dir: Path) -> dict:
    """Return {(sid, scen, gen, judge): {score, reasoning}} for canonical 3-judge ensemble.

    Combines per_judgment_opposite.jsonl (GPT + Claude rows; drops gemini-flash)
    and per_judgment_pro_audit.jsonl (Pro rows). Filters to rubric_plus_spec
    condition.
    """
    j: dict = {}
    counts = {"gpt": 0, "gemini-pro": 0, "claude": 0, "dropped": 0}

    opp_path = judgments_dir / "per_judgment_opposite.jsonl"
    with opp_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("condition") != CONDITION:
                continue
            judge = d.get("judge")
            if judge in ("gpt", "claude"):
                key = (d["statement_id"], d["scenario_idx"], d["generator"], judge)
                j[key] = {"score": d.get("score"), "reasoning": d.get("reasoning", "")}
                counts[judge] += 1
            else:
                counts["dropped"] += 1

    pro_path = judgments_dir / "per_judgment_pro_audit.jsonl"
    with pro_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("condition") != CONDITION:
                continue
            judge = d.get("judge")
            if judge == "gemini-pro":
                key = (d["statement_id"], d["scenario_idx"], d["generator"], judge)
                j[key] = {"score": d.get("score"), "reasoning": d.get("reasoning", "")}
                counts["gemini-pro"] += 1

    print(
        f"  loaded judgments: gpt={counts['gpt']}, pro={counts['gemini-pro']}, "
        f"claude={counts['claude']}, dropped(flash)={counts['dropped']}",
        file=sys.stderr,
    )
    return j


def load_responses(judgments_dir: Path) -> dict:
    """Return {(sid, scen, gen): {"user_query": ..., "response": ...}}."""
    r: dict = {}
    with (judgments_dir / "e8_responses.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            sid, scen = d["statement_id"], d["scenario_idx"]
            uq = d.get("user_query", "")
            for gen_key, field in [
                ("gpt-5.1", "response_gpt"),
                ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
                ("gemini-3-flash-preview", "response_gemini"),
            ]:
                resp = d.get(field, "")
                r[(sid, scen, gen_key)] = {"user_query": uq, "response": resp}
    with (judgments_dir / "e9_opposite_mode_responses.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            sid, scen = d["statement_id"], d["scenario_idx"]
            r[(sid, scen, d["generator"])] = {
                "user_query": d.get("user_query", ""),
                "response": d.get("response", ""),
            }
    print(f"  loaded responses: {len(r)} cells", file=sys.stderr)
    return r


def _load_phantom_full_spec(judgments_dir: Path) -> dict:
    """Load the canonical phantom_full_spec.jsonl that has all statement texts.

    Searches under repair_v0/ for any phantom_full_spec.jsonl. Returns {sid: spec_dict}.
    """
    repair_root = judgments_dir / "repair_v0"
    if not repair_root.exists():
        return {}
    for path in sorted(repair_root.rglob("phantom_full_spec.jsonl")):
        result: dict = {}
        try:
            for line in path.open():
                d = json.loads(line)
                result[d.get("id")] = d
            if result:
                return result
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def load_statement_texts(judgments_dir: Path, sids: list[str]) -> dict:
    """Return {sid: {"text": ..., "examples": [...]}}.

    Prefers dart_run10/round_1/{sid}/spec_with_examples_v10.json, falls back to
    dart_run9/{sid}/spec_with_examples_v9.json, then to the canonical
    phantom_full_spec.jsonl that has all 46 statements.
    """
    phantom = _load_phantom_full_spec(judgments_dir)
    texts = {}
    for sid in sids:
        candidates = [
            judgments_dir / "dart_run10" / "round_1" / sid / "spec_with_examples_v10.json",
            judgments_dir / "dart_run9" / sid / "spec_with_examples_v9.json",
        ]
        loaded = False
        for cand in candidates:
            if cand.exists():
                d = json.loads(cand.read_text())
                text = d.get("text") or d.get("v1_text") or d.get("v0_text") or ""
                examples = (d.get("metadata") or {}).get("examples", []) or []
                texts[sid] = {"text": text, "examples": examples, "source": cand.name}
                loaded = True
                break
        if not loaded and sid in phantom:
            d = phantom[sid]
            texts[sid] = {
                "text": d.get("text", ""),
                "examples": (d.get("metadata") or {}).get("examples", []) or [],
                "source": "phantom_full_spec.jsonl",
            }
            loaded = True
        if not loaded:
            print(f"  WARN: no spec text found for {sid}", file=sys.stderr)
            texts[sid] = {"text": "(spec text not found)", "examples": [], "source": None}
    return texts


def pwv(scores: list[int | None]) -> int:
    """Pairwise variance across 3 scores. Returns -1 if any missing."""
    if any(s is None for s in scores):
        return -1
    return sum((s_i - s_j) ** 2 for i, s_i in enumerate(scores) for s_j in scores[i + 1 :])


def select_cells_for_statement(sid: str, judgments: dict, responses: dict, rng: random.Random) -> dict:
    """Apply DisagreeMine + scenario dedup, pick contested + easy cells.

    Returns:
        {"contested": [cell_dict, ...], "easy": [cell_dict, ...]}
    where each cell_dict has scen, generator, scores per judge, pwv, user_query,
    response, anonymized_mapping (label->judge), judge_panels (in label order).
    """
    cells_with_pwv = []
    for scen in range(20):
        for gen in GEN_TO_SOURCE:
            scores = []
            reasonings = {}
            missing = False
            for judge in JUDGES:
                jrec = judgments.get((sid, scen, gen, judge))
                if jrec is None or jrec.get("score") is None:
                    missing = True
                    break
                scores.append(jrec["score"])
                reasonings[judge] = jrec.get("reasoning", "")
            if missing:
                continue
            resp = responses.get((sid, scen, gen))
            if resp is None or not resp.get("response"):
                continue
            cells_with_pwv.append(
                {
                    "scen": scen,
                    "generator": gen,
                    "scores": dict(zip(JUDGES, scores, strict=True)),
                    "reasonings": reasonings,
                    "pwv": pwv(scores),
                    "user_query": resp["user_query"],
                    "response": resp["response"],
                }
            )

    cells_with_pwv.sort(key=lambda c: -c["pwv"])

    contested_deduped = []
    seen_scen = set()
    for c in cells_with_pwv:
        if c["pwv"] <= 0:
            break
        if c["scen"] in seen_scen:
            continue
        seen_scen.add(c["scen"])
        contested_deduped.append(c)
        if len(contested_deduped) >= N_CONTESTED:
            break

    easy_pool = [c for c in cells_with_pwv if c["pwv"] == 0 and c["scen"] not in seen_scen]
    rng.shuffle(easy_pool)
    easy = []
    for c in easy_pool:
        if c["scen"] not in seen_scen:
            easy.append(c)
            seen_scen.add(c["scen"])
            if len(easy) >= N_EASY:
                break

    def anonymize(cells: list[dict], cell_type: str) -> list[dict]:
        anonymized = []
        for c in cells:
            labels = ["A", "B", "C"]
            rng.shuffle(labels)
            mapping = dict(zip(labels, JUDGES, strict=True))
            panels = []
            for label in ["A", "B", "C"]:
                judge = mapping[label]
                panels.append(
                    {
                        "label": label,
                        "score": c["scores"][judge],
                        "reasoning": c["reasonings"][judge],
                    }
                )
            anonymized.append(
                {
                    "cell_id": f"{sid}__scen{c['scen']:02d}__{c['generator']}",
                    "cell_type": cell_type,
                    "scen": c["scen"],
                    "generator": c["generator"],
                    "pwv": c["pwv"],
                    "user_query": c["user_query"],
                    "response": c["response"],
                    "judge_mapping": mapping,
                    "panels": panels,
                }
            )
        return anonymized

    return {
        "contested": anonymize(contested_deduped, "contested"),
        "easy": anonymize(easy, "easy"),
    }


def build_data_payload(sids: list[str], judgments_dir: Path, seed: int) -> dict:
    judgments = load_judgments(judgments_dir)
    responses = load_responses(judgments_dir)
    statement_texts = load_statement_texts(judgments_dir, sids)
    rng = random.Random(seed)

    statements = []
    for sid in sids:
        cells = select_cells_for_statement(sid, judgments, responses, rng)
        all_cells = cells["contested"] + cells["easy"]
        rng.shuffle(all_cells)
        statements.append(
            {
                "statement_id": sid,
                "spec_text": statement_texts[sid]["text"],
                "spec_examples": statement_texts[sid]["examples"],
                "n_contested": len(cells["contested"]),
                "n_easy": len(cells["easy"]),
                "cells": all_cells,
            }
        )

    return {
        "session_id": hashlib.sha256(f"{seed}-{','.join(sids)}".encode()).hexdigest()[:12],
        "seed": seed,
        "judges_in_ensemble": list(JUDGES),
        "judge_display_names_for_reveal": JUDGE_DISPLAY,
        "statements": statements,
        "thresholds_pre_registered": {
            "adopt_p_best": 0.85,
            "tentative_p_best": 0.65,
            "posterior_mean_floor_adopt": 0.50,
            "posterior_mean_floor_tentative": 0.40,
            "spec_broken_p_none_threshold": 0.30,
            "spec_broken_posterior_prob": 0.50,
            "fatigue_ratio": 0.60,
            "multimodal_within_cluster_dominance": 0.50,
            "kappa_good": 0.60,
            "kappa_tentative": 0.40,
            "retest_consistency": 0.80,
            "adaptive_n_early_stop": 0.90,
            "adaptive_n_expand": 0.70,
        },
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>DART Human-Apply Pilot Viewer</title>
<style>
  :root {
    --bg: #0f1115; --panel: #161922; --panel-2: #1d2230; --border: #2a3142;
    --text: #e6e8ec; --muted: #8e95a6; --accent: #6aa9ff; --accent-2: #a78bfa;
    --good: #34d399; --warn: #fbbf24; --bad: #f87171;
    --selected: #1e3a5f; --selected-border: #6aa9ff;
  }
  * { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--text); margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    line-height: 1.45; font-size: 14px; }
  .top-bar { padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--border);
    display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .top-bar h1 { font-size: 16px; margin: 0; color: var(--accent); }
  .top-bar select { background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    padding: 6px 10px; border-radius: 4px; font-size: 13px; }
  .top-bar button { background: var(--accent); color: var(--bg); border: none;
    padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: 600; }
  .top-bar button:hover { background: var(--accent-2); }
  .top-bar button.secondary { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); }
  .progress { color: var(--muted); font-size: 12px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  .spec-box { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 16px; margin-bottom: 20px; }
  .spec-box details { margin-top: 8px; }
  .spec-box summary { cursor: pointer; color: var(--accent); font-size: 12px; }
  .spec-text { color: var(--text); white-space: pre-wrap; }
  .cell-box { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 20px; margin-bottom: 20px; }
  .cell-header { color: var(--muted); font-size: 11px; margin-bottom: 12px;
    display: flex; gap: 16px; justify-content: space-between; }
  .qa-block { background: var(--panel-2); padding: 12px; border-radius: 4px; margin-bottom: 16px;
    border-left: 3px solid var(--accent-2); }
  .qa-block .label { color: var(--muted); font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }
  .qa-block .body { white-space: pre-wrap; word-wrap: break-word; max-height: 320px; overflow-y: auto; }
  .judges-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
  .judge-card { background: var(--panel-2); border: 2px solid var(--border); border-radius: 6px;
    padding: 14px; cursor: pointer; transition: border-color 0.15s; }
  .judge-card:hover { border-color: var(--accent); }
  .judge-card .judge-label { font-weight: 700; font-size: 18px; color: var(--accent); margin-bottom: 6px; }
  .judge-card .judge-score { display: inline-block; padding: 2px 8px; background: var(--bg);
    border-radius: 12px; font-size: 12px; color: var(--good); margin-bottom: 8px; }
  .judge-card .judge-reasoning { white-space: pre-wrap; word-wrap: break-word;
    max-height: 360px; overflow-y: auto; font-size: 13px; color: var(--text); }
  .picks-section { background: var(--panel-2); padding: 16px; border-radius: 6px; margin-bottom: 16px; }
  .picks-section h3 { margin: 0 0 12px 0; font-size: 13px; color: var(--accent); text-transform: uppercase; }
  .picks-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .pick-btn { background: var(--panel); color: var(--text); border: 2px solid var(--border);
    padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; }
  .pick-btn:hover { border-color: var(--accent); }
  .pick-btn.selected { background: var(--selected); border-color: var(--selected-border); color: var(--text); }
  .pick-btn.none-btn { color: var(--warn); }
  .pick-btn.tie-btn { color: var(--accent-2); }
  .meta-row { display: flex; gap: 16px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  .meta-row label { color: var(--muted); font-size: 12px; display: flex; align-items: center; gap: 6px; }
  .meta-row input[type=checkbox] { cursor: pointer; }
  .reason-row textarea { width: 100%; min-height: 50px; background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px; padding: 8px; font-family: inherit; font-size: 13px;
    resize: vertical; }
  .nav-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; }
  .nav-row button { padding: 8px 16px; }
  .save-status { font-size: 11px; color: var(--muted); }
  .save-status.saved { color: var(--good); }
  .completion-banner { background: var(--good); color: var(--bg); padding: 12px; border-radius: 6px;
    margin: 20px 0; font-weight: 600; }
  .empty-msg { color: var(--muted); padding: 40px; text-align: center; }
  .timing-debug { color: var(--muted); font-size: 11px; font-style: italic; }
  .cell-type-easy { border-left: 3px solid var(--good); }
  .cell-type-easy::before { content: ""; }
  /* Easy-cell visual marker is intentionally subtle; participants shouldn't pay attention to it. */
</style>
</head>
<body>
  <div class="top-bar">
    <h1>DART Human-Apply Pilot</h1>
    <select id="statement-select"></select>
    <span class="progress" id="progress"></span>
    <span class="save-status" id="save-status"></span>
    <button class="secondary" id="export-btn">Export picks JSONL</button>
    <button class="secondary" id="reset-btn" title="Clear local state for this statement">Reset statement</button>
  </div>
  <div class="container" id="main"></div>

<script>
const DATA = __DATA_PAYLOAD__;
const SESSION_KEY = "dart_human_apply::" + DATA.session_id;
const TIE_LABELS = {
  "TIE_AB": "Tie A & B",
  "TIE_AC": "Tie A & C",
  "TIE_BC": "Tie B & C",
  "TIE_ABC": "Tie all three",
};
const PICK_OPTIONS = ["A", "B", "C", "TIE_AB", "TIE_AC", "TIE_BC", "TIE_ABC", "NONE"];

function loadState() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return {picks: {}, current: {}};
    const s = JSON.parse(raw);
    if (!s.picks) s.picks = {};
    if (!s.current) s.current = {};
    return s;
  } catch (e) { return {picks: {}, current: {}}; }
}
function saveState(s) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(s));
  const el = document.getElementById("save-status");
  el.textContent = "✓ saved";
  el.classList.add("saved");
  setTimeout(() => el.classList.remove("saved"), 1200);
}
let STATE = loadState();

function getCurrentSid() {
  const sel = document.getElementById("statement-select");
  return sel.value;
}
function getStatement(sid) {
  return DATA.statements.find(s => s.statement_id === sid);
}
function getCellIdx(sid) {
  return STATE.current[sid] || 0;
}
function setCellIdx(sid, idx) {
  STATE.current[sid] = idx;
  saveState(STATE);
}

function renderStatementSelect() {
  const sel = document.getElementById("statement-select");
  sel.innerHTML = "";
  for (const s of DATA.statements) {
    const opt = document.createElement("option");
    const n_done = countPicks(s.statement_id);
    const n_total = s.cells.length;
    opt.value = s.statement_id;
    opt.textContent = `${s.statement_id} (${n_done}/${n_total})`;
    sel.appendChild(opt);
  }
  sel.onchange = render;
}

function countPicks(sid) {
  let n = 0;
  for (const c of getStatement(sid).cells) {
    if (STATE.picks[`${sid}::${c.cell_id}`]) n++;
  }
  return n;
}

function renderProgress(sid) {
  const s = getStatement(sid);
  const n_done = countPicks(sid);
  document.getElementById("progress").textContent =
    `cell ${getCellIdx(sid) + 1} of ${s.cells.length}, ${n_done} picked, ${s.cells.length - n_done} remaining`;
}

function render() {
  const sid = getCurrentSid();
  const s = getStatement(sid);
  const main = document.getElementById("main");
  const idx = getCellIdx(sid);
  if (idx >= s.cells.length) {
    main.innerHTML = `
      <div class="completion-banner">
        ✓ All ${s.cells.length} cells picked for <code>${sid}</code>.
        Export the JSONL via the top bar to run the analyzer.
      </div>
      <div class="spec-box">
        <strong>Spec statement:</strong>
        <div class="spec-text">${escapeHtml(s.spec_text)}</div>
        ${renderExamples(s.spec_examples)}
      </div>
      <button class="secondary" onclick="setCellIdx('${sid}', 0); render();">Review picks from start</button>
    `;
    renderProgress(sid);
    renderStatementSelect();
    return;
  }
  const c = s.cells[idx];
  const pickKey = `${sid}::${c.cell_id}`;
  const pick = STATE.picks[pickKey] || {};
  main.innerHTML = `
    <div class="spec-box">
      <strong>Spec statement:</strong> <code>${sid}</code>
      <div class="spec-text">${escapeHtml(s.spec_text)}</div>
      ${renderExamples(s.spec_examples)}
    </div>
    <div class="cell-box ${c.cell_type === 'easy' ? 'cell-type-easy' : ''}">
      <div class="cell-header">
        <span>cell ${idx + 1} of ${s.cells.length}</span>
        <span class="timing-debug">id: ${c.cell_id}</span>
      </div>
      <div class="qa-block">
        <div class="label">User prompt</div>
        <div class="body">${escapeHtml(c.user_query)}</div>
      </div>
      <div class="qa-block">
        <div class="label">Model response</div>
        <div class="body">${escapeHtml(c.response)}</div>
      </div>
      <div class="judges-grid">
        ${c.panels.map(p => `
          <div class="judge-card" onclick="selectJudge('${pickKey}', '${p.label}')">
            <div class="judge-label">Judge ${p.label}</div>
            <div class="judge-score">score: ${p.score}/5</div>
            <div class="judge-reasoning">${escapeHtml(p.reasoning)}</div>
          </div>
        `).join("")}
      </div>
      <div class="picks-section">
        <h3>Pick the judge whose reasoning + score best matches the spec's intent</h3>
        <div class="picks-grid">
          ${PICK_OPTIONS.map(opt => {
            const isSelected = pick.choice === opt;
            const klass = opt === "NONE" ? "pick-btn none-btn" :
                          opt.startsWith("TIE") ? "pick-btn tie-btn" : "pick-btn";
            const label = opt === "NONE" ? "None of these fit" : (TIE_LABELS[opt] || `Judge ${opt}`);
            return `<button class="${klass} ${isSelected ? 'selected' : ''}"
                       onclick="setPick('${pickKey}', '${opt}')">${label}</button>`;
          }).join("")}
        </div>
        <div class="meta-row">
          <label>
            <input type="checkbox" id="clear-call" ${pick.clear_or_close === 'close' ? '' : 'checked'}
                   onchange="setClearClose('${pickKey}', this.checked)" />
            <span>This was a clear call (uncheck if close)</span>
          </label>
        </div>
        <div class="reason-row">
          <div class="label" style="color: var(--muted); font-size: 11px; margin-bottom: 4px;">
            Why? (1 sentence — required even if 5 words)
          </div>
          <textarea id="reason-text"
                    placeholder="e.g., 'Judge A correctly flagged framing-as-laundering'"
                    onblur="setReason('${pickKey}', this.value)">${escapeHtml(pick.reason || "")}</textarea>
        </div>
      </div>
      <div class="nav-row">
        <button class="secondary" onclick="prevCell()">← Prev</button>
        <span>${idx + 1} / ${s.cells.length}</span>
        <button onclick="nextCell()">Next →</button>
      </div>
    </div>
  `;
  renderProgress(sid);
  renderStatementSelect();
  if (!pick.shown_timestamp) {
    pick.shown_timestamp = Date.now();
    STATE.picks[pickKey] = pick;
    saveState(STATE);
  }
}

function renderExamples(examples) {
  if (!examples || examples.length === 0) return "";
  return `<details><summary>Spec metadata.examples (${examples.length})</summary>
    <div style="margin-top: 8px;">
      ${examples.map((ex, i) => `
        <div style="border-left: 2px solid var(--border); padding-left: 10px; margin-bottom: 8px;">
          <div style="color: var(--muted); font-size: 11px;">example ${i + 1}</div>
          <div><strong>User:</strong> ${escapeHtml(ex.user_query || "")}</div>
          <div><strong>Good:</strong> ${escapeHtml(ex.good_response || "")}</div>
          <div><strong>Bad:</strong> ${escapeHtml(ex.bad_response || "")}</div>
          ${ex.description ? `<div style="color: var(--muted);"><em>${escapeHtml(ex.description)}</em></div>` : ""}
        </div>
      `).join("")}
    </div>
  </details>`;
}

function selectJudge(pickKey, label) { setPick(pickKey, label); }
function setPick(pickKey, choice) {
  const p = STATE.picks[pickKey] || {};
  p.choice = choice;
  p.pick_timestamp = Date.now();
  STATE.picks[pickKey] = p;
  saveState(STATE);
  render();
}
function setClearClose(pickKey, isClear) {
  const p = STATE.picks[pickKey] || {};
  p.clear_or_close = isClear ? "clear" : "close";
  STATE.picks[pickKey] = p;
  saveState(STATE);
}
function setReason(pickKey, reason) {
  const p = STATE.picks[pickKey] || {};
  p.reason = reason;
  STATE.picks[pickKey] = p;
  saveState(STATE);
}
function nextCell() {
  const sid = getCurrentSid();
  const s = getStatement(sid);
  setCellIdx(sid, Math.min(getCellIdx(sid) + 1, s.cells.length));
  render();
}
function prevCell() {
  const sid = getCurrentSid();
  setCellIdx(sid, Math.max(getCellIdx(sid) - 1, 0));
  render();
}

function exportJsonl() {
  const lines = [];
  for (const s of DATA.statements) {
    for (const c of s.cells) {
      const pickKey = `${s.statement_id}::${c.cell_id}`;
      const pick = STATE.picks[pickKey];
      if (!pick || !pick.choice) continue;
      lines.push(JSON.stringify({
        session_id: DATA.session_id,
        statement_id: s.statement_id,
        cell_id: c.cell_id,
        cell_type: c.cell_type,
        scen: c.scen,
        generator: c.generator,
        pwv: c.pwv,
        judge_mapping: c.judge_mapping,
        scores: Object.fromEntries(c.panels.map(p => [p.label, p.score])),
        pick: pick.choice,
        clear_or_close: pick.clear_or_close || "clear",
        reason: pick.reason || "",
        shown_timestamp: pick.shown_timestamp || null,
        pick_timestamp: pick.pick_timestamp || null,
      }));
    }
  }
  if (lines.length === 0) { alert("No picks to export yet."); return; }
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "application/jsonl"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `human_apply_picks_${DATA.session_id}.jsonl`;
  a.click();
}

function resetStatement() {
  const sid = getCurrentSid();
  if (!confirm(`Clear all picks for ${sid}? This cannot be undone.`)) return;
  for (const c of getStatement(sid).cells) {
    delete STATE.picks[`${sid}::${c.cell_id}`];
  }
  STATE.current[sid] = 0;
  saveState(STATE);
  render();
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

document.getElementById("export-btn").onclick = exportJsonl;
document.getElementById("reset-btn").onclick = resetStatement;
renderStatementSelect();
render();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--statements", nargs="+", help="Statement IDs to include")
    p.add_argument("--bucket-d", action="store_true", help="Use the canonical 15 Bucket D statements")
    p.add_argument("--output", "-o", required=True, help="Output HTML path")
    p.add_argument("--seed", type=int, default=42, help="Anonymization shuffle seed")
    p.add_argument(
        "--judgments-dir", default=str(SCRIPT_DIR), help="Directory containing per_judgment_*.jsonl and response files"
    )
    args = p.parse_args()

    if args.bucket_d:
        sids = BUCKET_D_15
    elif args.statements:
        sids = args.statements
    else:
        print("Specify --statements or --bucket-d", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.output)
    judgments_dir = Path(args.judgments_dir)

    print(f"Building viewer for {len(sids)} statements...", file=sys.stderr)
    data = build_data_payload(sids, judgments_dir, args.seed)

    n_contested_total = sum(s["n_contested"] for s in data["statements"])
    n_easy_total = sum(s["n_easy"] for s in data["statements"])
    print(
        f"  total cells: {n_contested_total} contested + {n_easy_total} easy " f"across {len(sids)} statements",
        file=sys.stderr,
    )

    payload_json = json.dumps(data, separators=(",", ":"))
    html_content = HTML_TEMPLATE.replace("__DATA_PAYLOAD__", payload_json)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content)
    print(f"  wrote {out_path} ({out_path.stat().st_size // 1024} KB; session_id={data['session_id']})", file=sys.stderr)


if __name__ == "__main__":
    main()
