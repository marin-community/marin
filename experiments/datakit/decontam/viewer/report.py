# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a decon run (or a two-run comparison) as a single self-contained HTML
report — the spot-check surface for the decon precision work.

    # single run: per-source flag table + click a row to read its flagged docs
    python experiments/datakit/decontam/viewer/report.py --run baseline.json --out report.html

    # compare two runs (e.g. before/after a fix): per-source Δ flag rate
    python experiments/datakit/decontam/viewer/report.py --run after.json --vs before.json --out cmp.html

Consumes the JSON exports from ``export_run.py``. All data is embedded and all
interactivity (sort / filter / expand) is client-side, so the file opens
anywhere and can be published as an Artifact.
"""

import argparse
import json

_CSS = """
*{box-sizing:border-box}body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#d7dbe0}
h1{font-size:18px;margin:0}.sub{color:#9aa4b2;font-size:12px;font-weight:400}
header{padding:12px 20px;background:#171a21;border-bottom:1px solid #262b34;position:sticky;top:0;z-index:5}
main{padding:14px 20px}input{background:#0b0d11;border:1px solid #2b3240;color:#d7dbe0;border-radius:5px;padding:5px 8px;width:240px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;border-bottom:1px solid #232833;text-align:left;vertical-align:top}
th{background:#141821;cursor:pointer;user-select:none;position:sticky;top:56px}
.num{text-align:right}.rate{font-weight:700}tr.src{cursor:pointer}tr.src:hover td{background:#161b24}
.chip{display:inline-block;background:#20303f;border:1px solid #2b4557;border-radius:10px;padding:0 7px;margin:1px;font-size:12px}
.docs{display:none;background:#0c0e12}.docs.open{display:table-row}
.doc{background:#141821;border:1px solid #262b34;border-radius:6px;padding:8px 10px;margin:8px 0}
.doc pre{white-space:pre-wrap;word-break:break-word;max-height:360px;overflow:auto;background:#0b0d11;padding:8px;border-radius:4px;margin:4px 0 0;font:12px/1.45 ui-monospace,monospace}
.meta{color:#9aa4b2;font-size:12px}.hi{color:#ffd479}.pos{color:#7bd88f}.neg{color:#ff6b6b}.zero{color:#6b7280}
.cols{display:flex;gap:12px;margin-top:8px}.col{flex:1 1 0;min-width:0}
.colh{color:#9aa4b2;font-size:12px;margin-bottom:2px;font-weight:600}
.evblock{margin-bottom:8px}.evh{color:#9aa4b2;font-size:12px;margin:4px 0 0}
.doc pre.ev{background:#0a1410}
mark{background:#8a6d00;color:#fff3cf;border-radius:2px;padding:0 1px}
@media(max-width:860px){.cols{flex-direction:column}}
</style>"""

_JS = """
<script>
function esc(t){return (t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function chips(fs){return (fs||[]).map(function(x){return '<span class=chip>'+esc(''+x[0])+' '+x[1]+'</span>'}).join(' ')}
// Highlight the overlapping n-gram spans inside `text`, preserving whitespace.
// Works at the token level so highlighting is robust to newlines/indentation:
// tokenize on whitespace, mark every token run equal to a matched n-gram, then
// coalesce (including the separators between two marked tokens) into <mark> spans.
function hl(text,ngrams){text=text||'';var parts=text.split(/(\\s+)/);
 var isTok=parts.map(function(p){return p.length>0&&!/^\\s+$/.test(p)});
 var vals=[],idxs=[];parts.forEach(function(p,i){if(isTok[i]){vals.push(p);idxs.push(i)}});
 var m=new Array(vals.length).fill(false);
 (ngrams||[]).forEach(function(ng){var g=ng.split(/\\s+/).filter(Boolean);if(!g.length)return;
  for(var s=0;s+g.length<=vals.length;s++){var ok=true;for(var j=0;j<g.length;j++){if(vals[s+j]!==g[j]){ok=false;break}}
   if(ok)for(var j=0;j<g.length;j++)m[s+j]=true}});
 var pm=new Array(parts.length).fill(false);idxs.forEach(function(pi,k){pm[pi]=m[k]});
 for(var i=1;i<parts.length-1;i++){if(!isTok[i]&&parts[i].length&&isTok[i-1]&&isTok[i+1]&&pm[i-1]&&pm[i+1])pm[i]=true}
 var out='',open=false;for(var i=0;i<parts.length;i++){
  if(pm[i]&&!open){out+='<mark>';open=true}if(!pm[i]&&open){out+='</mark>';open=false}out+=esc(parts[i])}
 if(open)out+='</mark>';return out}
function evcol(d){var ng=d.matched_ngrams||[];var e=(d.matched_evals||[]).map(function(e){
  return '<div class=evblock><div class=evh><span class=chip>'+esc(''+e.family)+' x'+e.hits+'</span> '+esc(e.eval_id)+
   '</div><pre class=ev>'+hl(e.text,ng)+'</pre></div>'}).join('');
 return e||'<span class=meta>(eval text unavailable)</span>'}
function toggle(i){var r=document.getElementById('d'+i);r.classList.toggle('open');
 if(r.dataset.filled)return;r.dataset.filled=1;var s=DATA.sources[i],h='';
 (s.samples||[]).forEach(function(d){var ng=d.matched_ngrams||[];
  h+='<div class=doc><span class=meta>overlap <span class=hi>'+d.max_overlap.toFixed(3)+
   '</span> · '+d.n_matched+' matched ngrams · '+esc(d.id)+' · '+chips(d.families)+'</span>'+
   '<div class=cols>'+
    '<div class=col><div class=colh>document · '+esc(s.name)+'</div><pre>'+hl(d.text,ng)+'</pre></div>'+
    '<div class=col><div class=colh>matched eval text</div>'+evcol(d)+'</div>'+
   '</div></div>'});
 r.cells[0].innerHTML=h||'<span class=meta>(no sampled docs)</span>'}
function sortBy(k){var S=DATA.sources.slice();var asc=(window._sk===k)?!window._asc:false;window._sk=k;window._asc=asc;
 S.sort(function(a,b){var x=a[k],y=b[k];if(typeof x==='string')return (x<y?-1:1)*(asc?1:-1);return (x-y)*(asc?1:-1)});
 render(S)}
function render(S){var f=(document.getElementById('flt').value||'').toLowerCase();var b='';
 S.forEach(function(s){var i=DATA.sources.indexOf(s);if(f&&s.name.toLowerCase().indexOf(f)<0)return;
  var rc=s.rate>=0.005?'neg':s.rate>0?'hi':'zero';
  b+='<tr class=src onclick=toggle('+i+')><td>'+esc(s.name)+'</td><td class=num>'+s.docs.toLocaleString()+
   '</td><td class=num>'+s.flagged.toLocaleString()+'</td><td class="num rate '+rc+'">'+(100*s.rate).toFixed(3)+
   '%</td><td>'+chips((s.top_families||[]).slice(0,6))+'</td></tr>'+
   '<tr class=docs id=d'+i+'><td colspan=5></td></tr>'});
 document.getElementById('tb').innerHTML=b}
document.addEventListener('DOMContentLoaded',function(){render(DATA.sources);document.getElementById('flt').oninput=function(){render(DATA.sources)}});
</script>"""


def _single(run: dict) -> str:
    td = sum(s["docs"] for s in run["sources"])
    tf = sum(s["flagged"] for s in run["sources"])
    rate = 100 * tf / td if td else 0
    head = (
        f"<header><h1>decon · {run['label']} "
        f"<span class=sub>target {run['target_tokens_b']}B · {len(run['sources'])} sources · "
        f"{tf:,}/{td:,} flagged = {rate:.4f}% · excl {', '.join(run['exclude']) or '—'}</span></h1>"
        f"<div style='margin-top:8px'><input id=flt placeholder='filter source…'></div></header>"
    )
    table = (
        "<table><thead><tr>"
        "<th onclick=sortBy('name')>source</th><th class=num onclick=sortBy('docs')>docs</th>"
        "<th class=num onclick=sortBy('flagged')>flagged</th><th class=num onclick=sortBy('rate')>rate</th>"
        "<th>top eval families</th></tr></thead><tbody id=tb></tbody></table>"
    )
    data = json.dumps(run).replace("</", "<\\/")  # script-safe embed
    return f"<!doctype html><meta charset=utf-8><title>decon {run['label']}</title><style>{_CSS}{head}<main>{table}</main><script>const DATA={data};</script>{_JS}"


def _compare(run: dict, base: dict) -> str:
    ma = {s["name"]: s for s in base["sources"]}
    mb = {s["name"]: s for s in run["sources"]}
    rows = []
    for name in set(ma) | set(mb):
        a, b = ma.get(name, {}), mb.get(name, {})
        rows.append(
            {
                "name": name,
                "ra": a.get("rate", 0),
                "rb": b.get("rate", 0),
                "d": b.get("rate", 0) - a.get("rate", 0),
                "fa": a.get("flagged", 0),
                "fb": b.get("flagged", 0),
            }
        )
    rows.sort(key=lambda r: -abs(r["d"]))
    body = ""
    for r in rows:
        cls = "neg" if r["d"] > 1e-9 else "pos" if r["d"] < -1e-9 else "zero"
        body += (
            f"<tr><td>{r['name']}</td><td class=num>{r['fa']:,}</td><td class=num>{r['fb']:,}</td>"
            f"<td class=num>{100*r['ra']:.3f}%</td><td class=num>{100*r['rb']:.3f}%</td>"
            f"<td class='num rate {cls}'>{100*r['d']:+.3f}%</td></tr>"
        )
    head = (
        f"<header><h1>decon compare <span class=sub>base=<b>{base['label']}</b> → run=<b>{run['label']}</b> · "
        f"<span class=pos>green = run flags fewer (FP removed)</span> · <span class=neg>red = more</span></span></h1></header>"
    )
    table = (
        f"<table><thead><tr><th>source</th><th class=num>{base['label']} n</th><th class=num>{run['label']} n</th>"
        f"<th class=num>{base['label']} %</th><th class=num>{run['label']} %</th><th class=num>Δ rate</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )
    return f"<!doctype html><meta charset=utf-8><title>decon compare</title><style>{_CSS}{head}<main>{table}</main>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--vs", default=None, help="base run JSON for a comparison report")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.run) as f:
        run = json.load(f)
    if args.vs:
        with open(args.vs) as f:
            html = _compare(run, json.load(f))
    else:
        html = _single(run)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
