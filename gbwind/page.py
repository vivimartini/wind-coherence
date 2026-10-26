"""The explorable results page (DESIGN.md section 8, deliverable 5).

One self contained HTML file with the data embedded. The default case is chosen
by the rule fixed in the execution note: among cases whose native portfolio
spread is above the period median, the one with the lowest observed output. The
rule never looks at coupling scores. Other low output cases and a random sample
are selectable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from gbwind import config

N_LOW, N_RANDOM = 25, 15


def _system_payload(sys_dir: Path, portfolio: pd.DataFrame) -> dict | None:
    summary = json.loads((sys_dir / "summary.json").read_text())
    if not summary.get("n_cases") or not (sys_dir / "couplings.zarr").exists():
        return None
    scores = pd.read_parquet(sys_dir / "scores.parquet")
    coupled = xr.open_zarr(sys_dir / "couplings.zarr").load()
    farms = [str(f) for f in coupled["farm"].values]

    native = scores[scores["coupling"] == "native"].reset_index(drop=True)
    uncertain = native["ens_sd_total"] > native["ens_sd_total"].median()
    low = native[uncertain].sort_values("obs_total").iloc[:N_LOW]
    rest = native.drop(low.index)
    rng = np.random.default_rng(config.SEED)
    pick = pd.concat([low, rest.iloc[rng.permutation(len(rest))[:N_RANDOM]]])

    case_key = {
        (pd.Timestamp(t), int(lead)): i
        for i, (t, lead) in enumerate(zip(coupled["init_time"].values, coupled["lead_h"].values, strict=True))
    }
    obs_lookup = scores.drop_duplicates(["init_time", "lead_h"]).set_index(["init_time", "lead_h"])
    per_case = scores.groupby(["init_time", "lead_h", "coupling"])["crps"].mean()

    cases = []
    for _, r in pick.iterrows():
        k = (pd.Timestamp(r["init_time"]), int(r["lead_h"]))
        if k not in case_key:
            continue
        arr = coupled["coupled_mw"].isel(case=case_key[k])
        members = {}
        for c in config.COUPLINGS:
            a = arr.sel(coupling=c).to_numpy()
            if np.isfinite(a).all():
                members[c] = np.round(a, 1).tolist()
        cases.append(
            {
                "init": str(k[0])[:16],
                "lead": k[1],
                "valid": str(k[0] + pd.Timedelta(hours=k[1]))[:16],
                "obs_total": round(float(obs_lookup.loc[k, "obs_total"]), 1),
                "members": members,
                "crps": {c: round(float(per_case.loc[(k[0], k[1], c)]), 5) for c in members},
            }
        )
    return {"summary": summary, "farms": farms, "cases": cases}


def build(period_dir: Path, portfolio: pd.DataFrame, synthetic: bool = False) -> Path:
    systems = {}
    for d in sorted(p for p in period_dir.iterdir() if p.is_dir() and (p / "summary.json").exists()):
        if "_" in d.name:
            continue
        payload = _system_payload(d, portfolio)
        if payload:
            systems[d.name] = payload
    if not systems:
        raise FileNotFoundError(f"no evaluated systems under {period_dir}")
    pf = portfolio.set_index("farm")
    farms = [
        {"id": f, "name": str(pf.loc[f, "name"]), "type": str(pf.loc[f, "type"]),
         "lat": float(pf.loc[f, "lat"]), "lon": float(pf.loc[f, "lon"]), "cap": float(pf.loc[f, "capacity_mw"])}
        for f in next(iter(systems.values()))["farms"]
    ]
    data = {"period": period_dir.name, "synthetic": synthetic, "farms": farms, "systems": systems}
    html = TEMPLATE.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = period_dir / "site" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>When do wind farms underproduce together?</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root { --ink:#1b1f24; --muted:#6b7280; --line:#e5e7eb; --native:#444; --historical:#1f77b4; --independent:#d62728; --cross:#2ca02c; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; color:var(--ink); background:#fafafa; }
  header { padding:20px 28px 8px; }
  h1 { margin:0 0 4px; font-size:24px; }
  .sub { color:var(--muted); max-width:900px; }
  .banner { background:#fff3cd; border:1px solid #f0d78c; padding:8px 12px; margin:10px 28px 0; border-radius:6px; }
  main { display:grid; grid-template-columns: 380px 1fr; gap:18px; padding:16px 28px 28px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
  .card h2 { font-size:15px; margin:0 0 8px; }
  #map { height:440px; border-radius:6px; }
  .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
  button.c { border:1px solid var(--line); background:#fff; padding:5px 10px; border-radius:16px; cursor:pointer; }
  button.c.on { color:#fff; border-color:transparent; }
  select { padding:4px 6px; }
  .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:18px; }
  svg text { font-size:11px; fill:var(--muted); }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  td, th { padding:4px 6px; border-bottom:1px solid var(--line); text-align:right; }
  td:first-child, th:first-child { text-align:left; }
  .big { font-size:28px; font-weight:600; }
  .note { color:var(--muted); font-size:13px; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } .grid2 { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>When do wind farms underproduce together?</h1>
  <div class="sub">Every farm keeps exactly the same 51 forecast values. Only the pairing of values across farms changes.
  Watch the spread of the portfolio total change while each farm's own forecast stays fixed.</div>
</header>
<div id="banner"></div>
<main>
  <div>
    <div class="card"><h2>Portfolio</h2><div id="map"></div>
      <div class="note">Circle area is capacity. Click two farms to compare how their members pair up.</div></div>
    <div class="card" style="margin-top:18px"><h2>Full evaluation, <span id="period"></span></h2>
      <div id="headline"></div></div>
  </div>
  <div>
    <div class="card">
      <div class="controls">
        <label>System <select id="system"></select></label>
        <label>Case <select id="case"></select></label>
      </div>
      <div class="controls" id="couplings"></div>
      <div class="grid2">
        <div><h2>Portfolio total, 51 members</h2><svg id="hist" width="100%" height="260"></svg>
          <div class="note">Black line: metered output. <span id="lowp"></span></div></div>
        <div><h2>Pairing between two farms</h2><svg id="scatter" width="100%" height="260"></svg>
          <div class="note" id="pairnote"></div></div>
      </div>
      <div id="casestats" style="margin-top:10px"></div>
    </div>
    <div class="card" style="margin-top:18px"><h2>By aggregation level</h2>
      <svg id="agg" width="100%" height="220"></svg>
      <div class="note">CRPS minus native, fraction of subset capacity, 95% block bootstrap interval. Below zero means the coupling beats native.</div></div>
  </div>
</main>
<script>
const DATA = /*DATA*/null;
const COL = {native:'#444', historical:'#1f77b4', independent:'#d62728', cross:'#2ca02c'};
const LABEL = {native:'Native', historical:'Historical', independent:'Independent', cross:'Cross model'};
function farthestPair() {
  let best = [0, 1], d = -1;
  DATA.farms.forEach((a, i) => DATA.farms.forEach((b, j) => {
    const dd = (a.lat - b.lat) ** 2 + ((a.lon - b.lon) * Math.cos(a.lat * Math.PI / 180)) ** 2;
    if (j > i && dd > d) { d = dd; best = [i, j]; }
  }));
  return best;
}
let state = {system: Object.keys(DATA.systems)[0], caseIdx: 0, coupling: 'native', pair: farthestPair()};
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
const capTotal = DATA.farms.reduce((a, f) => a + f.cap, 0);

if (DATA.synthetic) $('banner').innerHTML = '<div class="banner"><b>Synthetic demo.</b> Built from a simulated world with a planted dependence error. These are not results about real forecasts.</div>';
$('period').textContent = DATA.period;

const map = L.map('map', {scrollWheelZoom:false}).setView([55, -2.5], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution:'&copy; OpenStreetMap'}).addTo(map);
const markers = DATA.farms.map((f, i) => {
  const m = L.circleMarker([f.lat, f.lon], {radius: 4 + Math.sqrt(f.cap) / 3, color: f.type === 'offshore' ? '#1f77b4' : '#8c564b', weight: 2, fillOpacity: 0.35}).addTo(map);
  m.bindTooltip(`${f.name}, ${Math.round(f.cap)} MW`);
  m.on('click', () => { state.pair = [state.pair[1], i]; draw(); });
  return m;
});

for (const s of Object.keys(DATA.systems)) { const o = document.createElement('option'); o.value = s; o.textContent = s.toUpperCase(); $('system').appendChild(o); }
$('system').onchange = e => { state.system = e.target.value; state.caseIdx = 0; fillCases(); draw(); };
function fillCases() {
  const sel = $('case'); sel.innerHTML = '';
  DATA.systems[state.system].cases.forEach((c, i) => {
    const o = document.createElement('option'); o.value = i;
    o.textContent = `${c.valid} UTC (issued ${c.init.slice(0,10)}, +${c.lead}h), observed ${Math.round(c.obs_total)} MW` + (i === 0 ? '  [illustrative case, fixed rule]' : '');
    sel.appendChild(o);
  });
  sel.value = state.caseIdx;
}
$('case').onchange = e => { state.caseIdx = +e.target.value; draw(); };
function fillCouplings() {
  const box = $('couplings'); box.innerHTML = '';
  const c = DATA.systems[state.system].cases[state.caseIdx];
  for (const k of Object.keys(LABEL)) {
    if (!c.members[k]) continue;
    const b = document.createElement('button'); b.className = 'c' + (state.coupling === k ? ' on' : '');
    b.textContent = LABEL[k]; if (state.coupling === k) b.style.background = COL[k];
    b.onclick = () => { state.coupling = k; draw(); }; box.appendChild(b);
  }
}

function drawHist(c) {
  const svg = $('hist'); clear(svg);
  const W = svg.clientWidth || 420, H = 260, P = {l:40, r:10, t:10, b:30};
  const totals = {};
  for (const k in c.members) totals[k] = c.members[k].map(row => row.reduce((a, b) => a + b, 0));
  const all = Object.values(totals).flat().concat([c.obs_total]);
  const lo = 0, hi = Math.max(capTotal * 0.2, Math.max(...all) * 1.05);
  const nb = 24, bw = (hi - lo) / nb;
  const x = v => P.l + (v - lo) / (hi - lo) * (W - P.l - P.r);
  const counts = Array(nb).fill(0);
  totals[state.coupling].forEach(v => counts[Math.min(nb - 1, Math.floor((v - lo) / bw))]++);
  const ymax = Math.max(...counts, 1);
  const y = v => H - P.b - v / ymax * (H - P.t - P.b);
  counts.forEach((n, i) => el('rect', {x: x(lo + i * bw) + 1, y: y(n), width: Math.max(1, x(lo + bw) - x(lo) - 2), height: H - P.b - y(n), fill: COL[state.coupling], opacity: 0.75}, svg));
  if (state.coupling !== 'native') {
    const cn = Array(nb).fill(0); totals.native.forEach(v => cn[Math.min(nb - 1, Math.floor((v - lo) / bw))]++);
    let d = ''; cn.forEach((n, i) => { d += `${i ? 'L' : 'M'}${x(lo + i * bw)},${y(n)} L${x(lo + (i + 1) * bw)},${y(n)} `; });
    el('path', {d, fill: 'none', stroke: COL.native, 'stroke-width': 1.5, 'stroke-dasharray': '4 3'}, svg);
  }
  el('line', {x1: x(c.obs_total), x2: x(c.obs_total), y1: P.t, y2: H - P.b, stroke: '#000', 'stroke-width': 2}, svg);
  el('line', {x1: P.l, x2: W - P.r, y1: H - P.b, y2: H - P.b, stroke: '#999'}, svg);
  for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4; el('text', {x: x(v), y: H - 12, 'text-anchor': 'middle'}, svg).textContent = Math.round(v) + ' MW'; }
  const thr = 0.2 * capTotal;
  const p = k => (totals[k].filter(v => v < thr).length / totals[k].length * 100).toFixed(0);
  $('lowp').innerHTML = `Chance the portfolio is below 20% of capacity: ` + Object.keys(totals).map(k => `<b style="color:${COL[k]}">${LABEL[k]} ${p(k)}%</b>`).join(', ') + (state.coupling !== 'native' ? '. Dashed: native.' : '.');
}

function drawScatter(c) {
  const svg = $('scatter'); clear(svg);
  const W = svg.clientWidth || 420, H = 260, P = {l:44, r:10, t:10, b:30};
  const [a, b] = state.pair, fa = DATA.farms[a], fb = DATA.farms[b];
  const m = c.members[state.coupling];
  const x = v => P.l + v / fa.cap * (W - P.l - P.r), y = v => H - P.b - v / fb.cap * (H - P.t - P.b);
  el('rect', {x: P.l, y: P.t, width: W - P.l - P.r, height: H - P.t - P.b, fill: 'none', stroke: '#ddd'}, svg);
  m.forEach(row => el('circle', {cx: x(row[a]), cy: y(row[b]), r: 3.5, fill: COL[state.coupling], opacity: 0.7}, svg));
  el('text', {x: (W + P.l) / 2, y: H - 8, 'text-anchor': 'middle'}, svg).textContent = `${fa.name} (MW, 0 to ${Math.round(fa.cap)})`;
  const t = el('text', {x: 12, y: H / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${H / 2})`}, svg); t.textContent = `${fb.name} (MW)`;
  markers.forEach((mk, i) => mk.setStyle({weight: (i === a || i === b) ? 5 : 2}));
  $('pairnote').textContent = 'Each dot is one member. The horizontal and vertical spreads never change; only which values sit together does.';
}

function drawCaseStats(c) {
  const rows = Object.keys(c.crps).map(k => `<tr><td style="color:${COL[k]}">${LABEL[k]}</td><td>${(c.crps[k] * 100).toFixed(2)}%</td></tr>`).join('');
  $('casestats').innerHTML = `<table><tr><th>This case</th><th>Portfolio CRPS, % of capacity (mean over template draws)</th></tr>${rows}</table>`;
}

function drawHeadline() {
  const s = DATA.systems[state.system].summary, d = s.differences_vs_native.crps, n = s.metrics.crps.native.estimate;
  const fmt = v => (v * 100).toFixed(2);
  let rows = `<tr><td>Native</td><td>${fmt(n)}%</td><td></td></tr>`;
  for (const k of ['historical', 'independent', 'cross']) if (d[k]) rows += `<tr><td style="color:${COL[k]}">${LABEL[k]}</td><td>${fmt(s.metrics.crps[k].estimate)}%</td><td>${fmt(d[k].estimate)} [${fmt(d[k].lo)}, ${fmt(d[k].hi)}]</td></tr>`;
  const vs = s.differences_vs_native.vs.historical;
  $('headline').innerHTML = `<table><tr><th>${state.system.toUpperCase()}</th><th>Portfolio CRPS</th><th>minus native, 95% CI</th></tr>${rows}</table>
    <p class="note">${s.n_cases} cases over ${s.n_days} issue days. Block bootstrap over days, block ${s.block_length_days} days.
    Variogram score, historical minus native: ${vs ? vs.estimate.toFixed(4) + ' [' + vs.lo.toFixed(4) + ', ' + vs.hi.toFixed(4) + ']' : 'n/a'}.
    Per farm CRPS is identical across couplings by construction and checked for every case.</p>`;
}

function drawAgg() {
  const svg = $('agg'); clear(svg);
  const s = DATA.systems[state.system].summary.aggregation;
  const W = svg.clientWidth || 800, H = 220, P = {l:60, r:10, t:10, b:30};
  const sizes = Object.keys(s).map(Number).sort((a, b) => a - b);
  let vals = [];
  for (const k of ['historical', 'independent', 'cross']) sizes.forEach(z => { if (s[z][k]) vals.push(s[z][k].lo, s[z][k].hi); });
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const x = z => P.l + (z - sizes[0]) / Math.max(1, sizes[sizes.length - 1] - sizes[0]) * (W - P.l - P.r);
  const y = v => P.t + (hi - v) / (hi - lo || 1) * (H - P.t - P.b);
  el('line', {x1: P.l, x2: W - P.r, y1: y(0), y2: y(0), stroke: COL.native}, svg);
  for (const k of ['historical', 'independent', 'cross']) {
    if (!s[sizes[0]][k]) continue;
    let band = '', line = '';
    sizes.forEach((z, i) => { line += `${i ? 'L' : 'M'}${x(z)},${y(s[z][k].estimate)} `; band += `${i ? 'L' : 'M'}${x(z)},${y(s[z][k].hi)} `; });
    [...sizes].reverse().forEach(z => { band += `L${x(z)},${y(s[z][k].lo)} `; });
    el('path', {d: band + 'Z', fill: COL[k], opacity: 0.15}, svg);
    el('path', {d: line, fill: 'none', stroke: COL[k], 'stroke-width': 2}, svg);
  }
  sizes.forEach(z => el('text', {x: x(z), y: H - 10, 'text-anchor': 'middle'}, svg).textContent = z);
  [lo, 0, hi].forEach(v => el('text', {x: P.l - 6, y: y(v) + 4, 'text-anchor': 'end'}, svg).textContent = (v * 100).toFixed(2) + '%');
}

function draw() {
  const c = DATA.systems[state.system].cases[state.caseIdx];
  if (!c.members[state.coupling]) state.coupling = 'native';
  fillCouplings(); drawHist(c); drawScatter(c); drawCaseStats(c); drawHeadline(); drawAgg();
}
fillCases(); draw();
window.addEventListener('resize', draw);
</script>
</body>
</html>
"""
