#!/usr/bin/env python3
"""Local web front end for the LDO design flow.

Every Run click starts a fresh `ldo_run.py` job (a cold optimisation). The page only ever shows
the output of the job it launched itself: results are read from the exact file that job reports.
Serves on 127.0.0.1 only. Needs: pip install flask
"""
import json, os, re, signal, subprocess, sys, threading, time
from flask import Flask, Response, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("LDO_GUI_PORT", "8765"))
app = Flask(__name__)
app.json.sort_keys = False

JOB = dict(proc=None, log=None, started=0.0, args=None, state="idle", final=None, error=None)
LOCK = threading.Lock()

RE_STAGE = re.compile(r"\[stage (\d)\] (\d+)/(\d+)\s+best=([-\d.eE+]+)\s+screened-passing=(\d+)\s+(\d+)s")
RE_ATT = re.compile(r"=== attempt (\d+)/(\d+)")
RE_STATUS = re.compile(r"^STATUS: (\S+)")
RE_FINAL = re.compile(r"^FINAL_JSON: (.+)$")
RE_SLICE = re.compile(r"stage 1 slice (\d+)/(\d+)")
LIMITS = dict(vout=(1.8, 5.5), cout=(0.5, 100.0), psrr=(10.0, 80.0), line=(0.05, 10.0), tol=(0.2, 10.0),
              pm=(30.0, 90.0), attempts=(1, 10), workers=(1, 16))


def num(d, key, lo, hi, cast=float):
    try:
        v = cast(d[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}")
    return v


def build_cmd(d):
    pdk = d.get("pdk")
    if pdk not in ("gf180", "sky130"):
        raise ValueError("unknown PDK")
    vout = num(d, "vout", *LIMITS["vout"])
    if (pdk == "gf180" and not any(abs(vout - v) < 1e-9 for v in (3.3, 5.0))) or (pdk == "sky130" and abs(vout - 3.3) > 1e-9):
        raise ValueError("this PDK supports VOUT 3.3 V and 5.0 V (SKY130: 3.3 V only)")
    cmd = [sys.executable, "-u", os.path.join(HERE, "ldo_run.py"), "--pdk", pdk, "--vout", f"{vout:g}",
           "--cout", f"{num(d, 'cout', *LIMITS['cout']):g}", "--psrr", f"{num(d, 'psrr', *LIMITS['psrr']):g}",
           "--line-pct", f"{num(d, 'line', *LIMITS['line']):g}", "--tol-pct", f"{num(d, 'tol', *LIMITS['tol']):g}",
           "--pm-floor", f"{num(d, 'pm', *LIMITS['pm']):g}",
           "--attempts", str(num(d, "attempts", *LIMITS["attempts"], cast=int))]
    if d.get("workers"):
        cmd += ["--workers", str(num(d, "workers", *LIMITS["workers"], cast=int))]
    return cmd


def refresh_state():
    """Call with LOCK held. Updates state when the process has ended."""
    p = JOB["proc"]
    if p is not None and JOB["state"] == "running" and p.poll() is not None:
        JOB["log"].close()
        txt = read_log()
        m = None
        for line in txt.splitlines():
            mm = RE_FINAL.match(line.strip())
            if mm:
                m = mm.group(1).strip()
        if JOB.get("stopped"):
            JOB["state"] = "stopped"
        elif m and os.path.exists(m) and os.path.getmtime(m) >= JOB["started"] - 1:
            JOB["state"], JOB["final"] = "done", m
        else:
            JOB["state"], JOB["error"] = "error", "The flow ended without producing a result. See the log below."


def read_log():
    try:
        return open(JOB["logpath"], errors="replace").read()
    except OSError:
        return ""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.post("/run")
def run():
    with LOCK:
        refresh_state()
        if JOB["state"] == "running":
            return jsonify(error="A design run is already in progress."), 409
        try:
            cmd = build_cmd(request.get_json(force=True, silent=True) or {})
        except ValueError as e:
            return jsonify(error=str(e)), 400
        logpath = os.path.join(HERE, "gui_job.log")
        JOB.update(proc=None, state="running", final=None, error=None, stopped=False,
                   started=time.time(), args=cmd, logpath=logpath)
        JOB["log"] = open(logpath, "w")
        body = request.get_json(force=True, silent=True) or {}
        JOB["warm"] = bool(body.get("warm"))
        env = dict(os.environ, PYTHONUNBUFFERED="1", LDO_WARM="1" if JOB["warm"] else "0")
        JOB["proc"] = subprocess.Popen(cmd, cwd=HERE, stdout=JOB["log"], stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
        return jsonify(ok=True)


@app.post("/stop")
def stop():
    with LOCK:
        refresh_state()
        p = JOB["proc"]
        if JOB["state"] == "running" and p is not None:
            JOB["stopped"] = True
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            refresh_state()
        return jsonify(ok=True)


@app.get("/status")
def status():
    with LOCK:
        refresh_state()
        out = dict(state=JOB["state"], error=JOB["error"],
                   started_at=time.strftime("%H:%M:%S", time.localtime(JOB["started"])) if JOB["started"] else None)
        if JOB["state"] == "idle":
            return jsonify(out)
        txt = read_log()
        lines = txt.splitlines()
        att, n_att, stage, slc = 0, JOB["args"][JOB["args"].index("--attempts") + 1], 0, None
        evals = planned = passing = 0
        best = None
        t_stage = 0
        attempts = []
        for ln in lines:
            m = RE_ATT.search(ln)
            if m:
                att, n_att = int(m.group(1)), int(m.group(2))
                stage, evals, planned, best, passing, slc = 0, 0, 0, None, 0, None
                continue
            m = RE_SLICE.search(ln)
            if m:
                slc = (int(m.group(1)), int(m.group(2)))
            m = RE_STAGE.search(ln)
            if m:
                stage, evals, planned = int(m.group(1)), int(m.group(2)), int(m.group(3))
                best, passing, t_stage = float(m.group(4)), int(m.group(5)), int(m.group(6))
            m = RE_STATUS.match(ln.strip())
            if m:
                attempts.append(dict(n=att, status=m.group(1)))
            if ln.startswith("verified candidate"):
                stage = 3
        out.update(attempt=att, attempts=int(n_att), stage=stage, slice=slc, evals=evals, planned=planned,
                   best=best, passing=passing, attempt_results=attempts,
                   elapsed=round(time.time() - JOB["started"]), log_tail=lines[-6:], spec_args=JOB["args"][3:],
                   warm=JOB.get("warm", False), history=run_history(lines))
        if JOB["state"] == "done":
            try:
                out["result"] = json.load(open(JOB["final"]))
            except (OSError, ValueError):
                out.update(state="error", error="The result file could not be read.")
        return jsonify(out)


RE_RUNDIR = re.compile(r"^run folder: (.+)$")


def run_history(lines):
    """Convergence curve and MLP predicted-vs-simulated rows for the latest attempt of THIS job."""
    rd = None
    for ln in lines:
        m = RE_RUNDIR.match(ln.strip())
        if m:
            rd = m.group(1).strip()
    if not rd or not os.path.isdir(rd) or os.path.getmtime(rd) < JOB["started"] - 5:
        return None
    ev = []
    try:
        for line in open(os.path.join(rd, "evals.jsonl"), errors="replace"):
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    best, curve, firstpass = None, [], None
    for e in ev:
        t = e.get("total", 1e6)
        best = t if best is None else min(best, t)
        curve.append([e.get("i"), round(min(t, 1e6), 5), round(best, 5), e.get("src", "bo"), bool(e.get("screen"))])
        if firstpass is None and e.get("screen"):
            firstpass = e.get("i")
    if len(curve) > 1500:                       # keep the payload small
        step = len(curve) // 1500 + 1
        curve = [c for j, c in enumerate(curve) if j % step == 0 or c[3] != "bo"]
    rows = []
    try:
        ws = json.load(open(os.path.join(rd, "warm_seeds.json")))
    except (OSError, ValueError):
        ws = None
    if ws:
        sim = {}
        for e in ev:                                  # first simulation of each point (stage 2 re-runs some)
            sim.setdefault(tuple(round(v, 4) for v in e["x"]), e)
        for r in ws.get("rows", []):
            e = sim.get(tuple(round(v, 4) for v in r["x"]))
            rows.append(dict(kind=r["kind"], pred=r["pred"], pred_margin=r.get("pred_margin_spec", r["pred_margin"]),
                             sim=e["m"] if e else None, screen=e["screen"] if e else None,
                             i=e["i"] if e else None))
    return dict(curve=curve, first_pass=firstpass, warm_rows=rows,
                warm_status=ws.get("status") if ws else ("mlp-requested" if JOB.get("warm") else "cold"))


def model_info(pdk):
    p = os.environ.get("LDO_SURROGATE") or os.path.join(HERE, "surrogate_ws", f"{pdk}.pt")
    try:
        m = json.load(open(p + ".json"))
        return dict(exists=os.path.exists(p), n_train=m.get("n_train"), trained_at=m.get("trained_at"),
                    val_r2=m.get("val_r2"), coverage=m.get("coverage"))
    except (OSError, ValueError):
        return dict(exists=False)


@app.get("/model")
def model():
    return jsonify(model_info(request.args.get("pdk", "gf180")))


def _final():
    with LOCK:
        refresh_state()
        if JOB["state"] != "done":
            return None
        return json.load(open(JOB["final"]))


@app.get("/result.json")
def result_json():
    r = _final()
    if r is None:
        return jsonify(error="no finished run"), 404
    return Response(json.dumps(r, indent=1), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=ldo_result.json"})


@app.get("/netlist")
def netlist():
    r = _final()
    if r is None or not r.get("design"):
        return jsonify(error="no finished design"), 404
    lst = r.get("pareto") or [r["design"]]
    i = request.args.get("i", type=int)
    try:
        sys.path.insert(0, HERE)
        import ldo_pdk as L
        s, d = r["spec"], (lst[i] if i is not None and 0 <= i < len(lst) else r["design"])
        sp = L.make_spec(s["vout"], dropout=round(s["vin_min"] - s["vout"], 3),
                         vin_over=round(s["vin_max"] - s["vout"], 3))
        p = L.base_params(sp, d["sizes"]); p["R1_val"] = f"{d['R1_ohm']:.1f}"
        text = L.build(s["pdk"], p, d["n_pass"], "tt", 27, "", cout=s["cout_uF"] * 1e-6, esr=sp["ESR"])
    except Exception as e:
        return jsonify(error=f"could not build the netlist: {e}"), 500
    return Response(text + "\n", mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename=ldo_{s['pdk']}_{s['vout']:g}V.cir"})


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LDO Designer</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16191d;--mut:#5b6572;--line:#e1e5ea;--acc:#1d5fd1;--accink:#fff;
--ok:#12794a;--okbg:#e3f5ec;--bad:#a4291d;--badbg:#fdecea;--warn:#8a5a00;--warnbg:#fff3d6;--bar:#e6eaf0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111418;--card:#1a1f26;--ink:#e8ecf1;--mut:#9aa6b4;
--line:#2c343e;--acc:#6aa2ff;--accink:#0b1220;--ok:#5fd29b;--okbg:#14301f;--bad:#ff8b7e;--badbg:#3a1a17;--warn:#f2c46b;--warnbg:#3a2d10;--bar:#2a313b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mut);margin:0 0 18px}
.grid{display:grid;grid-template-columns:340px 1fr;gap:16px}@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.card h2{font-size:15px;margin:0 0 10px}
label{display:block;font-size:13px;color:var(--mut);margin:10px 0 3px}
input,select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);font:inherit}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.seg{display:flex;gap:6px}.seg button{flex:1;padding:8px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);cursor:pointer;font:inherit}
.seg button.on{background:var(--acc);color:var(--accink);border-color:var(--acc)}.seg button:disabled{opacity:.4;cursor:not-allowed}
.btn{width:100%;margin-top:16px;padding:11px;border:0;border-radius:8px;background:var(--acc);color:var(--accink);font:600 15px system-ui;cursor:pointer}
.btn.stop{background:var(--bad);color:#fff;margin-top:8px}.btn:disabled{opacity:.5;cursor:not-allowed}
.fixed{font-size:13px;color:var(--mut);margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
details{margin-top:12px}summary{cursor:pointer;color:var(--mut);font-size:13px}
.note{font-size:13px;border-radius:7px;padding:8px 10px;margin-top:10px;background:var(--warnbg);color:var(--warn)}
.bar{height:10px;background:var(--bar);border-radius:6px;overflow:hidden;margin:8px 0}.bar i{display:block;height:100%;background:var(--acc);width:0;transition:width .4s}
.stats{display:flex;flex-wrap:wrap;gap:6px 22px;font-size:14px;margin-top:6px}.stats b{font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;border:1px solid var(--line);margin-right:6px;color:var(--mut)}
.pill.on{background:var(--acc);color:var(--accink);border-color:var(--acc)}
.banner{padding:12px 14px;border-radius:8px;font-weight:600;margin-bottom:12px}.banner.ok{background:var(--okbg);color:var(--ok)}
.banner.bad{background:var(--badbg);color:var(--bad)}.banner small{display:block;font-weight:400;margin-top:3px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:500;font-size:13px}td.r,th.r{text-align:right}.ok{color:var(--ok)}.bad{color:var(--bad)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{border:1px solid var(--line);border-radius:8px;padding:9px 11px}.tile span{display:block;font-size:12px;color:var(--mut)}.tile b{font-size:17px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:8px;font-size:12px;overflow:auto;max-height:130px;margin:10px 0 0}
a.dl{display:inline-block;margin:8px 10px 0 0;color:var(--acc);text-decoration:none;font-weight:600}
.front{margin:6px 0 4px}.front svg{width:100%;height:auto;display:block}
.front text{fill:var(--mut);font-size:11px}.front circle{fill:var(--acc);fill-opacity:.55;stroke:var(--acc);cursor:pointer}
.front circle.sel{fill-opacity:1;stroke:var(--ink);stroke-width:2}.front line{stroke:var(--line)}
tr.pick{cursor:pointer}tr.pick.sel td{background:var(--okbg)}tr.pick:hover td{background:var(--bg)}
.tag{font-size:11px;border:1px solid var(--line);border-radius:99px;padding:0 7px;color:var(--mut);margin-left:6px}
.empty{color:var(--mut);padding:30px 6px;text-align:center}.hide{display:none}.scroll{overflow-x:auto}
</style></head><body><div class="wrap">
<h1>LDO Designer</h1>
<p class="sub">Enter a spec and run. Every click starts a fresh optimisation; nothing is loaded from earlier runs.</p>
<div class="grid">
<div class="card"><h2>Specification</h2>
<label>Technology</label>
<select id="pdk"><option value="gf180">GF180MCU (3.3 V or 5 V)</option><option value="sky130">SKY130 (3.3 V only, experimental, very slow)</option></select>
<label>Output voltage</label>
<div class="seg" id="vseg"><button data-v="3.3" class="on">3.3 V</button><button data-v="5.0">5.0 V</button></div>
<div class="row"><div><label>Output capacitor (µF)</label><input id="cout" type="number" step="any" value="4"></div>
<div><label>PSRR at 10 kHz (dB ≥)</label><input id="psrr" type="number" step="any" value="40"></div></div>
<div class="row"><div><label>Line regulation (%/V ≤)</label><input id="line" type="number" step="any" value="1.0"></div>
<div><label>VOUT accuracy (± %)</label><input id="tol" type="number" step="any" value="1.5"></div></div>
<label>Phase margin floor (deg)</label><input id="pm" type="number" step="any" value="45">
<label>Starting points for the search</label>
<div class="seg" id="wseg"><button data-w="0" class="on">Cold (random)</button><button data-w="1">MLP warm start</button></div>
<div id="wnote" class="fixed" style="margin-top:6px;border:0;padding-top:0"></div>
<details><summary>Advanced</summary>
<div class="row"><div><label>Max attempts</label><input id="attempts" type="number" value="5" min="1" max="10"></div>
<div><label>Parallel workers</label><input id="workers" type="number" placeholder="auto" min="1" max="16"></div></div></details>
<div id="pdknote" class="note hide">SKY130 simulations take minutes each; a run can take hours and may not find a design. GF180 is recommended.</div>
<button class="btn" id="run">Run optimisation</button><button class="btn stop hide" id="stop">Stop</button>
<div class="fixed">Fixed: load 100 mA, VIN = VOUT + 0.3 to 0.4 V, load regulation 1% of VOUT, Iq ≤ 100 µA, external capacitor only.<br>Verified at 27 °C across five process corners (tt, ss, ff, sf, fs). Temperature is not checked.</div>
<div id="err" class="note hide" style="background:var(--badbg);color:var(--bad)"></div></div>
<div class="card" id="right"><div class="empty">No run yet. Set a spec and press Run.</div></div>
</div></div>
<script>
const $=id=>document.getElementById(id);let vout=3.3,timer=null,lastState='idle';
function setV(v){vout=v;document.querySelectorAll('#vseg button').forEach(b=>b.classList.toggle('on',parseFloat(b.dataset.v)===v))}
document.querySelectorAll('#vseg button').forEach(b=>b.onclick=()=>setV(parseFloat(b.dataset.v)));
let warm=false;
function setW(w){warm=w;document.querySelectorAll('#wseg button').forEach(b=>b.classList.toggle('on',(b.dataset.w==='1')===w))}
document.querySelectorAll('#wseg button').forEach(b=>b.onclick=()=>{if(!b.disabled)setW(b.dataset.w==='1')});
async function modelNote(){let m={exists:false};try{m=await (await fetch('/model?pdk='+$('pdk').value)).json()}catch(e){}
 const wb=document.querySelector('#wseg button[data-w="1"]');wb.disabled=!m.exists;if(!m.exists)setW(false);
 $('wnote').textContent=m.exists?('MLP trained on '+m.n_train+' simulated designs ('+m.trained_at+'). It only proposes starting points; every result shown is freshly simulated.')
  :'No MLP model for this PDK yet (python surrogate_ws.py train). Cold start only.'}
$('pdk').onchange=()=>{const s=$('pdk').value==='sky130';$('pdknote').classList.toggle('hide',!s);
 document.querySelector('#vseg button[data-v="5.0"]').disabled=s;if(s)setV(3.3);modelNote()};
modelNote();
function spec(){const g=id=>parseFloat($(id).value);return{pdk:$('pdk').value,vout:vout,cout:g('cout'),psrr:g('psrr'),line:g('line'),tol:g('tol'),pm:g('pm'),
 attempts:parseInt($('attempts').value||'5'),workers:$('workers').value?parseInt($('workers').value):null,warm:warm}}
function convChart(c,fp){if(!c||c.length<2)return '';const W=560,H=200,L=46,R=12,T=10,B=34,n=c[c.length-1][0]||c.length;
 const ly=v=>Math.log10(Math.max(1e-3,Math.min(v,1e3)));const ys=c.map(r=>ly(r[2])).concat(c.map(r=>ly(r[1])));
 const y0=Math.min(...ys),y1=Math.max(...ys),px=i=>L+(i/n)*(W-L-R),py=v=>H-B-(y1===y0?.5:(v-y0)/(y1-y0))*(H-B-T);
 let g='';for(let k=0;k<=3;k++){const yv=y0+(y1-y0)*k/3;g+='<line x1="'+L+'" y1="'+py(yv)+'" x2="'+(W-R)+'" y2="'+py(yv)+'"/><text x="'+(L-5)+'" y="'+(py(yv)+4)+'" text-anchor="end">'+Math.pow(10,yv).toPrecision(2)+'</text>'}
 for(let k=0;k<=4;k++){const xv=Math.round(n*k/4);g+='<text x="'+px(xv)+'" y="'+(H-B+14)+'" text-anchor="middle">'+xv+'</text>'}
 c.forEach(r=>{const m=r[3]!=='bo'&&r[3]!=='seed';g+='<circle cx="'+px(r[0])+'" cy="'+py(ly(r[1]))+'" r="'+(m?3.5:1.6)+'" style="fill:'+(m?'var(--warn)':'var(--mut)')+';fill-opacity:'+(m?.95:.35)+';stroke:none"></circle>'});
 g+='<polyline fill="none" style="stroke:var(--acc);stroke-width:2" points="'+c.map(r=>px(r[0])+','+py(ly(r[2]))).join(' ')+'"/>';
 if(fp)g+='<line x1="'+px(fp)+'" y1="'+T+'" x2="'+px(fp)+'" y2="'+(H-B)+'" style="stroke:var(--ok);stroke-dasharray:4 3"/><text x="'+(px(fp)+4)+'" y="'+(T+10)+'" style="fill:var(--ok)">first pass @'+fp+'</text>';
 g+='<text x="'+(L+(W-L-R)/2)+'" y="'+(H-4)+'" text-anchor="middle">SPICE evaluations</text>';
 return '<div class="front"><svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Convergence">'+g+'</svg></div>'}
function histSection(s){const h=s.history;if(!h)return '';const mode=h.warm_status==='mlp'?'MLP warm start':(h.warm_status==='cold'?'cold start':'warm start requested: '+h.warm_status);
 let o='<h2 style="margin-top:16px">Search progress, latest attempt <span class="tag">'+mode+'</span></h2><p class="sub" style="margin:0 0 4px">Line: best score so far (log scale). Grey dots: BO/random evaluations. Orange dots: points proposed by the MLP.</p>'+convChart(h.curve,h.first_pass);
 if(h.warm_rows&&h.warm_rows.length){const f=(v,d)=>v==null?'–':fmt(v,d);
  const sm=h.warm_rows.filter(r=>r.screen!=null),pp=sm.filter(r=>r.pred_margin>0),ok=sm.filter(r=>r.screen);
  o+='<h2 style="margin-top:10px">MLP prediction vs SPICE</h2><p class="sub" style="margin:0 0 4px">'+sm.length+' MLP proposals simulated; '+ok.length+' passed the screen in SPICE'
   +(pp.length?(' ('+pp.filter(r=>r.screen).length+' of the '+pp.length+' predicted to pass)'):'')+'.</p><details><summary>Show every proposal</summary><div class="scroll"><table><tr><th>Source</th><th class="r">Eval #</th><th class="r">PSRR pred / sim (dB)</th><th class="r">PM pred / sim (°)</th><th class="r">Iq pred / sim (µA)</th><th class="r">Line pred / sim (mV)</th><th>Pred. pass</th><th>Sim. pass</th></tr>'
  +h.warm_rows.map(r=>{const m=r.sim||{};return '<tr><td>'+(r.kind==='init'?'initial':'online')+'</td><td class="r">'+(r.i||'pending')+'</td><td class="r">'+f(r.pred.psrr,1)+' / '+f(m.psrr,1)+'</td><td class="r">'+f(r.pred.pm,0)+' / '+f(m.pm,0)+'</td><td class="r">'+f(r.pred.iq,0)+' / '+f(m.iq,0)+'</td><td class="r">'+f(r.pred.line,1)+' / '+f(m.line,1)+'</td><td>'+(r.pred_margin>0?'yes':'no')+'</td><td class="'+(r.screen?'ok':'bad')+'">'+(r.screen==null?'–':(r.screen?'yes':'no'))+'</td></tr>'}).join('')+'</table></div></details>'}
 return o}
function err(t){$('err').textContent=t;$('err').classList.toggle('hide',!t)}
$('run').onclick=async()=>{err('');const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(spec())});
 const j=await r.json();if(!r.ok){err(j.error);return}lastState='running';SEL=null;$('right').innerHTML='<div class="empty">Starting…</div>';poll()};
$('stop').onclick=async()=>{await fetch('/stop',{method:'POST'});poll()};
const fmt=(v,d=2)=>(typeof v==='number')?v.toFixed(d):'–';
let SEL=null,RES=null;
const chk=(d,k)=>{const c=d.checks.find(c=>c.name.startsWith(k));return c?c.value:null};
const AX={area:['Active area (µm²)',d=>d.area_um2],iq:['Quiescent current (µA)',d=>chk(d,'Quiescent')],psrr:['PSRR at 10 kHz (dB)',d=>chk(d,'PSRR')],pm:['Phase margin (deg)',d=>chk(d,'Phase')]};
function frontChart(list,names){const nx=names[0],ny=names[names.length-1];if(!AX[nx]||!AX[ny]||nx===ny)return '';
 const W=560,H=230,L=52,R=14,T=10,B=36,xs=list.map(d=>AX[nx][1](d)),ys=list.map(d=>AX[ny][1](d));
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),px=v=>L+(x1===x0?.5:(v-x0)/(x1-x0))*(W-L-R-20)+10,py=v=>H-B-(y1===y0?.5:(v-y0)/(y1-y0))*(H-B-T-20)-10;
 let g='';for(let k=0;k<=3;k++){const xv=x0+(x1-x0)*k/3,yv=y0+(y1-y0)*k/3;
  g+='<line x1="'+px(xv)+'" y1="'+T+'" x2="'+px(xv)+'" y2="'+(H-B)+'"/><text x="'+px(xv)+'" y="'+(H-B+14)+'" text-anchor="middle">'+fmt(xv,0)+'</text>'
   +'<line x1="'+L+'" y1="'+py(yv)+'" x2="'+(W-R)+'" y2="'+py(yv)+'"/><text x="'+(L-6)+'" y="'+(py(yv)+4)+'" text-anchor="end">'+fmt(yv,1)+'</text>'}
 g+='<text x="'+(L+(W-L-R)/2)+'" y="'+(H-4)+'" text-anchor="middle">'+AX[nx][0]+'</text><text transform="translate(12,'+(T+(H-B-T)/2)+') rotate(-90)" text-anchor="middle">'+AX[ny][0]+'</text>';
 list.forEach((d,i)=>{g+='<circle class="'+(i===SEL?'sel':'')+'" data-i="'+i+'" cx="'+px(AX[nx][1](d))+'" cy="'+py(AX[ny][1](d))+'" r="'+(i===SEL?8:6)+'"><title>'+d.n_pass+' pass units</title></circle>'});
 return '<div class="front"><svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Trade-off front">'+g+'</svg></div>'}
function frontSection(r,list){if(list.length<2)return '';const names=r.pareto_objectives||['area','iq','psrr'];
 const rec=r.design?JSON.stringify(r.design.x):null;
 return '<h2 style="margin-top:4px">Trade-off front</h2><p class="sub" style="margin:0 0 4px">'+list.length+' verified designs that each meet the spec. Smaller area and lower current cost PSRR margin. Click a design to inspect it.</p>'
  +frontChart(list,names)+'<div class="scroll"><table><tr><th>Pass units</th><th class="r">Area (µm²)</th><th class="r">Iq (µA)</th><th class="r">PSRR (dB)</th><th class="r">Phase margin (°)</th></tr>'
  +list.map((d,i)=>'<tr class="pick'+(i===SEL?' sel':'')+'" data-i="'+i+'"><td>'+d.n_pass+(JSON.stringify(d.x)===rec?'<span class="tag">best margin</span>':'')+'</td><td class="r">'+Math.round(d.area_um2)+'</td><td class="r">'+fmt(chk(d,'Quiescent'),1)+'</td><td class="r">'+fmt(chk(d,'PSRR'),1)+'</td><td class="r">'+fmt(chk(d,'Phase'),1)+'</td></tr>').join('')+'</table></div><hr style="border:0;border-top:1px solid var(--line);margin:14px 0">'}
function top3Section(r){const L=(r.top_verified&&r.top_verified.length)?r.top_verified:(r.design?[r.design]:[]);if(!L.length)return '';
 const sp=r.spec,C=['tt','ss','ff','sf','fs'],nc=L.length+1;
 const cell=(v,d,bad)=>'<td class="r'+(bad?' bad':'')+'">'+fmt(v,d)+'</td>';
 const row=(lab,f)=>'<tr><td>'+lab+'</td>'+L.map(f).join('')+'</tr>';
 const sub=t=>'<tr><th colspan="'+nc+'" style="padding-top:12px">'+t+'</th></tr>';
 let o='<h2 style="margin-top:16px">Top '+L.length+' verified designs</h2><p class="sub" style="margin:0 0 4px">All simulated in this run, shown whether or not they meet the spec. Ranked: full passes first, then by checks passed and worst margin. #1 is the design this run reports. Red = misses the spec.</p>'
  +'<div class="scroll"><table><tr><th></th>'+L.map((d,i)=>'<th class="r">#'+(i+1)+'</th>').join('')+'</tr>';
 o+=row('Result',d=>{const n=d.checks.filter(c=>!c.ok).length;return '<td class="r '+(n?'bad':'ok')+'">'+(n?n+' FAIL':'all pass')+'</td>'});
 o+=row('Worst margin (%)',d=>cell(d.worst_margin*100,1,d.worst_margin<0));
 o+=row('Pass units',d=>'<td class="r">'+d.n_pass+'</td>');
 o+=row('Active area (µm²)',d=>'<td class="r">'+Math.round(d.area_um2)+'</td>');
 o+=sub('Checks (worst case over corners)');
 L[0].checks.forEach(c0=>{o+=row(c0.name+' <span class="tag">'+c0.op.replace('<','&lt;')+' '+fmt(c0.limit,1)+'</span>',d=>{const x=d.checks.find(y=>y.name===c0.name);return x?cell(x.value,2,!x.ok):'<td class="r">–</td>'})});
 o+=sub('PSRR at 10 kHz (dB) by corner and load <span class="tag">≥ '+fmt(sp.psrr_min_dB,1)+'</span>');
 C.forEach(c=>['100mA','2mA'].forEach(l=>{o+=row(c+' · '+l.replace('mA',' mA'),d=>{const v=((d.psrr_dB||{})[c]||{})[l];return cell(v,1,v<sp.psrr_min_dB)})}));
 o+=sub('Phase margin at '+sp.cout_uF+' µF (deg, worst load) <span class="tag">≥ '+fmt(sp.pm_floor_deg,0)+'</span>');
 C.forEach(c=>{o+=row(c,d=>{const t=d.phase_margin_by_cout||{},k=Object.keys(t).find(k=>parseFloat(k)===sp.cout_uF),e=k?t[k][c]:null;
  return e?'<td class="r'+(e.pm<sp.pm_floor_deg?' bad':'')+'">'+fmt(e.pm,1)+' <span class="tag">'+e.at_load_mA+' mA</span></td>':'<td class="r">–</td>'})});
 if(L.some(d=>d.meas_failed&&d.meas_failed.length))o+=row('Failed measurements',d=>'<td class="r'+((d.meas_failed||[]).length?' bad':'')+'">'+((d.meas_failed||[]).join(', ')||'–')+'</td>');
 return o+'</table></div>'}
function mmss(s){return Math.floor(s/60)+':'+String(s%60).padStart(2,'0')}
function pill(t,on){return '<span class="pill'+(on?' on':'')+'">'+t+'</span>'}
function renderRunning(s){const st=s.stage,names=['Starting','1 · Broad search','2 · Refine','3 · Verify'];
 const pct=s.planned?Math.min(100,100*s.evals/s.planned):0;
 let h='<h2>Running</h2><div>'+[1,2,3].map(i=>pill(names[i],st===i)).join('')+'</div>';
 h+='<div class="bar"><i style="width:'+(st===3?100:pct)+'%"></i></div><div class="stats"><span>Attempt <b>'+s.attempt+' / '+s.attempts+'</b></span>';
 if(st===1&&s.slice)h+='<span>Slice <b>'+s.slice[0]+' / '+s.slice[1]+'</b></span>';
 h+='<span>Evaluations <b>'+s.evals+(s.planned?' / '+s.planned:'')+'</b></span><span>Best score <b>'+(s.best==null?'–':fmt(s.best,3))+'</b></span>'
 +'<span>Designs passing the screen <b>'+s.passing+'</b></span><span>Elapsed <b>'+mmss(s.elapsed)+'</b></span></div>';
 if(s.attempt_results.length)h+='<p class="sub" style="margin-top:12px">Earlier attempts: '+s.attempt_results.map(a=>'#'+a.n+' '+(a.status==='met_spec'?'met spec':'no design')).join(', ')+'</p>';
 h+=histSection(s);
 h+='<pre>'+s.log_tail.join('\n').replace(/</g,'&lt;')+'</pre>';$('right').innerHTML=h}
function renderResult(s){RES=s;const r=s.result,ok=r.status==='met_spec',sp=r.spec,list=(r.pareto&&r.pareto.length)?r.pareto:(r.design?[r.design]:[]);
 if(SEL===null||SEL>=list.length){const k=r.design?list.findIndex(x=>JSON.stringify(x.x)===JSON.stringify(r.design.x)):-1;SEL=k<0?0:k}
 const d=list.length?list[SEL]:null;
 let h='<div class="banner '+(ok?'ok':'bad')+'">'+(ok?'Design found. It meets every spec.':'No design met the full spec. Closest design shown.')
  +'<small>'+sp.pdk+' · '+sp.vout+' V · '+sp.cout_uF+' µF · attempts used: '+(r.attempts||1)+' · total time '+mmss(s.elapsed)+'<br>Result of the run you started at '+s.started_at+' (this session). Press Run for a new optimisation.</small></div>';
 if(r.suggestion)h+='<div class="note">'+r.suggestion+'</div>';
 if(d){h+=frontSection(r,list);h+='<div class="tiles"><div class="tile"><span>Pass transistor</span><b>'+d.n_pass+' × '+d.pass_unit_um[0]+'/'+d.pass_unit_um[1]+' µm</b></div>'
  +'<div class="tile"><span>Feedback R1 / R2</span><b>'+fmt(d.R1_ohm/1e3,1)+'k / '+fmt(d.R2_ohm/1e3,0)+'k</b></div>'
  +'<div class="tile"><span>Min Cout for phase margin</span><b>'+(d.min_cout_uF==null?'–':d.min_cout_uF+' µF')+'</b></div>'
  +'<div class="tile"><span>Active area</span><b>'+Math.round(d.area_um2)+' µm²</b></div></div>';
  h+='<h2>Checks (worst case over corners)</h2><div class="scroll"><table><tr><th>Check</th><th class="r">Value</th><th class="r">Limit</th><th></th></tr>'
  +d.checks.map(c=>'<tr><td>'+c.name+'</td><td class="r">'+fmt(c.value,3)+'</td><td class="r">'+c.op+' '+fmt(c.limit,3)+'</td><td class="'+(c.ok?'ok':'bad')+'">'+(c.ok?'pass':'FAIL')+'</td></tr>').join('')+'</table></div>';
  h+='<h2 style="margin-top:16px">By process corner</h2><div class="scroll"><table><tr><th>Corner</th><th class="r">VOUT full load (V)</th><th class="r">Load reg (mV)</th><th class="r">Line reg (mV)</th><th class="r">Iq (µA)</th><th class="r">PSRR 100 mA (dB)</th><th class="r">PSRR 2 mA (dB)</th></tr>'
  +d.corners.map(c=>{const p=d.psrr_dB[c.corner]||{};return '<tr><td>'+c.corner+'</td><td class="r">'+fmt(c.vout_full,4)+'</td><td class="r">'+fmt(c.load_mV,1)+'</td><td class="r">'+fmt(c.line_mV,2)+'</td><td class="r">'+fmt(c.iq_uA,1)+'</td><td class="r">'+fmt(p['100mA'],1)+'</td><td class="r">'+fmt(p['2mA'],1)+'</td></tr>'}).join('')+'</table></div>';
  h+='<h2 style="margin-top:16px">Device sizes (µm)</h2><div class="scroll"><table><tr>'+Object.keys(d.sizes).map(k=>'<th>'+k+'</th>').join('')+'</tr><tr>'
  +Object.values(d.sizes).map(v=>'<td>'+v+'</td>').join('')+'</tr></table></div>';
  h+=histSection(s);
  h+=top3Section(r);
  h+='<br><a class="dl" href="/netlist?i='+SEL+'">Download netlist (.cir)</a><a class="dl" href="/result.json">Download result (.json)</a>'}
 $('right').innerHTML=h;
 $('right').querySelectorAll('[data-i]').forEach(el=>el.onclick=()=>{SEL=parseInt(el.dataset.i);renderResult(RES)})}
async function poll(){clearTimeout(timer);let s;try{s=await (await fetch('/status')).json()}catch(e){timer=setTimeout(poll,3000);return}
 const running=s.state==='running';$('run').disabled=running;$('stop').classList.toggle('hide',!running);
 if(running){renderRunning(s);timer=setTimeout(poll,1500);return}
 if(s.state==='done')renderResult(s);
 else if(s.state==='error')$('right').innerHTML='<div class="banner bad">'+s.error+'</div>'+(s.log_tail?'<pre>'+s.log_tail.join('\n').replace(/</g,'&lt;')+'</pre>':'');
 else if(s.state==='stopped')$('right').innerHTML='<div class="empty">Run stopped.</div>';
 lastState=s.state}
poll();
</script></body></html>"""

if __name__ == "__main__":
    print(f"LDO Designer: open http://localhost:{PORT} in your browser (Ctrl-C here to quit)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)

