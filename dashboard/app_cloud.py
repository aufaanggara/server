import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Simulasi Antrian Server",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown(
    """
<style>
.metric-card {
    background: var(--background-color);
    border: 1px solid rgba(100,181,246,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size: 10px; color: #78909c; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 700; color: #64b5f6; line-height: 1; }
.metric-sub { font-size: 11px; color: #90a4ae; margin-top: 4px; }
.metric-good .metric-value { color: #4caf50; }
.metric-warn .metric-value { color: #ff9800; }
.metric-bad .metric-value { color: #f44336; }
.rho-card {
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-family: monospace;
    font-size: 13px;
    border-left: 3px solid #64b5f6;
    background: var(--secondary-background-color);
}
.littles-card {
    border: 1px solid rgba(100,181,246,0.2);
    border-left: 4px solid #64b5f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
    background: var(--secondary-background-color);
}
.littles-card .eq { font-size: 20px; font-weight: 700; color: #64b5f6; font-family: monospace; }
.alert-stable { background: rgba(46,125,50,0.15); border: 1px solid #2e7d32; border-radius: 8px; padding: 10px 16px; color: #a5d6a7; font-size: 13px; }
.alert-critical { background: rgba(198,40,40,0.15); border: 1px solid #c62828; border-radius: 8px; padding: 10px 16px; color: #ef9a9a; font-size: 13px; }
.alert-warn { background: rgba(230,81,0,0.15); border: 1px solid #e65100; border-radius: 8px; padding: 10px 16px; color: #ffcc80; font-size: 13px; }
.section-header { font-size: 12px; color: #78909c; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px; padding-bottom: 4px; border-bottom: 1px solid rgba(100,181,246,0.15); }
</style>
""",
    unsafe_allow_html=True,
)


# ── Simulation logic ──────────────────────────────────────────────
def select_worker(algorithm, request_id, workers_queue, num_workers):
    if algorithm == "round_robin":
        return request_id % num_workers
    elif algorithm == "random":
        return np.random.randint(0, num_workers)
    elif algorithm == "least_connection":
        return int(np.argmin(workers_queue))
    return request_id % num_workers


def run_simulation(lambda_rate, num_workers, algorithm, duration, mu):
    results = []
    workers_queue = [0] * num_workers
    workers_busy_until = [0.0] * num_workers
    current_time = 0.0
    request_id = 0
    base_time = datetime.now()

    while current_time < duration:
        inter_arrival = np.random.exponential(1.0 / lambda_rate)
        current_time += inter_arrival
        if current_time > duration:
            break
        worker_id = select_worker(algorithm, request_id, workers_queue, num_workers)
        wait_time = max(0.0, workers_busy_until[worker_id] - current_time)
        service_time = np.random.exponential(1.0 / mu)
        start_service = current_time + wait_time
        finish_time = start_service + service_time
        workers_busy_until[worker_id] = finish_time
        workers_queue[worker_id] += 1
        results.append(
            {
                "request_id": request_id,
                "worker_id": worker_id,
                "arrive_time": round(current_time, 4),
                "wait_time": round(wait_time, 4),
                "service_time": round(service_time, 4),
                "total_time": round(wait_time + service_time, 4),
                "timestamp": (base_time + timedelta(seconds=current_time)).isoformat(),
            }
        )
        request_id += 1

    for r in results:
        workers_queue[r["worker_id"]] = max(0, workers_queue[r["worker_id"]] - 1)
    return results


def run_all_algorithms(lambda_rate, num_workers, duration, mu):
    return {
        algo: run_simulation(lambda_rate, num_workers, algo, duration, mu)
        for algo in ["round_robin", "random", "least_connection"]
    }


def plotly_dark(fig, title=""):
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1422",
        font_color="#b0bec5",
        title=dict(text=title, font=dict(color="#90caf9", size=13)),
        xaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        yaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="#0d1422", bordercolor="#1a2a3a"),
    )
    return fig


def get_animation_html(lambda_rate, num_workers, mu, algorithm):
    algo_map = {
        "round_robin": "round_robin",
        "random": "random",
        "least_connection": "least_connection",
    }
    algo_js = algo_map[algorithm]

    # Semua kode HTML animasi, tapi variabel awal diinjek dari Python
    return f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="UTF-8"/>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:100%; overflow:hidden; background:#0a0e1a; font-family:'Segoe UI',Arial,sans-serif; color:#e0e0e0; display:flex; flex-direction:column; }}
    header {{ flex-shrink:0; padding:6px 14px; background:linear-gradient(90deg,#0d1b2a,#1a2a4a); border-bottom:1px solid #1e3a5f; display:flex; align-items:center; justify-content:space-between; }}
    header h1 {{ font-size:13px; font-weight:600; color:#64b5f6; }}
    header .hsub {{ font-size:10px; color:#78909c; }}
    #controls {{ flex-shrink:0; display:flex; flex-wrap:nowrap; gap:8px; padding:6px 12px; background:#0d1422; border-bottom:1px solid #1a2a3a; align-items:flex-end; overflow-x:auto; }}
    .ctrl-group {{ display:flex; flex-direction:column; gap:2px; min-width:90px; flex-shrink:0; }}
    .ctrl-group label {{ font-size:9px; color:#78909c; text-transform:uppercase; letter-spacing:.8px; }}
    .ctrl-group .val {{ font-size:12px; color:#64b5f6; font-weight:600; }}
    input[type=range] {{ width:100%; accent-color:#64b5f6; cursor:pointer; height:16px; }}
    select {{ background:#1a2a3a; color:#e0e0e0; border:1px solid #2a3a4a; padding:4px 6px; border-radius:4px; font-size:11px; cursor:pointer; }}
    .btn-group {{ display:flex; gap:6px; margin-left:auto; align-items:flex-end; flex-shrink:0; }}
    button {{ padding:5px 11px; border:none; border-radius:5px; cursor:pointer; font-size:11px; font-weight:600; transition:all .2s; white-space:nowrap; }}
    #btnPlay {{ background:#1565c0; color:#fff; }} #btnPlay:hover {{ background:#1976d2; }} #btnPlay.paused {{ background:#2e7d32; }}
    #btnReset {{ background:#37474f; color:#cfd8dc; }} #btnReset:hover {{ background:#455a64; }}
    #btnStress {{ background:#7b3b00; color:#ffcc80; }} #btnStress.active {{ background:#e65100; color:#fff; animation:stressPulse .8s infinite alternate; }}
    @keyframes stressPulse {{ from{{box-shadow:0 0 4px rgba(230,81,0,.4)}} to{{box-shadow:0 0 14px rgba(230,81,0,.9)}} }}
    #formulaBar {{ flex-shrink:0; background:#0a1628; border-bottom:1px solid #1a2a3a; padding:4px 12px; font-size:10px; color:#78909c; display:flex; align-items:center; gap:8px; }}
    #formulaBar .desc {{ color:#90caf9; }} #timerBar {{ margin-left:auto; display:flex; align-items:center; gap:12px; font-family:monospace; font-size:10px; }}
    #timerBar .timer {{ color:#80cbc4; }} #timerBar .rhoF {{ color:#a5d6a7; }}
    #main {{ flex:1; display:flex; overflow:hidden; min-height:0; }}
    #canvasWrap {{ position:relative; flex:0 0 auto; display:flex; }}
    #alertBanner {{ display:none; position:absolute; top:8px; left:50%; transform:translateX(-50%); background:rgba(183,28,28,.9); color:#fff; padding:5px 16px; border-radius:20px; font-size:11px; font-weight:600; border:1px solid #ef5350; animation:pulse 1s infinite alternate; z-index:10; white-space:nowrap; }}
    #autopauseMsg {{ display:none; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); background:rgba(10,14,26,.95); border:1px solid #ef5350; border-radius:12px; padding:18px 24px; text-align:center; z-index:20; }}
    #autopauseMsg h3 {{ color:#ef5350; font-size:13px; margin-bottom:6px; }}
    #autopauseMsg p {{ color:#90a4ae; font-size:11px; line-height:1.6; }}
    #autopauseMsg button {{ margin-top:10px; background:#1565c0; color:#fff; padding:5px 16px; border:none; border-radius:6px; cursor:pointer; font-size:11px; }}
    @keyframes pulse {{ from{{box-shadow:0 0 6px rgba(239,83,80,.4)}} to{{box-shadow:0 0 20px rgba(239,83,80,.95)}} }}
    #rightPanel {{ flex:1; display:flex; flex-direction:column; overflow:hidden; border-left:1px solid #1a2a3a; min-width:0; }}
    #statsGrid {{ flex-shrink:0; display:grid; grid-template-columns:repeat(3,1fr); border-bottom:1px solid #1a2a3a; }}
    .stat-item {{ padding:8px 10px; border-right:1px solid #1a2a3a; display:flex; flex-direction:column; gap:1px; }}
    .stat-item:nth-child(3n) {{ border-right:none; }}
    .stat-label {{ font-size:9px; color:#546e7a; text-transform:uppercase; letter-spacing:.8px; }}
    .stat-value {{ font-size:16px; font-weight:700; color:#64b5f6; transition:color .3s; }}
    .stat-sub {{ font-size:9px; color:#78909c; }}
    .stat-formula {{ font-size:9px; color:#455a64; font-family:monospace; }}
    #algoTable {{ flex-shrink:0; padding:6px 10px 4px; border-bottom:1px solid #1a2a3a; }}
    .at-title {{ font-size:9px; color:#546e7a; text-transform:uppercase; letter-spacing:.8px; margin-bottom:4px; }}
    table.at {{ width:100%; border-collapse:collapse; font-size:10px; }}
    table.at th {{ color:#546e7a; font-weight:500; text-align:left; padding:2px 6px; border-bottom:1px solid #1a2a3a; font-size:9px; }}
    table.at td {{ padding:3px 6px; color:#90a4ae; border-bottom:1px solid #0d1422; }}
    table.at tr.active-row td {{ color:#e0e0e0; background:#0d1a2e; }}
    table.at td.num {{ font-family:monospace; color:#64b5f6; }}
    table.at td.num.good {{ color:#4caf50; }} table.at td.num.warn {{ color:#ff9800; }} table.at td.num.bad {{ color:#f44336; }}
    #chartArea {{ flex:1; display:flex; flex-direction:column; overflow:hidden; padding:6px 10px 8px; gap:6px; min-height:0; }}
    .chart-col {{ flex:1; display:flex; flex-direction:column; min-height:0; }}
    .chart-title {{ font-size:9px; color:#546e7a; text-transform:uppercase; letter-spacing:.8px; margin-bottom:3px; flex-shrink:0; }}
    canvas.chart {{ flex:1; width:100%; display:block; min-height:0; }}
  </style>
</head>
<body>
<header>
  <span style="font-size:16px"></span>
  <h1>Simulasi Stokastik Sistem Antrian Server</h1>
  <span class="hsub">Distribusi Poisson &amp; Eksponensial — M/M/c Queue</span>
</header>
<div id="controls">
  <div class="ctrl-group">
    <label>λ — Arrival Rate</label>
    <input type="range" id="slLambda" min="0.5" max="5" step="0.5" value="{lambda_rate}">
    <div class="val"><span id="valLambda">{lambda_rate:.1f}</span> req/s</div>
  </div>
  <div class="ctrl-group">
    <label>Workers (c)</label>
    <input type="range" id="slWorkers" min="1" max="5" step="1" value="{num_workers}">
    <div class="val"><span id="valWorkers">{num_workers}</span> server</div>
  </div>
  <div class="ctrl-group">
    <label>μ — Service Rate</label>
    <input type="range" id="slMu" min="0.5" max="3" step="0.5" value="{mu}">
    <div class="val"><span id="valMu">{mu:.1f}</span> req/s</div>
  </div>
  <div class="ctrl-group">
    <label>Speed</label>
    <input type="range" id="slSpeed" min="0.5" max="3" step="0.5" value="1">
    <div class="val"><span id="valSpeed">1.0</span>x</div>
  </div>
  <div class="ctrl-group">
    <label>Algoritma</label>
    <select id="selAlgo">
      <option value="round_robin" {'selected' if algo_js=='round_robin' else ''}>Round Robin</option>
      <option value="random" {'selected' if algo_js=='random' else ''}>Random</option>
      <option value="least_connection" {'selected' if algo_js=='least_connection' else ''}>Least Connection</option>
    </select>
  </div>
  <div class="btn-group">
    <button id="btnPlay" onclick="togglePlay()">⏸ Pause</button>
    <button id="btnReset" onclick="resetSim()">↺ Reset</button>
    <button id="btnStress" onclick="toggleStress()">📈 Stress</button>
  </div>
</div>
<div id="formulaBar">
  <span style="color:#546e7a;font-size:9px;text-transform:uppercase;letter-spacing:.8px">Algoritma:</span>
  <span class="desc" id="algoDescText"></span>
  <div id="timerBar">
    <span class="timer">⏱ <span id="timerVal">00:00</span></span>
    <span class="rhoF" id="rhoFormula"></span>
  </div>
</div>
<div id="main">
  <div id="canvasWrap">
    <div id="alertBanner">🔴 SISTEM KRITIS — ρ ≥ 1!</div>
    <div id="autopauseMsg">
      <h3>⚠️ Sistem Tidak Stabil (ρ ≥ 1)</h3>
      <p>Antrian akan terus membesar.<br>Tambah worker, kurangi λ, atau tingkatkan μ.</p>
      <button onclick="resumeFromPause()">▶ Lanjutkan</button>
    </div>
  </div>
  <div id="rightPanel">
    <div id="statsGrid">
      <div class="stat-item"><div class="stat-label">Total Request</div><div class="stat-value" id="stTotal">0</div><div class="stat-sub">sejak reset</div></div>
      <div class="stat-item"><div class="stat-label">Selesai</div><div class="stat-value" id="stDone">0</div><div class="stat-sub" id="stDonePct">0% success</div></div>
      <div class="stat-item"><div class="stat-label">Throughput</div><div class="stat-value" id="stTput">0.00</div><div class="stat-sub">req/s aktual</div></div>
      <div class="stat-item"><div class="stat-label">Utilisasi (ρ)</div><div class="stat-value" id="stRho">0.67</div><div class="stat-sub" id="stRhoStatus">✅ stabil</div><div class="stat-formula" id="stRhoFormula"></div></div>
      <div class="stat-item"><div class="stat-label">Avg Wait (Wq)</div><div class="stat-value" id="stWait">0.00</div><div class="stat-sub">detik</div></div>
      <div class="stat-item"><div class="stat-label">Little's L=λW</div><div class="stat-value" id="stL">0.00</div><div class="stat-sub" id="stLsub">λ×W = L</div></div>
    </div>
    <div id="algoTable">
      <div class="at-title">Perbandingan Algoritma (Live)</div>
      <table class="at">
        <thead><tr><th>Algoritma</th><th>Total</th><th>Tput</th><th>Avg Wait</th><th>Avg Svc</th><th>L=λW</th><th>Status</th></tr></thead>
        <tbody>
          <tr id="row_round_robin"><td>Round Robin</td><td class="num" id="at_rr_total">—</td><td class="num" id="at_rr_tput">—</td><td class="num" id="at_rr_wait">—</td><td class="num" id="at_rr_svc">—</td><td class="num" id="at_rr_L">—</td><td id="at_rr_status">—</td></tr>
          <tr id="row_random"><td>Random</td><td class="num" id="at_rd_total">—</td><td class="num" id="at_rd_tput">—</td><td class="num" id="at_rd_wait">—</td><td class="num" id="at_rd_svc">—</td><td class="num" id="at_rd_L">—</td><td id="at_rd_status">—</td></tr>
          <tr id="row_least_connection"><td>Least Conn.</td><td class="num" id="at_lc_total">—</td><td class="num" id="at_lc_tput">—</td><td class="num" id="at_lc_wait">—</td><td class="num" id="at_lc_svc">—</td><td class="num" id="at_lc_L">—</td><td id="at_lc_status">—</td></tr>
        </tbody>
      </table>
    </div>
    <div id="chartArea">
      <div class="chart-col"><div class="chart-title">Throughput Real-time (req/s)</div><canvas id="chartTput" class="chart"></canvas></div>
      <div class="chart-col"><div class="chart-title">Distribusi Service Time</div><canvas id="chartSvc" class="chart"></canvas></div>
    </div>
  </div>
</div>
<script>
  let lambda={lambda_rate}, numWorkers={num_workers}, mu={mu}, speed=1;
  let algorithm="{algo_js}";
  let paused=false, autoPaused=false, userOverride=false, stressActive=false;
  let packets=[], workers=[], trails=[];
  let stats={{total:0,done:0,waitTimes:[],serviceTimes:[],startTime:performance.now()}};
  let nextSpawn=0, reqId=0, lineAnimOffset=0;
  let tputData=[], lastChartUpdate=0;
  const algoStats={{
    round_robin:{{total:0,done:0,waitTimes:[],serviceTimes:[],startTime:performance.now()}},
    random:{{total:0,done:0,waitTimes:[],serviceTimes:[],startTime:performance.now()}},
    least_connection:{{total:0,done:0,waitTimes:[],serviceTimes:[],startTime:performance.now()}}
  }};
  const ALGO_DESC={{round_robin:"Round Robin — bergiliran ke setiap worker",random:"Random — worker dipilih secara acak",least_connection:"Least Connection — worker dengan antrian terpendek"}};
  const ALGO_SHORT={{round_robin:"RR",random:"RND",least_connection:"LC"}};
  const ALGO_KEY={{round_robin:"rr",random:"rd",least_connection:"lc"}};

  let audioCtx=null;
  function ensureAudio(){{if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();}}
  function playDone(){{try{{ensureAudio();let o=audioCtx.createOscillator(),g=audioCtx.createGain();o.connect(g);g.connect(audioCtx.destination);o.frequency.value=880;g.gain.setValueAtTime(0.05,audioCtx.currentTime);g.gain.exponentialRampToValueAtTime(0.001,audioCtx.currentTime+0.07);o.start();o.stop(audioCtx.currentTime+0.07);}}catch(e){{}}}}
  function playAlarm(){{try{{ensureAudio();[440,330].forEach((f,i)=>{{let o=audioCtx.createOscillator(),g=audioCtx.createGain();o.connect(g);g.connect(audioCtx.destination);o.frequency.value=f;g.gain.setValueAtTime(0.1,audioCtx.currentTime+i*.15);g.gain.exponentialRampToValueAtTime(0.001,audioCtx.currentTime+i*.15+.18);o.start(audioCtx.currentTime+i*.15);o.stop(audioCtx.currentTime+i*.15+.18);}});}}catch(e){{}}}}
  let lastAlarm=0;

  function bind(id,valId,dec,cb){{const el=document.getElementById(id);el.oninput=()=>{{document.getElementById(valId).innerText=parseFloat(el.value).toFixed(dec);cb(parseFloat(el.value));updateFormulaBar();}};}}
  bind("slLambda","valLambda",1,v=>lambda=v);
  bind("slWorkers","valWorkers",0,v=>{{numWorkers=v;initWorkers();}});
  bind("slMu","valMu",1,v=>mu=v);
  bind("slSpeed","valSpeed",1,v=>speed=v);
  document.getElementById("selAlgo").onchange=e=>{{algorithm=e.target.value;document.getElementById("algoDescText").textContent=ALGO_DESC[algorithm];highlightAlgoRow();}};

  function updateFormulaBar(){{let rho=lambda/(numWorkers*mu);document.getElementById("rhoFormula").textContent=`ρ = ${{lambda.toFixed(1)}}/${{numWorkers}}×${{mu.toFixed(1)}}) = ${{rho.toFixed(2)}}`;}}
  function highlightAlgoRow(){{["round_robin","random","least_connection"].forEach(a=>document.getElementById(`row_${{a}}`).classList.toggle("active-row",a===algorithm));}}
  function togglePlay(){{if(autoPaused)return;paused=!paused;const btn=document.getElementById("btnPlay");btn.textContent=paused?"▶ Play":"⏸ Pause";btn.classList.toggle("paused",paused);}}
  function resumeFromPause(){{autoPaused=false;userOverride=true;paused=false;document.getElementById("autopauseMsg").style.display="none";document.getElementById("btnPlay").textContent="⏸ Pause";document.getElementById("btnPlay").classList.remove("paused");}}
  function resetSim(){{packets=[];trails=[];stats={{total:0,done:0,waitTimes:[],serviceTimes:[],startTime:performance.now()}};tputData=[];reqId=0;nextSpawn=0;autoPaused=false;userOverride=false;document.getElementById("autopauseMsg").style.display="none";initWorkers();["chartTput","chartSvc"].forEach(id=>{{const c=document.getElementById(id);if(c.width)c.getContext("2d").clearRect(0,0,c.width,c.height);}});updateFormulaBar();}}
  function initWorkers(){{workers=Array.from({{length:numWorkers}},(_,i)=>{{return{{id:i,queue:0,busy:false,busyTimer:0,processed:0,pendingQueue:[]}}}});}}
  function selectWorker(){{if(algorithm==="round_robin")return reqId%numWorkers;if(algorithm==="random")return Math.floor(Math.random()*numWorkers);return workers.reduce((a,b)=>b.queue<a.queue?b:a).id;}}
  function expRandom(rate){{return-Math.log(Math.random())/rate;}}

  let stressInterval=null;
  function toggleStress(){{stressActive=!stressActive;const btn=document.getElementById("btnStress");btn.classList.toggle("active",stressActive);if(stressActive){{btn.textContent="⏹ Stop Stress";resetSim();paused=false;document.getElementById("btnPlay").textContent="⏸ Pause";document.getElementById("btnPlay").classList.remove("paused");lambda=0.5;document.getElementById("slLambda").value=0.5;document.getElementById("valLambda").textContent="0.5";updateFormulaBar();stressInterval=setInterval(()=>{{if(!stressActive)return;let newL=Math.min(lambda+0.5,5);lambda=newL;document.getElementById("slLambda").value=newL;document.getElementById("valLambda").textContent=newL.toFixed(1);updateFormulaBar();if(newL>=5)clearInterval(stressInterval);}},3000);}}else{{btn.textContent="📈 Stress";clearInterval(stressInterval);}}}}

  function formatTimer(s){{let m=Math.floor(s/60),sec=Math.floor(s%60);return`${{String(m).padStart(2,"0")}}:${{String(sec).padStart(2,"0")}}`;}}
  setInterval(()=>{{if(!paused&&!autoPaused){{let s=(performance.now()-stats.startTime)/1000;document.getElementById("timerVal").textContent=formatTimer(s);}}}},500);

  function drawTputChart(){{const el=document.getElementById("chartTput");const ctx=el.getContext("2d");const W=el.offsetWidth,H=el.offsetHeight||60;el.width=W;el.height=H;ctx.clearRect(0,0,W,H);if(tputData.length<2)return;const max=Math.max(...tputData,.1),step=W/(tputData.length-1);ctx.strokeStyle="rgba(30,58,95,.5)";ctx.lineWidth=.5;for(let i=0;i<=3;i++){{const y=H/3*i;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}const g=ctx.createLinearGradient(0,0,0,H);g.addColorStop(0,"rgba(100,181,246,.3)");g.addColorStop(1,"rgba(100,181,246,.02)");ctx.fillStyle=g;ctx.beginPath();ctx.moveTo(0,H);tputData.forEach((v,i)=>ctx.lineTo(i*step,H-(v/max)*(H-8)));ctx.lineTo((tputData.length-1)*step,H);ctx.closePath();ctx.fill();ctx.strokeStyle="#64b5f6";ctx.lineWidth=1.5;ctx.lineJoin="round";ctx.beginPath();tputData.forEach((v,i)=>{{const x=i*step,y=H-(v/max)*(H-8);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});ctx.stroke();const lx=(tputData.length-1)*step,ly=H-(tputData[tputData.length-1]/max)*(H-8);ctx.fillStyle="#64b5f6";ctx.beginPath();ctx.arc(lx,ly,3,0,Math.PI*2);ctx.fill();ctx.fillStyle="#546e7a";ctx.font="9px Segoe UI";ctx.fillText(max.toFixed(1),2,10);ctx.fillText("0",2,H-2);ctx.fillStyle="#90caf9";ctx.fillText(tputData[tputData.length-1].toFixed(2)+" req/s",W-66,10);}}
  function drawSvcChart(){{const el=document.getElementById("chartSvc");const ctx=el.getContext("2d");const W=el.offsetWidth,H=el.offsetHeight||60;el.width=W;el.height=H;ctx.clearRect(0,0,W,H);const data=stats.serviceTimes.slice(-300);if(data.length<5)return;const bins=14,maxV=Math.min(Math.max(...data),5),binW=maxV/bins;const counts=new Array(bins).fill(0);data.forEach(v=>{{const b=Math.min(Math.floor(v/binW),bins-1);counts[b]++;}});const maxC=Math.max(...counts,1),bw=W/bins;counts.forEach((c,i)=>{{const x=i*bw,h=(c/maxC)*(H-8),alpha=Math.max(.15,1-i/bins*.65);ctx.fillStyle=`rgba(100,181,246,${{alpha}})`;ctx.fillRect(x+1,H-h,bw-2,h);}});ctx.strokeStyle="#ef9f27";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);ctx.beginPath();for(let i=0;i<=bins;i++){{const x=(i/bins)*W,v=(i/bins)*maxV,ey=mu*Math.exp(-mu*v),normY=Math.min(ey/mu,1),y=H-normY*(H-8);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}}ctx.stroke();ctx.setLineDash([]);}}

  function updateAlgoTable(rho){{const key=ALGO_KEY[algorithm];const as=algoStats[algorithm];const elapsed=(performance.now()-as.startTime)/1000;const atput=elapsed>0?as.done/elapsed:0;const arec=as.waitTimes.slice(-30);const await_=arec.length>0?arec.reduce((a,b)=>a+b,0)/arec.length:0;const asvc=as.serviceTimes.length>0?as.serviceTimes.slice(-30).reduce((a,b)=>a+b,0)/Math.min(30,as.serviceTimes.length):1/mu;const aL=atput*(await_+asvc);const setCell=(id,val,cls)=>{{const el=document.getElementById(id);if(el){{el.textContent=val;el.className="num"+(cls?" "+cls:"");}}}};setCell(`at_${{key}}_total`,as.total);setCell(`at_${{key}}_tput`,atput.toFixed(2));setCell(`at_${{key}}_wait`,await_.toFixed(3),await_<0.3?"good":await_<1?"warn":"bad");setCell(`at_${{key}}_svc`,asvc.toFixed(3));setCell(`at_${{key}}_L`,aL.toFixed(2));const statusEl=document.getElementById(`at_${{key}}_status`);if(statusEl)statusEl.textContent=rho<0.7?"✅ stabil":rho<1?"⚠️ kritis":"🔴 collapse";highlightAlgoRow();}}

  new p5(function(p){{
    let W,H;
    p.setup=function(){{
      const wrap=document.getElementById("canvasWrap");
      W=wrap.offsetWidth||600; H=wrap.offsetHeight||400;
      if(W===0)W=Math.floor(window.innerWidth*.58);
      if(H===0)H=window.innerHeight-120;
      let c=p.createCanvas(W,H); c.parent("canvasWrap");
      p.textFont("Segoe UI"); initWorkers(); updateFormulaBar(); highlightAlgoRow();
      document.getElementById("algoDescText").textContent=ALGO_DESC[algorithm];
      new ResizeObserver(()=>{{const wr=document.getElementById("canvasWrap");let nw=wr.offsetWidth,nh=wr.offsetHeight;if(nw>10&&nh>10){{W=nw;H=nh;p.resizeCanvas(W,H);}}}}).observe(document.getElementById("canvasWrap"));
    }};
    p.draw=function(){{
      if(paused||autoPaused)return;
      let dt=(p.deltaTime/1000)*speed;
      lineAnimOffset=(lineAnimOffset+dt*28)%12;
      p.background(10,14,26); drawGrid(p,W,H);
      let lbX=W*.32,lbY=H/2,genX=W*.08,genY=H/2;
      let wSpacing=H/(numWorkers+1);
      let rho=lambda/(numWorkers*mu);
      document.getElementById("alertBanner").style.display=rho>=1?"block":"none";
      if(rho>=1){{let pulse=(Math.sin(p.millis()/300)+1)/2;p.noStroke();p.fill(180,20,20,pulse*18);p.rect(0,0,W,H);let now=performance.now();if(!autoPaused&&!userOverride&&stats.total>8){{autoPaused=true;playAlarm();document.getElementById("autopauseMsg").style.display="block";}}else if(userOverride&&now-lastAlarm>4000){{playAlarm();lastAlarm=now;}}}}
      for(let w of workers){{if(w.busy){{w.busyTimer-=dt;if(w.busyTimer<=0){{w.busy=false;w.queue=Math.max(0,w.queue-1);w.processed++;stats.done++;algoStats[algorithm].done++;playDone();if(w.pendingQueue.length>0){{let nx=w.pendingQueue.shift();let waitT=(performance.now()-nx.arriveT)/1000;stats.waitTimes.push(waitT);algoStats[algorithm].waitTimes.push(waitT);let svcT=expRandom(mu);stats.serviceTimes.push(svcT);algoStats[algorithm].serviceTimes.push(svcT);w.busy=true;w.busyTimer=svcT;}}}}}}}}
      nextSpawn-=dt;
      if(nextSpawn<=0){{let wid=selectWorker();let wx=W*.78,wy=wSpacing*(wid+1);let arriveT=performance.now();packets.push({{x:genX,y:genY,tx:wx,ty:wy,wid,phase:"toLB",alpha:255,id:reqId,size:10,arriveT,waitHeat:0}});workers[wid].queue++;if(!workers[wid].busy){{let svcT=expRandom(mu);stats.serviceTimes.push(svcT);algoStats[algorithm].serviceTimes.push(svcT);workers[wid].busy=true;workers[wid].busyTimer=svcT;stats.waitTimes.push(0);algoStats[algorithm].waitTimes.push(0);}}else{{workers[wid].pendingQueue.push({{arriveT}});}}stats.total++;algoStats[algorithm].total++;reqId++;nextSpawn=expRandom(lambda);}}
      for(let t of trails){{t.alpha-=8;if(t.alpha>0){{p.noStroke();p.fill(100,180,255,t.alpha*.2);p.ellipse(t.x,t.y,t.size*.5);}}}}
      trails=trails.filter(t=>t.alpha>0);
      drawAnimLine(p,genX+28,genY,lbX-38,lbY,lineAnimOffset);
      for(let i=0;i<numWorkers;i++)drawAnimLine(p,lbX+38,lbY,W*.78-44,wSpacing*(i+1),lineAnimOffset);
      for(let i=0;i<numWorkers;i++){{let w=workers[i],wx=W*.78,wy=wSpacing*(i+1),qLen=w.pendingQueue.length;for(let q=0;q<Math.min(qLen,6);q++){{let age=w.pendingQueue[Math.min(q,w.pendingQueue.length-1)]?(performance.now()-w.pendingQueue[Math.min(q,w.pendingQueue.length-1)].arriveT)/1000:0;let heat=Math.min(age/3,1);let r=Math.round(241+heat*-10),g2=Math.round(196-heat*120),b=60;p.noStroke();p.fill(r,g2,b,190);p.ellipse(wx-50-q*11,wy,9);p.fill(255,200);p.textSize(6);p.textAlign(p.CENTER,p.CENTER);p.text(q+1,wx-50-q*11,wy);}}if(qLen>6){{p.fill(231,76,60,200);p.textSize(8);p.textAlign(p.LEFT,p.CENTER);p.text(`+${{qLen-6}}`,wx-50-6*11,wy);}}}}
      for(let pk of packets){{pk.waitHeat=Math.min((performance.now()-pk.arriveT)/3000,1);let r=Math.round(150+pk.waitHeat*81),gb=Math.round(210-pk.waitHeat*134);if(pk.phase==="toLB"){{pk.x=p.lerp(pk.x,lbX,.09*speed);pk.y=p.lerp(pk.y,lbY,.09*speed);if(p.dist(pk.x,pk.y,lbX,lbY)<8)pk.phase="toWorker";}}else if(pk.phase==="toWorker"){{pk.x=p.lerp(pk.x,pk.tx,.07*speed);pk.y=p.lerp(pk.y,pk.ty,.07*speed);if(p.dist(pk.x,pk.y,pk.tx,pk.ty)<8)pk.phase="arrive";}}else if(pk.phase==="arrive"){{pk.size=p.lerp(pk.size,20,.2);pk.alpha-=12;if(pk.alpha<=0)pk.phase="done";}}if(pk.phase!=="done"&&pk.alpha>0){{trails.push({{x:pk.x,y:pk.y,alpha:pk.alpha*.4,size:pk.size}});p.noStroke();p.fill(r,gb,gb*.6,pk.alpha*.1);p.ellipse(pk.x,pk.y,pk.size*3);p.fill(r,gb,gb*.6,pk.alpha*.22);p.ellipse(pk.x,pk.y,pk.size*2);p.fill(r,gb,255-pk.waitHeat*200,pk.alpha);p.ellipse(pk.x,pk.y,pk.size);p.fill(255,pk.alpha);p.textSize(7);p.textAlign(p.CENTER,p.CENTER);p.text(pk.id,pk.x,pk.y);}}}}
      packets=packets.filter(pk=>pk.phase!=="done");
      drawBox(p,genX,genY,56,52,[30,80,160],[64,148,230],"REQUEST\\nGENERATOR",`λ=${{lambda}}/s`);
      drawBox(p,lbX,lbY,76,58,[80,40,160],[150,100,230],"LOAD\\nBALANCER",ALGO_SHORT[algorithm]);
      for(let i=0;i<numWorkers;i++){{let w=workers[i],wx=W*.78,wy=wSpacing*(i+1),col=workerColor(w);if(w.busy){{p.noStroke();p.fill(col[0],col[1],col[2],22+Math.sin(p.millis()/200)*12);p.rect(wx-58,wy-38,116,76,14);}}drawWorkerBox(p,wx,wy,w,col,i);}}
      updateStatsUI(rho);
    }};
    function drawAnimLine(p,x1,y1,x2,y2,off){{let d=p.dist(x1,y1,x2,y2);p.stroke(30,70,120,70);p.strokeWeight(1);p.line(x1,y1,x2,y2);let steps=Math.ceil(d/12);for(let i=0;i<steps;i++){{let t1=((i*12+off)%d)/d,t2=((i*12+6+off)%d)/d;if(t1<0||t1>1||t2<0||t2>1)continue;let ax=p.lerp(x1,x2,t1),ay=p.lerp(y1,y2,t1),bx=p.lerp(x1,x2,t2),by=p.lerp(y1,y2,t2);p.stroke(100,180,255,170);p.strokeWeight(1.8);p.line(ax,ay,bx,by);}}}}
    function drawGrid(p,W,H){{p.stroke(18,28,46);p.strokeWeight(.5);for(let x=0;x<W;x+=40)p.line(x,0,x,H);for(let y=0;y<H;y+=40)p.line(0,y,W,y);}}
    function drawBox(p,x,y,w,h,cD,cL,title,sub){{p.noStroke();p.fill(0,0,0,60);p.rect(x-w/2+3,y-h/2+3,w,h,10);p.fill(cD[0],cD[1],cD[2]);p.rect(x-w/2,y-h/2,w,h,10);p.fill(cL[0],cL[1],cL[2],40);p.rect(x-w/2,y-h/2,w,h/2,10,10,0,0);p.stroke(cL[0],cL[1],cL[2],130);p.strokeWeight(1.5);p.noFill();p.rect(x-w/2,y-h/2,w,h,10);p.noStroke();p.fill(220,235,255);p.textSize(9);p.textAlign(p.CENTER,p.CENTER);p.text(title,x,y-8);p.fill(cL[0],cL[1],cL[2]);p.textSize(10);p.textStyle(p.BOLD);p.text(sub,x,y+9);p.textStyle(p.NORMAL);}}
    function drawWorkerBox(p,x,y,w,col,idx){{let bw=88,bh=60;p.noStroke();p.fill(0,0,0,60);p.rect(x-bw/2+3,y-bh/2+3,bw,bh,10);p.fill(col[0]*.35,col[1]*.35,col[2]*.35);p.rect(x-bw/2,y-bh/2,bw,bh,10);p.fill(col[0],col[1],col[2],50);p.rect(x-bw/2,y-bh/2,bw,bh/2,10,10,0,0);p.stroke(col[0],col[1],col[2],180);p.strokeWeight(1.5);p.noFill();p.rect(x-bw/2,y-bh/2,bw,bh,10);p.noStroke();p.fill(w.busy?[231,76,60]:[46,204,113]);p.ellipse(x+bw/2-10,y-bh/2+10,7);p.fill(220,235,255);p.textSize(9);p.textAlign(p.CENTER,p.CENTER);p.text(`WORKER ${{idx}}`,x,y-14);p.fill(col[0]+80,col[1]+80,col[2]+80);p.textSize(10);p.textStyle(p.BOLD);p.text(`Queue: ${{w.queue}}`,x,y+2);p.textStyle(p.NORMAL);p.textSize(8);p.fill(w.busy?[231,120,100]:[100,200,130]);p.text(w.busy?"● BUSY":"● IDLE",x,y+17);if(w.busy&&w.busyTimer>0){{let prog=Math.max(0,Math.min(1,1-w.busyTimer/(1/mu)));p.noStroke();p.fill(30,40,60);p.rect(x-30,y+26,60,5,3);p.fill(col[0],col[1],col[2],200);p.rect(x-30,y+26,60*prog,5,3);}}}}
    function workerColor(w){{if(w.queue===0)return[46,204,113];if(w.queue<=2)return[241,196,15];return[231,76,60];}}
    function updateStatsUI(rho){{let elapsed=(performance.now()-stats.startTime)/1000;let tput=elapsed>0?stats.done/elapsed:0;let recent=stats.waitTimes.slice(-30);let avgWait=recent.length>0?recent.reduce((a,b)=>a+b,0)/recent.length:0;let avgSvc=stats.serviceTimes.length>0?stats.serviceTimes.slice(-30).reduce((a,b)=>a+b,0)/Math.min(30,stats.serviceTimes.length):1/mu;let W_total=avgWait+avgSvc,L=tput*W_total;document.getElementById("stTotal").textContent=stats.total;document.getElementById("stDone").textContent=stats.done;document.getElementById("stDonePct").textContent=stats.total>0?`${{Math.round(stats.done/stats.total*100)}}% success`:"0% success";document.getElementById("stTput").textContent=tput.toFixed(2);let rhoEl=document.getElementById("stRho");rhoEl.textContent=rho.toFixed(2);rhoEl.style.color=rho<0.7?"#4caf50":rho<1?"#ff9800":"#f44336";document.getElementById("stRhoStatus").textContent=rho<0.7?"✅ stabil":rho<0.9?"⚠️ mendekati kritis":"🔴 kritis!";document.getElementById("stRhoFormula").textContent=`${{lambda.toFixed(1)}}/(${{numWorkers}}×${{mu.toFixed(1)}})`;document.getElementById("stWait").textContent=avgWait.toFixed(2);document.getElementById("stL").textContent=L.toFixed(2);document.getElementById("stLsub").textContent=`${{tput.toFixed(2)}}×${{W_total.toFixed(2)}}=${{L.toFixed(2)}}`;let now=performance.now();if(now-lastChartUpdate>1000){{tputData.push(tput);if(tputData.length>60)tputData.shift();lastChartUpdate=now;drawTputChart();drawSvcChart();}}updateAlgoTable(rho);}}
  }});
  window.addEventListener("load",()=>{{const wrap=document.getElementById("canvasWrap");const main=document.getElementById("main");const rightW=380;wrap.style.width=main.offsetWidth-rightW+"px";wrap.style.height=main.offsetHeight+"px";document.getElementById("rightPanel").style.width=rightW+"px";}});
  window.addEventListener("resize",()=>{{const wrap=document.getElementById("canvasWrap");const main=document.getElementById("main");const rightW=380;wrap.style.width=main.offsetWidth-rightW+"px";wrap.style.height=main.offsetHeight+"px";}});
</script>
</body>
</html>"""


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")
    st.markdown(
        '<div class="section-header">Parameter Stokastik</div>', unsafe_allow_html=True
    )

    lambda_rate = st.slider("λ — Arrival Rate (req/s)", 0.5, 5.0, 2.0, 0.5)
    mu = st.slider("μ — Service Rate (req/s)", 0.5, 3.0, 1.0, 0.5)
    num_workers = st.slider("Jumlah Worker (c)", 1, 5, 3)

    rho = lambda_rate / (num_workers * mu)
    rho_color = "#4caf50" if rho < 0.7 else "#ff9800" if rho < 1 else "#f44336"
    rho_status = (
        "✅ Stabil"
        if rho < 0.7
        else "⚠️ Mendekati kritis" if rho < 1 else "🔴 Tidak stabil!"
    )
    st.markdown(
        f"""
    <div class="rho-card">
        ρ = λ/(c·μ) = {lambda_rate:.1f}/({num_workers}×{mu:.1f}) = 
        <span style="color:{rho_color};font-weight:700;font-size:16px">{rho:.2f}</span><br>
        <span style="color:{rho_color};font-size:12px">{rho_status}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">Konfigurasi Simulasi</div>', unsafe_allow_html=True
    )
    algorithm = st.selectbox(
        "Algoritma Load Balancer",
        ["round_robin", "random", "least_connection"],
        format_func=lambda x: {
            "round_robin": "Round Robin",
            "random": "Random",
            "least_connection": "Least Connection",
        }[x],
    )
    duration = st.slider("Durasi Simulasi (detik)", 10, 60, 30, 10)

    st.markdown("---")
    run = st.button("▶ Jalankan Simulasi")

    st.markdown(
        '<div class="section-header">Tentang Model</div>', unsafe_allow_html=True
    )
    st.markdown(
        """
    <div style="font-size:11px;color:#546e7a;line-height:1.7">
    <b style="color:#78909c">M/M/c Queue:</b><br>
    • Arrival: Distribusi Poisson (λ)<br>
    • Service: Distribusi Eksponensial (μ)<br>
    • c: Jumlah worker paralel<br>
    • ρ &lt; 1 → sistem stabil
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── MAIN ──────────────────────────────────────────────────────────
st.markdown("# Simulasi Stokastik Sistem Antrian Server")
st.markdown(
    "**Pemodelan & Simulasi Stokastik** — Distribusi Poisson & Eksponensial | Model M/M/c Queue"
)
st.divider()

if not run:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:20px">M/M/c</div>
            <div class="metric-sub">Kendall's Notation</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-card {'metric-good' if rho < 0.7 else 'metric-warn' if rho < 1 else 'metric-bad'}">
            <div class="metric-label">Utilisasi ρ saat ini</div>
            <div class="metric-value">{rho:.2f}</div>
            <div class="metric-sub">{rho_status}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Algoritma tersedia</div>
            <div class="metric-value" style="font-size:20px">3</div>
            <div class="metric-sub">RR · Random · LC</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-left:4px solid #64b5f6;border-radius:10px;padding:20px 24px;margin-bottom:16px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Cara Penggunaan</div>
        <div style="font-size:13px;color:#90a4ae;line-height:2.2">
        1. Atur <b style="color:#64b5f6">λ</b> — arrival rate request per detik (Distribusi Poisson)<br>
        2. Atur <b style="color:#64b5f6">μ</b> — service rate per worker (Distribusi Eksponensial)<br>
        3. Atur <b style="color:#64b5f6">c</b> — jumlah worker paralel<br>
        4. Pastikan <b style="color:#4caf50">ρ = λ/(c·μ) &lt; 1</b> agar sistem stabil<br>
        5. Pilih algoritma load balancing<br>
        6. Klik <b style="color:#64b5f6">▶ Jalankan Simulasi</b> di sidebar
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-radius:10px;padding:20px 24px;margin-bottom:16px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Rumus Utama</div>
        <div style="font-family:monospace;font-size:13px;color:#90caf9;line-height:2.2">
        ρ = λ / (c × μ) &nbsp;&nbsp;&nbsp; ← utilisasi sistem (harus &lt; 1)<br>
        L = λ × W &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ← Little's Law (rata-rata request dalam sistem)<br>
        W = Wq + 1/μ &nbsp;&nbsp;&nbsp;&nbsp; ← total time = wait + service
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-radius:10px;padding:20px 24px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Algoritma Load Balancing</div>
        <div style="font-size:13px;color:#90a4ae;line-height:2.2">
            <b style="color:#64b5f6">Round Robin</b> — request didistribusikan bergiliran, cocok untuk beban homogen<br>
            <b style="color:#64b5f6">Random</b> — worker dipilih acak, variansi tinggi, overhead nol<br>
            <b style="color:#64b5f6">Least Connection</b> — selalu ke worker paling kosong, optimal untuk beban heterogen
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # st.markdown("<br>", unsafe_allow_html=True)
    # st.link_button("Buka Animasi Interaktif", "https://aufaanggara.github.io/server/dashboard/animation.html")
    st.markdown("### Animasi Interaktif")
    st.components.v1.html(
        get_animation_html(lambda_rate, num_workers, mu, algorithm),
        height=650,
        scrolling=False,
    )

else:
    # ── Run simulation ────────────────────────────────────────────
    with st.spinner("⏳ Menjalankan simulasi stokastik..."):
        results = run_simulation(lambda_rate, num_workers, algorithm, duration, mu)
        all_results = run_all_algorithms(lambda_rate, num_workers, duration, mu)

    df = pd.DataFrame(results)
    tput = len(df) / duration
    avg_wait = df["wait_time"].mean()
    avg_svc = df["service_time"].mean()
    avg_total = df["total_time"].mean()
    L = tput * avg_total
    rho = lambda_rate / (num_workers * mu)

    # ── Stability alert ───────────────────────────────────────────
    if rho >= 1:
        st.markdown(
            f'<div class="alert-critical">🔴 <b>SISTEM TIDAK STABIL</b> — ρ = {rho:.2f} ≥ 1. Antrian akan terus membesar tanpa batas. Tambah worker atau kurangi λ.</div>',
            unsafe_allow_html=True,
        )
    elif rho >= 0.8:
        st.markdown(
            f'<div class="alert-warn">⚠️ <b>Mendekati kritis</b> — ρ = {rho:.2f}. Sistem mulai jenuh, wait time meningkat signifikan.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-stable">✅ <b>Sistem stabil</b> — ρ = {rho:.2f}. Kapasitas server mencukupi untuk menangani beban.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "Total Request", len(df), "sejak reset", ""),
        (c2, "Throughput", f"{tput:.2f}", "req/s aktual", ""),
        (c3, "Avg Service Time", f"{avg_svc:.3f}s", "distribusi Exp(μ)", ""),
        (
            c4,
            "Avg Wait Time",
            f"{avg_wait:.3f}s",
            "Wq rata-rata",
            "warn" if avg_wait > 1 else "good",
        ),
        (
            c5,
            "Utilisasi ρ",
            f"{rho:.2f}",
            rho_status,
            "bad" if rho >= 1 else "warn" if rho >= 0.8 else "good",
        ),
        (c6, "Little's L", f"{L:.2f}", f"λ×W = {tput:.2f}×{avg_total:.2f}", ""),
    ]
    for col, label, value, sub, cls in metrics:
        col.markdown(
            f"""
        <div class="metric-card {'metric-'+cls if cls else ''}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        df["time_bucket"] = (df["arrive_time"] // 5) * 5
        tput_df = df.groupby("time_bucket").size().reset_index(name="count")
        fig1 = px.area(
            tput_df,
            x="time_bucket",
            y="count",
            labels={"time_bucket": "Waktu (s)", "count": "Jumlah Request"},
        )
        fig1.update_traces(line_color="#64b5f6", fillcolor="rgba(100,181,246,0.15)")
        plotly_dark(fig1, "Request per 5 Detik")
        st.plotly_chart(fig1, use_container_width=True, key="fig1")

    with col_r:
        fig2 = px.histogram(
            df,
            x="service_time",
            nbins=25,
            labels={"service_time": "Service Time (s)", "count": "Frekuensi"},
        )
        fig2.update_traces(marker_color="#64b5f6", marker_opacity=0.7)

        # Overlay kurva Eksponensial teoritis
        x_range = np.linspace(0, df["service_time"].max(), 200)
        y_exp = mu * np.exp(-mu * x_range) * len(df) * (df["service_time"].max() / 25)
        fig2.add_trace(
            go.Scatter(
                x=x_range,
                y=y_exp,
                mode="lines",
                line=dict(color="#ef9f27", width=2, dash="dot"),
                name="Exp(μ) teoritis",
            )
        )
        plotly_dark(fig2, "Distribusi Service Time + Kurva Teoritis")
        st.plotly_chart(fig2, use_container_width=True, key="fig2")

    # ── Charts row 2 ──────────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        worker_df = df.groupby("worker_id").size().reset_index(name="requests")
        colors = [
            "#4caf50" if q == 0 else "#ff9800" if q <= 2 else "#f44336"
            for q in worker_df["requests"]
        ]
        fig3 = px.bar(
            worker_df,
            x="worker_id",
            y="requests",
            labels={"worker_id": "Worker ID", "requests": "Jumlah Request"},
        )
        fig3.update_traces(marker_color=colors)
        plotly_dark(fig3, f"Request per Worker ({algorithm.replace('_',' ').title()})")
        st.plotly_chart(fig3, use_container_width=True, key="fig3")

    with col_r2:
        # Scatter: arrive time vs wait time
        fig4 = px.scatter(
            df,
            x="arrive_time",
            y="wait_time",
            color="worker_id",
            labels={
                "arrive_time": "Waktu Kedatangan (s)",
                "wait_time": "Wait Time (s)",
            },
            color_continuous_scale="Blues",
        )
        plotly_dark(fig4, "Wait Time vs Waktu Kedatangan")
        st.plotly_chart(fig4, use_container_width=True, key="fig4")

    st.divider()

    # ── Utilisasi per interval ────────────────────────────────────
    df["interval"] = (df["arrive_time"] // 5) * 5
    util_df = (
        df.groupby("interval")
        .apply(lambda g: min(len(g) / (5 * lambda_rate), 1.5))
        .reset_index(name="utilisasi")
    )
    fig_util = px.area(
        util_df,
        x="interval",
        y="utilisasi",
        labels={"interval": "Waktu (s)", "utilisasi": "Utilisasi Estimasi"},
    )
    fig_util.update_traces(line_color="#ff9800", fillcolor="rgba(255,152,0,0.1)")
    fig_util.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="#f44336",
        annotation_text="ρ = 1 (batas kritis)",
        annotation_position="top right",
    )
    fig_util.add_hline(
        y=rho,
        line_dash="dash",
        line_color="#4caf50",
        annotation_text=f"ρ teoritis = {rho:.2f}",
        annotation_position="bottom right",
    )
    plotly_dark(fig_util, "Estimasi Utilisasi per Interval Waktu")
    st.plotly_chart(fig_util, use_container_width=True, key="fig_util")

    st.divider()

    # ── Perbandingan algoritma ────────────────────────────────────
    st.markdown("### Perbandingan Algoritma Load Balancer")
    comparison = []
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            r = lambda_rate / (num_workers * mu)
            comparison.append(
                {
                    "Algoritma": algo.replace("_", " ").title(),
                    "Total Request": len(d),
                    "Throughput (req/s)": round(len(d) / duration, 3),
                    "Avg Service Time (s)": round(d["service_time"].mean(), 4),
                    "Avg Wait Time (s)": round(d["wait_time"].mean(), 4),
                    "Avg Total Time (s)": round(d["total_time"].mean(), 4),
                    "L = λW": round((len(d) / duration) * d["total_time"].mean(), 3),
                }
            )
    comp_df = pd.DataFrame(comparison)

    col_tbl, col_bar = st.columns([1.2, 1])
    with col_tbl:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    with col_bar:
        fig5 = px.bar(
            comp_df,
            x="Algoritma",
            y="Throughput (req/s)",
            color="Algoritma",
            color_discrete_sequence=["#64b5f6", "#4caf50", "#ff9800"],
        )
        plotly_dark(fig5, "Throughput per Algoritma")
        st.plotly_chart(fig5, use_container_width=True, key="fig5")

    # Wait time comparison
    fig6 = go.Figure()
    algo_colors = {
        "Round Robin": "#64b5f6",
        "Random": "#4caf50",
        "Least Connection": "#ff9800",
    }
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            fig6.add_trace(
                go.Box(
                    y=d["wait_time"],
                    name=algo.replace("_", " ").title(),
                    marker_color=algo_colors.get(
                        algo.replace("_", " ").title(), "#64b5f6"
                    ),
                    boxmean=True,
                )
            )
    plotly_dark(fig6, "Distribusi Wait Time per Algoritma (Box Plot)")
    st.plotly_chart(fig6, use_container_width=True, key="fig6")

    st.divider()

    # ── Little's Law ──────────────────────────────────────────────
    st.markdown("### Validasi Little's Law")
    st.markdown(
        f"""
    <div class="littles-card">
        <div style="font-size:12px;color:#546e7a;margin-bottom:8px">L = λ × W</div>
        <div class="eq">L = {tput:.3f} × {avg_total:.3f} = {L:.3f}</div>
        <div style="font-size:12px;color:#78909c;margin-top:8px">
        λ (arrival rate aktual) = <b style="color:#64b5f6">{tput:.3f} req/s</b> &nbsp;|&nbsp;
        W (avg total time) = <b style="color:#64b5f6">{avg_total:.3f}s</b> &nbsp;|&nbsp;
        L (rata-rata dalam sistem) = <b style="color:#64b5f6">{L:.3f} request</b>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Animasi link ──────────────────────────────────────────────
    st.markdown("### Animasi Interaktif")
    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-left:4px solid #64b5f6;border-radius:10px;padding:16px 20px;margin-bottom:12px">
        <div style="font-size:13px;color:#90a4ae">
        Visualisasi packet flow secara real-time — paket data mengalir dari request generator ke load balancer ke worker, 
        dengan warna worker yang berubah sesuai beban (hijau → kuning → merah).
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.components.v1.html(
        get_animation_html(lambda_rate, num_workers, mu, algorithm),
        height=650,
        scrolling=False,
    )

    # ── Log tabel ─────────────────────────────────────────────────
    st.markdown("### Log Hasil Simulasi")
    col_log, col_dl = st.columns([4, 1])
    with col_log:
        st.dataframe(
            df[
                [
                    "request_id",
                    "worker_id",
                    "arrive_time",
                    "wait_time",
                    "service_time",
                    "total_time",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with col_dl:
        st.markdown("<br><br>", unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "simulation_results.csv", "text/csv")
        st.success("✅ Simulasi selesai!")
