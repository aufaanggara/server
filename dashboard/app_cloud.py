import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
import html
import io
import json
import math

def erlang_c_metrics(lam, c, mu):
    """
    Hitung metrik M/M/c menggunakan formula Erlang-C yang tepat.
    Mengembalikan dict berisi: rho, P0, C (Erlang-C prob), Wq, W, Lq, L
    Mengembalikan None jika sistem tidak stabil (rho >= 1).
    """
    rho = lam / (c * mu)
    if rho >= 1.0:
        return None  # tidak stabil, antrian tak terbatas

    sum_terms = sum((c * rho) ** k / math.factorial(k) for k in range(c))
    last_term = (c * rho) ** c / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / (sum_terms + last_term)

    C_erlang = last_term * P0

    Wq = C_erlang / (c * mu - lam)
    W  = Wq + 1.0 / mu

    Lq = lam * Wq
    L  = lam * W

    return {"rho": rho, "P0": P0, "C_erlang": C_erlang,
            "Wq": Wq, "W": W, "Lq": Lq, "L": L}

st.set_page_config(
    page_title="Simulasi Antrian Server",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.metric-card {
    background: #0d1422;
    border: 1px solid rgba(100,181,246,0.18);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size: 10px; color: #78909c; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
.metric-value { font-size: 26px; font-weight: 700; color: #64b5f6; line-height: 1; font-family: 'JetBrains Mono', monospace; }
.metric-sub { font-size: 11px; color: #546e7a; margin-top: 4px; }
.metric-good .metric-value { color: #4caf50; }
.metric-warn .metric-value { color: #ff9800; }
.metric-bad  .metric-value { color: #f44336; }

.rho-card {
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    border-left: 3px solid #64b5f6;
    background: #0d1422;
}
.littles-card {
    border: 1px solid rgba(100,181,246,0.2);
    border-left: 4px solid #64b5f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
    background: #0d1422;
}
.littles-card .eq { font-size: 20px; font-weight: 700; color: #64b5f6; font-family: 'JetBrains Mono', monospace; }

.alert-stable   { background: rgba(46,125,50,0.15);  border: 1px solid #2e7d32; border-radius: 8px; padding: 10px 16px; color: #a5d6a7; font-size: 13px; }
.alert-critical { background: rgba(198,40,40,0.15);  border: 1px solid #c62828; border-radius: 8px; padding: 10px 16px; color: #ef9a9a; font-size: 13px; }
.alert-warn     { background: rgba(230,81,0,0.15);   border: 1px solid #e65100; border-radius: 8px; padding: 10px 16px; color: #ffcc80; font-size: 13px; }

.section-header { font-size: 11px; color: #546e7a; text-transform: uppercase; letter-spacing: 1.5px; margin: 16px 0 8px; padding-bottom: 4px; border-bottom: 1px solid rgba(100,181,246,0.12); font-family: 'JetBrains Mono', monospace; }

.rank-card {
    background: #0d1422;
    border: 1px solid rgba(100,181,246,0.15);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.rank-medal { font-size: 22px; }
.rank-algo  { font-size: 15px; font-weight: 700; color: #90caf9; }
.rank-reason { font-size: 11px; color: #546e7a; margin-top: 4px; }
.rank-badge { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 20px; margin-left: 8px; }
.badge-best { background: rgba(76,175,80,0.2); color: #a5d6a7; border: 1px solid #4caf50; }
.badge-fast { background: rgba(100,181,246,0.15); color: #90caf9; border: 1px solid #64b5f6; }
.badge-warn { background: rgba(255,152,0,0.15); color: #ffcc80; border: 1px solid #ff9800; }

.stress-active {
    background: rgba(230,81,0,.15);
    border: 1px solid #e65100;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    color: #ffcc80;
    margin-top: 4px;
}

.tab-intro {
    background: #0d1422;
    border: 1px solid #1a2a3a;
    border-left: 4px solid #64b5f6;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
    font-size: 13px;
    color: #90a4ae;
    line-height: 1.9;
}
</style>
""", unsafe_allow_html=True)

# ── session_state defaults ─────────────────────────────────────────────────────
for k, v in {
    "sim_paused": False,
    "sim_reset_key": 0,
    "sim_stress": False,
    "last_results": None,
    "last_params": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def select_worker(algorithm, request_id, workers_queue, num_workers):
    if algorithm == "round_robin":        return request_id % num_workers
    elif algorithm == "random":           return np.random.randint(0, num_workers)
    elif algorithm == "least_connection": return int(np.argmin(workers_queue))
    return request_id % num_workers


def run_simulation(lambda_rate, num_workers, algorithm, duration, mu, seed=None):
    if seed is not None:
        np.random.seed(seed)
    results = []
    workers_queue      = [0]   * num_workers
    workers_busy_until = [0.0] * num_workers
    current_time = 0.0
    request_id   = 0
    base_time    = datetime.now()

    while current_time < duration:
        inter_arrival = np.random.exponential(1.0 / lambda_rate)
        current_time += inter_arrival
        if current_time > duration:
            break
        worker_id    = select_worker(algorithm, request_id, workers_queue, num_workers)
        wait_time    = max(0.0, workers_busy_until[worker_id] - current_time)
        service_time = np.random.exponential(1.0 / mu)
        start_service = current_time + wait_time
        finish_time   = start_service + service_time
        workers_busy_until[worker_id] = finish_time
        workers_queue[worker_id] += 1
        results.append({
            "request_id":   request_id,
            "worker_id":    worker_id,
            "arrive_time":  round(current_time,  4),
            "wait_time":    round(wait_time,      4),
            "service_time": round(service_time,   4),
            "total_time":   round(wait_time + service_time, 4),
            "timestamp":    (base_time + timedelta(seconds=current_time)).isoformat(),
        })
        request_id += 1

    for r in results:
        workers_queue[r["worker_id"]] = max(0, workers_queue[r["worker_id"]] - 1)
    return results


def run_all_algorithms(lambda_rate, num_workers, duration, mu):
    return {
        algo: run_simulation(lambda_rate, num_workers, algo, duration, mu)
        for algo in ["round_robin", "random", "least_connection"]
    }


def run_simulation_multi(lambda_rate, num_workers, algorithm, duration, mu, n_runs=8):
    """
    Jalankan simulasi n_runs kali dengan seed berbeda.
    Kembalikan dict metrik dengan mean ± std untuk confidence interval.
    """
    metrics = {"throughput": [], "avg_wait": [], "avg_total": [], "L": [], "p99_wait": []}
    for i in range(n_runs):
        res = run_simulation(lambda_rate, num_workers, algorithm, duration, mu, seed=i * 17 + 3)
        if not res:
            continue
        d   = pd.DataFrame(res)
        tput = len(d) / duration
        metrics["throughput"].append(tput)
        metrics["avg_wait"].append(d["wait_time"].mean())
        metrics["avg_total"].append(d["total_time"].mean())
        metrics["L"].append(tput * d["total_time"].mean())
        metrics["p99_wait"].append(d["wait_time"].quantile(0.99))

    result = {}
    for k, vals in metrics.items():
        if vals:
            result[k + "_mean"] = np.mean(vals)
            result[k + "_std"]  = np.std(vals)
            result[k + "_ci95"] = 1.96 * np.std(vals) / np.sqrt(len(vals))
            result[k + "_runs"] = vals
    return result


def plotly_dark(fig, title=""):
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1422",
        font_color="#b0bec5",
        font_family="DM Sans",
        title=dict(text=title, font=dict(color="#90caf9", size=14, family="JetBrains Mono")),
        xaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        yaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(bgcolor="#0d1422", bordercolor="#1a2a3a", font=dict(size=12)),
    )
    return fig


def rank_algorithms(all_results, duration, lambda_rate, rho):
    """Auto-rank algorithms and return recommendations."""
    scores = {}
    for algo, res in all_results.items():
        if not res:
            continue
        d = pd.DataFrame(res)
        tput = len(d) / duration
        avg_wait = d["wait_time"].mean()
        p99_wait = d["wait_time"].quantile(0.99)
        fairness = d.groupby("worker_id").size().std() if len(d.groupby("worker_id")) > 1 else 0
        # Composite score: lower wait + lower p99 + higher throughput + lower std dev
        score = tput - avg_wait * 2 - p99_wait * 0.5 - fairness * 0.01
        scores[algo] = {
            "score": score,
            "tput": tput,
            "avg_wait": avg_wait,
            "p99_wait": p99_wait,
            "fairness": fairness,
        }
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return ranked


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATION HTML (UPGRADED: real-time stats + queue visualisation + timeline)
# ══════════════════════════════════════════════════════════════════════════════
def get_animation_html(lambda_rate, num_workers, mu, algorithm_left, algorithm_right=None, speed=1.0, paused=False, split_mode=False, reset_key=0):
    algo_label = {
        "round_robin": "Round Robin",
        "random": "Random",
        "least_connection": "Least Connection",
    }[algorithm_left]
    # Use explicit right algorithm if provided (from sidebar). Otherwise fall back to legacy mapping.
    if algorithm_right:
        algo2 = algorithm_right
    else:
        algo2 = {"round_robin": "least_connection", "random": "round_robin", "least_connection": "random"}[algorithm_left]
    algo2_label = {"round_robin": "Round Robin", "least_connection": "Least Connection", "random": "Random"}[algo2]

    rho = lambda_rate / (num_workers * mu)
    rho_color = "#4caf50" if rho < 0.7 else "#ff9800" if rho < 1 else "#f44336"
    rho_sub = "✅ stabil" if rho < 0.7 else "⚠️ mendekati kritis" if rho < 1 else "🔴 kritis!"
    init_paused = "true" if paused else "false"
    split_js = "true" if split_mode else "false"
    reset_js = int(reset_key)

    html = """<!doctype html>
  <html lang="id">
  <head>
  <meta charset="UTF-8"/>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:100%; overflow:hidden; background:#0a0e1a;
                  font-family:'Segoe UI',Arial,sans-serif; color:#e0e0e0;
                  display:flex; flex-direction:column; }}
    #infoBar {{
      flex-shrink:0; display:flex; align-items:stretch;
      background:#0d1422; border-bottom:1px solid #1a2a3a; min-height:68px;
    }}
    .info-item {{
      display:flex; flex-direction:column; justify-content:center;
      gap:2px; padding:8px 14px; border-right:1px solid #1a2a3a; flex:1;
    }}
    .info-item:last-child {{ border-right:none; }}
    .info-label {{ font-size:9px; color:#546e7a; text-transform:uppercase; letter-spacing:1px; font-weight:500; }}
    .info-value {{ font-size:18px; font-weight:700; color:#64b5f6; font-family:monospace; line-height:1.1; }}
    .info-unit  {{ font-size:11px; color:#37474f; font-weight:400; }}
    .info-sub   {{ font-size:9px; color:#37474f; }}
    #formulaBar {{
      flex-shrink:0; background:#080c18; border-bottom:1px solid #111827;
      padding:4px 14px; font-size:10px; color:#546e7a;
      display:flex; align-items:center; gap:8px; font-family:monospace; flex-wrap:wrap;
    }}
    #formulaBar .rhoVal {{ color:#a5d6a7; font-weight:700; font-size:11px; }}
    #formulaBar .sep    {{ color:#1e3a5f; }}
    #rtStats {{
      flex-shrink:0; display:flex; background:#060a14; border-bottom:1px solid #0f1d2e;
      padding:5px 14px; gap:20px; align-items:center;
    }}
    .rt-item {{ display:flex; flex-direction:column; align-items:center; min-width:70px; }}
    .rt-label {{ font-size:8px; color:#37474f; text-transform:uppercase; letter-spacing:.8px; }}
    .rt-val   {{ font-size:15px; font-weight:700; font-family:monospace; }}
    #timelineWrap {{
      flex-shrink:0; height:40px; background:#060a14; border-top:1px solid #0f1d2e;
      position:relative; overflow:hidden;
    }}
    #timelineCanvas {{ width:100%; height:100%; }}
    #main {{ flex:1; position:relative; overflow:hidden; min-height:0; }}
    #canvasWrap {{ width:100%; height:100%; position:absolute; inset:0; }}
    #splitLabel {{
      display:none; position:absolute; top:6px; left:0; width:100%;
      pointer-events:none; z-index:5;
    }}
    #splitLabel.on {{ display:flex; justify-content:space-around; }}
    .split-lbl {{
      font-size:11px; font-family:monospace; color:#ce93d8;
      background:rgba(10,14,26,.7); padding:3px 10px; border-radius:20px;
      border:1px solid rgba(206,147,216,.3);
    }}
    #alertBanner {{
      display:none; position:absolute; top:8px; left:50%; transform:translateX(-50%);
      background:rgba(183,28,28,.92); color:#fff; padding:5px 18px;
      border-radius:20px; font-size:11px; font-weight:700;
      border:1px solid #ef5350; z-index:10; white-space:nowrap; pointer-events:none;
      animation:abPulse 1s infinite alternate;
    }}
    #pauseOverlay {{
      display:none; position:absolute; inset:0;
      background:rgba(10,14,26,.65); backdrop-filter:blur(2px);
      z-index:8; align-items:center; justify-content:center;
      flex-direction:column; gap:8px;
    }}
    #pauseOverlay.visible {{ display:flex; }}
    #pauseOverlay span  {{ font-size:26px; font-weight:700; color:#64b5f6; letter-spacing:4px; opacity:.9; }}
    #pauseOverlay small {{ font-size:10px; color:#546e7a; }}
    @keyframes abPulse {{
      from {{ box-shadow:0 0 6px rgba(239,83,80,.4); }}
      to   {{ box-shadow:0 0 18px rgba(239,83,80,.95); }}
    }}
  </style>
</head>
<body>
<div id="infoBar">
  <div class="info-item">
    <div class="info-label">λ — Arrival</div>
    <div class="info-value" id="info_lambda">2.0 <span class="info-unit">req/s</span></div>
    <div class="info-sub">Distribusi Poisson</div>
  </div>
  <div class="info-item">
    <div class="info-label">μ — Service</div>
    <div class="info-value" id="info_mu">1.0 <span class="info-unit">req/s</span></div>
    <div class="info-sub">Dist. Eksponensial</div>
  </div>
  <div class="info-item">
    <div class="info-label">Workers (c)</div>
    <div class="info-value" id="info_workers">3</div>
    <div class="info-sub">Paralel aktif</div>
  </div>
  <div class="info-item">
    <div class="info-label">Algoritma</div>
    <div class="info-value" style="font-size:13px;color:#ce93d8" id="info_algo">Round Robin</div>
    <div class="info-sub" id="algo2subInfo">Single mode</div>
  </div>
  <div class="info-item" style="flex:.7">
    <div class="info-label">ρ Utilisasi</div>
    <div class="info-value" id="info_rho" style="color:#4caf50">0.67</div>
    <div class="info-sub" id="info_rho_sub">✅ stabil</div>
  </div>
</div>
<div id="formulaBar">
  <span id="formula_lambda">ρ = λ/(c·μ) = 2.0/(3×1.0) =</span>
  <span class="rhoVal" id="formula_rho">0.67</span>
  <span class="sep">|</span>
  <span>L = λW</span>
  <span class="sep">|</span>
  <span>W = Wq + 1/μ</span>
  <span class="sep">|</span>
  <span>M/M/c Queue Model</span>
</div>
<div id="rtStats">
  <div class="rt-item"><div class="rt-label">Total Req</div><div class="rt-val" id="rt_total" style="color:#64b5f6">0</div></div>
  <div class="rt-item"><div class="rt-label">Throughput</div><div class="rt-val" id="rt_tput" style="color:#4caf50">0.0/s</div></div>
  <div class="rt-item"><div class="rt-label">Avg Wait</div><div class="rt-val" id="rt_wait" style="color:#ff9800">0.00s</div></div>
  <div class="rt-item"><div class="rt-label">In Queue</div><div class="rt-val" id="rt_queue" style="color:#ef5350">0</div></div>
  <div class="rt-item"><div class="rt-label">Little's L</div><div class="rt-val" id="rt_L" style="color:#ce93d8">0.00</div></div>
  <div class="rt-item"><div class="rt-label">Completed</div><div class="rt-val" id="rt_done" style="color:#a5d6a7">0</div></div>
  <div style="margin-left:auto;font-size:9px;color:#1e3a5f;font-family:monospace">LIVE STATS — attn real-time</div>
</div>
<div id="main">
  <div id="canvasWrap">
    <div id="resetSignal" data-reset="0" style="display:none"></div>
    <div id="pauseSignal" data-paused="false" style="display:none"></div>
    <div id="splitLabel" class="{("on" if split_mode else "")}">
      <div class="split-lbl">◀ {algo_label}</div>
      <div class="split-lbl">{algo2_label} ▶</div>
    </div>
    <div id="alertBanner">🔴 SISTEM KRITIS — ρ ≥ 1!</div>
    <div id="pauseOverlay" class="{'visible' if paused else ''}">
      <span>⏸ PAUSED</span>
      <small>Lanjutkan dari sidebar Streamlit</small>
    </div>
  </div>
</div>
<div id="timelineWrap">
  <canvas id="timelineCanvas"></canvas>
</div>

<script>
  let lambda = {js_lambda};
  let numWorkers = {js_num_workers};
  let mu = {js_mu};
  let algorithm = '{js_algorithm}';
  let algorithm2 = '{js_algorithm2}';
  let splitMode = {js_split};
  let speed = {js_speed};
  let paused = {js_paused};
  let resetKey = {js_reset};

  function updatePauseOverlay() {
    const ov = document.getElementById('pauseOverlay');
    if (ov) ov.className = paused ? 'visible' : '';
  }

  function setInfoText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function setInfoHtml(id, htmlValue) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = htmlValue;
  }

  function updateConfigDisplay() {
    setInfoHtml('info_lambda', `${{lambda.toFixed(1)}} <span class="info-unit">req/s</span>`);
    setInfoHtml('info_mu', `${{mu.toFixed(1)}} <span class="info-unit">req/s</span>`);
    setInfoText('info_workers', numWorkers);
    setInfoText('info_algo', algorithm.replace('_', ' ').replace(/\\b\\w/g, c => c.toUpperCase()));
    setInfoText('algo2subInfo', splitMode ? `vs ${{algorithm2.replace('_', ' ').replace(/\\b\\w/g, c => c.toUpperCase())}}` : 'Single mode');

    const rhoVal = lambda / (numWorkers * mu);
    const rhoText = rhoVal.toFixed(3);
    const rhoColor = rhoVal < 0.7 ? '#4caf50' : rhoVal < 1 ? '#ff9800' : '#f44336';
    const rhoSub = rhoVal < 0.7 ? '✅ stabil' : rhoVal < 1 ? '⚠️ mendekati kritis' : '🔴 kritis!';
    const rhoEl = document.getElementById('info_rho');
    if (rhoEl) {
      rhoEl.textContent = rhoText;
      rhoEl.style.color = rhoColor;
    }
    setInfoText('info_rho_sub', rhoSub);
    setInfoText('formula_rho', rhoText);
    setInfoHtml('formula_lambda', `ρ = λ/(c·μ) = ${{lambda.toFixed(1)}}/(${{numWorkers}}×${{mu.toFixed(1)}}) =`);
    document.getElementById('alertBanner').style.display = rhoVal >= 1 ? 'block' : 'none';
  }

  updatePauseOverlay();
  updateConfigDisplay();

  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg || msg.source !== 'wrapper') return;
    if (msg.type === 'pause') {
      paused = !!(msg.payload && msg.payload.paused);
      updatePauseOverlay();
      console.info('Frame received pause:', paused);
    } else if (msg.type === 'reset') {
      doReset();
      console.info('Frame received reset');
    } else if (msg.type === 'config' || msg.type === 'init') {
      const cfg = msg.payload || {};
      if (cfg.lambda !== undefined) lambda = Number(cfg.lambda) || lambda;
      if (cfg.mu !== undefined) mu = Number(cfg.mu) || mu;
      if (cfg.numWorkers !== undefined) numWorkers = Number(cfg.numWorkers) || numWorkers;
      if (cfg.algorithm) algorithm = cfg.algorithm;
      if (cfg.algorithm2) algorithm2 = cfg.algorithm2;
      if (cfg.splitMode !== undefined) splitMode = Boolean(cfg.splitMode);
      if (cfg.speed !== undefined) speed = Number(cfg.speed) || speed;
      if (cfg.paused !== undefined) paused = Boolean(cfg.paused);
      if (cfg.resetKey !== undefined) {
        const nextKey = Number(cfg.resetKey);
        if (nextKey !== resetKey) {
          resetKey = nextKey;
          doReset();
        }
      }
      updatePauseOverlay();
      updateConfigDisplay();
      console.info('Frame received config/init:', msg.type, cfg);
    }
  });

  function doReset() {
    try {
      workers = makeWorkers(numWorkers);
      workers2 = makeWorkers(numWorkers);
      packets = [];
      packets2 = [];
      trails = [];
      trails2 = [];
      nextSpawn = 0; nextSpawn2 = 0; reqId = 0; reqId2 = 0;
      totalReq = 0; totalReq2 = 0; completed = 0; completed2 = 0;
      waitSamples = []; totalWaitSum = 0;
      tlHistory = [];
      lineAnimOffset = 0;
      simStart = performance.now();
      drawTimeline();
      console.info('Animation reset (resetKey=' + resetKey + ')');
    } catch (e) {
      console.warn('doReset error', e);
    }
  }
  
  // Try to restore previous state (if available) so pause/resend doesn't reset
  // (will attempt to run after simulation variables are initialized)

  // ── Stats tracking ──────────────────────────────────────────────
  let totalReq=0, completed=0, waitSamples=[], totalWaitSum=0;
  let simTime=0, simStart=performance.now();
  let totalReq2=0, completed2=0;
  const ALGO_SHORT={{ round_robin:"RR", random:"RND", least_connection:"LC" }};

  // ── Timeline ────────────────────────────────────────────────────
  const tlCanvas = document.getElementById("timelineCanvas");
  const tlCtx    = tlCanvas.getContext("2d");
  let tlHistory  = [];  // {{t, util}}
  let tlLastT    = 0;

  function resizeTL(){{
    tlCanvas.width  = tlCanvas.offsetWidth;
    tlCanvas.height = tlCanvas.offsetHeight;
  }}
  resizeTL();
  window.addEventListener("resize", resizeTL);

  function drawTimeline(){{
    const W=tlCanvas.width, H=tlCanvas.height;
    tlCtx.clearRect(0,0,W,H);
    tlCtx.fillStyle="#060a14";
    tlCtx.fillRect(0,0,W,H);
    // Grid
    tlCtx.strokeStyle="#0f1d2e"; tlCtx.lineWidth=.5;
    tlCtx.beginPath(); tlCtx.moveTo(0,H*.5); tlCtx.lineTo(W,H*.5); tlCtx.stroke();
    tlCtx.beginPath(); tlCtx.moveTo(0,H*.25); tlCtx.lineTo(W,H*.25); tlCtx.stroke();
    // Threshold line ρ=1
    const maxUtil = Math.max(2, ...tlHistory.map(h=>h.util));
    const scaleY = (u) => H - (u/1.5)*H*.8 - H*.08;
    // Fill area
    if(tlHistory.length>1){{
      tlCtx.beginPath();
      tlCtx.moveTo(0, H);
      for(let i=0;i<tlHistory.length;i++){{
        const x = (i/Math.max(tlHistory.length-1,1))*W;
        const y = scaleY(tlHistory[i].util);
        tlCtx.lineTo(x, y);
      }}
      tlCtx.lineTo(W,H); tlCtx.closePath();
      const grad = tlCtx.createLinearGradient(0,0,0,H);
      grad.addColorStop(0,"rgba(255,152,0,.4)");
      grad.addColorStop(1,"rgba(255,152,0,.02)");
      tlCtx.fillStyle=grad; tlCtx.fill();
      // Line
      tlCtx.beginPath();
      for(let i=0;i<tlHistory.length;i++){{
        const x=(i/Math.max(tlHistory.length-1,1))*W;
        const y=scaleY(tlHistory[i].util);
        i===0?tlCtx.moveTo(x,y):tlCtx.lineTo(x,y);
      }}
      tlCtx.strokeStyle="#ff9800"; tlCtx.lineWidth=1.5; tlCtx.stroke();
    }}
    // ρ=1 line
    const y1 = scaleY(1.0);
    tlCtx.setLineDash([4,4]); tlCtx.strokeStyle="rgba(244,67,54,.6)"; tlCtx.lineWidth=1;
    tlCtx.beginPath(); tlCtx.moveTo(0,y1); tlCtx.lineTo(W,y1); tlCtx.stroke();
    tlCtx.setLineDash([]);
    tlCtx.fillStyle="rgba(244,67,54,.7)"; tlCtx.font="8px monospace";
    tlCtx.fillText("ρ=1",4,y1-3);
    // Label
    tlCtx.fillStyle="#37474f"; tlCtx.font="9px monospace";
    tlCtx.fillText("UTILISASI TIMELINE",4,H-4);
  }}

  // ── Workers state ───────────────────────────────────────────────
  function makeWorkers(n){{
    return Array.from({{length:n}},(_,i)=>({{
      id:i, queue:0, busy:false, busyTimer:0, processed:0,
      pendingQueue:[], busyFrac:0
    }}));
  }}

  let workers  = makeWorkers(numWorkers);
  let workers2 = makeWorkers(numWorkers);
  let packets=[], packets2=[], trails=[], trails2=[];
  let nextSpawn=0, nextSpawn2=0, reqId=0, reqId2=0;
  let lineAnimOffset=0;
  let expRandom = (rate) => -Math.log(Math.random())/rate;

  // Now that core arrays exist, animation is ready to receive configuration.
  console.info('Animation initialized');

  function selectWorker(algo, rid, wArr){{
    if(algo==="round_robin")      return rid % wArr.length;
    if(algo==="random")           return Math.floor(Math.random()*wArr.length);
    return wArr.reduce((a,b)=>b.queue<a.queue?b:a).id;
  }}

  // ── Update real-time stats ───────────────────────────────────────
  function updateStats(){{
    const tElapsed=(performance.now()-simStart)/1000;
    const tput=tElapsed>0?(totalReq/tElapsed):0;
    const avgW=waitSamples.length>0?(totalWaitSum/waitSamples.length):0;
    const inQueue=workers.reduce((s,w)=>s+w.queue,0);
    const L=tput*(avgW+1/mu);
    document.getElementById("rt_total").textContent=totalReq;
    document.getElementById("rt_tput").textContent=tput.toFixed(1)+"/s";
    document.getElementById("rt_wait").textContent=avgW.toFixed(2)+"s";
    document.getElementById("rt_queue").textContent=inQueue;
    document.getElementById("rt_L").textContent=L.toFixed(2);
    document.getElementById("rt_done").textContent=completed;
    // Timeline every second
    if(tElapsed-tlLastT>=0.5){{
      tlLastT=tElapsed;
      const util=Math.min(inQueue/(numWorkers)+workers.filter(w=>w.busy).length/numWorkers,2);
      tlHistory.push({{t:tElapsed,util:util}});
      if(tlHistory.length>200) tlHistory.shift();
      drawTimeline();
    }}
  }}

  new p5(function(p){{
    let W,H;

    p.setup=function(){{
      const wrap=document.getElementById("canvasWrap");
      W=wrap.offsetWidth||800; H=wrap.offsetHeight||380;
      let c=p.createCanvas(W,H); c.parent("canvasWrap");
      p.textFont("Segoe UI");
      new ResizeObserver(()=>{{
        const wr=document.getElementById("canvasWrap");
        let nw=wr.offsetWidth,nh=wr.offsetHeight;
        if(nw>10&&nh>10){{W=nw;H=nh;p.resizeCanvas(W,H);}}
      }}).observe(document.getElementById("canvasWrap"));
    }};

    p.draw=function(){{
      if(paused) return;
      const simDt = p.deltaTime/1000;
      const animDt = simDt * speed;
      lineAnimOffset=(lineAnimOffset+animDt*28)%12;
      p.background(10,14,26);
      drawGrid(p,W,H);

      const rho=lambda/(numWorkers*mu);
      document.getElementById("alertBanner").style.display=rho>=1?"block":"none";
      if(rho>=1){{
        let pulse=(Math.sin(p.millis()/300)+1)/2;
        p.noStroke(); p.fill(180,20,20,pulse*18); p.rect(0,0,W,H);
      }}

      if(splitMode){{
        // Split divider
        p.stroke(30,60,90); p.strokeWeight(1.5);
        p.line(W/2,0,W/2,H);
        drawPanel(p,0,0,W/2,H,workers,packets,trails,algorithm,reqId,nextSpawn,false,simDt,animDt);
        drawPanel(p,W/2,0,W/2,H,workers2,packets2,trails2,algorithm2,reqId2,nextSpawn2,true,simDt,animDt);
        // Update spawn for both
        nextSpawn-=simDt; nextSpawn2-=simDt;
        spawnIfNeeded(0,W/2,H,false,simDt);
        spawnIfNeeded(W/2,W/2,H,true,simDt);
      }} else {{
        drawPanel(p,0,0,W,H,workers,packets,trails,algorithm,reqId,nextSpawn,false,simDt,animDt);
        nextSpawn-=simDt;
        spawnIfNeeded(0,W,H,false,simDt);
      }}
      updateStats();
    }};

    function spawnIfNeeded(offsetX,panelW,panelH,isRight,dt){{
      const ns = isRight ? nextSpawn2 : nextSpawn;
      if(ns<=0){{
        const wArr=isRight?workers2:workers;
        const rid=isRight?reqId2:reqId;
        const wid=selectWorker(isRight?algorithm2:algorithm,rid,wArr);
        const genX=offsetX+panelW*.12, genY=panelH/2;
        const wX=offsetX+panelW*.82;
        const wSpacing=panelH/(numWorkers+1), wy=wSpacing*(wid+1);
        const pk={{ x:genX,y:genY,tx:wX,ty:wy,wid,phase:"toLB",alpha:255,id:rid,size:10,arriveT:performance.now(),waitHeat:0 }};
        if(isRight){{ packets2.push(pk); workers2[wid].queue++; reqId2++; nextSpawn2=expRandom(lambda);
          totalReq2++;
          if(!workers2[wid].busy){{ workers2[wid].busy=true; workers2[wid].busyTimer=expRandom(mu); }}
          else{{ workers2[wid].pendingQueue.push({{arriveT:performance.now()}}); }}
        }} else {{
          packets.push(pk); workers[wid].queue++; reqId++; nextSpawn=expRandom(lambda);
          totalReq++; waitSamples.push(0);
          if(!workers[wid].busy){{ workers[wid].busy=true; workers[wid].busyTimer=expRandom(mu); }}
          else{{ workers[wid].pendingQueue.push({{arriveT:performance.now()}}); }}
        }}
      }}
    }}

    function drawPanel(p,offX,offY,panelW,panelH,wArr,pkts,trls,algo,_rid,_ns,isRight,simDt,animDt){{
      const lbX=offX+panelW*.36, lbY=panelH/2;
      const genX=offX+panelW*.12, genY=panelH/2;
      const wX=offX+panelW*.82;
      const wSpacing=panelH/(numWorkers+1);
      const moveFactor = Math.min(animDt * 4, 1);

      // Update workers
      for(let w of wArr){{
        if(w.busy){{
          w.busyTimer-=simDt;
          if(w.busyTimer<=0){{
            w.busy=false; w.queue=Math.max(0,w.queue-1);
            if(!isRight){{ completed++; }}
            else{{ completed2++; }}
            w.processed++;
            if(w.pendingQueue.length>0){{
              const pq=w.pendingQueue.shift();
              const waited=(performance.now()-pq.arriveT)/1000;
              if(!isRight){{ totalWaitSum+=waited; }}
              w.busy=true; w.busyTimer=expRandom(mu);
            }}
          }}
        }}
      }}

      // Trails
      for(let t of trls){{
        t.alpha-=10;
        if(t.alpha>0){{ p.noStroke(); p.fill(100,180,255,t.alpha*.18); p.ellipse(t.x,t.y,t.size*.5); }}
      }}
      trls.splice(0,trls.length,...trls.filter(t=>t.alpha>0));

      // Lines
      drawAnimLine(p,genX+26,genY,lbX-36,lbY,lineAnimOffset);
      for(let i=0;i<numWorkers;i++)
        drawAnimLine(p,lbX+36,lbY,wX-52,wSpacing*(i+1),lineAnimOffset);

      // Queue visualisation (stacked dots)
      for(let i=0;i<numWorkers;i++){{
        const w=wArr[i], wy=wSpacing*(i+1), qLen=w.pendingQueue.length;
        const maxShow=8;
        for(let q=0;q<Math.min(qLen,maxShow);q++){{
          const age=w.pendingQueue[q]?(performance.now()-w.pendingQueue[q].arriveT)/1000:0;
          const heat=Math.min(age/4,1);
          const r2=Math.round(100+heat*131),g2=Math.round(180-heat*130),b2=Math.round(255-heat*200);
          // Stack horizontally to the left of worker
          const qx=wX-62-(q*13);
          p.noStroke(); p.fill(r2,g2,b2,210); p.ellipse(qx,wy,11);
          p.fill(255,200); p.textSize(7); p.textAlign(p.CENTER,p.CENTER);
          p.text(q+1,qx,wy);
        }}
        if(qLen>maxShow){{
          p.fill(231,76,60,220); p.textSize(9); p.textAlign(p.LEFT,p.CENTER);
          p.text(`+${{qLen-maxShow}}`,wX-62-maxShow*13,wy);
        }}
        // Queue length bar
        if(qLen>0){{
          const barW=Math.min(qLen/8,1)*50;
          p.noStroke(); p.fill(231,76,60,60); p.rect(wX-62-maxShow*13-8,wy+8,50,4,2);
          p.fill(231,76,60,180); p.rect(wX-62-maxShow*13-8,wy+8,barW,4,2);
        }}
      }}

      // Packets
      for(let pk of pkts){{
        pk.waitHeat=Math.min((performance.now()-pk.arriveT)/3000,1);
        const r3=Math.round(150+pk.waitHeat*81),gb=Math.round(210-pk.waitHeat*134);
        if(pk.phase==="toLB"){{
          pk.x=p.lerp(pk.x,lbX,moveFactor); pk.y=p.lerp(pk.y,lbY,moveFactor);
          if(p.dist(pk.x,pk.y,lbX,lbY)<8) pk.phase="toWorker";
        }} else if(pk.phase==="toWorker"){{
          pk.x=p.lerp(pk.x,pk.tx,moveFactor); pk.y=p.lerp(pk.y,pk.ty,moveFactor);
          if(p.dist(pk.x,pk.y,pk.tx,pk.ty)<8) pk.phase="arrive";
        }} else if(pk.phase==="arrive"){{
          pk.size=p.lerp(pk.size,20,.2); pk.alpha-=12;
          if(pk.alpha<=0) pk.phase="done";
        }}
        if(pk.phase!=="done"&&pk.alpha>0){{
          trls.push({{x:pk.x,y:pk.y,alpha:pk.alpha*.4,size:pk.size}});
          p.noStroke();
          p.fill(r3,gb,gb*.6,pk.alpha*.1); p.ellipse(pk.x,pk.y,pk.size*3);
          p.fill(r3,gb,gb*.6,pk.alpha*.22); p.ellipse(pk.x,pk.y,pk.size*2);
          p.fill(r3,gb,255-pk.waitHeat*200,pk.alpha); p.ellipse(pk.x,pk.y,pk.size);
          p.fill(255,pk.alpha); p.textSize(7); p.textAlign(p.CENTER,p.CENTER);
          p.text(pk.id%1000,pk.x,pk.y);
        }}
      }}
      pkts.splice(0,pkts.length,...pkts.filter(pk=>pk.phase!=="done"));

      // Boxes
      const algoShort=ALGO_SHORT[algo]||"?";
      drawBox(p,genX,genY,54,50,[30,80,160],[64,148,230],"REQ GEN",`λ=${{lambda}}/s`);
      drawBox(p,lbX,lbY,70,56,[80,40,160],[150,100,230],"BALANCER",algoShort);
      for(let i=0;i<numWorkers;i++){{
        const w=wArr[i], wy=wSpacing*(i+1), col=workerColor(w);
        if(w.busy){{
          p.noStroke(); p.fill(col[0],col[1],col[2],20+Math.sin(p.millis()/200)*10);
          p.rect(wX-48,wy-32,96,64,12);
        }}
        drawWorkerBox(p,wX,wy,w,col,i);
      }}
    }}

    function drawAnimLine(p,x1,y1,x2,y2,off){{
      const d=p.dist(x1,y1,x2,y2);
      p.stroke(30,70,120,70); p.strokeWeight(1); p.line(x1,y1,x2,y2);
      const steps=Math.ceil(d/12);
      for(let i=0;i<steps;i++){{
        const t1=((i*12+off)%d)/d, t2=((i*12+6+off)%d)/d;
        if(t1<0||t1>1||t2<0||t2>1) continue;
        const ax=p.lerp(x1,x2,t1),ay=p.lerp(y1,y2,t1),bx=p.lerp(x1,x2,t2),by=p.lerp(y1,y2,t2);
        p.stroke(100,180,255,170); p.strokeWeight(1.8); p.line(ax,ay,bx,by);
      }}
    }}
    function drawGrid(p,W,H){{
      p.stroke(18,28,46); p.strokeWeight(.5);
      for(let x=0;x<W;x+=40) p.line(x,0,x,H);
      for(let y=0;y<H;y+=40) p.line(0,y,W,y);
    }}
    function drawBox(p,x,y,w,h,cD,cL,title,sub){{
      p.noStroke(); p.fill(0,0,0,60); p.rect(x-w/2+3,y-h/2+3,w,h,10);
      p.fill(cD[0],cD[1],cD[2]); p.rect(x-w/2,y-h/2,w,h,10);
      p.fill(cL[0],cL[1],cL[2],40); p.rect(x-w/2,y-h/2,w,h/2,10,10,0,0);
      p.stroke(cL[0],cL[1],cL[2],130); p.strokeWeight(1.5); p.noFill(); p.rect(x-w/2,y-h/2,w,h,10);
      p.noStroke(); p.fill(220,235,255); p.textSize(9); p.textAlign(p.CENTER,p.CENTER); p.text(title,x,y-8);
      p.fill(cL[0],cL[1],cL[2]); p.textSize(10); p.textStyle(p.BOLD); p.text(sub,x,y+9); p.textStyle(p.NORMAL);
    }}
    function drawWorkerBox(p,x,y,w,col,idx){{
      const bw=82,bh=58;
      p.noStroke(); p.fill(0,0,0,60); p.rect(x-bw/2+3,y-bh/2+3,bw,bh,10);
      p.fill(col[0]*.35,col[1]*.35,col[2]*.35); p.rect(x-bw/2,y-bh/2,bw,bh,10);
      p.fill(col[0],col[1],col[2],50); p.rect(x-bw/2,y-bh/2,bw,bh/2,10,10,0,0);
      p.stroke(col[0],col[1],col[2],180); p.strokeWeight(1.5); p.noFill(); p.rect(x-bw/2,y-bh/2,bw,bh,10);
      p.noStroke();
      p.fill(w.busy?[231,76,60]:[46,204,113]); p.ellipse(x+bw/2-9,y-bh/2+9,7);
      p.fill(220,235,255); p.textSize(8); p.textAlign(p.CENTER,p.CENTER); p.text(`W${{idx}}`,x,y-15);
      p.fill(col[0]+80,col[1]+80,col[2]+80); p.textSize(9); p.textStyle(p.BOLD);
      p.text(`Q:${{w.queue}}`,x-12,y+2); p.textStyle(p.NORMAL);
      p.fill(160,200,255); p.textSize(8); p.text(`✓${{w.processed}}`,x+14,y+2);
      p.textSize(7); p.fill(w.busy?[231,120,100]:[100,200,130]); p.text(w.busy?"●BUSY":"●IDLE",x,y+15);
      if(w.busy&&w.busyTimer>0){{
        const prog=Math.max(0,Math.min(1,1-w.busyTimer/(1/mu)));
        p.noStroke(); p.fill(30,40,60); p.rect(x-28,y+24,56,4,2);
        p.fill(col[0],col[1],col[2],200); p.rect(x-28,y+24,56*prog,4,2);
      }}
    }}
    function workerColor(w){{
      if(w.queue===0) return[46,204,113];
      if(w.queue<=2)  return[241,196,15];
      return[231,76,60];
    }}
  }});
    </script>
    </body>
    </html>"""

    # Normalize doubled braces (from previous f-string escaping) back to single braces
    html = html.replace('{{', '{').replace('}}', '}')

    # Replace Python-template placeholders (avoid using f-strings over large JS blobs)
    split_class = "on" if split_mode else ""
    html = html.replace('{algo_label}', algo_label)
    html = html.replace('{algo2_label}', algo2_label)
    html = html.replace('{("on" if split_mode else "")}', split_class)
    html = html.replace('{rho_color}', rho_color)
    html = html.replace('{rho_sub}', rho_sub)
    html = html.replace('{js_lambda}', f"{lambda_rate:.4f}")
    html = html.replace('{js_num_workers}', str(num_workers))
    html = html.replace('{js_mu}', f"{mu:.4f}")
    html = html.replace('{js_algorithm}', algorithm_left)
    html = html.replace('{js_algorithm2}', algo2)
    html = html.replace('{js_split}', 'true' if split_mode else 'false')
    html = html.replace('{js_speed}', f"{speed:.4f}")
    html = html.replace('{js_paused}', init_paused)
    html = html.replace('{js_reset}', str(reset_js))
    html = html.replace('{lambda_rate:.1f}', f"{lambda_rate:.1f}")
    html = html.replace('{num_workers}', str(num_workers))
    html = html.replace('{mu:.1f}', f"{mu:.1f}")
    html = html.replace('{rho:.3f}', f"{rho:.3f}")

    return html


def get_animation_wrapper_html(lambda_rate=2.0, num_workers=3, mu=1.0,
                                algorithm_left='round_robin', algorithm_right='least_connection',
                                speed=1.0, paused=False, split_mode=False, reset_key=0):
    # Stable wrapper HTML; dynamic config is sent via postMessage from the
    # Streamlit page to avoid reloading the iframe on every rerun.
    initial_config = {
        "lambda": 2.0,
        "mu": 1.0,
        "numWorkers": 3,
        "algorithm": "round_robin",
        "algorithm2": "least_connection",
        "splitMode": False,
        "speed": 1.0,
        "paused": False,
        "resetKey": 0,
    }
    frame_html = get_animation_html(
        lambda_rate=2.0,
        num_workers=3,
        mu=1.0,
        algorithm_left="round_robin",
        algorithm_right="least_connection",
        speed=1.0,
        paused=False,
        split_mode=False,
        reset_key=0,
    )
    frame_src = "data:text/html;base64," + base64.b64encode(frame_html.encode("utf-8")).decode("ascii")
    initial_config_json = json.dumps(initial_config)

    return f"""<!doctype html>
<html lang=\"id\">
<head>
  <meta charset=\"UTF-8\"/>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; background:#0a0e1a; overflow:hidden; }}
    body {{ color:#e0e0e0; font-family:'Segoe UI',Arial,sans-serif; }}
    #wrapper {{ width:100%; height:100%; position:relative; }}
    #animFrame {{ width:100%; height:100%; border:none; background:#0a0e1a; }}
  </style>
</head>
<body>
<div id=\"wrapper\">
  <iframe id=\"animFrame\" title=\"Simulasi Antrian\" src=\"{frame_src}\"></iframe>
</div>
<script>
  const initialConfig = {initial_config_json};
  const iframe = document.getElementById('animFrame');

  function sendToFrame(message) {{
    if(iframe && iframe.contentWindow) {{
      iframe.contentWindow.postMessage(Object.assign({{ source:'wrapper' }}, message), '*');
    }}
  }}

  window.addEventListener('message', (event) => {{
    const msg = event.data;
    if(!msg || msg.source !== 'page') return;
    if(msg.type === 'config') {{
      sendToFrame({{ type:'config', payload: msg.payload }});
    }} else if(msg.type === 'reset') {{
      sendToFrame({{ type:'reset' }});
    }} else if(msg.type === 'pause') {{
      sendToFrame(msg);
    }} else if(msg.type === 'init') {{
      sendToFrame(msg);
    }}
  }});

  function sendInit() {{
    sendToFrame({{ type:'init', payload: initialConfig }});
    sendToFrame({{ type:'pause', payload: {{ paused: initialConfig.paused }} }});
  }}

  iframe.addEventListener('load', () => {{
    sendInit();
  }});

  window.addEventListener('message', (event) => {{
    const msg = event.data;
    if(!msg || msg.source !== 'animFrame') return;
    if(msg.type === 'ready') {{
      sendInit();
    }}
  }});

</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(params, df, all_results, duration, rho, L):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                             leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle('TitleCustom', parent=styles['Title'],
                                  fontSize=18, spaceAfter=4, textColor=colors.HexColor("#1a237e"))
    h1_style    = ParagraphStyle('H1Custom', parent=styles['Heading1'],
                                  fontSize=13, textColor=colors.HexColor("#0d47a1"), spaceBefore=14, spaceAfter=4)
    body_style  = ParagraphStyle('BodyCustom', parent=styles['Normal'],
                                  fontSize=10, leading=16, textColor=colors.HexColor("#263238"))
    mono_style  = ParagraphStyle('Mono', parent=styles['Code'],
                                  fontSize=9, leading=14, textColor=colors.HexColor("#1565c0"),
                                  backColor=colors.HexColor("#e3f2fd"), borderPadding=6)

    story.append(Paragraph("Laporan Simulasi Antrian Server", title_style))
    story.append(Paragraph("Pemodelan Stokastik — Model M/M/c Queue", styles['Heading3']))
    story.append(Paragraph(f"Tanggal: {datetime.now().strftime('%d %B %Y, %H:%M')}", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a237e"), spaceAfter=12))

    # Parameters
    story.append(Paragraph("1. Parameter Simulasi", h1_style))
    p = params
    param_data = [
        ["Parameter", "Nilai", "Keterangan"],
        ["λ (Arrival Rate)", f"{p['lambda_rate']:.1f} req/s", "Distribusi Poisson"],
        ["μ (Service Rate)", f"{p['mu']:.1f} req/s", "Distribusi Eksponensial"],
        ["c (Workers)", str(p['num_workers']), "Server paralel aktif"],
        ["Durasi Simulasi", f"{p['duration']} detik", ""],
        ["Algoritma", p['algorithm'].replace("_"," ").title(), "Load balancing strategy"],
        ["ρ = λ/(c·μ)", f"{rho:.4f}", "< 1 = stabil" if rho < 1 else ">= 1 = tidak stabil"],
    ]
    tbl = Table(param_data, colWidths=[5*cm, 4*cm, 7*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a237e")),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('FONTSIZE',   (0,1), (-1,-1), 9),
        ('PADDING',    (0,0), (-1,-1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Results
    story.append(Paragraph("2. Hasil Simulasi", h1_style))
    tput = len(df) / duration
    avg_wait  = df["wait_time"].mean()
    avg_svc   = df["service_time"].mean()
    avg_total = df["total_time"].mean()
    res_data = [
        ["Metrik", "Nilai Empiris", "Interpretasi"],
        ["Total Request",      str(len(df)),           "Request diproses"],
        ["Throughput",         f"{tput:.3f} req/s",    "Request selesai per detik"],
        ["Avg Service Time",   f"{avg_svc:.4f}s",      f"Teoritis: {1/p['mu']:.4f}s"],
        ["Avg Wait Time (Wq)", f"{avg_wait:.4f}s",     "Waktu antre sebelum dilayani"],
        ["Avg Total Time (W)", f"{avg_total:.4f}s",    "Wq + 1/μ"],
        ["Little's L",         f"{L:.4f}",             f"λ × W = {tput:.3f} × {avg_total:.3f}"],
    ]
    tbl2 = Table(res_data, colWidths=[5.5*cm, 4.5*cm, 6*cm])
    tbl2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d47a1")),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#e8f5e9"), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('PADDING',    (0,0), (-1,-1), 5),
    ]))
    story.append(tbl2)
    story.append(Spacer(1, 12))

    # Little's Law
    story.append(Paragraph("3. Validasi Little's Law", h1_style))
    story.append(Paragraph(
        f"Little's Law menyatakan: L = lambda * W, di mana L adalah rata-rata jumlah request dalam sistem, "
        f"lambda adalah throughput aktual, dan W adalah rata-rata waktu total dalam sistem.",
        body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"L = {tput:.3f} x {avg_total:.3f} = {L:.3f} request",
        mono_style))
    story.append(Spacer(1, 12))

    # Algorithm comparison
    story.append(Paragraph("4. Perbandingan Algoritma", h1_style))
    comp_data = [["Algoritma","Throughput","Avg Wait","Avg Total","Little's L","Rank"]]
    from functools import reduce
    ranked = rank_algorithms(all_results, duration, p['lambda_rate'], rho)
    rank_map = {r[0]: i+1 for i, r in enumerate(ranked)}
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            t = len(d)/duration
            comp_data.append([
                algo.replace("_"," ").title(),
                f"{t:.3f}",
                f"{d['wait_time'].mean():.4f}s",
                f"{d['total_time'].mean():.4f}s",
                f"{t*d['total_time'].mean():.3f}",
                f"#{rank_map.get(algo,'?')}",
            ])
    tbl3 = Table(comp_data, colWidths=[4.5*cm, 3*cm, 3*cm, 3*cm, 2.5*cm, 2*cm])
    tbl3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#311b92")),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ede7f6"), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('PADDING',    (0,0), (-1,-1), 5),
    ]))
    story.append(tbl3)
    story.append(Spacer(1, 12))

    # Stability verdict
    story.append(Paragraph("5. Kesimpulan", h1_style))
    if rho >= 1:
        verdict = f"Sistem TIDAK STABIL. rho = {rho:.3f} >= 1. Antrian akan terus membesar tanpa batas."
        status_color = colors.HexColor("#b71c1c")
    elif rho >= 0.8:
        verdict = f"Sistem mendekati kritis. rho = {rho:.3f}. Disarankan menambah worker atau mengurangi lambda."
        status_color = colors.HexColor("#e65100")
    else:
        verdict = f"Sistem STABIL. rho = {rho:.3f} < 1. Kapasitas mencukupi untuk menangani traffic."
        status_color = colors.HexColor("#1b5e20")

    story.append(Paragraph(verdict, ParagraphStyle('verdict', parent=body_style,
                            textColor=status_color, fontSize=11, fontName='Helvetica-Bold')))

    if ranked:
        best_algo, best_stats = ranked[0]
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Algoritma terbaik untuk kondisi ini: <b>{best_algo.replace('_',' ').title()}</b> "
            f"(throughput: {best_stats['tput']:.3f} req/s, avg wait: {best_stats['avg_wait']:.4f}s)",
            body_style))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b0bec5")))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Dibuat dengan Simulasi Antrian Server — Pemodelan Stokastik M/M/c", 
                            ParagraphStyle('footer', parent=body_style, fontSize=8, textColor=colors.HexColor("#90a4ae"))))

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")

    st.markdown('<div class="section-header">Parameter Stokastik</div>', unsafe_allow_html=True)
    lambda_rate = st.slider("λ — Arrival Rate (req/s)", 0.5, 5.0, 2.0, 0.5)
    mu          = st.slider("μ — Service Rate (req/s)", 0.5, 3.0, 1.0, 0.5)
    num_workers = st.slider("Jumlah Worker (c)", 1, 5, 3)
    speed       = st.slider("Speed animasi", 0.5, 3.0, 1.0, 0.5, format="%.1fx")

    rho = lambda_rate / (num_workers * mu)
    rho_color  = "#4caf50" if rho < 0.7 else "#ff9800" if rho < 1 else "#f44336"
    rho_status = "✅ Stabil" if rho < 0.7 else "⚠️ Mendekati kritis" if rho < 1 else "🔴 Tidak stabil!"
    st.markdown(f"""
    <div class="rho-card">
        ρ = λ/(c·μ) = {lambda_rate:.1f}/({num_workers}×{mu:.1f}) =
        <span style="color:{rho_color};font-weight:700;font-size:16px">{rho:.3f}</span><br>
        <span style="color:{rho_color};font-size:12px">{rho_status}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Konfigurasi Simulasi</div>', unsafe_allow_html=True)
    algorithm = st.selectbox(
        "Algoritma Load Balancer",
        ["round_robin", "random", "least_connection"],
        format_func=lambda x: {"round_robin":"Round Robin","random":"Random","least_connection":"Least Connection"}[x],
    )
    duration = st.slider("Durasi Simulasi (detik)", 10, 120, 30, 10)

    st.markdown('<div class="section-header">Kontrol Animasi</div>', unsafe_allow_html=True)
    split_mode = st.toggle("🔀 Split-screen (2 algoritma)", value=False)

    # Jika split mode aktif, berikan pilihan algoritma kanan
    algorithm_right = None
    if split_mode:
      algos = ["round_robin", "random", "least_connection"]
      fmt = lambda x: {"round_robin":"Round Robin","random":"Random","least_connection":"Least Connection"}[x]
      # default pilihan kanan berdasarkan mapping lama
      default_map = {"round_robin": "least_connection", "random": "round_robin", "least_connection": "random"}
      try:
        default_idx = algos.index(default_map.get(algorithm, algos[0]))
      except Exception:
        default_idx = 0
      algorithm_right = st.selectbox("Algoritma (kanan)", algos, index=default_idx, format_func=fmt)

    if st.button("⏸ Pause" if not st.session_state.sim_paused else "▶ Play",
                 use_container_width=True, key="btn_pause"):
        st.session_state.sim_paused = not st.session_state.sim_paused
        st.rerun()

    if st.button("↺ Reset Animasi", use_container_width=True, key="btn_reset"):
        st.session_state.sim_reset_key += 1
        st.session_state.sim_paused = False
        st.session_state.sim_stress = False
        st.rerun()

    stress_label = "⏹ Stop Stress Test" if st.session_state.sim_stress else "📈 Stress Test"
    if st.button(stress_label, use_container_width=True, key="btn_stress"):
        st.session_state.sim_stress = not st.session_state.sim_stress
        st.rerun()

    if st.session_state.sim_stress:
        lambda_rate = 5.0
        st.markdown('<div class="stress-active">📈 Stress aktif — λ dipaksa ke 5.0 req/s</div>', unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶ Jalankan & Tampilkan Hasil", use_container_width=True, type="primary")

    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    run_sens = st.button("📊 Jalankan Sensitivity Analysis", use_container_width=True)

    st.markdown('<div class="section-header">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#546e7a;line-height:1.7">
    <b style="color:#78909c">M/M/c Queue:</b><br>
    • Arrival: Poisson (λ)<br>
    • Service: Eksponensial (μ)<br>
    • c: Workers paralel<br>
    • ρ = λ/(c·μ) &lt; 1 → stabil
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# ⚙️ Simulasi Stokastik Sistem Antrian Server")
st.markdown("**Pemodelan & Simulasi Stokastik** — Distribusi Poisson & Eksponensial | Model M/M/c Queue")
st.divider()

# ── Tab Navigation ────────────────────────────────────────────────────────────
tab_anim, tab_hasil, tab_compare, tab_sensitivity = st.tabs([
    "🎬 Animasi",
    "📊 Hasil Simulasi",
    "⚖️ Perbandingan Algoritma",
    "🔬 Sensitivity Analysis",
])


# ════════════════════════════════════════════════════════
# TAB 1 — ANIMASI
# ════════════════════════════════════════════════════════
with tab_anim:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:20px">M/M/c</div>
            <div class="metric-sub">Kendall's Notation</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card {'metric-good' if rho<0.7 else 'metric-warn' if rho<1 else 'metric-bad'}">
            <div class="metric-label">Utilisasi ρ saat ini</div>
            <div class="metric-value">{rho:.3f}</div>
            <div class="metric-sub">{rho_status}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        mode_txt = "Split-Screen ON 🔀" if split_mode else "Single Mode"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Mode Animasi</div>
            <div class="metric-value" style="font-size:16px">{mode_txt}</div>
            <div class="metric-sub">{algorithm.replace("_"," ").title()}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="tab-intro">
        <b style="color:#64b5f6">Cara Membaca Animasi:</b><br>
        • <b>Bola berwarna</b> = request yang bergerak dari generator → load balancer → worker<br>
        • <b>Tumpukan titik di kiri worker</b> = antrian menunggu (makin merah = makin lama menunggu)<br>
        • <b>Progress bar bawah worker</b> = proses service sedang berjalan<br>
        • <b>Timeline bawah</b> = sejarah utilisasi sistem per detik<br>
        • <b>LIVE STATS (bar atas)</b> = total req, throughput, avg wait, Little's L secara real-time
    </div>""", unsafe_allow_html=True)

    # Embed the animation HTML directly (avoid nested data: iframes/sandbox issues)
    frame_html = get_animation_html(
      lambda_rate=lambda_rate,
      num_workers=num_workers,
      mu=mu,
      algorithm_left=algorithm,
      algorithm_right=algorithm_right,
      speed=speed,
      paused=st.session_state.sim_paused,
      split_mode=split_mode,
      reset_key=st.session_state.sim_reset_key,
    )
    import streamlit.components.v1 as components
    components.html(frame_html, height=640, scrolling=True)



# ════════════════════════════════════════════════════════
# TAB 2 — HASIL SIMULASI
# ════════════════════════════════════════════════════════
with tab_hasil:
    if not run_btn and st.session_state.last_results is None:
        st.markdown("""<div class="tab-intro">
            Klik <b style="color:#64b5f6">▶ Jalankan & Tampilkan Hasil</b> di sidebar untuk menjalankan simulasi
            dan melihat analisis lengkap di sini.
        </div>""", unsafe_allow_html=True)
    else:
        if run_btn:
            with st.spinner("⏳ Menjalankan simulasi stokastik..."):
                results     = run_simulation(lambda_rate, num_workers, algorithm, duration, mu)
                all_results = run_all_algorithms(lambda_rate, num_workers, duration, mu)
            st.session_state.last_results = (results, all_results)
            st.session_state.last_params  = dict(
                lambda_rate=lambda_rate, mu=mu, num_workers=num_workers,
                duration=duration, algorithm=algorithm
            )

        results, all_results = st.session_state.last_results
        p_saved = st.session_state.last_params

        df        = pd.DataFrame(results)
        tput      = len(df) / p_saved["duration"]
        avg_wait  = df["wait_time"].mean()
        avg_svc   = df["service_time"].mean()
        avg_total = df["total_time"].mean()
        rho_saved = p_saved["lambda_rate"] / (p_saved["num_workers"] * p_saved["mu"])
        L         = tput * avg_total

        # Alert
        if rho_saved >= 1:
            st.markdown(f'<div class="alert-critical">🔴 <b>SISTEM TIDAK STABIL</b> — ρ = {rho_saved:.3f} ≥ 1. Antrian akan terus membesar.</div>', unsafe_allow_html=True)
        elif rho_saved >= 0.8:
            st.markdown(f'<div class="alert-warn">⚠️ <b>Mendekati kritis</b> — ρ = {rho_saved:.3f}. Sistem mulai jenuh.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-stable">✅ <b>Sistem stabil</b> — ρ = {rho_saved:.3f}. Kapasitas mencukupi.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
        metrics_data = [
            (c1, "Total Request",      len(df),         "request diproses", ""),
            (c2, "Throughput",         f"{tput:.2f}/s",  "req/s aktual",    ""),
            (c3, "Avg Service",        f"{avg_svc:.3f}s","Exp(μ) empiris",  ""),
            (c4, "Avg Wait Wq",        f"{avg_wait:.3f}s","waktu antre",    "warn" if avg_wait > 0.5 else "good"),
            (c5, "Avg Total W",        f"{avg_total:.3f}s","Wq + 1/μ",     ""),
            (c6, "Utilisasi ρ",        f"{rho_saved:.3f}", rho_status,      "bad" if rho_saved>=1 else "warn" if rho_saved>=0.8 else "good"),
            (c7, "Little's L",         f"{L:.2f}",       f"λW",            ""),
        ]
        for col, label, value, sub, cls in metrics_data:
            col.markdown(f"""<div class="metric-card {'metric-'+cls if cls else ''}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Charts
        col_l, col_r = st.columns(2)
        with col_l:
            df["time_bucket"] = (df["arrive_time"] // 5) * 5
            tput_df = df.groupby("time_bucket").size().reset_index(name="count")
            fig1 = px.area(tput_df, x="time_bucket", y="count",
                           labels={"time_bucket":"Waktu (s)","count":"Jumlah Request"})
            fig1.update_traces(line_color="#64b5f6", fillcolor="rgba(100,181,246,0.15)")
            plotly_dark(fig1, "Request per 5 Detik")
            st.plotly_chart(fig1, use_container_width=True)

        with col_r:
            fig2 = px.histogram(df, x="service_time", nbins=25,
                                labels={"service_time":"Service Time (s)","count":"Frekuensi"})
            fig2.update_traces(marker_color="#64b5f6", marker_opacity=0.7)
            x_range = np.linspace(0, df["service_time"].max(), 200)
            y_exp   = p_saved["mu"] * np.exp(-p_saved["mu"]*x_range) * len(df) * (df["service_time"].max()/25)
            fig2.add_trace(go.Scatter(x=x_range, y=y_exp, mode="lines",
                                       line=dict(color="#ef9f27",width=2,dash="dot"),
                                       name="Exp(μ) teoritis"))
            plotly_dark(fig2, "Distribusi Service Time + Kurva Teoritis")
            st.plotly_chart(fig2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            worker_df = df.groupby("worker_id").size().reset_index(name="requests")
            colors_bar = ["#4caf50" if q<=worker_df["requests"].mean()*0.8
                          else "#ff9800" if q<=worker_df["requests"].mean()*1.2
                          else "#f44336" for q in worker_df["requests"]]
            fig3 = px.bar(worker_df, x="worker_id", y="requests",
                          labels={"worker_id":"Worker ID","requests":"Jumlah Request"})
            fig3.update_traces(marker_color=colors_bar)
            plotly_dark(fig3, f"Request per Worker ({p_saved['algorithm'].replace('_',' ').title()})")
            st.plotly_chart(fig3, use_container_width=True)

        with col_r2:
            fig4 = px.scatter(df, x="arrive_time", y="wait_time", color="worker_id",
                              labels={"arrive_time":"Waktu Kedatangan (s)","wait_time":"Wait Time (s)"},
                              color_continuous_scale="Blues")
            plotly_dark(fig4, "Wait Time vs Waktu Kedatangan")
            st.plotly_chart(fig4, use_container_width=True)

        # Utilisasi timeline
        df["interval"] = (df["arrive_time"] // 5) * 5
        util_df = (df.groupby("interval")
                     .apply(lambda g: min(len(g)/(5*p_saved["lambda_rate"]), 1.5))
                     .reset_index(name="utilisasi"))
        fig_util = px.area(util_df, x="interval", y="utilisasi",
                           labels={"interval":"Waktu (s)","utilisasi":"Utilisasi Estimasi"})
        fig_util.update_traces(line_color="#ff9800", fillcolor="rgba(255,152,0,0.1)")
        fig_util.add_hline(y=1.0, line_dash="dot", line_color="#f44336",
                           annotation_text="ρ = 1 (batas kritis)", annotation_position="top right")
        fig_util.add_hline(y=rho_saved, line_dash="dash", line_color="#4caf50",
                           annotation_text=f"ρ teoritis = {rho_saved:.3f}", annotation_position="bottom right")
        plotly_dark(fig_util, "Estimasi Utilisasi per Interval Waktu")
        st.plotly_chart(fig_util, use_container_width=True)

        # Little's Law validation
        st.divider()
        st.markdown("### Validasi Little's Law")
        st.markdown(f"""<div class="littles-card">
            <div style="font-size:12px;color:#546e7a;margin-bottom:8px">L = λ × W (Little's Law)</div>
            <div class="eq">L = {tput:.3f} × {avg_total:.3f} = {L:.3f}</div>
            <div style="font-size:12px;color:#78909c;margin-top:8px">
            λ aktual = <b style="color:#64b5f6">{tput:.3f} req/s</b> &nbsp;|&nbsp;
            W = <b style="color:#64b5f6">{avg_total:.3f}s</b> &nbsp;|&nbsp;
            L = <b style="color:#64b5f6">{L:.3f} request</b>
            </div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Stability check: bandingkan satu run vs multi-run ─────────────────
        st.markdown("### 🔄 Stabilitas Hasil (Single vs Multi-Run)")
        with st.expander("Lihat analisis stabilitas hasil simulasi ini", expanded=False):
            with st.spinner("Menjalankan 5 ulangan simulasi untuk cek stabilitas..."):
                stab_data = run_simulation_multi(
                    p_saved["lambda_rate"], p_saved["num_workers"],
                    p_saved["algorithm"], p_saved["duration"], p_saved["mu"], n_runs=5
                )
            col_s1, col_s2, col_s3 = st.columns(3)
            metrics_stab = [
                (col_s1, "Throughput", "throughput", "/s"),
                (col_s2, "Avg Wait",   "avg_wait",   "s"),
                (col_s3, "Little's L", "L",          ""),
            ]
            for col_s, label_s, key_s, unit_s in metrics_stab:
                mean_s = stab_data.get(key_s + "_mean", 0)
                std_s  = stab_data.get(key_s + "_std", 0)
                cv_s   = (std_s / mean_s * 100) if mean_s > 0 else 0
                color_cv = "#4caf50" if cv_s < 5 else "#ff9800" if cv_s < 15 else "#f44336"
                col_s.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label_s}</div>
                    <div class="metric-value">{mean_s:.3f}{unit_s}</div>
                    <div class="metric-sub">CV: <span style="color:{color_cv}">{cv_s:.1f}%</span> 
                    {"✅ stabil" if cv_s<5 else "⚠️ moderat" if cv_s<15 else "🔴 tinggi variasi"}</div>
                </div>""", unsafe_allow_html=True)
            runs_tput = stab_data.get("throughput_runs", [])
            if runs_tput:
                fig_stab = go.Figure()
                fig_stab.add_trace(go.Scatter(
                    y=runs_tput, x=list(range(1, len(runs_tput)+1)),
                    mode="lines+markers",
                    line=dict(color="#64b5f6", width=1.5),
                    marker=dict(size=8, color="#64b5f6"),
                    name="Throughput per run",
                ))
                mean_line = np.mean(runs_tput)
                std_line  = np.std(runs_tput)
                fig_stab.add_hline(y=mean_line, line_dash="dash", line_color="#4caf50",
                                   annotation_text=f"mean={mean_line:.3f}")
                fig_stab.add_hrect(y0=mean_line-std_line, y1=mean_line+std_line,
                                   fillcolor="rgba(100,181,246,0.08)", line_width=0,
                                   annotation_text="±1σ", annotation_position="top right")
                plotly_dark(fig_stab, "Throughput per Run (Stabilitas)")
                fig_stab.update_xaxes(title="Run ke-")
                fig_stab.update_yaxes(title="Throughput (req/s)")
                st.plotly_chart(fig_stab, use_container_width=True)

        st.divider()

        # Export PDF
        st.markdown("### 📄 Export Laporan")
        col_pdf, col_csv = st.columns(2)
        with col_pdf:
            if st.button("📄 Generate PDF Laporan", use_container_width=True):
                with st.spinner("Membuat PDF..."):
                    pdf_bytes = generate_pdf_report(
                        p_saved, df, all_results, p_saved["duration"], rho_saved, L
                    )
                st.download_button(
                    "⬇️ Download PDF",
                    pdf_bytes,
                    f"laporan_simulasi_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
        with col_csv:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "simulation_results.csv", "text/csv",
                               use_container_width=True)

        st.divider()

        # Log
        st.markdown("### Log Hasil Simulasi")
        st.dataframe(
            df[["request_id","worker_id","arrive_time","wait_time","service_time","total_time"]],
            use_container_width=True, hide_index=True,
        )


# ════════════════════════════════════════════════════════
# TAB 3 — PERBANDINGAN ALGORITMA
# ════════════════════════════════════════════════════════
with tab_compare:
    if st.session_state.last_results is None:
        st.markdown("""<div class="tab-intro">
            Jalankan simulasi terlebih dahulu dari sidebar untuk melihat perbandingan algoritma.
        </div>""", unsafe_allow_html=True)
    else:
        _, all_results = st.session_state.last_results
        p_saved = st.session_state.last_params
        dur = p_saved["duration"]
        rho_s = p_saved["lambda_rate"] / (p_saved["num_workers"] * p_saved["mu"])

        ranked = rank_algorithms(all_results, dur, p_saved["lambda_rate"], rho_s)

        # Auto ranking
        st.markdown("### 🏆 Ranking Algoritma Otomatis")
        medals = ["🥇", "🥈", "🥉"]
        badge_styles = ["badge-best", "badge-fast", "badge-warn"]
        badge_labels = ["TERBAIK", "RUNNER-UP", "KE-3"]
        cols_rank = st.columns(3)
        for i, (algo, stats) in enumerate(ranked):
            with cols_rank[i]:
                st.markdown(f"""<div class="rank-card">
                    <div class="rank-medal">{medals[i]} <span class="rank-algo">{algo.replace("_"," ").title()}</span>
                    <span class="rank-badge {badge_styles[i]}">{badge_labels[i]}</span></div>
                    <div style="margin-top:8px;font-size:12px;color:#78909c;font-family:monospace">
                        Throughput: <b style="color:#64b5f6">{stats['tput']:.3f}</b> req/s<br>
                        Avg Wait: <b style="color:#ff9800">{stats['avg_wait']:.4f}</b>s<br>
                        P99 Wait: <b style="color:#ef5350">{stats['p99_wait']:.4f}</b>s<br>
                        Composite Score: <b style="color:#a5d6a7">{stats['score']:.3f}</b>
                    </div>
                    <div class="rank-reason">Score = Throughput − 2×AvgWait − 0.5×P99Wait</div>
                </div>""", unsafe_allow_html=True)

        # Recommendation box
        best_algo = ranked[0][0]
        best_stats = ranked[0][1]
        st.markdown(f"""<div class="alert-stable" style="margin-top:16px">
            ✅ <b>Rekomendasi untuk λ={p_saved['lambda_rate']}, c={p_saved['num_workers']}, μ={p_saved['mu']}:</b>
            Gunakan <b>{best_algo.replace('_',' ').title()}</b> —
            throughput terbaik {best_stats['tput']:.3f} req/s dengan rata-rata wait {best_stats['avg_wait']:.4f}s.
            {"Least Connection unggul saat ada perbedaan beban antar worker." if best_algo=="least_connection" else
             "Round Robin efisien saat beban seragam dan service time homogen." if best_algo=="round_robin" else
             "Random memberikan distribusi probabilistik yang merata pada kondisi ini."}
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Comparison table
        st.markdown("### Tabel Perbandingan Lengkap")
        comparison = []
        for algo, res in all_results.items():
            if res:
                d = pd.DataFrame(res)
                t = len(d)/dur
                comparison.append({
                    "Algoritma":              algo.replace("_"," ").title(),
                    "Total Request":          len(d),
                    "Throughput (req/s)":     round(t, 3),
                    "Avg Service Time (s)":   round(d["service_time"].mean(), 4),
                    "Avg Wait Time (s)":      round(d["wait_time"].mean(), 4),
                    "P99 Wait Time (s)":      round(d["wait_time"].quantile(0.99), 4),
                    "Avg Total Time (s)":     round(d["total_time"].mean(), 4),
                    "Little's L":             round(t * d["total_time"].mean(), 3),
                    "Rank":                   f"#{[r[0] for r in ranked].index(algo)+1}",
                })
        comp_df = pd.DataFrame(comparison)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()

        # Charts
        col_bar, col_box = st.columns(2)
        with col_bar:
            fig5 = px.bar(comp_df, x="Algoritma", y="Throughput (req/s)", color="Algoritma",
                          color_discrete_sequence=["#64b5f6","#4caf50","#ff9800"])
            plotly_dark(fig5, "Throughput per Algoritma")
            st.plotly_chart(fig5, use_container_width=True)

        with col_box:
            fig6 = go.Figure()
            algo_colors_map = {"Round Robin":"#64b5f6","Random":"#4caf50","Least Connection":"#ff9800"}
            for algo, res in all_results.items():
                if res:
                    d = pd.DataFrame(res)
                    name = algo.replace("_"," ").title()
                    fig6.add_trace(go.Box(y=d["wait_time"], name=name,
                                           marker_color=algo_colors_map.get(name,"#64b5f6"), boxmean=True))
            plotly_dark(fig6, "Distribusi Wait Time per Algoritma")
            st.plotly_chart(fig6, use_container_width=True)

        # ── Confidence Interval Chart ──────────────────────────────────────────
        st.markdown("### 📊 Perbandingan Metrik dengan Confidence Interval (95%)")
        st.markdown("""<div class="tab-intro" style="font-size:12px;padding:10px 16px">
            Setiap algoritma dijalankan <b>8 kali</b> dengan seed berbeda. 
            Error bar menunjukkan 95% confidence interval — semakin pendek = semakin konsisten.
        </div>""", unsafe_allow_html=True)

        with st.spinner("Menghitung confidence interval (8 runs × 3 algoritma)..."):
            ci_results = {}
            for algo in ["round_robin", "random", "least_connection"]:
                ci_results[algo] = run_simulation_multi(
                    p_saved["lambda_rate"], p_saved["num_workers"],
                    algo, p_saved["duration"], p_saved["mu"], n_runs=8
                )

        ci_metric_tabs = st.tabs(["Throughput", "Avg Wait Time", "Little's L"])

        ci_colors = {"round_robin": "#64b5f6", "random": "#4caf50", "least_connection": "#ff9800"}
        ci_labels = {"round_robin": "Round Robin", "random": "Random", "least_connection": "Least Connection"}

        ci_metric_keys = [
            ("throughput", "Throughput (req/s)", ci_metric_tabs[0]),
            ("avg_wait",   "Avg Wait Time (s)",  ci_metric_tabs[1]),
            ("L",          "Little's L",         ci_metric_tabs[2]),
        ]

        for metric_key, metric_label, tab_obj in ci_metric_keys:
            with tab_obj:
                fig_ci = go.Figure()
                for algo, ci_data in ci_results.items():
                    if not ci_data:
                        continue
                    mean_val = ci_data.get(metric_key + "_mean", 0)
                    ci95_val = ci_data.get(metric_key + "_ci95", 0)
                    runs_val = ci_data.get(metric_key + "_runs", [])
                    fig_ci.add_trace(go.Bar(
                        x=[ci_labels[algo]],
                        y=[mean_val],
                        error_y=dict(type="data", array=[ci95_val], visible=True,
                                     color="rgba(255,255,255,0.6)", thickness=2, width=10),
                        name=ci_labels[algo],
                        marker_color=ci_colors[algo],
                        marker_opacity=0.85,
                    ))
                    fig_ci.add_trace(go.Scatter(
                        x=[ci_labels[algo]] * len(runs_val),
                        y=runs_val,
                        mode="markers",
                        name=f"{ci_labels[algo]} (runs)",
                        marker=dict(color=ci_colors[algo], size=7, opacity=0.6,
                                    symbol="circle-open", line=dict(width=1.5)),
                        showlegend=False,
                    ))
                plotly_dark(fig_ci, f"{metric_label} — Mean ± 95% CI (n=8 runs)")
                fig_ci.update_layout(barmode="group", showlegend=False)
                st.plotly_chart(fig_ci, use_container_width=True)

        st.markdown("#### Tabel Ringkasan Confidence Interval")
        ci_table_rows = []
        for algo, ci_data in ci_results.items():
            if ci_data:
                ci_table_rows.append({
                    "Algoritma":          ci_labels[algo],
                    "Throughput mean":    f"{ci_data.get('throughput_mean',0):.3f}",
                    "Throughput ±CI95":   f"±{ci_data.get('throughput_ci95',0):.4f}",
                    "Avg Wait mean (s)":  f"{ci_data.get('avg_wait_mean',0):.4f}",
                    "Avg Wait ±CI95":     f"±{ci_data.get('avg_wait_ci95',0):.5f}",
                    "L mean":             f"{ci_data.get('L_mean',0):.3f}",
                    "L ±CI95":            f"±{ci_data.get('L_ci95',0):.4f}",
                })
        if ci_table_rows:
            st.dataframe(pd.DataFrame(ci_table_rows), use_container_width=True, hide_index=True)

        st.markdown("### 📈 Grafik L vs ρ — Teoritis (Erlang-C M/M/c) vs Empiris")
        st.markdown("""<div class="tab-intro" style="font-size:12px;padding:12px 16px">
            Kurva teoritis dihitung menggunakan formula <b>Erlang-C M/M/c</b> yang akurat
            (bukan aproksimasi M/M/1). Semakin dekat titik empiris ke kurva, semakin valid simulasi.
            Garis putus-putus menunjukkan batas ρ = 1 (sistem tidak stabil).
        </div>""", unsafe_allow_html=True)

        c_val  = p_saved["num_workers"]
        mu_val = p_saved["mu"]

        rho_range = np.linspace(0.05, 0.97, 100)
        fig_lrho  = go.Figure()

        worker_plot_options = sorted(set([1, 2, c_val, min(c_val+1, 5)]))
        line_colors_theory  = ["#546e7a", "#64b5f6", "#4caf50", "#ce93d8"]

        for idx_w, c_plot in enumerate(worker_plot_options):
            L_theory = []
            rho_valid = []
            for r in rho_range:
                lam_t = r * c_plot * mu_val
                res_t = erlang_c_metrics(lam_t, c_plot, mu_val)
                if res_t:
                    L_theory.append(res_t["L"])
                    rho_valid.append(r)
            if L_theory:
                dash_style = "solid" if c_plot == c_val else "dot"
                width_style = 2.5 if c_plot == c_val else 1.5
                fig_lrho.add_trace(go.Scatter(
                    x=rho_valid, y=L_theory,
                    mode="lines",
                    name=f"c={c_plot} teoritis (Erlang-C)",
                    line=dict(color=line_colors_theory[idx_w % len(line_colors_theory)],
                              width=width_style, dash=dash_style),
                ))

        emp_rho_pts, emp_L_pts, emp_L_err = [], [], []
        for lam_scan in np.linspace(0.3, c_val * mu_val * 0.95, 14):
            runs = []
            for seed_i in range(5):
                res_s = run_simulation(lam_scan, c_val, p_saved["algorithm"], 30, mu_val, seed=seed_i*7)
                if res_s:
                    d_s = pd.DataFrame(res_s)
                    t_s = len(d_s) / 30
                    runs.append(t_s * d_s["total_time"].mean())
            if runs:
                emp_rho_pts.append(lam_scan / (c_val * mu_val))
                emp_L_pts.append(np.mean(runs))
                emp_L_err.append(np.std(runs))

        fig_lrho.add_trace(go.Scatter(
            x=emp_rho_pts, y=emp_L_pts,
            error_y=dict(type="data", array=emp_L_err, visible=True,
                         color="rgba(76,175,80,0.5)", thickness=1.5, width=6),
            mode="markers+lines",
            name=f"c={c_val} empiris (simulasi, n=5)",
            marker=dict(color="#4caf50", size=9, symbol="circle",
                        line=dict(color="#2e7d32", width=1.5)),
            line=dict(color="#4caf50", width=1.5),
        ))

        fig_lrho.add_vline(x=1.0, line_dash="dot", line_color="#f44336",
                           annotation_text="ρ = 1 (kritis)", annotation_position="top right")
        plotly_dark(fig_lrho, f"Little's L vs ρ — Erlang-C M/M/c Teoritis vs Empiris (c={c_val}, μ={mu_val})")
        fig_lrho.update_xaxes(title="ρ (utilisasi)", range=[0, 1.02])
        fig_lrho.update_yaxes(title="L (avg request dalam sistem)")
        st.plotly_chart(fig_lrho, use_container_width=True)

        st.markdown("#### Tabel Validasi Erlang-C")
        validation_rows = []
        for r_check in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            lam_check = r_check * c_val * mu_val
            theory    = erlang_c_metrics(lam_check, c_val, mu_val)
            sim_runs  = []
            for s in range(5):
                res_v = run_simulation(lam_check, c_val, p_saved["algorithm"], 30, mu_val, seed=s*13)
                if res_v:
                    d_v   = pd.DataFrame(res_v)
                    t_v   = len(d_v) / 30
                    sim_runs.append({"W": d_v["total_time"].mean(), "Wq": d_v["wait_time"].mean(), "L": t_v * d_v["total_time"].mean()})
            if theory and sim_runs:
                avg_W_emp  = np.mean([r["W"] for r in sim_runs])
                avg_Wq_emp = np.mean([r["Wq"] for r in sim_runs])
                avg_L_emp  = np.mean([r["L"] for r in sim_runs])
                validation_rows.append({
                    "ρ": r_check,
                    "Wq teoritis (s)": round(theory["Wq"], 4),
                    "Wq empiris (s)":  round(avg_Wq_emp, 4),
                    "W teoritis (s)":  round(theory["W"], 4),
                    "W empiris (s)":   round(avg_W_emp, 4),
                    "L teoritis":      round(theory["L"], 3),
                    "L empiris":       round(avg_L_emp, 3),
                    "Error L (%)":     round(abs(theory["L"] - avg_L_emp) / max(theory["L"], 0.001) * 100, 1),
                })
        if validation_rows:
            val_df = pd.DataFrame(validation_rows)
            st.dataframe(
                val_df.style.background_gradient(subset=["Error L (%)"], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True
            )


# ════════════════════════════════════════════════════════
# TAB 4 — SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════
with tab_sensitivity:
    st.markdown("""<div class="tab-intro">
        <b style="color:#64b5f6">Sensitivity Analysis</b> menjalankan simulasi secara otomatis
        untuk berbagai kombinasi λ dan c, lalu menampilkan hasilnya sebagai heatmap dan line chart multi-series.
        Ini membantu memahami bagaimana sistem merespons perubahan beban dan kapasitas.
    </div>""", unsafe_allow_html=True)

    if run_sens:
        lambda_values  = np.arange(0.5, 5.5, 0.5)
        worker_options = [1, 2, 3, 4, 5]
        sens_mu        = mu

        with st.spinner("⏳ Menjalankan sensitivity analysis (semua kombinasi λ × c)..."):
            sens_records = []
            for c in worker_options:
                for lam in lambda_values:
                    rho_s = lam / (c * sens_mu)
                    res_s = run_simulation(lam, c, "round_robin", 30, sens_mu, seed=42)
                    if res_s:
                        d_s = pd.DataFrame(res_s)
                        t_s = len(d_s)/30
                        sens_records.append({
                            "λ": round(lam, 1),
                            "Workers (c)": c,
                            "ρ": round(rho_s, 3),
                            "Throughput": round(t_s, 3),
                            "Avg Wait": round(d_s["wait_time"].mean(), 4),
                            "Little's L": round(t_s * d_s["total_time"].mean(), 3),
                            "Stable": rho_s < 1,
                        })

        sens_df = pd.DataFrame(sens_records)
        st.success(f"✅ Selesai — {len(sens_records)} kombinasi diuji.")

        # Heatmap Avg Wait
        st.markdown("### Heatmap Avg Wait Time (λ × c)")
        pivot_wait = sens_df.pivot(index="Workers (c)", columns="λ", values="Avg Wait")
        fig_heat = go.Figure(go.Heatmap(
            z=pivot_wait.values,
            x=pivot_wait.columns.astype(str),
            y=pivot_wait.index,
            colorscale="RdYlGn_r",
            zmin=0,
            colorbar=dict(title="Avg Wait (s)", tickfont=dict(color="#90a4ae")),
            hoverongaps=False,
            text=np.round(pivot_wait.values, 3),
            texttemplate="%{text}s",
            textfont=dict(size=10),
        ))
        fig_heat.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1422", font_color="#b0bec5",
            font_family="DM Sans",
            xaxis=dict(title="λ (req/s)", gridcolor="#1a2a3a"),
            yaxis=dict(title="Workers (c)", gridcolor="#1a2a3a"),
            title=dict(text="Avg Wait Time Heatmap — merah=tinggi, hijau=rendah",
                       font=dict(color="#90caf9", size=14, family="JetBrains Mono")),
            margin=dict(l=60,r=30,t=60,b=50),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Heatmap Utilisasi
        st.markdown("### Heatmap Utilisasi ρ (λ × c)")
        pivot_rho = sens_df.pivot(index="Workers (c)", columns="λ", values="ρ")
        fig_heat2 = go.Figure(go.Heatmap(
            z=pivot_rho.values,
            x=pivot_rho.columns.astype(str),
            y=pivot_rho.index,
            colorscale="RdYlGn_r",
            zmin=0, zmax=1.5,
            colorbar=dict(title="ρ utilisasi", tickfont=dict(color="#90a4ae")),
            text=np.round(pivot_rho.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
        ))
        fig_heat2.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1422", font_color="#b0bec5",
            font_family="DM Sans",
            xaxis=dict(title="λ (req/s)", gridcolor="#1a2a3a"),
            yaxis=dict(title="Workers (c)", gridcolor="#1a2a3a"),
            title=dict(text="Utilisasi ρ Heatmap — merah=kritis (≥1), hijau=stabil",
                       font=dict(color="#90caf9", size=14, family="JetBrains Mono")),
            margin=dict(l=60,r=30,t=60,b=50),
        )
        st.plotly_chart(fig_heat2, use_container_width=True)

        # Line chart multi-series — Throughput vs λ per worker count
        st.markdown("### Throughput vs λ (multi-series per c)")
        fig_line = go.Figure()
        line_colors = ["#64b5f6","#4caf50","#ff9800","#ce93d8","#ef5350"]
        for i, c in enumerate(worker_options):
            sub = sens_df[sens_df["Workers (c)"] == c]
            fig_line.add_trace(go.Scatter(
                x=sub["λ"], y=sub["Throughput"],
                mode="lines+markers", name=f"c={c} workers",
                line=dict(color=line_colors[i], width=2),
                marker=dict(size=7),
            ))
        plotly_dark(fig_line, "Throughput vs λ per Jumlah Worker")
        fig_line.update_xaxes(title="λ (req/s)")
        fig_line.update_yaxes(title="Throughput (req/s)")
        st.plotly_chart(fig_line, use_container_width=True)

        # Line chart — Avg Wait vs λ per worker count
        st.markdown("### Avg Wait Time vs λ (multi-series per c)")
        fig_line2 = go.Figure()
        for i, c in enumerate(worker_options):
            sub = sens_df[sens_df["Workers (c)"] == c]
            fig_line2.add_trace(go.Scatter(
                x=sub["λ"], y=sub["Avg Wait"],
                mode="lines+markers", name=f"c={c} workers",
                line=dict(color=line_colors[i], width=2),
                marker=dict(size=7),
            ))
        plotly_dark(fig_line2, "Avg Wait Time vs λ per Jumlah Worker")
        fig_line2.update_xaxes(title="λ (req/s)")
        fig_line2.update_yaxes(title="Avg Wait Time (s)")
        st.plotly_chart(fig_line2, use_container_width=True)

        st.divider()
        st.markdown("### Tabel Sensitivity Analysis Lengkap")
        st.dataframe(
            sens_df.style.background_gradient(subset=["Avg Wait","ρ"], cmap="RdYlGn_r")
                         .format({"ρ":"{:.3f}","Throughput":"{:.3f}","Avg Wait":"{:.4f}","Little's L":"{:.3f}"}),
            use_container_width=True, hide_index=True
        )

        csv_sens = sens_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Sensitivity CSV", csv_sens,
                           "sensitivity_analysis.csv", "text/csv")

    else:
        st.info("Klik **📊 Jalankan Sensitivity Analysis** di sidebar untuk memulai analisis.")
        st.markdown("""<div style="background:#0d1422;border:1px solid #1a2a3a;border-radius:10px;padding:20px;margin-top:16px">
            <div style="font-size:12px;color:#546e7a;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px">Preview: Apa yang akan dianalisis</div>
            <div style="font-family:monospace;font-size:12px;color:#90a4ae;line-height:2.2">
            λ ∈ {{0.5, 1.0, 1.5, ..., 5.0}} — 10 nilai arrival rate<br>
            c ∈ {{1, 2, 3, 4, 5}} — 5 konfigurasi worker<br>
            ──────────────────────────────<br>
            Total: 50 kombinasi simulasi<br>
            Output: 2 heatmap + 2 line chart + tabel + CSV export
            </div>
        </div>""", unsafe_allow_html=True)