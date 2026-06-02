import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
import io
import math

# ══════════════════════════════════════════════════════════════════════════════
# ERLANG-C
# ══════════════════════════════════════════════════════════════════════════════
def erlang_c_metrics(lam, c, mu):
    rho = lam / (c * mu)
    if rho >= 1.0:
        return None
    sum_terms = sum((c * rho) ** k / math.factorial(k) for k in range(c))
    last_term  = (c * rho) ** c / (math.factorial(c) * (1 - rho))
    P0         = 1.0 / (sum_terms + last_term)
    C_erlang   = last_term * P0
    Wq = C_erlang / (c * mu - lam)
    W  = Wq + 1.0 / mu
    Lq = lam * Wq
    L  = lam * W
    return {"rho": rho, "P0": P0, "C_erlang": C_erlang,
            "Wq": Wq, "W": W, "Lq": Lq, "L": L}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Simulasi Antrian Server",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLES  (color palette: dark theme dengan aksen biru terang)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0f0f0f !important; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: 1px solid #404040;
}
[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #82aaff !important;
}
/* Slider track fill (left side - hijau) */
[data-testid="stSidebar"] .stSlider svg [fill="#FB8500"],
[data-testid="stSidebar"] .stSlider svg circle {
    fill: #66bb6a !important;
}
/* Slider thumb/circle */
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background-color: #66bb6a !important;
}
/* Slider min/max values - ALL inputs and text */
[data-testid="stSidebar"] .stSlider input {
    color: #ffffff !important;
    background-color: #1a1a1a !important;
    opacity: 0 !important;
    font-weight: 700 !important;
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}
[data-testid="stSidebar"] .stSlider input[type="text"] {
    display: none !important;
    opacity: 0 !important;
}
}
[data-testid="stSidebar"] .stSlider input[type="number"] {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSlider span {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stSlider div span,
[data-testid="stSidebar"] .stSlider div p,
[data-testid="stSidebar"] .stSlider p {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stSlider {
    color: #ffffff !important;
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    background: #242424;
    border: 1px solid #404040;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
.metric-label { font-size: 10px; color: #a0a0a0; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
.metric-value { font-size: 26px; font-weight: 700; color: #82aaff; line-height: 1;
    font-family: 'JetBrains Mono', monospace; }
.metric-sub   { font-size: 11px; color: #808080; margin-top: 4px; }
.metric-good  .metric-value { color: #66bb6a; }
.metric-warn  .metric-value { color: #ce93d8; }
.metric-bad   .metric-value { color: #ef5350; }

/* ── Rho card ───────────────────────────────────────────── */
.rho-card {
    border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
    border-left: 3px solid #82aaff;
    background: #2a2a2a; color: #e8e8e8;
}

/* ── Alert banners ──────────────────────────────────────── */
.alert-stable   { background:rgba(102,187,106,0.15); border:1px solid #66bb6a; border-radius:8px;
    padding:10px 16px; color:#66bb6a; font-size:13px; }
.alert-critical { background:rgba(239,83,80,0.15); border:1px solid #ef5350; border-radius:8px;
    padding:10px 16px; color:#ef5350; font-size:13px; }
.alert-warn     { background:rgba(206,147,216,0.15); border:1px solid #ce93d8; border-radius:8px;
    padding:10px 16px; color:#ce93d8; font-size:13px; }

/* ── Section header ─────────────────────────────────────── */
.section-header { font-size:11px; color:#a0a0a0; text-transform:uppercase;
    letter-spacing:1.5px; margin:16px 0 8px; padding-bottom:4px;
    border-bottom:1px solid #404040; font-family:'JetBrains Mono',monospace; }

/* ── Rank cards ─────────────────────────────────────────── */
.rank-card { background:#242424; border:1px solid #404040; border-radius:10px;
    padding:16px 20px; margin-bottom:8px; box-shadow:0 1px 4px rgba(0,0,0,0.3); }
.rank-algo  { font-size:15px; font-weight:700; color:#82aaff; }
.rank-reason { font-size:11px; color:#a0a0a0; margin-top:4px; }
.rank-badge { display:inline-block; font-size:10px; padding:2px 8px;
    border-radius:20px; margin-left:8px; }
.badge-best { background:rgba(102,187,106,0.2); color:#66bb6a; border:1px solid #66bb6a; }
.badge-fast { background:rgba(130,170,255,0.2); color:#82aaff; border:1px solid #82aaff; }
.badge-warn { background:rgba(206,147,216,0.2); color:#ce93d8; border:1px solid #ce93d8; }

/* ── Little's law card ──────────────────────────────────── */
.littles-card { border:1px solid #404040; border-left:4px solid #82aaff;
    border-radius:10px; padding:16px 20px; margin:8px 0;
    font-size:14px; background:#2a2a2a; }
.littles-card .eq { font-size:20px; font-weight:700; color:#82aaff;
    font-family:'JetBrains Mono',monospace; }

/* ── Stress badge ───────────────────────────────────────── */
.stress-active { background:rgba(206,147,216,0.15); border:1px solid #ce93d8; border-radius:6px;
    padding:6px 10px; font-size:11px; color:#ce93d8; margin-top:4px; }

/* ── Tab intro ──────────────────────────────────────────── */
.tab-intro { background:#2a2a2a; border:1px solid #404040; border-left:4px solid #82aaff;
    border-radius:10px; padding:18px 22px; margin-bottom:18px;
    font-size:13px; color:#d0d0d0; line-height:1.9; }

/* ── Conclusion card ────────────────────────────────────── */
.conclusion-card { border-radius:12px; padding:20px 24px; margin:12px 0;
    border-left:5px solid #82aaff; background:#2a2a2a; }
.conclusion-title { font-size:16px; font-weight:700; color:#e8e8e8; margin-bottom:8px; }
.conclusion-body  { font-size:13px; color:#d0d0d0; line-height:1.8; }
.conclusion-good  { border-left-color:#66bb6a; background:rgba(102,187,106,0.1); }
.conclusion-warn  { border-left-color:#ce93d8; background:rgba(206,147,216,0.1); }
.conclusion-bad   { border-left-color:#ef5350; background:rgba(239,83,80,0.1); }

/* ── Guide steps ────────────────────────────────────────── */
.guide-step { display:flex; gap:14px; align-items:flex-start;
    padding:12px 0; border-bottom:1px solid #3a3a3a; }
.guide-step:last-child { border-bottom:none; }
.step-num { min-width:28px; height:28px; border-radius:50%; background:#82aaff;
    color:#0f0f0f; display:flex; align-items:center; justify-content:center;
    font-size:13px; font-weight:700; margin-top:2px; }
.step-content { flex:1; }
.step-title { font-size:14px; font-weight:600; color:#e8e8e8; margin-bottom:3px; }
.step-desc  { font-size:12px; color:#a0a0a0; line-height:1.6; }

/* ── Theory vs empirical table ──────────────────────────── */
.theory-table { width:100%; border-collapse:collapse; font-size:13px;
    font-family:'JetBrains Mono',monospace; }
.theory-table th { background:#404040; color:#82aaff; padding:8px 12px;
    text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; }
.theory-table td { padding:8px 12px; border-bottom:1px solid #3a3a3a; color:#d0d0d0; }
.theory-table tr:nth-child(even) td { background:#2a2a2a; }
.match-good { color:#66bb6a; font-weight:600; }
.match-ok   { color:#ce93d8; font-weight:600; }
.match-bad  { color:#ef5350; font-weight:600; }

/* ── Button styling ─────────────────────────────────────── */
[data-testid="stSidebar"] button {
    background: #2a2a2a !important;
    border: 1px solid #505050 !important;
    color: #e8e8e8 !important;
}
[data-testid="stSidebar"] button:hover {
    background: #333333 !important;
    border-color: #606060 !important;
}
button {
    background: #2a2a2a !important;
    border: 1px solid #505050 !important;
    color: #e8e8e8 !important;
}
button:hover {
    background: #333333 !important;
    border-color: #606060 !important;
}
</style>

<script>
// Force slider number visibility on every interaction
function fixSliderNumbers() {
    const sliderInputs = document.querySelectorAll('[data-testid="stSidebar"] .stSlider input[type="text"]');
    sliderInputs.forEach(input => {
        input.style.color = '#ffffff';
        input.style.backgroundColor = '#242424';
        input.style.fontWeight = '700';
        input.style.opacity = '1';
    });
}

// Run immediately
fixSliderNumbers();

// Run on page changes/rerenders
document.addEventListener('DOMContentLoaded', fixSliderNumbers);
window.addEventListener('load', fixSliderNumbers);

// Observer untuk mutation pada DOM
const observer = new MutationObserver(fixSliderNumbers);
observer.observe(document.body, {childList: true, subtree: true});
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "sim_paused":      False,
    "sim_reset_key":   0,
    "sim_stress":      False,
    "stress_step":     0,
    "last_results":    None,
    "last_params":     None,
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
    results, workers_queue, workers_busy_until = [], [0]*num_workers, [0.0]*num_workers
    current_time, request_id, base_time = 0.0, 0, datetime.now()
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
    return {algo: run_simulation(lambda_rate, num_workers, algo, duration, mu)
            for algo in ["round_robin", "random", "least_connection"]}


def run_simulation_multi(lambda_rate, num_workers, algorithm, duration, mu, n_runs=8):
    metrics = {"throughput": [], "avg_wait": [], "avg_total": [], "L": [], "p99_wait": []}
    for i in range(n_runs):
        res = run_simulation(lambda_rate, num_workers, algorithm, duration, mu, seed=i*17+3)
        if not res:
            continue
        d = pd.DataFrame(res)
        t = len(d) / duration
        metrics["throughput"].append(t)
        metrics["avg_wait"].append(d["wait_time"].mean())
        metrics["avg_total"].append(d["total_time"].mean())
        metrics["L"].append(t * d["total_time"].mean())
        metrics["p99_wait"].append(d["wait_time"].quantile(0.99))
    result = {}
    for k, vals in metrics.items():
        if vals:
            result[k+"_mean"] = np.mean(vals)
            result[k+"_std"]  = np.std(vals)
            result[k+"_ci95"] = 1.96 * np.std(vals) / np.sqrt(len(vals))
            result[k+"_runs"] = vals
    return result


def plotly_light(fig, title=""):
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8faff",
        font_color="#374151", font_family="Inter",
        title=dict(text=title, font=dict(color="#1e2a4a", size=14, family="JetBrains Mono")),
        xaxis=dict(gridcolor="#e8eeff", linecolor="#dde3f5"),
        yaxis=dict(gridcolor="#e8eeff", linecolor="#dde3f5"),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(bgcolor="#f8faff", bordercolor="#dde3f5", font=dict(size=12)),
    )
    return fig


def rank_algorithms(all_results, duration, lambda_rate, rho):
    scores = {}
    for algo, res in all_results.items():
        if not res:
            continue
        d = pd.DataFrame(res)
        tput      = len(d) / duration
        avg_wait  = d["wait_time"].mean()
        p99_wait  = d["wait_time"].quantile(0.99)
        fairness  = d.groupby("worker_id").size().std() if len(d.groupby("worker_id")) > 1 else 0
        score     = tput - avg_wait*2 - p99_wait*0.5 - fairness*0.01
        scores[algo] = {"score": score, "tput": tput, "avg_wait": avg_wait,
                        "p99_wait": p99_wait, "fairness": fairness}
    return sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATION HTML  — server rack + stochastic visuals
# ══════════════════════════════════════════════════════════════════════════════
def get_animation_html(lambda_rate, num_workers, mu, algorithm,
                       speed=1.0, paused=False, split_mode=False,
                       algo_left="round_robin", algo_right="least_connection",
                       stress_lambda=None):

    algo_labels = {"round_robin":"Round Robin","random":"Random","least_connection":"Least Connection"}
    eff_lambda  = stress_lambda if stress_lambda is not None else lambda_rate
    rho         = eff_lambda / (num_workers * mu)
    rho_color   = "#16a34a" if rho < 0.7 else "#d97706" if rho < 1 else "#dc2626"
    rho_sub     = "✅ stabil" if rho < 0.7 else "⚠️ mendekati kritis" if rho < 1 else "🔴 kritis!"
    init_paused = "true" if paused else "false"
    split_js    = "true" if split_mode else "false"
    algoL_js    = algo_left  if split_mode else algorithm
    algoR_js    = algo_right if split_mode else "round_robin"
    stress_lam_js = "null" if stress_lambda is None else str(stress_lambda)

    return f"""<!doctype html>
<html lang="id">
<head>
<meta charset="UTF-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{width:100%;height:100%;overflow:hidden;background:#0f0f0f;
  font-family:'Segoe UI',Arial,sans-serif;color:#e8e8e8;
  display:flex;flex-direction:column;}}

/* ── Top info bar ─────────────────────────────────────── */
#infoBar{{flex-shrink:0;display:flex;align-items:stretch;
  background:#1a1a1a;border-bottom:2px solid #404040;min-height:64px;}}
.info-item{{display:flex;flex-direction:column;justify-content:center;gap:2px;
  padding:8px 14px;border-right:1px solid #3a3a3a;flex:1;}}
.info-item:last-child{{border-right:none;}}
.info-label{{font-size:9px;color:#a0a0a0;text-transform:uppercase;letter-spacing:1px;font-weight:500;}}
.info-value{{font-size:18px;font-weight:700;color:#82aaff;font-family:monospace;line-height:1.1;}}
.info-unit {{font-size:11px;color:#808080;font-weight:400;}}
.info-sub  {{font-size:9px;color:#808080;}}

/* ── Formula bar ──────────────────────────────────────── */
#formulaBar{{flex-shrink:0;background:#2a2a2a;border-bottom:1px solid #3a3a3a;
  padding:4px 14px;font-size:10px;color:#a0a0a0;
  display:flex;align-items:center;gap:8px;font-family:monospace;flex-wrap:wrap;}}
#formulaBar .rhoVal{{color:#66bb6a;font-weight:700;font-size:11px;}}
#formulaBar .sep{{color:#505050;}}

/* ── RT stats bar ─────────────────────────────────────── */
#rtStats{{flex-shrink:0;display:flex;background:#1a1a1a;border-bottom:1px solid #404040;
  padding:5px 14px;gap:20px;align-items:center;}}
.rt-item{{display:flex;flex-direction:column;align-items:center;min-width:70px;}}
.rt-label{{font-size:8px;color:#a0a0a0;text-transform:uppercase;letter-spacing:.8px;}}
.rt-val  {{font-size:15px;font-weight:700;font-family:monospace;color:#82aaff;}}

/* ── Dist panel ───────────────────────────────────────── */
#distPanel{{position:absolute;right:0;top:0;bottom:0;width:200px;
  background:rgba(26,26,26,0.98);border-left:1px solid #404040;
  padding:8px 10px;overflow:hidden;font-family:monospace;font-size:10px;color:#d0d0d0;}}
#distPanel h4{{font-size:9px;text-transform:uppercase;letter-spacing:1px;
  color:#a0a0a0;margin-bottom:6px;border-bottom:1px solid #3a3a3a;padding-bottom:4px;}}
.dist-hist-wrap{{margin-bottom:10px;}}
.dist-label{{font-size:8px;color:#a0a0a0;margin-bottom:2px;}}
#iatCanvas,#svcCanvas{{width:100%;height:52px;border-radius:4px;
  background:#242424;display:block;}}
.rho-bar-bg{{height:6px;background:#3a3a3a;border-radius:3px;margin:3px 0;}}
.formula-line{{font-size:9px;color:#808080;margin:2px 0;}}

/* ── Timeline ─────────────────────────────────────────── */
#timelineWrap{{flex-shrink:0;height:44px;background:#1a1a1a;border-top:1px solid #404040;position:relative;}}
#timelineCanvas{{width:100%;height:100%;}}

/* ── Main canvas ──────────────────────────────────────── */
#main{{flex:1;position:relative;overflow:hidden;min-height:0;}}
#canvasWrap{{width:100%;height:100%;position:absolute;inset:0;padding-right:200px;}}

/* ── Split labels ─────────────────────────────────────── */
#splitLabel{{display:none;position:absolute;top:6px;left:0;width:calc(100%-200px);
  pointer-events:none;z-index:5;}}
#splitLabel.on{{display:flex;justify-content:space-around;}}
.split-lbl{{font-size:11px;font-family:monospace;color:#82aaff;
  background:rgba(26,26,26,0.9);padding:3px 10px;border-radius:20px;
  border:1px solid #505050;}}

/* ── Overlays ─────────────────────────────────────────── */
#alertBanner{{display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);
  background:rgba(239,83,80,0.92);color:#fff;padding:5px 18px;
  border-radius:20px;font-size:11px;font-weight:700;
  border:1px solid #ef5350;z-index:10;white-space:nowrap;pointer-events:none;
  animation:abPulse 1s infinite alternate;}}
#pauseOverlay{{display:none;position:absolute;inset:0;
  background:rgba(26,26,26,0.8);backdrop-filter:blur(2px);
  z-index:8;align-items:center;justify-content:center;flex-direction:column;gap:8px;}}
#pauseOverlay.visible{{display:flex;}}
#pauseOverlay span{{font-size:26px;font-weight:700;color:#82aaff;letter-spacing:4px;}}
#pauseOverlay small{{font-size:10px;color:#a0a0a0;}}
#stressLabel{{display:none;position:absolute;bottom:50px;left:14px;
  background:rgba(255,183,77,0.15);border:1px solid #ffb74d;border-radius:6px;
  padding:4px 10px;font-size:11px;color:#ffb74d;z-index:6;}}
#stressLabel.on{{display:block;}}
@keyframes abPulse{{from{{box-shadow:0 0 6px rgba(239,83,80,.4);}}to{{box-shadow:0 0 18px rgba(239,83,80,.9);}}}}
</style>
</head>
<body>

<!-- Info bar -->
<div id="infoBar">
  <div class="info-item">
    <div class="info-label">λ — Arrival</div>
    <div class="info-value" id="ib_lambda">{eff_lambda:.1f} <span class="info-unit">req/s</span></div>
    <div class="info-sub">Distribusi Poisson</div>
  </div>
  <div class="info-item">
    <div class="info-label">μ — Service</div>
    <div class="info-value">{mu:.1f} <span class="info-unit">req/s</span></div>
    <div class="info-sub">Dist. Eksponensial</div>
  </div>
  <div class="info-item">
    <div class="info-label">Workers (c)</div>
    <div class="info-value">{num_workers}</div>
    <div class="info-sub">Paralel aktif</div>
  </div>
  <div class="info-item">
    <div class="info-label">Algoritma</div>
    <div class="info-value" id="ib_algo" style="font-size:13px;color:#7c3aed">{algo_labels.get(algorithm, algorithm)}</div>
    <div class="info-sub" id="ib_algo2">{'Split mode aktif' if split_mode else 'Single mode'}</div>
  </div>
  <div class="info-item" style="flex:.7">
    <div class="info-label">ρ Utilisasi</div>
    <div class="info-value" id="ib_rho" style="color:{rho_color}">{rho:.2f}</div>
    <div class="info-sub" id="ib_rho_sub">{rho_sub}</div>
  </div>
</div>

<!-- Formula bar -->
<div id="formulaBar">
  <span>ρ = λ/(c·μ)</span>
  <span class="sep">|</span>
  <span class="rhoVal" id="fb_rho">{rho:.3f}</span>
  <span class="sep">|</span>
  <span>L = λW</span>
  <span class="sep">|</span>
  <span>W = Wq + 1/μ</span>
  <span class="sep">|</span>
  <span>M/M/c Queue Model</span>
</div>

<!-- RT Stats -->
<div id="rtStats">
  <div class="rt-item"><div class="rt-label">Total Req</div><div class="rt-val" id="rt_total" style="color:#2563eb">0</div></div>
  <div class="rt-item"><div class="rt-label">Throughput</div><div class="rt-val" id="rt_tput" style="color:#16a34a">0.0/s</div></div>
  <div class="rt-item"><div class="rt-label">Avg Wait</div><div class="rt-val" id="rt_wait" style="color:#d97706">0.00s</div></div>
  <div class="rt-item"><div class="rt-label">In Queue</div><div class="rt-val" id="rt_queue" style="color:#dc2626">0</div></div>
  <div class="rt-item"><div class="rt-label">Little's L</div><div class="rt-val" id="rt_L" style="color:#7c3aed">0.00</div></div>
  <div class="rt-item"><div class="rt-label">Completed</div><div class="rt-val" id="rt_done" style="color:#16a34a">0</div></div>
  <div style="margin-left:auto;font-size:9px;color:#9aa4c8;font-family:monospace">LIVE</div>
</div>

<!-- Main area -->
<div id="main">
  <div id="canvasWrap">
    <div id="splitLabel" class="{'on' if split_mode else ''}">
      <div class="split-lbl" id="sl_left">◀ {algo_labels.get(algo_left, algo_left)}</div>
      <div class="split-lbl" id="sl_right">{algo_labels.get(algo_right, algo_right)} ▶</div>
    </div>
    <div id="alertBanner">🔴 SISTEM KRITIS — ρ ≥ 1!</div>
    <div id="pauseOverlay" class="{'visible' if paused else ''}">
      <span>⏸ PAUSED</span>
      <small>Lanjutkan dari sidebar</small>
    </div>
    <div id="stressLabel" class="{'on' if stress_lambda is not None else ''}">
      📈 Stress Test — λ = <span id="stress_lam_val">{eff_lambda:.1f}</span> req/s
    </div>
  </div>

  <!-- Distribution panel -->
  <div id="distPanel">
    <h4>Distribusi Stokastik</h4>
    <div class="dist-hist-wrap">
      <div class="dist-label">Inter-arrival time (IAT)</div>
      <canvas id="iatCanvas"></canvas>
      <div class="formula-line">~ Exp(λ) → Poisson arrivals</div>
    </div>
    <div class="dist-hist-wrap">
      <div class="dist-label">Service time</div>
      <canvas id="svcCanvas"></canvas>
      <div class="formula-line">~ Exp(μ) → M/M/c model</div>
    </div>
    <div style="margin-top:6px">
      <div class="formula-line">ρ = λ/(c·μ)</div>
      <div class="rho-bar-bg">
        <div id="rhoBars" style="height:6px;border-radius:3px;background:#16a34a;width:{min(rho,1)*100:.0f}%"></div>
      </div>
      <div class="formula-line" id="rhoValText">ρ = {rho:.3f} {'⚠' if rho > 0.8 else ''}</div>
    </div>
    <div style="margin-top:8px;border-top:1px solid #e8eeff;padding-top:6px">
      <div class="formula-line">Little's Law: L = λ·W</div>
      <div id="littles_live" style="font-size:11px;color:#2563eb;font-weight:700;margin-top:2px">L = –</div>
    </div>
    <!-- IAT timeline dots -->
    <div style="margin-top:8px;border-top:1px solid #e8eeff;padding-top:6px">
      <div class="formula-line">Arrival timeline (Poisson)</div>
      <canvas id="iatTimelineCanvas" style="width:100%;height:24px;background:#f0f4ff;border-radius:3px;display:block;margin-top:3px;"></canvas>
    </div>
    <!-- Per-worker service time -->
    <div style="margin-top:8px;border-top:1px solid #e8eeff;padding-top:6px">
      <div class="formula-line">Service time per worker</div>
      <div id="workerSvcList" style="font-size:9px;color:#374151;line-height:1.8;margin-top:2px;"></div>
    </div>
  </div>
</div>

<!-- Timeline -->
<div id="timelineWrap">
  <canvas id="timelineCanvas"></canvas>
</div>

<script>
const lambda0={lambda_rate}, mu={mu}, numWorkers={num_workers};
const algorithm="{algorithm}", algoLeft="{algoL_js}", algoRight="{algoR_js}";
const splitMode={split_js};
let paused={init_paused};
let stressLambda={stress_lam_js};
let stressStart=null, stressStep=0;
const STRESS_STEPS=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5];
const STRESS_INTERVAL=1.0; // seconds per step

// ── State ──────────────────────────────────────────────────
let totalReq=0, completed=0, waitSamples=[], totalWaitSum=0;
let simStart=performance.now(), simTime=0;
let currentLambda=stressLambda!==null?stressLambda:lambda0;

// ── Stochastic data ────────────────────────────────────────
let iatHistory=[]; // inter-arrival times
let svcHistory=[]; // service times (all workers)
let workerSvcTimes=Array.from({{length:numWorkers}},()=>[]); // per worker
let tlHistory=[];  // utilisasi timeline
let tlLastT=0;
let iatBins=new Array(20).fill(0);
let svcBins=new Array(20).fill(0);
let lastArriveTime=0;
const MAX_HIST=200;

// ── Canvas setup ───────────────────────────────────────────
const tlCanvas=document.getElementById('timelineCanvas');
const tlCtx=tlCanvas.getContext('2d');
const iatCanvas=document.getElementById('iatCanvas');
const iatCtx=iatCanvas.getContext('2d');
const svcCanvas=document.getElementById('svcCanvas');
const svcCtx=svcCanvas.getContext('2d');
const iatTlCanvas=document.getElementById('iatTimelineCanvas');
const iatTlCtx=iatTlCanvas.getContext('2d');

function resizeCanvases(){{
  [tlCanvas,iatCanvas,svcCanvas,iatTlCanvas].forEach(c=>{{
    c.width=c.offsetWidth*2; c.height=c.offsetHeight*2;
    c.getContext('2d').scale(2,2);
  }});
}}
resizeCanvases();
window.addEventListener('resize',resizeCanvases);

// ── Draw helpers ───────────────────────────────────────────
function drawHistogram(ctx, bins, color, W, H){{
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#f0f4ff'; ctx.fillRect(0,0,W,H);
  const maxB=Math.max(...bins,1);
  const bw=W/bins.length;
  bins.forEach((b,i)=>{{
    const bh=(b/maxB)*(H-4);
    ctx.fillStyle=color+'aa';
    ctx.fillRect(i*bw+1,H-bh,bw-2,bh);
  }});
  // Theoretical Exp curve overlay
  const rate=color==="#2563eb"?currentLambda:mu;
  ctx.strokeStyle=color; ctx.lineWidth=1.5;
  ctx.beginPath();
  for(let x=0;x<=W;x+=2){{
    const t=x/W*4;
    const y=rate*Math.exp(-rate*t);
    const yn=Math.max(0,(y/rate)*(H-4));
    x===0?ctx.moveTo(x,H-yn):ctx.lineTo(x,H-yn);
  }}
  ctx.stroke();
}}

function drawTimeline(){{
  const W=tlCanvas.offsetWidth, H=tlCanvas.offsetHeight;
  tlCtx.clearRect(0,0,W,H);
  tlCtx.fillStyle='#f0f4ff'; tlCtx.fillRect(0,0,W,H);
  if(tlHistory.length<2) return;
  const maxU=Math.max(1.5,...tlHistory.map(h=>h.util));
  const scaleY=u=>H-(u/maxU)*(H*0.85)-H*0.07;
  tlCtx.beginPath(); tlCtx.moveTo(0,H);
  tlHistory.forEach((h,i)=>{{
    const x=(i/Math.max(tlHistory.length-1,1))*W;
    tlCtx.lineTo(x,scaleY(h.util));
  }});
  tlCtx.lineTo(W,H); tlCtx.closePath();
  const grad=tlCtx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'rgba(74,111,165,0.35)');
  grad.addColorStop(1,'rgba(74,111,165,0.03)');
  tlCtx.fillStyle=grad; tlCtx.fill();
  tlCtx.beginPath();
  tlHistory.forEach((h,i)=>{{
    const x=(i/Math.max(tlHistory.length-1,1))*W;
    i===0?tlCtx.moveTo(x,scaleY(h.util)):tlCtx.lineTo(x,scaleY(h.util));
  }});
  tlCtx.strokeStyle='#4a6fa5'; tlCtx.lineWidth=1.5; tlCtx.stroke();
  const y1=scaleY(1.0);
  tlCtx.setLineDash([4,4]); tlCtx.strokeStyle='rgba(220,38,38,0.6)'; tlCtx.lineWidth=1;
  tlCtx.beginPath(); tlCtx.moveTo(0,y1); tlCtx.lineTo(W,y1); tlCtx.stroke();
  tlCtx.setLineDash([]);
  tlCtx.fillStyle='rgba(220,38,38,0.7)'; tlCtx.font='8px monospace';
  tlCtx.fillText('ρ=1',4,y1-3);
  tlCtx.fillStyle='#6b7a99'; tlCtx.font='8px monospace';
  tlCtx.fillText('UTILISASI TIMELINE',4,H-4);
}}

function drawIATTimeline(){{
  const W=iatTlCanvas.offsetWidth, H=iatTlCanvas.offsetHeight;
  iatTlCtx.clearRect(0,0,W,H);
  iatTlCtx.fillStyle='#f0f4ff'; iatTlCtx.fillRect(0,0,W,H);
  const last=iatHistory.slice(-30);
  if(last.length<2) return;
  let cum=0;
  const total=last.reduce((a,b)=>a+b,0);
  iatTlCtx.strokeStyle='#6b7a99'; iatTlCtx.lineWidth=0.5;
  iatTlCtx.beginPath(); iatTlCtx.moveTo(0,H/2); iatTlCtx.lineTo(W,H/2); iatTlCtx.stroke();
  last.forEach(iat=>{{
    cum+=iat;
    const x=(cum/total)*W;
    iatTlCtx.fillStyle='#2563eb';
    iatTlCtx.beginPath(); iatTlCtx.arc(x,H/2,3,0,Math.PI*2); iatTlCtx.fill();
    const barH=Math.min(iat/2*H,H/2-2);
    iatTlCtx.fillStyle='#2563eb88';
    iatTlCtx.fillRect(x-1,H/2,2,barH);
  }});
}}

// ── Worker state ───────────────────────────────────────────
function makeWorkers(n){{
  return Array.from({{length:n}},(_,i)=>{{
    const wy=0; // set in draw
    return {{id:i,queue:0,busy:false,busyTimer:0,processed:0,
      pendingQueue:[],busyFrac:0,lastSvcTime:0,
      ledPhase:Math.random()*Math.PI*2,
      fanAngle:Math.random()*Math.PI*2,
      diskAngle:Math.random()*Math.PI*2,
      heatLevel:0}};
  }});
}}

let workers=makeWorkers(numWorkers);
let workers2=makeWorkers(numWorkers);
let packets=[],packets2=[],trails=[],trails2=[];
let nextSpawn=0,nextSpawn2=0,reqId=0,reqId2=0;
let lineAnimOffset=0;
const expRandom=rate=>-Math.log(Math.random())/rate;

function selectWorker(algo,rid,wArr){{
  if(algo==='round_robin')      return rid%wArr.length;
  if(algo==='random')           return Math.floor(Math.random()*wArr.length);
  return wArr.reduce((a,b)=>b.queue<a.queue?b:a).id;
}}

// ── Update stats ───────────────────────────────────────────
function updateStats(){{
  const elapsed=(performance.now()-simStart)/1000;
  const tput=elapsed>0?(totalReq/elapsed):0;
  const avgW=waitSamples.length>0?(totalWaitSum/waitSamples.length):0;
  const inQ=workers.reduce((s,w)=>s+w.queue,0);
  const L=tput*(avgW+1/mu);
  document.getElementById('rt_total').textContent=totalReq;
  document.getElementById('rt_tput').textContent=tput.toFixed(1)+'/s';
  document.getElementById('rt_wait').textContent=avgW.toFixed(2)+'s';
  document.getElementById('rt_queue').textContent=inQ;
  document.getElementById('rt_L').textContent=L.toFixed(2);
  document.getElementById('rt_done').textContent=completed;
  document.getElementById('littles_live').textContent=`L = ${{tput.toFixed(2)}} × ${{(avgW+1/mu).toFixed(2)}} = ${{L.toFixed(2)}}`;

  if(elapsed-tlLastT>=0.5){{
    tlLastT=elapsed;
    const util=Math.min(inQ/numWorkers+workers.filter(w=>w.busy).length/numWorkers,2);
    tlHistory.push({{t:elapsed,util}});
    if(tlHistory.length>200) tlHistory.shift();
    drawTimeline();
    // Draw histograms
    drawHistogram(iatCtx,iatBins,'#2563eb',iatCanvas.offsetWidth,iatCanvas.offsetHeight);
    drawHistogram(svcCtx,svcBins,'#0891b2',svcCanvas.offsetWidth,svcCanvas.offsetHeight);
    drawIATTimeline();
    // Rho bar update
    const rho=currentLambda/(numWorkers*mu);
    const rhoCol=rho<0.7?'#16a34a':rho<1?'#d97706':'#dc2626';
    document.getElementById('rhoBars').style.width=Math.min(rho,1)*100+'%';
    document.getElementById('rhoBars').style.background=rhoCol;
    document.getElementById('rhoValText').textContent=`ρ = ${{rho.toFixed(3)}} ${{rho>1?'⚠ KRITIS':rho>0.8?'⚠ tinggi':''}}`;
    document.getElementById('rhoValText').style.color=rhoCol;
    // Worker svc times
    const wList=document.getElementById('workerSvcList');
    wList.innerHTML=workerSvcTimes.map((arr,i)=>{{
      const avg=arr.length>0?(arr.slice(-10).reduce((a,b)=>a+b,0)/Math.min(arr.length,10)):0;
      return `W${{i}}: <span style="color:#2563eb">${{avg.toFixed(3)}}s</span> avg (Exp μ=${{mu}})`;
    }}).join('<br>');
  }}
}}

// ── Stress test logic ──────────────────────────────────────
function updateStress(dt){{
  if(stressLambda===null) return;
  if(stressStart===null) stressStart=0;
  stressStart+=dt;
  const step=Math.min(Math.floor(stressStart/STRESS_INTERVAL),STRESS_STEPS.length-1);
  currentLambda=STRESS_STEPS[step];
  // Update info bar
  document.getElementById('ib_lambda').innerHTML=currentLambda.toFixed(1)+' <span class="info-unit">req/s</span>';
  const stressEl=document.getElementById('stressLabel');
  if(stressEl){{
    stressEl.style.display='block';
    document.getElementById('stress_lam_val').textContent=currentLambda.toFixed(1);
  }}
  const rho=currentLambda/(numWorkers*mu);
  const rc=rho<0.7?'#16a34a':rho<1?'#d97706':'#dc2626';
  document.getElementById('ib_rho').textContent=rho.toFixed(2);
  document.getElementById('ib_rho').style.color=rc;
  document.getElementById('fb_rho').textContent=rho.toFixed(3);
  document.getElementById('fb_rho').style.color=rc;
}}

// ── p5 sketch ──────────────────────────────────────────────
new p5(function(p){{
  let W,H;

  p.setup=function(){{
    const wrap=document.getElementById('canvasWrap');
    W=wrap.offsetWidth||700; H=wrap.offsetHeight||420;
    let cv=p.createCanvas(W,H); cv.parent('canvasWrap');
    p.textFont('monospace');
    new ResizeObserver(()=>{{
      const wr=document.getElementById('canvasWrap');
      const nw=wr.offsetWidth,nh=wr.offsetHeight;
      if(nw>10&&nh>10){{W=nw;H=nh;p.resizeCanvas(W,H);}}
    }}).observe(document.getElementById('canvasWrap'));
  }};

  p.draw=function(){{
    if(paused) return;
    const dt=Math.min(p.deltaTime/1000,0.05);
    lineAnimOffset=(lineAnimOffset+dt*28)%12;
    updateStress(dt);
    p.background(240,244,255);
    drawGrid(p,W,H);

    const rho=currentLambda/(numWorkers*mu);
    document.getElementById('alertBanner').style.display=rho>=1?'block':'none';
    if(rho>=1){{
      const pulse=(Math.sin(p.millis()/300)+1)/2;
      p.noStroke(); p.fill(220,38,38,pulse*12); p.rect(0,0,W,H);
    }}

    if(splitMode){{
      p.stroke(200,210,230); p.strokeWeight(1.5);
      p.line(W/2,0,W/2,H);
      drawPanel(p,0,0,W/2,H,workers,packets,trails,algoLeft,reqId,false,dt);
      drawPanel(p,W/2,0,W/2,H,workers2,packets2,trails2,algoRight,reqId2,true,dt);
      nextSpawn-=dt; nextSpawn2-=dt;
      spawnIfNeeded(0,W/2,H,false,dt);
      spawnIfNeeded(W/2,W/2,H,true,dt);
    }} else {{
      drawPanel(p,0,0,W,H,workers,packets,trails,algorithm,reqId,false,dt);
      nextSpawn-=dt;
      spawnIfNeeded(0,W,H,false,dt);
    }}
    updateStats();
  }};

  function spawnIfNeeded(offsetX,panelW,panelH,isRight,dt){{
    const ns=isRight?nextSpawn2:nextSpawn;
    if(ns<=0){{
      const wArr=isRight?workers2:workers;
      const rid=isRight?reqId2:reqId;
      const wid=selectWorker(isRight?algoRight:algoLeft,rid,wArr);
      const genX=offsetX+panelW*0.10, genY=panelH/2;
      const wX=offsetX+panelW*0.70;
      const wSpacing=panelH/(numWorkers+1);
      const wy=wSpacing*(wid+1);
      const iat=expRandom(currentLambda);
      // Record IAT for histogram
      if(!isRight){{
        iatHistory.push(iat);
        if(iatHistory.length>MAX_HIST) iatHistory.shift();
        const bin=Math.min(Math.floor(iat*5),iatBins.length-1);
        iatBins[bin]++;
        totalReq++;
        waitSamples.push(0);
      }}
      const pk={{x:genX,y:genY,tx:wX,ty:wy,wid,phase:'toLB',alpha:255,
        id:rid,size:9,arriveT:performance.now(),waitHeat:0,progress:0}};
      if(isRight){{packets2.push(pk);workers2[wid].queue++;reqId2++;nextSpawn2=iat;
        if(!workers2[wid].busy){{workers2[wid].busy=true;
          const svc=expRandom(mu);workers2[wid].busyTimer=svc;workers2[wid].lastSvcTime=svc;}}
        else workers2[wid].pendingQueue.push({{arriveT:performance.now()}});
      }}else{{packets.push(pk);workers[wid].queue++;reqId++;nextSpawn=iat;
        const svcT=expRandom(mu);
        if(!workers[wid].busy){{workers[wid].busy=true;workers[wid].busyTimer=svcT;
          workers[wid].lastSvcTime=svcT;workers[wid].diskAngle+=0.1;
          svcBins[Math.min(Math.floor(svcT*3),svcBins.length-1)]++;
          workerSvcTimes[wid].push(svcT);if(workerSvcTimes[wid].length>50) workerSvcTimes[wid].shift();
        }} else workers[wid].pendingQueue.push({{arriveT:performance.now()}});
      }}
    }}
  }}

  function drawPanel(p,offX,offY,panelW,panelH,wArr,pkts,trls,algo,_rid,isRight,dt){{
    const lbX=offX+panelW*0.30, lbY=panelH/2;
    const genX=offX+panelW*0.10, genY=panelH/2;
    const wX=offX+panelW*0.70;
    const wSpacing=panelH/(numWorkers+1);

    // Update workers
    for(let w of wArr){{
      w.fanAngle+=(w.busy?5:0.5)*dt;
      w.diskAngle+=(w.busy?3:0.2)*dt;
      w.ledPhase+=dt*2;
      w.heatLevel=w.heatLevel*0.98+(w.busy?0.02:0);
      if(w.busy){{
        w.busyTimer-=dt;
        if(w.busyTimer<=0){{
          w.busy=false; w.queue=Math.max(0,w.queue-1);
          if(!isRight){{completed++;totalWaitSum+=0;}}
          w.processed++;
          if(w.pendingQueue.length>0){{
            const pq=w.pendingQueue.shift();
            const waited=(performance.now()-pq.arriveT)/1000;
            if(!isRight) totalWaitSum+=waited;
            const svc=expRandom(mu);
            w.busy=true;w.busyTimer=svc;w.lastSvcTime=svc;
            if(!isRight){{svcBins[Math.min(Math.floor(svc*3),svcBins.length-1)]++;
              workerSvcTimes[w.id].push(svc);if(workerSvcTimes[w.id].length>50)workerSvcTimes[w.id].shift();}}
          }}
        }}
      }}
    }}

    // Trails
    for(let t2 of trls){{t2.alpha-=8;}}
    trls.splice(0,trls.length,...trls.filter(t2=>t2.alpha>0));

    // Connections: gen → lb dashed
    drawAnimDash(p,genX+20,genY,lbX-18,lbY,lineAnimOffset);

    // Pipes lb → workers (bezier, thickness = queue)
    for(let i=0;i<numWorkers;i++){{
      const wy=wSpacing*(i+1);
      const w=wArr[i];
      const load=Math.min(w.queue/5,1);
      const thick=2+w.queue*2.5;
      const r=Math.round(22+load*193), g=Math.round(163-load*131), b=Math.round(74-load*74);
      p.stroke(r,g,b,60); p.strokeWeight(thick); p.noFill();
      p.beginShape();
      p.vertex(lbX+18,lbY);
      p.bezierVertex(lbX+60,lbY,wX-60,wy,wX-44,wy);
      p.endShape();
      p.stroke(r,g,b,180); p.strokeWeight(1.2);
      p.beginShape();
      p.vertex(lbX+18,lbY);
      p.bezierVertex(lbX+60,lbY,wX-60,wy,wX-44,wy);
      p.endShape();
      // Flow dots inside pipe
      for(let d=0;d<3;d++){{
        const phase=((p.millis()/1200+d/3+i*0.5)%1);
        const bx=p.bezierPoint(lbX+18,lbX+60,wX-60,wX-44,phase);
        const by=p.bezierPoint(lbY,lbY,wy,wy,phase);
        p.noStroke(); p.fill(r,g,b,160); p.circle(bx,by,4);
      }}
      // Queue dots (waiting requests)
      for(let q=0;q<Math.min(w.pendingQueue.length,6);q++){{
        const waited=w.pendingQueue[q]?(performance.now()-w.pendingQueue[q].arriveT)/1000:0;
        const heat=Math.min(waited/4,1);
        const qr=Math.round(22+heat*198),qg=Math.round(163-heat*131),qb=Math.round(74-heat*74);
        const qx=wX-52-(q*11);
        p.noStroke(); p.fill(qr,qg,qb,200); p.circle(qx,wy,8);
      }}
    }}

    // Packets
    for(let pk of pkts){{
      pk.waitHeat=Math.min((performance.now()-pk.arriveT)/3000,1);
      const r=Math.round(37+pk.waitHeat*183),g=Math.round(138-pk.waitHeat*106),b=Math.round(219-pk.waitHeat*183);
      if(pk.phase==='toLB'){{
        pk.x=p.lerp(pk.x,lbX,.09); pk.y=p.lerp(pk.y,lbY,.09);
        if(p.dist(pk.x,pk.y,lbX,lbY)<8) pk.phase='toWorker';
      }} else if(pk.phase==='toWorker'){{
        const wy2=wSpacing*(pk.wid+1);
        pk.progress=Math.min((pk.progress||0)+0.04,1);
        pk.x=p.bezierPoint(lbX+18,lbX+60,wX-60,wX-44,pk.progress);
        pk.y=p.bezierPoint(lbY,lbY,wy2,wy2,pk.progress);
        if(pk.progress>=1) pk.phase='arrive';
      }} else if(pk.phase==='arrive'){{
        pk.size=p.lerp(pk.size,22,.2); pk.alpha-=12;
        if(pk.alpha<=0) pk.phase='done';
      }}
      if(pk.phase!=='done'&&pk.alpha>0){{
        trls.push({{x:pk.x,y:pk.y,alpha:pk.alpha*.3,size:pk.size}});
        p.noStroke();
        p.fill(r,g,b,pk.alpha*.08); p.circle(pk.x,pk.y,pk.size*3.5);
        p.fill(r,g,b,pk.alpha*.2);  p.circle(pk.x,pk.y,pk.size*2);
        p.fill(r,g,b,pk.alpha);      p.circle(pk.x,pk.y,pk.size);
        p.fill(255,pk.alpha*.9); p.textSize(6); p.textAlign(p.CENTER,p.CENTER);
        p.text(pk.id%1000,pk.x,pk.y);
      }}
    }}
    pkts.splice(0,pkts.length,...pkts.filter(pk=>pk.phase!=='done'));

    // Generator (server-like shape)
    drawGenerator(p,genX,genY,algo,offX,panelW);
    // Load balancer
    drawLoadBalancer(p,lbX,lbY,algo);
    // Server racks
    for(let i=0;i<numWorkers;i++) drawServerRack(p,wX,wSpacing*(i+1),wArr[i],i);
  }}

  function drawGenerator(p,x,y,algo,offX,panelW){{
    // Outer pill
    p.fill(219,234,254); p.stroke(37,99,235); p.strokeWeight(1.2);
    p.rect(x-24,y-22,48,44,10);
    // Signal arcs
    for(let r2=1;r2<=3;r2++){{
      const ph=(p.millis()/800+r2*0.4)%1;
      p.noFill(); p.stroke(37,99,235,Math.round((1-ph)*160));
      p.strokeWeight(1.2);
      p.arc(x,y+4,r2*12,r2*12,-p.PI*0.85,-p.PI*0.15);
    }}
    p.fill(37,99,235); p.noStroke(); p.circle(x,y+4,5);
    p.fill(37,99,235); p.textSize(7); p.textAlign(p.CENTER,p.CENTER);
    p.text('REQ GEN',x,y-10);
    p.fill(107,122,153); p.textSize(7);
    p.text(`λ=${{currentLambda.toFixed(1)}}/s`,x,y+14);
    // IAT mini bars (last 6)
    const last6=iatHistory.slice(-6);
    last6.forEach((iat,i)=>{{
      const bh=Math.min(iat/2*16,14);
      p.fill(37,99,235,160);
      p.rect(x-18+i*7,y+22,5,-bh,1);
    }});
  }}

  function drawLoadBalancer(p,x,y,algo){{
    // Spinning ring segments
    p.noFill(); p.stroke(124,58,237); p.strokeWeight(1.5);
    for(let seg=0;seg<8;seg++){{
      const a1=(p.TWO_PI/8)*seg+p.millis()/1200;
      const a2=a1+p.TWO_PI/8-0.25;
      p.arc(x,y,46,46,a1,a2);
    }}
    // Core hexagon
    p.fill(237,233,254); p.stroke(124,58,237); p.strokeWeight(1.5);
    p.beginShape();
    for(let i=0;i<6;i++){{
      const a=(p.TWO_PI/6)*i-p.PI/6;
      p.vertex(x+20*Math.cos(a),y+20*Math.sin(a));
    }}
    p.endShape(p.CLOSE);
    // Inner lines
    for(let i=0;i<3;i++){{
      const a=(p.TWO_PI/3)*i+p.millis()/2000;
      p.stroke(124,58,237,160); p.strokeWeight(1);
      p.line(x,y,x+13*Math.cos(a),y+13*Math.sin(a));
    }}
    const short={{'round_robin':'RR','random':'RND','least_connection':'LC'}};
    p.fill(91,33,182); p.noStroke(); p.textSize(8); p.textAlign(p.CENTER,p.CENTER);
    p.text('BALANCER',x,y-4);
    p.fill(124,58,237); p.textSize(9); p.textStyle(p.BOLD);
    p.text(short[algo]||algo,x,y+7); p.textStyle(p.NORMAL);
  }}

  function drawServerRack(p,cx,cy,w,idx){{
    const rW=80, rH=56;
    const x0=cx-rW/2, y0=cy-rH/2;

    // Shadow
    p.noStroke(); p.fill(0,0,0,15); p.rect(x0+3,y0+3,rW,rH,8);

    // Chassis body
    const bodyCol=w.busy?[220,230,255]:[240,244,255];
    p.fill(...bodyCol); p.stroke(150,170,210); p.strokeWeight(1.2);
    p.rect(x0,y0,rW,rH,8);

    // Top stripe (rack unit)
    p.fill(200,215,240); p.noStroke();
    p.rect(x0,y0,rW,7,8,8,0,0);

    // Screw holes
    [[x0+6,y0+3.5],[x0+rW-6,y0+3.5]].forEach(([sx,sy])=>{{
      p.fill(160,180,220); p.circle(sx,sy,4);
      p.fill(120,140,180); p.circle(sx,sy,1.5);
    }});

    // Heat glow
    if(w.heatLevel>0.2){{
      p.noStroke(); p.fill(220,38,38,Math.round(w.heatLevel*25));
      p.rect(x0,y0,rW,rH,8);
    }}

    // Fan
    const fanX=x0+rW-16, fanY=cy-4;
    p.fill(200,215,240); p.stroke(150,170,210); p.strokeWeight(0.8);
    p.circle(fanX,fanY,16);
    for(let b=0;b<5;b++){{
      const ba=w.fanAngle+(p.TWO_PI/5)*b;
      p.stroke(120,140,180); p.strokeWeight(1.3);
      p.line(fanX,fanY,fanX+6*Math.cos(ba),fanY+6*Math.sin(ba));
    }}
    p.noStroke(); p.fill(170,185,220); p.circle(fanX,fanY,4);

    // Disk LEDs
    for(let d=0;d<3;d++){{
      const active=w.busy&&Math.sin(w.diskAngle*3+d*1.7)>0.3;
      p.noStroke(); p.fill(active?[0,180,100]:[180,200,230]);
      p.rect(x0+8+d*9,y0+11,6,3,1);
    }}

    // CPU bar
    const cpuLoad=w.busy?(0.4+0.6*Math.abs(Math.sin(w.diskAngle*2))):0.04;
    const bw2=44, bh2=4;
    p.fill(210,220,240); p.noStroke(); p.rect(x0+8,y0+20,bw2,bh2,2);
    const r2=Math.round(22+cpuLoad*193), g2=Math.round(163-cpuLoad*131), b2=74;
    p.fill(r2,g2,b2); p.rect(x0+8,y0+20,bw2*cpuLoad,bh2,2);
    p.fill(107,122,153); p.textSize(6); p.textAlign(p.LEFT,p.CENTER);
    p.text('CPU',x0+8,y0+15);

    // Service progress bar
    if(w.busy&&w.lastSvcTime>0){{
      const prog=1-(w.busyTimer/w.lastSvcTime);
      p.fill(210,220,240); p.noStroke(); p.rect(x0+8,y0+rH-14,50,4,2);
      p.fill(14,165,233); p.rect(x0+8,y0+rH-14,50*prog,4,2);
      p.fill(107,122,153); p.textSize(5.5); p.textAlign(p.LEFT,p.CENTER);
      p.text(`Exp(μ)=${{w.lastSvcTime.toFixed(2)}}s`,x0+8,y0+rH-21);
    }}

    // Status LED
    const ledPulse=(Math.sin(p.millis()/200+w.ledPhase)+1)/2;
    const ledR=w.busy?Math.round(22+w.heatLevel*193):22;
    const ledG=w.busy?Math.round(163-w.heatLevel*131):163;
    const ledB=w.busy?74:74;
    p.noStroke(); p.fill(ledR,ledG,ledB,w.busy?80+ledPulse*120:60);
    p.circle(x0+rW-8,y0+rH-8,7);
    p.fill(ledR,ledG,ledB); p.circle(x0+rW-8,y0+rH-8,4);

    // Labels
    p.fill(30,42,74); p.noStroke(); p.textSize(8); p.textAlign(p.LEFT,p.CENTER);
    p.text(`SRV-${{idx}}`,x0+8,y0+rH-23);
    p.fill(w.busy?[217,119,6]:[22,163,74]); p.textSize(7);
    p.text(w.busy?'PROCESSING':'IDLE',x0+8,y0+rH-14);

    // Queue badge
    if(w.queue>0){{
      const badgeR=8+(w.queue>3?2:0);
      p.fill(w.queue>=4?[220,38,38]:[217,119,6]);
      p.noStroke(); p.circle(x0-8,y0+10,badgeR*2);
      p.fill(255); p.textSize(8); p.textAlign(p.CENTER,p.CENTER);
      p.text(w.queue,x0-8,y0+10);
    }}

    // Processed count
    p.fill(107,122,153); p.textSize(7); p.textAlign(p.RIGHT,p.CENTER);
    p.text(`✓${{w.processed}}`,x0+rW-22,y0+rH-8);
  }}

  function drawAnimDash(p,x1,y1,x2,y2,off){{
    const d=p.dist(x1,y1,x2,y2);
    p.stroke(150,170,210,100); p.strokeWeight(1);
    p.drawingContext.setLineDash([5,6]);
    p.line(x1,y1,x2,y2);
    p.drawingContext.setLineDash([]);
    const steps=Math.ceil(d/12);
    for(let i=0;i<steps;i++){{
      const t1=((i*12+off)%d)/d, t2=((i*12+6+off)%d)/d;
      if(t1<0||t1>1||t2<0||t2>1) continue;
      const ax=p.lerp(x1,x2,t1),ay=p.lerp(y1,y2,t1);
      const bx=p.lerp(x1,x2,t2),by=p.lerp(y1,y2,t2);
      p.stroke(37,99,235,200); p.strokeWeight(2);
      p.line(ax,ay,bx,by);
    }}
  }}

  function drawGrid(p,W,H){{
    p.stroke(220,228,242); p.strokeWeight(0.5);
    for(let x=0;x<W;x+=36) p.line(x,0,x,H);
    for(let y=0;y<H;y+=36) p.line(0,y,W,y);
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

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                              leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle('T', parent=styles['Title'],
                                  fontSize=18, spaceAfter=4, textColor=colors.HexColor("#1e2a4a"))
    h1_style    = ParagraphStyle('H1', parent=styles['Heading1'],
                                  fontSize=13, textColor=colors.HexColor("#2563eb"), spaceBefore=14, spaceAfter=4)
    body_style  = ParagraphStyle('B', parent=styles['Normal'],
                                  fontSize=10, leading=16, textColor=colors.HexColor("#374151"))
    mono_style  = ParagraphStyle('M', parent=styles['Code'],
                                  fontSize=9, leading=14, textColor=colors.HexColor("#1d4ed8"),
                                  backColor=colors.HexColor("#eff6ff"), borderPadding=6)

    story.append(Paragraph("Laporan Simulasi Antrian Server", title_style))
    story.append(Paragraph("Pemodelan Stokastik — Model M/M/c Queue", styles['Heading3']))
    story.append(Paragraph(f"Tanggal: {datetime.now().strftime('%d %B %Y, %H:%M')}", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4a6fa5"), spaceAfter=12))

    p = params
    story.append(Paragraph("1. Parameter Simulasi", h1_style))
    param_data = [
        ["Parameter","Nilai","Keterangan"],
        ["λ (Arrival Rate)", f"{p['lambda_rate']:.1f} req/s","Distribusi Poisson"],
        ["μ (Service Rate)", f"{p['mu']:.1f} req/s","Distribusi Eksponensial"],
        ["c (Workers)", str(p['num_workers']),"Server paralel aktif"],
        ["Durasi Simulasi", f"{p['duration']} detik",""],
        ["Algoritma", p['algorithm'].replace("_"," ").title(),"Load balancing strategy"],
        ["ρ = λ/(c·μ)", f"{rho:.4f}","< 1 = stabil" if rho<1 else ">= 1 = tidak stabil"],
    ]
    t1 = Table(param_data, colWidths=[5*cm,4*cm,7*cm])
    t1.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#4a6fa5")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor("#f8faff"),colors.white]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor("#dde3f5")),
        ('FONTSIZE',(0,0),(-1,-1),9),('PADDING',(0,0),(-1,-1),5),
    ]))
    story.append(t1); story.append(Spacer(1,12))

    story.append(Paragraph("2. Hasil Simulasi", h1_style))
    tput=len(df)/duration; avg_wait=df["wait_time"].mean()
    avg_svc=df["service_time"].mean(); avg_total=df["total_time"].mean()
    res_data=[
        ["Metrik","Nilai Empiris","Interpretasi"],
        ["Total Request",str(len(df)),"Request diproses"],
        ["Throughput",f"{tput:.3f} req/s","Request selesai per detik"],
        ["Avg Service Time",f"{avg_svc:.4f}s",f"Teoritis: {1/p['mu']:.4f}s"],
        ["Avg Wait Time (Wq)",f"{avg_wait:.4f}s","Waktu antre sebelum dilayani"],
        ["Avg Total Time (W)",f"{avg_total:.4f}s","Wq + 1/μ"],
        ["Little's L",f"{L:.4f}",f"λ × W = {tput:.3f} × {avg_total:.3f}"],
    ]
    t2=Table(res_data,colWidths=[5.5*cm,4.5*cm,6*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#2563eb")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor("#eff6ff"),colors.white]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor("#dde3f5")),
        ('FONTSIZE',(0,0),(-1,-1),9),('PADDING',(0,0),(-1,-1),5),
    ]))
    story.append(t2); story.append(Spacer(1,12))

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🖥️ Panel Kontrol")

    st.markdown('<div class="section-header">Parameter Stokastik</div>', unsafe_allow_html=True)
    lambda_rate = st.slider("λ — Arrival Rate (req/s)", 0.5, 5.0, 2.0, 0.5)
    st.markdown('<div style="display:flex;justify-content:space-between;color:#808080;font-size:12px;margin-top:-8px;margin-bottom:12px;"><span>0.5</span><span>5.0</span></div>', unsafe_allow_html=True)
    
    mu          = st.slider("μ — Service Rate (req/s)", 0.5, 3.0, 1.0, 0.5)
    st.markdown('<div style="display:flex;justify-content:space-between;color:#808080;font-size:12px;margin-top:-8px;margin-bottom:12px;"><span>0.5</span><span>3.0</span></div>', unsafe_allow_html=True)
    
    num_workers = st.slider("Jumlah Worker (c)", 1, 5, 3)
    st.markdown('<div style="display:flex;justify-content:space-between;color:#808080;font-size:12px;margin-top:-8px;margin-bottom:12px;"><span>1</span><span>5</span></div>', unsafe_allow_html=True)
    
    speed       = st.slider("Speed simulasi", 0.5, 3.0, 1.0, 0.5, format="%.1fx")
    st.markdown('<div style="display:flex;justify-content:space-between;color:#808080;font-size:12px;margin-top:-8px;margin-bottom:12px;"><span>0.5x</span><span>3.0x</span></div>', unsafe_allow_html=True)

    rho = lambda_rate / (num_workers * mu)
    rho_color  = "#16a34a" if rho < 0.7 else "#d97706" if rho < 1 else "#dc2626"
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
        ["round_robin","random","least_connection"],
        format_func=lambda x: {"round_robin":"Round Robin","random":"Random",
                                "least_connection":"Least Connection"}[x],
    )
    duration = st.slider("Durasi Simulasi (detik)", 10, 120, 30, 10)

    st.markdown('<div class="section-header">Kontrol Animasi</div>', unsafe_allow_html=True)
    split_mode = st.toggle("🔀 Split-screen (2 algoritma)", value=False)

    if split_mode:
        st.markdown("**Pilih algoritma perbandingan:**")
        algo_options = {"round_robin":"Round Robin","random":"Random","least_connection":"Least Connection"}
        algo_left  = st.selectbox("◀ Kiri",  list(algo_options.keys()),
                                   format_func=lambda x: algo_options[x], index=0, key="algo_left")
        algo_right = st.selectbox("▶ Kanan", list(algo_options.keys()),
                                   format_func=lambda x: algo_options[x], index=2, key="algo_right")
    else:
        algo_left  = algorithm
        algo_right = "least_connection"

    if st.button("⏸ Pause" if not st.session_state.sim_paused else "▶ Play",
                 use_container_width=True, key="btn_pause"):
        st.session_state.sim_paused = not st.session_state.sim_paused
        st.rerun()

    if st.button("↺ Reset Animasi", use_container_width=True, key="btn_reset"):
        st.session_state.sim_reset_key += 1
        st.session_state.sim_paused = False
        st.session_state.sim_stress = False
        st.rerun()

    stress_label = "⏹ Stop Stress Test" if st.session_state.sim_stress else "📈 Stress Test (bertahap)"
    if st.button(stress_label, use_container_width=True, key="btn_stress"):
        st.session_state.sim_stress = not st.session_state.sim_stress
        st.session_state.sim_reset_key += 1
        st.rerun()

    if st.session_state.sim_stress:
        st.markdown("""<div class="stress-active">
            📈 Stress aktif — λ naik bertahap<br>
            0.5 → 1.0 → 1.5 → … → 5.0 req/s<br>
            (tiap 1 detik, selama 10 langkah)
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    run_btn  = st.button("▶ Jalankan & Tampilkan Hasil", use_container_width=True)
    run_sens = st.button("📊 Analisis Sensitivitas", use_container_width=True)

    st.markdown('<div class="section-header">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#d0d0d0;line-height:1.7">
    <b style="color:#4a6fa5">M/M/c Queue:</b><br>
    • Arrival: Poisson (λ)<br>
    • Service: Eksponensial (μ)<br>
    • c: Workers paralel<br>
    • ρ = λ/(c·μ) &lt; 1 → stabil
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🖥️ Simulasi Stokastik Sistem Antrian Server")
st.markdown("**Pemodelan & Simulasi Stokastik** — M/M/c Queue | Poisson Arrivals | Exponential Service")
st.divider()

tab_guide, tab_anim, tab_hasil, tab_compare, tab_sensitivity = st.tabs([
    "📖 Panduan",
    "🎬 Animasi",
    "📊 Hasil Simulasi",
    "⚖️ Perbandingan Algoritma",
    "🔬 Analisis Sensitivitas",
])


# ════════════════════════════════════════════════════════
# TAB 0 — PANDUAN
# ════════════════════════════════════════════════════════
with tab_guide:
    col_g1, col_g2 = st.columns([1.2, 1])

    with col_g1:
        st.markdown("### 🚀 Cara Menggunakan Aplikasi")
        st.markdown("""
        <div style="background:#242424;border:1px solid #404040;border-radius:12px;padding:20px 24px">
        <div class="guide-step">
          <div class="step-num">1</div>
          <div class="step-content">
            <div class="step-title">Atur parameter di sidebar</div>
            <div class="step-desc">Geser slider <b>λ (arrival rate)</b> untuk mengatur seberapa cepat request datang,
            <b>μ (service rate)</b> untuk kecepatan tiap server, dan <b>c (workers)</b> untuk jumlah server paralel.
            Perhatikan nilai <b>ρ</b> — jika ρ ≥ 1, sistem tidak stabil.</div>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">2</div>
          <div class="step-content">
            <div class="step-title">Lihat animasi real-time di tab Animasi</div>
            <div class="step-desc">Panel kanan menampilkan <b>histogram distribusi stokastik</b> secara live —
            histogram inter-arrival time (IAT) dan service time beserta kurva teoritis Exp(λ/μ) sebagai overlay.
            Lebar pipa = beban antrian. Warna paket biru→merah = semakin lama menunggu.</div>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">3</div>
          <div class="step-content">
            <div class="step-title">Gunakan Split-screen untuk membandingkan algoritma</div>
            <div class="step-desc">Aktifkan toggle <b>Split-screen</b> di sidebar, lalu pilih algoritma untuk
            panel kiri dan kanan secara terpisah. Tersedia: Round Robin, Random, Least Connection.</div>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">4</div>
          <div class="step-content">
            <div class="step-title">Jalankan Stress Test untuk melihat perilaku sistem</div>
            <div class="step-desc">Klik <b>Stress Test</b> — λ akan naik bertahap dari 0.5 → 1.0 → … → 5.0 req/s
            (tiap 1 detik). Perhatikan pipa menebal, warna LED server berubah, dan ρ mendekati kritis.</div>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">5</div>
          <div class="step-content">
            <div class="step-title">Jalankan simulasi dan lihat hasil</div>
            <div class="step-desc">Klik <b>▶ Jalankan & Tampilkan Hasil</b> di sidebar, lalu buka tab
            <b>Hasil Simulasi</b>. Di sana ada metrik lengkap, grafik distribusi, validasi Little's Law,
            dan <b>kesimpulan otomatis</b> yang membandingkan hasil empiris dengan teori Erlang-C M/M/c.</div>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">6</div>
          <div class="step-content">
            <div class="step-title">Bandingkan ketiga algoritma sekaligus</div>
            <div class="step-desc">Tab <b>Perbandingan Algoritma</b> menampilkan ranking otomatis, tabel
            perbandingan, box plot distribusi wait time, dan kurva L vs ρ teoritis vs empiris.</div>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">7</div>
          <div class="step-content">
            <div class="step-title">Jalankan Analisis Sensitivitas</div>
            <div class="step-desc">Tab <b>Analisis Sensitivitas</b> menguji 50 kombinasi λ × c secara otomatis
            dan menampilkan heatmap serta line chart multi-series.</div>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("### 📐 Konsep Kunci")
        st.markdown("""
        <div style="background:#242424;border:1px solid #404040;border-radius:12px;padding:20px 24px;font-size:13px;line-height:1.9;color:#d0d0d0">

        <b style="color:#82aaff">Model M/M/c Queue</b><br>
        Sistem antrian dengan:<br>
        &nbsp;&nbsp;• <b>M</b> = Markovian (Poisson) arrivals<br>
        &nbsp;&nbsp;• <b>M</b> = Markovian (Exponential) service<br>
        &nbsp;&nbsp;• <b>c</b> = jumlah server paralel<br><br>

        <b style="color:#82aaff">Utilisasi ρ</b><br>
        ρ = λ/(c·μ) — proporsi waktu server sibuk<br>
        &nbsp;&nbsp;• ρ &lt; 0.7 → stabil, performa baik<br>
        &nbsp;&nbsp;• 0.7 ≤ ρ &lt; 1 → mendekati kritis<br>
        &nbsp;&nbsp;• ρ ≥ 1 → tidak stabil, antrian tak terbatas<br><br>

        <b style="color:#82aaff">Little's Law</b><br>
        L = λ · W<br>
        &nbsp;&nbsp;• L = avg request dalam sistem<br>
        &nbsp;&nbsp;• λ = throughput aktual<br>
        &nbsp;&nbsp;• W = avg waktu total dalam sistem<br><br>

        <b style="color:#82aaff">Erlang-C Formula</b><br>
        Menghitung probabilitas request harus menunggu
        dan rata-rata waktu tunggu teoritis (Wq).<br><br>

        <b style="color:#82aaff">Algoritma Load Balancing</b><br>
        &nbsp;&nbsp;• <b>Round Robin</b>: distribusi merata bergilir<br>
        &nbsp;&nbsp;• <b>Random</b>: pilihan acak<br>
        &nbsp;&nbsp;• <b>Least Connection</b>: pilih server paling sedikit antrian<br><br>

        <b style="color:#2563eb">Unsur Stokastik</b><br>
        Inter-arrival time dan service time bersifat acak,
        masing-masing mengikuti distribusi eksponensial.
        Histogram di panel animasi menampilkan distribusi
        empiris vs kurva teoritis secara real-time.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎨 Panduan Animasi")
        st.markdown("""
        <div style="background:#242424;border:1px solid #404040;border-radius:12px;padding:16px 20px;font-size:12px;color:#d0d0d0;line-height:2">
        🔵 <b>Paket biru</b> — request baru (belum lama menunggu)<br>
        🔴 <b>Paket merah</b> — request lama (mendekati SLA breach)<br>
        🟢 <b>LED hijau server</b> — server idle<br>
        🔴 <b>LED merah server</b> — server busy/panas<br>
        <b>Pipa tebal</b> — antrian panjang di jalur tersebut<br>
        <b>Dot mengalir di pipa</b> — throughput aktif<br>
        📊 <b>Panel kanan</b> — histogram IAT & service time live<br>
        📈 <b>Timeline bawah</b> — utilisasi sistem historis
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 1 — ANIMASI
# ════════════════════════════════════════════════════════
with tab_anim:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:20px">M/M/{num_workers}</div>
            <div class="metric-sub">Kendall's Notation</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        cls = 'metric-good' if rho<0.7 else 'metric-warn' if rho<1 else 'metric-bad'
        st.markdown(f"""<div class="metric-card {cls}">
            <div class="metric-label">Utilisasi ρ</div>
            <div class="metric-value">{rho:.3f}</div>
            <div class="metric-sub">{rho_status}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        mode_txt = f"Split: {algo_left.replace('_',' ').title()} vs {algo_right.replace('_',' ').title()}" if split_mode else "Single Mode"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Mode</div>
            <div class="metric-value" style="font-size:14px">{mode_txt}</div>
            <div class="metric-sub">{algorithm.replace('_',' ').title()}</div>
        </div>""", unsafe_allow_html=True)

    stress_lambda_val = 0.5 if st.session_state.sim_stress else None

    st.components.v1.html(
        get_animation_html(
            lambda_rate=lambda_rate, num_workers=num_workers, mu=mu,
            algorithm=algorithm, speed=speed,
            paused=st.session_state.sim_paused,
            split_mode=split_mode,
            algo_left=algo_left, algo_right=algo_right,
            stress_lambda=stress_lambda_val,
        ),
        height=660, scrolling=False,
    )


# ════════════════════════════════════════════════════════
# TAB 2 — HASIL SIMULASI
# ════════════════════════════════════════════════════════
with tab_hasil:
    if not run_btn and st.session_state.last_results is None:
        st.markdown("""<div class="tab-intro">
            Klik <b style="color:#2563eb">▶ Jalankan & Tampilkan Hasil</b> di sidebar untuk menjalankan simulasi
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
                duration=duration, algorithm=algorithm)

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
            st.markdown(f'<div class="alert-critical">🔴 <b>SISTEM TIDAK STABIL</b> — ρ = {rho_saved:.3f} ≥ 1.</div>', unsafe_allow_html=True)
        elif rho_saved >= 0.8:
            st.markdown(f'<div class="alert-warn">⚠️ <b>Mendekati kritis</b> — ρ = {rho_saved:.3f}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-stable">✅ <b>Sistem stabil</b> — ρ = {rho_saved:.3f}.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        cols7 = st.columns(7)
        mdata = [
            (cols7[0],"Total Request",len(df),"request",""),
            (cols7[1],"Throughput",f"{tput:.2f}/s","req/s",""),
            (cols7[2],"Avg Service",f"{avg_svc:.3f}s","Exp(μ) empiris",""),
            (cols7[3],"Avg Wait Wq",f"{avg_wait:.3f}s","waktu antre","warn" if avg_wait>0.5 else "good"),
            (cols7[4],"Avg Total W",f"{avg_total:.3f}s","Wq + 1/μ",""),
            (cols7[5],"Utilisasi ρ",f"{rho_saved:.3f}",rho_status,"bad" if rho_saved>=1 else "warn" if rho_saved>=0.8 else "good"),
            (cols7[6],"Little's L",f"{L:.2f}","λW",""),
        ]
        for col, label, value, sub, cls in mdata:
            col.markdown(f"""<div class="metric-card {'metric-'+cls if cls else ''}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── KESIMPULAN KUALITAS ANTRIAN ──────────────────────────────────────
        st.markdown("### 🎯 Kesimpulan: Kualitas Antrian & Validasi Teori")

        theory = erlang_c_metrics(p_saved["lambda_rate"], p_saved["num_workers"], p_saved["mu"])

        # Kualitas antrian
        def queue_quality_verdict(rho, avg_wait, avg_svc, tput, lam):
            if rho >= 1:
                return "bad", "Sistem Tidak Stabil", (
                    f"Dengan ρ = {rho:.3f} ≥ 1, antrian akan terus membesar tanpa batas. "
                    f"Kapasitas server ({p_saved['num_workers']} worker × μ={p_saved['mu']}/s = {p_saved['num_workers']*p_saved['mu']:.1f} req/s) "
                    f"tidak cukup untuk menangani arrival rate λ = {lam:.1f} req/s. "
                    f"<b>Solusi: tambah worker atau tingkatkan μ.</b>"
                )
            elif rho >= 0.85:
                return "warn", "Antrian Mendekati Jenuh", (
                    f"ρ = {rho:.3f} berada di zona kuning (0.85–1.0). Sistem masih stabil namun sensitif terhadap lonjakan traffic. "
                    f"Rata-rata wait time {avg_wait:.3f}s ({avg_wait/avg_svc*100:.0f}% dari service time). "
                    f"Disarankan menambah 1 worker untuk menurunkan ρ ke {lam/((p_saved['num_workers']+1)*p_saved['mu']):.3f}."
                )
            elif rho >= 0.6:
                return "warn", "Antrian Moderat — Performa Cukup", (
                    f"ρ = {rho:.3f}, sistem beroperasi dalam zona aman namun mulai ada antrian. "
                    f"Rata-rata {avg_wait:.3f}s per request dihabiskan untuk menunggu. "
                    f"Throughput aktual {tput:.2f} req/s dari kapasitas teoretis {lam:.1f} req/s."
                )
            else:
                return "good", "Antrian Sehat — Performa Optimal", (
                    f"ρ = {rho:.3f}, sistem beroperasi nyaman di bawah 70% kapasitas. "
                    f"Rata-rata wait time hanya {avg_wait:.3f}s, sangat kecil dibanding service time {avg_svc:.3f}s. "
                    f"Throughput {tput:.2f} req/s sesuai arrival rate {lam:.1f} req/s."
                )

        q_cls, q_title, q_body = queue_quality_verdict(rho_saved, avg_wait, avg_svc, tput, p_saved["lambda_rate"])
        st.markdown(f"""
        <div class="conclusion-card conclusion-{q_cls}">
            <div class="conclusion-title">{'✅' if q_cls=='good' else '⚠️' if q_cls=='warn' else '🔴'} {q_title}</div>
            <div class="conclusion-body">{q_body}</div>
        </div>
        """, unsafe_allow_html=True)

        # Theory vs Empirical table
        if theory:
            st.markdown("#### 📐 Perbandingan: Simulasi vs Teori Erlang-C M/M/c")

            def err_class(err_pct):
                if err_pct < 5:  return "match-good", f"{err_pct:.1f}% ✓"
                if err_pct < 15: return "match-ok",   f"{err_pct:.1f}% ~"
                return "match-bad", f"{err_pct:.1f}% ✗"

            rows = [
                ("Wq (avg wait time)", f"{theory['Wq']:.4f}s", f"{avg_wait:.4f}s",
                 abs(theory['Wq']-avg_wait)/max(theory['Wq'],0.001)*100),
                ("W (avg total time)", f"{theory['W']:.4f}s", f"{avg_total:.4f}s",
                 abs(theory['W']-avg_total)/max(theory['W'],0.001)*100),
                ("L (avg in system)", f"{theory['L']:.4f}", f"{L:.4f}",
                 abs(theory['L']-L)/max(theory['L'],0.001)*100),
                ("ρ (utilisasi)", f"{theory['rho']:.4f}", f"{rho_saved:.4f}",
                 abs(theory['rho']-rho_saved)/max(theory['rho'],0.001)*100),
            ]

            table_rows = ""
            for label, theo_val, emp_val, err in rows:
                cls, err_txt = err_class(err)
                table_rows += f"""<tr>
                    <td>{label}</td>
                    <td>{theo_val}</td>
                    <td>{emp_val}</td>
                    <td class="{cls}">{err_txt}</td>
                </tr>"""

            st.markdown(f"""
            <table class="theory-table">
              <thead><tr>
                <th>Metrik</th><th>Teori (Erlang-C)</th>
                <th>Empiris (Simulasi)</th><th>Error</th>
              </tr></thead>
              <tbody>{table_rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

            # Interpretation
            avg_err = np.mean([r[3] for r in rows])
            if avg_err < 5:
                interp = f"✅ Rata-rata error {avg_err:.1f}% — simulasi <b>sangat konvergen</b> ke teori M/M/c. Model stokastik berjalan dengan benar."
            elif avg_err < 15:
                interp = f"⚠️ Rata-rata error {avg_err:.1f}% — simulasi <b>cukup sesuai</b> teori. Deviasi wajar karena variansi sampel (durasi {p_saved['duration']}s)."
            else:
                interp = f"🔴 Rata-rata error {avg_err:.1f}% — deviasi cukup besar. Coba <b>perpanjang durasi simulasi</b> agar sampel lebih representatif."
            st.markdown(f"<div style='font-size:13px;color:#374151;margin-top:10px;padding:10px 14px;background:#f8faff;border-radius:8px;border:1px solid #dde3f5'>{interp}</div>", unsafe_allow_html=True)

            # Recommendation
            st.markdown("#### 💡 Rekomendasi Kapasitas")
            c_min_stable = math.ceil(p_saved["lambda_rate"] / p_saved["mu"])
            c_rec_70     = math.ceil(p_saved["lambda_rate"] / (0.70 * p_saved["mu"]))
            c_rec_80     = math.ceil(p_saved["lambda_rate"] / (0.80 * p_saved["mu"]))

            col_r1, col_r2, col_r3 = st.columns(3)
            for col, label, val, desc, color in [
                (col_r1, "c minimum (stabil)", c_min_stable, "ρ < 1", "#dc2626"),
                (col_r2, "c optimal (ρ < 0.80)", c_rec_80, "zona aman", "#d97706"),
                (col_r3, "c ideal (ρ < 0.70)", c_rec_70, "performa terbaik", "#16a34a"),
            ]:
                col.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">{val}</div>
                    <div class="metric-sub">{desc}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-critical">
                ⚠️ Sistem tidak stabil (ρ ≥ 1) — teori Erlang-C tidak dapat dihitung.
                Kurangi λ atau tambah worker.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Charts
        col_l, col_r = st.columns(2)
        with col_l:
            df["time_bucket"] = (df["arrive_time"] // 5) * 5
            tput_df = df.groupby("time_bucket").size().reset_index(name="count")
            fig1 = px.area(tput_df, x="time_bucket", y="count",
                           labels={"time_bucket":"Waktu (s)","count":"Jumlah Request"})
            fig1.update_traces(line_color="#4a6fa5", fillcolor="rgba(74,111,165,0.12)")
            plotly_light(fig1, "Request per 5 Detik")
            st.plotly_chart(fig1, use_container_width=True)
        with col_r:
            fig2 = px.histogram(df, x="service_time", nbins=25,
                                labels={"service_time":"Service Time (s)","count":"Frekuensi"})
            fig2.update_traces(marker_color="#4a6fa5", marker_opacity=0.7)
            x_range = np.linspace(0, df["service_time"].max(), 200)
            y_exp   = p_saved["mu"] * np.exp(-p_saved["mu"]*x_range) * len(df) * (df["service_time"].max()/25)
            fig2.add_trace(go.Scatter(x=x_range, y=y_exp, mode="lines",
                                       line=dict(color="#d97706",width=2,dash="dot"),
                                       name="Exp(μ) teoritis"))
            plotly_light(fig2, "Distribusi Service Time + Kurva Teoritis")
            st.plotly_chart(fig2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            worker_df = df.groupby("worker_id").size().reset_index(name="requests")
            fig3 = px.bar(worker_df, x="worker_id", y="requests",
                          color="requests", color_continuous_scale="Blues",
                          labels={"worker_id":"Worker ID","requests":"Jumlah Request"})
            plotly_light(fig3, f"Request per Worker ({p_saved['algorithm'].replace('_',' ').title()})")
            st.plotly_chart(fig3, use_container_width=True)
        with col_r2:
            fig4 = px.scatter(df, x="arrive_time", y="wait_time", color="worker_id",
                              labels={"arrive_time":"Waktu Kedatangan (s)","wait_time":"Wait Time (s)"},
                              color_continuous_scale="Blues")
            plotly_light(fig4, "Wait Time vs Waktu Kedatangan")
            st.plotly_chart(fig4, use_container_width=True)

        # Utilisasi timeline
        df["interval"] = (df["arrive_time"] // 5) * 5
        util_df = (df.groupby("interval")
                     .apply(lambda g: min(len(g)/(5*p_saved["lambda_rate"]), 1.5))
                     .reset_index(name="utilisasi"))
        fig_util = px.area(util_df, x="interval", y="utilisasi",
                           labels={"interval":"Waktu (s)","utilisasi":"Utilisasi Estimasi"})
        fig_util.update_traces(line_color="#4a6fa5", fillcolor="rgba(74,111,165,0.10)")
        fig_util.add_hline(y=1.0, line_dash="dot", line_color="#dc2626",
                           annotation_text="ρ = 1 (batas kritis)", annotation_position="top right")
        fig_util.add_hline(y=rho_saved, line_dash="dash", line_color="#16a34a",
                           annotation_text=f"ρ teoritis = {rho_saved:.3f}", annotation_position="bottom right")
        plotly_light(fig_util, "Estimasi Utilisasi per Interval Waktu")
        st.plotly_chart(fig_util, use_container_width=True)

        # Little's Law
        st.divider()
        st.markdown("### Validasi Little's Law")
        st.markdown(f"""<div class="littles-card">
            <div style="font-size:12px;color:#6b7a99;margin-bottom:8px">L = λ × W (Little's Law)</div>
            <div class="eq">L = {tput:.3f} × {avg_total:.3f} = {L:.3f}</div>
            <div style="font-size:12px;color:#6b7a99;margin-top:8px">
            λ aktual = <b style="color:#2563eb">{tput:.3f} req/s</b> &nbsp;|&nbsp;
            W = <b style="color:#2563eb">{avg_total:.3f}s</b> &nbsp;|&nbsp;
            L = <b style="color:#2563eb">{L:.3f} request</b>
            </div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Export
        st.markdown("### 📄 Export")
        col_pdf, col_csv = st.columns(2)
        with col_pdf:
            if st.button("📄 Generate PDF", use_container_width=True):
                with st.spinner("Membuat PDF..."):
                    pdf_bytes = generate_pdf_report(p_saved, df, all_results, p_saved["duration"], rho_saved, L)
                st.download_button("⬇️ Download PDF", pdf_bytes,
                    f"laporan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf",
                    use_container_width=True)
        with col_csv:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "simulation_results.csv", "text/csv",
                               use_container_width=True)

        st.divider()
        st.markdown("### Log Hasil Simulasi")
        st.dataframe(
            df[["request_id","worker_id","arrive_time","wait_time","service_time","total_time"]],
            use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════
# TAB 3 — PERBANDINGAN ALGORITMA
# ════════════════════════════════════════════════════════
with tab_compare:
    if st.session_state.last_results is None:
        st.markdown("""<div class="tab-intro">
            Jalankan simulasi terlebih dahulu dari sidebar.
        </div>""", unsafe_allow_html=True)
    else:
        _, all_results = st.session_state.last_results
        p_saved = st.session_state.last_params
        dur = p_saved["duration"]
        rho_s = p_saved["lambda_rate"] / (p_saved["num_workers"] * p_saved["mu"])
        ranked = rank_algorithms(all_results, dur, p_saved["lambda_rate"], rho_s)

        st.markdown("### 🏆 Ranking Algoritma")
        medals = ["🥇","🥈","🥉"]
        badge_styles = ["badge-best","badge-fast","badge-warn"]
        badge_labels = ["TERBAIK","RUNNER-UP","KE-3"]
        cols_rank = st.columns(3)
        for i, (algo, stats) in enumerate(ranked):
            with cols_rank[i]:
                st.markdown(f"""<div class="rank-card">
                    <div style="font-size:22px;margin-bottom:4px">{medals[i]}
                    <span class="rank-algo">{algo.replace("_"," ").title()}</span>
                    <span class="rank-badge {badge_styles[i]}">{badge_labels[i]}</span></div>
                    <div style="font-size:12px;color:#6b7a99;font-family:monospace">
                        Throughput: <b style="color:#2563eb">{stats['tput']:.3f}</b> req/s<br>
                        Avg Wait: <b style="color:#d97706">{stats['avg_wait']:.4f}</b>s<br>
                        P99 Wait: <b style="color:#dc2626">{stats['p99_wait']:.4f}</b>s<br>
                        Score: <b style="color:#16a34a">{stats['score']:.3f}</b>
                    </div>
                    <div class="rank-reason">Score = Throughput − 2×AvgWait − 0.5×P99</div>
                </div>""", unsafe_allow_html=True)

        best_algo, best_stats = ranked[0]
        st.markdown(f"""<div class="alert-stable" style="margin-top:16px">
            ✅ <b>Rekomendasi untuk λ={p_saved['lambda_rate']}, c={p_saved['num_workers']}, μ={p_saved['mu']}:</b>
            Gunakan <b>{best_algo.replace('_',' ').title()}</b> —
            throughput {best_stats['tput']:.3f} req/s, avg wait {best_stats['avg_wait']:.4f}s.
            {"Least Connection unggul saat ada perbedaan beban antar worker." if best_algo=="least_connection" else
             "Round Robin efisien saat beban seragam dan service time homogen." if best_algo=="round_robin" else
             "Random memberikan distribusi probabilistik merata pada kondisi ini."}
        </div>""", unsafe_allow_html=True)

        st.divider()
        comparison = []
        for algo, res in all_results.items():
            if res:
                d = pd.DataFrame(res); t = len(d)/dur
                comparison.append({
                    "Algoritma": algo.replace("_"," ").title(),
                    "Total Request": len(d),
                    "Throughput (req/s)": round(t,3),
                    "Avg Service Time (s)": round(d["service_time"].mean(),4),
                    "Avg Wait Time (s)": round(d["wait_time"].mean(),4),
                    "P99 Wait Time (s)": round(d["wait_time"].quantile(0.99),4),
                    "Avg Total Time (s)": round(d["total_time"].mean(),4),
                    "Little's L": round(t*d["total_time"].mean(),3),
                    "Rank": f"#{[r[0] for r in ranked].index(algo)+1}",
                })
        comp_df = pd.DataFrame(comparison)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        col_bar, col_box = st.columns(2)
        with col_bar:
            fig5 = px.bar(comp_df, x="Algoritma", y="Throughput (req/s)", color="Algoritma",
                          color_discrete_sequence=["#4a6fa5","#16a34a","#d97706"])
            plotly_light(fig5, "Throughput per Algoritma")
            st.plotly_chart(fig5, use_container_width=True)
        with col_box:
            fig6 = go.Figure()
            for algo, res in all_results.items():
                if res:
                    d = pd.DataFrame(res)
                    fig6.add_trace(go.Box(y=d["wait_time"], name=algo.replace("_"," ").title(), boxmean=True))
            plotly_light(fig6, "Distribusi Wait Time per Algoritma")
            st.plotly_chart(fig6, use_container_width=True)

        # CI comparison
        st.markdown("### 📊 Confidence Interval (95%, n=8 runs)")
        with st.spinner("Menghitung..."):
            ci_results = {algo: run_simulation_multi(
                p_saved["lambda_rate"], p_saved["num_workers"],
                algo, p_saved["duration"], p_saved["mu"], n_runs=8)
                for algo in ["round_robin","random","least_connection"]}

        ci_labels = {"round_robin":"Round Robin","random":"Random","least_connection":"Least Connection"}
        ci_colors = {"round_robin":"#4a6fa5","random":"#16a34a","least_connection":"#d97706"}
        ci_tabs = st.tabs(["Throughput","Avg Wait Time","Little's L"])
        for (mkey, mlabel, tab_obj) in [
            ("throughput","Throughput (req/s)",ci_tabs[0]),
            ("avg_wait","Avg Wait Time (s)",ci_tabs[1]),
            ("L","Little's L",ci_tabs[2])]:
            with tab_obj:
                fig_ci = go.Figure()
                for algo, ci_data in ci_results.items():
                    if not ci_data: continue
                    fig_ci.add_trace(go.Bar(
                        x=[ci_labels[algo]], y=[ci_data.get(mkey+"_mean",0)],
                        error_y=dict(type="data",array=[ci_data.get(mkey+"_ci95",0)],visible=True),
                        name=ci_labels[algo], marker_color=ci_colors[algo], marker_opacity=0.85))
                plotly_light(fig_ci, f"{mlabel} — Mean ± 95% CI")
                st.plotly_chart(fig_ci, use_container_width=True)

        # L vs rho
        st.markdown("### 📈 L vs ρ — Erlang-C Teoritis vs Empiris")
        c_val, mu_val = p_saved["num_workers"], p_saved["mu"]
        rho_range = np.linspace(0.05, 0.97, 100)
        fig_lrho  = go.Figure()
        for c_plot in sorted(set([1, 2, c_val, min(c_val+1,5)])):
            L_theory, rho_valid = [], []
            for r in rho_range:
                res_t = erlang_c_metrics(r*c_plot*mu_val, c_plot, mu_val)
                if res_t:
                    L_theory.append(res_t["L"]); rho_valid.append(r)
            if L_theory:
                fig_lrho.add_trace(go.Scatter(x=rho_valid, y=L_theory, mode="lines",
                    name=f"c={c_plot} teoritis",
                    line=dict(width=2.5 if c_plot==c_val else 1.5, dash="solid" if c_plot==c_val else "dot")))
        emp_pts = []
        for lam_scan in np.linspace(0.3, c_val*mu_val*0.95, 14):
            runs = [pd.DataFrame(run_simulation(lam_scan, c_val, "round_robin", 30, mu_val, seed=s*7))
                    for s in range(5)]
            runs = [r for r in runs if len(r) > 0]
            if runs:
                vals = [(len(r)/30)*r["total_time"].mean() for r in runs]
                emp_pts.append((lam_scan/(c_val*mu_val), np.mean(vals), np.std(vals)))
        if emp_pts:
            ex,ey,ee = zip(*emp_pts)
            fig_lrho.add_trace(go.Scatter(x=list(ex), y=list(ey),
                error_y=dict(type="data",array=list(ee),visible=True),
                mode="markers+lines", name=f"c={c_val} empiris",
                marker=dict(size=8,color="#16a34a")))
        fig_lrho.add_vline(x=1.0, line_dash="dot", line_color="#dc2626",
                           annotation_text="ρ=1", annotation_position="top right")
        plotly_light(fig_lrho, f"L vs ρ — M/M/c Teoritis vs Empiris (c={c_val}, μ={mu_val})")
        fig_lrho.update_xaxes(title="ρ", range=[0,1.02])
        fig_lrho.update_yaxes(title="L (avg dalam sistem)")
        st.plotly_chart(fig_lrho, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 4 — Analisis Sensitivitas  (versi lengkap)
# Ganti seluruh blok `with tab_sensitivity:` di kode 3 dengan ini
# ════════════════════════════════════════════════════════
with tab_sensitivity:
    st.markdown("""<div class="tab-intro">
        <b style="color:#2563eb">Analisis Sensitivitas</b> menguji semua kombinasi λ × c secara otomatis
        dan menampilkan hasilnya sebagai heatmap dan line chart multi-series.
        Ini membantu memahami bagaimana sistem merespons perubahan beban dan kapasitas.
    </div>""", unsafe_allow_html=True)

    if run_sens:
        lambda_values  = np.arange(0.5, 5.5, 0.5)
        worker_options = [1, 2, 3, 4, 5]
        sens_mu        = mu

        with st.spinner("⏳ Menjalankan Analisis Sensitivitas (semua kombinasi λ × c)..."):
            sens_records = []
            for c in worker_options:
                for lam in lambda_values:
                    rho_s2 = lam / (c * sens_mu)
                    res_s  = run_simulation(lam, c, "round_robin", 30, sens_mu, seed=42)
                    if res_s:
                        d_s = pd.DataFrame(res_s)
                        t_s = len(d_s) / 30
                        sens_records.append({
                            "λ":          round(lam, 1),
                            "Workers (c)": c,
                            "ρ":          round(rho_s2, 3),
                            "Throughput": round(t_s, 3),
                            "Avg Wait":   round(d_s["wait_time"].mean(), 4),
                            "Little's L": round(t_s * d_s["total_time"].mean(), 3),
                            "Stable":     rho_s2 < 1,
                        })

        sens_df = pd.DataFrame(sens_records)
        st.success(f"✅ Selesai — {len(sens_records)} kombinasi diuji.")

        # ── Heatmap Avg Wait Time ────────────────────────────────────────────
        st.markdown("### Heatmap Avg Wait Time (λ × c)")
        pivot_wait = sens_df.pivot(index="Workers (c)", columns="λ", values="Avg Wait")
        fig_h1 = go.Figure(go.Heatmap(
            z=pivot_wait.values,
            x=pivot_wait.columns.astype(str),
            y=pivot_wait.index,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Avg Wait (s)", tickfont=dict(color="#374151")),
            text=np.round(pivot_wait.values, 3),
            texttemplate="%{text}s",
            textfont=dict(size=10),
            hoverongaps=False,
        ))
        fig_h1.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8faff", font_color="#374151",
            xaxis=dict(title="λ (req/s)", gridcolor="#e8eeff"),
            yaxis=dict(title="Workers (c)", gridcolor="#e8eeff"),
            title=dict(
                text="Avg Wait Time Heatmap — merah=tinggi, hijau=rendah",
                font=dict(color="#1e2a4a", size=14, family="JetBrains Mono")
            ),
            margin=dict(l=60, r=30, t=60, b=50),
        )
        st.plotly_chart(fig_h1, use_container_width=True)

        # ── Heatmap Utilisasi ρ ──────────────────────────────────────────────
        st.markdown("### Heatmap Utilisasi ρ (λ × c)")
        pivot_rho = sens_df.pivot(index="Workers (c)", columns="λ", values="ρ")
        fig_h2 = go.Figure(go.Heatmap(
            z=pivot_rho.values,
            x=pivot_rho.columns.astype(str),
            y=pivot_rho.index,
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=1.5,
            colorbar=dict(title="ρ utilisasi", tickfont=dict(color="#374151")),
            text=np.round(pivot_rho.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
        ))
        fig_h2.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8faff", font_color="#374151",
            xaxis=dict(title="λ (req/s)", gridcolor="#e8eeff"),
            yaxis=dict(title="Workers (c)", gridcolor="#e8eeff"),
            title=dict(
                text="Utilisasi ρ Heatmap — merah=kritis (≥1), hijau=stabil",
                font=dict(color="#1e2a4a", size=14, family="JetBrains Mono")
            ),
            margin=dict(l=60, r=30, t=60, b=50),
        )
        st.plotly_chart(fig_h2, use_container_width=True)

        # ── Line Chart: Throughput vs λ ──────────────────────────────────────
        line_colors = ["#4a6fa5", "#16a34a", "#d97706", "#7c3aed", "#dc2626"]

        st.markdown("### Throughput vs λ (multi-series per c)")
        fig_l1 = go.Figure()
        for i, c in enumerate(worker_options):
            sub = sens_df[sens_df["Workers (c)"] == c]
            fig_l1.add_trace(go.Scatter(
                x=sub["λ"], y=sub["Throughput"],
                mode="lines+markers",
                name=f"c={c} workers",
                line=dict(color=line_colors[i], width=2),
                marker=dict(size=7),
            ))
        plotly_light(fig_l1, "Throughput vs λ per Jumlah Worker")
        fig_l1.update_xaxes(title="λ (req/s)")
        fig_l1.update_yaxes(title="Throughput (req/s)")
        st.plotly_chart(fig_l1, use_container_width=True)

        # ── Line Chart: Avg Wait vs λ ────────────────────────────────────────
        st.markdown("### Avg Wait Time vs λ (multi-series per c)")
        fig_l2 = go.Figure()
        for i, c in enumerate(worker_options):
            sub = sens_df[sens_df["Workers (c)"] == c]
            fig_l2.add_trace(go.Scatter(
                x=sub["λ"], y=sub["Avg Wait"],
                mode="lines+markers",
                name=f"c={c} workers",
                line=dict(color=line_colors[i], width=2),
                marker=dict(size=7),
            ))
        plotly_light(fig_l2, "Avg Wait Time vs λ per Jumlah Worker")
        fig_l2.update_xaxes(title="λ (req/s)")
        fig_l2.update_yaxes(title="Avg Wait Time (s)")
        st.plotly_chart(fig_l2, use_container_width=True)

        # ── Tabel Validasi Erlang-C per ρ ────────────────────────────────────
        st.markdown("### 📐 Tabel Validasi Erlang-C per Nilai ρ")
        st.markdown("""<div class="tab-intro" style="font-size:12px;padding:10px 16px">
            Membandingkan hasil simulasi (n=5 run) dengan prediksi teori <b>Erlang-C M/M/c</b>
            pada berbagai nilai ρ. Kolom Error (%) menunjukkan seberapa dekat simulasi ke teori.
            Error &lt;5% = konvergen baik; &gt;15% = perlu durasi simulasi lebih panjang.
        </div>""", unsafe_allow_html=True)

        # gunakan c dan mu dari parameter sidebar saat sensitivity dijalankan
        c_val_sens  = num_workers
        mu_val_sens = mu

        validation_rows = []
        for r_check in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            lam_check = r_check * c_val_sens * mu_val_sens
            theory    = erlang_c_metrics(lam_check, c_val_sens, mu_val_sens)
            sim_runs  = []
            for s in range(5):
                res_v = run_simulation(lam_check, c_val_sens, "round_robin", 30, mu_val_sens, seed=s * 13)
                if res_v:
                    d_v  = pd.DataFrame(res_v)
                    t_v  = len(d_v) / 30
                    sim_runs.append({
                        "W":  d_v["total_time"].mean(),
                        "Wq": d_v["wait_time"].mean(),
                        "L":  t_v * d_v["total_time"].mean(),
                    })
            if theory and sim_runs:
                avg_W_emp  = np.mean([r["W"]  for r in sim_runs])
                avg_Wq_emp = np.mean([r["Wq"] for r in sim_runs])
                avg_L_emp  = np.mean([r["L"]  for r in sim_runs])
                err_L      = abs(theory["L"] - avg_L_emp) / max(theory["L"], 0.001) * 100
                validation_rows.append({
                    "ρ":              r_check,
                    "λ (req/s)":      round(lam_check, 3),
                    "Wq teori (s)":   round(theory["Wq"], 4),
                    "Wq empiris (s)": round(avg_Wq_emp, 4),
                    "W teori (s)":    round(theory["W"], 4),
                    "W empiris (s)":  round(avg_W_emp, 4),
                    "L teori":        round(theory["L"], 3),
                    "L empiris":      round(avg_L_emp, 3),
                    "Error L (%)":    round(err_L, 1),
                })

        if validation_rows:
            val_df = pd.DataFrame(validation_rows)
            st.dataframe(
                val_df.style.background_gradient(subset=["Error L (%)"], cmap="RdYlGn_r")
                            .format({
                                "ρ": "{:.2f}",
                                "λ (req/s)": "{:.3f}",
                                "Wq teori (s)": "{:.4f}",
                                "Wq empiris (s)": "{:.4f}",
                                "W teori (s)": "{:.4f}",
                                "W empiris (s)": "{:.4f}",
                                "L teori": "{:.3f}",
                                "L empiris": "{:.3f}",
                                "Error L (%)": "{:.1f}%",
                            }),
                use_container_width=True,
                hide_index=True,
            )
            # interpretasi rata-rata error
            avg_err_val = val_df["Error L (%)"].mean()
            if avg_err_val < 5:
                val_interp = f"✅ Rata-rata error {avg_err_val:.1f}% — simulasi <b>sangat konvergen</b> ke teori Erlang-C M/M/c."
                val_cls = "alert-stable"
            elif avg_err_val < 15:
                val_interp = f"⚠️ Rata-rata error {avg_err_val:.1f}% — konvergensi <b>cukup baik</b>. Perpanjang durasi simulasi untuk akurasi lebih tinggi."
                val_cls = "alert-warn"
            else:
                val_interp = f"🔴 Rata-rata error {avg_err_val:.1f}% — deviasi tinggi. Gunakan durasi ≥60 detik dan lebih banyak run."
                val_cls = "alert-critical"
            st.markdown(f'<div class="{val_cls}" style="margin-top:10px">{val_interp}</div>', unsafe_allow_html=True)

        # ── Tabel Sensitivity Lengkap + Download ─────────────────────────────
        st.divider()
        st.markdown("### Tabel Analisis Sensitivitas Lengkap")
        st.dataframe(
            sens_df.style
                   .background_gradient(subset=["Avg Wait", "ρ"], cmap="RdYlGn_r")
                   .format({
                       "ρ":          "{:.3f}",
                       "Throughput": "{:.3f}",
                       "Avg Wait":   "{:.4f}",
                       "Little's L": "{:.3f}",
                   }),
            use_container_width=True,
            hide_index=True,
        )

        csv_sens = sens_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Sensitivity CSV",
            csv_sens,
            "sensitivity_analysis.csv",
            "text/csv",
            use_container_width=True,
        )

    else:
        # ── Preview state (belum dijalankan) ─────────────────────────────────
        st.info("Klik **📊 Analisis Sensitivitas** di sidebar untuk memulai analisis.")
        st.markdown("""
        <div style="background:#f8faff;border:1px solid #dde3f5;border-radius:12px;
                    padding:20px 24px;margin-top:16px">
            <div style="font-size:11px;color:#6b7a99;margin-bottom:12px;
                        text-transform:uppercase;letter-spacing:1.5px;
                        font-family:'JetBrains Mono',monospace;
                        border-bottom:1px solid #dde3f5;padding-bottom:6px">
                Preview: Apa yang akan dianalisis
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                        color:#374151;line-height:2.2">
                λ ∈ {0.5, 1.0, 1.5, …, 5.0} — <b style="color:#2563eb">10 nilai</b> arrival rate<br>
                c ∈ {1, 2, 3, 4, 5} — <b style="color:#2563eb">5 konfigurasi</b> worker<br>
                <span style="color:#dde3f5">──────────────────────────────────────</span><br>
                Total: <b style="color:#16a34a">50 kombinasi</b> simulasi<br>
                Output: 2 heatmap + 2 line chart + tabel validasi Erlang-C + CSV export
            </div>
        </div>
        """, unsafe_allow_html=True)